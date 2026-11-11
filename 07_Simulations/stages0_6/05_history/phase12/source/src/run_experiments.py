#!/usr/bin/env python3
"""Execute the frozen phase-1/2 ABM matrix and write restartable root blocks.

python src/run_experiments.py --workers 4
python src/run_experiments.py --workers 4 --resume
python src/run_experiments.py --seeds 2026101201 --workers 1
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse,csv,gzip,hashlib,json,time,sys,platform,datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import numba
import engine
from records import ranks,cohorts,metrics,array_hash,savez,write_csv,density_arrays
ROOT=Path(__file__).resolve().parents[1]
CONFIG_PATH=ROOT/'config/experiment_config.json'
CFG=json.loads(CONFIG_PATH.read_text())

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_hash():
    h=hashlib.sha256()
    for name in ['engine.py','records.py','run_experiments.py']:
        h.update(name.encode());h.update((ROOT/'src'/name).read_bytes())
    return h.hexdigest()

def stream_seed(seed,k,stage):
    return int(np.random.SeedSequence([seed,k,stage]).generate_state(1)[0])

def write_gzip_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    with gzip.open(tmp,'wt',encoding='utf-8',newline='',compresslevel=1) as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    os.replace(tmp,path)

def write_anchor(path,w,a,u,p,r0,meta):
    lab=np.where(r0<len(w)//2,0,np.where(r0>=len(w)-len(w)//100,2,1)).astype(np.uint8)
    savez(path,w=w,a=a,u=u,permutation=p.astype(np.int32),id=np.arange(len(w),dtype=np.int32),
       rank0=r0.astype(np.int32),cohort_code=lab,
       metadata_json=np.array(json.dumps(meta,sort_keys=True)),
       rng_state_json=np.array(json.dumps(engine.rng_state(),sort_keys=True)))

def prepare(seed):
    m=CFG['model']; n=m['N']; args=(m['epsilon'],m['h'],m['mu'],m['r'])
    w2,a,u,p=engine.initialise(n,seed,1);w=w2[:,0].copy()
    states=[];trace=[];j=0
    for rnd in range(1,20001):
        engine.neutral_round(w,a,u,p,*args)
        g=float(engine.gini_numba(w))
        trace.append(dict(root_seed=seed,round=rnd,time=rnd/m['r'],gini=g))
        if j<len(CFG['targets']) and g>=CFG['targets'][j]:
            states.append((w.copy(),a.copy(),u.copy(),p.copy(),rnd,g,engine.rng_state()))
            j+=1
        if j==len(CFG['targets']):break
    if j!=len(CFG['targets']):raise RuntimeError('Neutral targets not reached')
    return states,trace

def compute_root(seed):
    clock=time.perf_counter()
    if seed not in CFG['developer_seeds']+CFG['holdout_seeds']:raise ValueError('Unregistered seed')
    m=CFG['model'];n=m['N'];r=m['r'];args=(m['epsilon'],m['h'],m['mu'],r)
    dev=seed in CFG['developer_seeds'];root_split='development' if dev else 'holdout_locked'
    states,prep_trace=prepare(seed)
    rows={'development':[],'holdout_locked':[]};budgetrows={'development':[],'holdout_locked':[]}
    prep_rows=[];stage_audits=[];checkpoint_paths=[];density_paths=[];micro_paths=[]
    maxmass=0.;max_qerr=0.;max_verr=0.;neg=0;zero=0;ties=0
    cfgsha=digest(CONFIG_PATH); codesha=source_hash()
    for ki,target in enumerate(CFG['targets']):
        nw,na,nu,np_,prepround,prepg,preprng=states[ki]
        prep_rows.append(dict(root_seed=seed,target_G=target,prep_rounds=prepround,prep_time=prepround/r,actual_G=prepg,
            initial_state_hash=array_hash(nw,na,nu,np_)))
        parent_hash=None
        for stage,si in [('selection',1),('tax',2)]:
            params=CFG['lambdas'] if stage=='selection' else CFG['chis']
            la=np.array(params if stage=='selection' else [4.]*len(params),dtype=np.float64)
            ch=np.array([0.]*len(params) if stage=='selection' else params,dtype=np.float64)
            if stage=='selection':w0=nw.copy();a=na.copy();u=nu.copy();p=np_.copy()
            else:w0=anchor_B[0].copy();a=anchor_B[1].copy();u=anchor_B[2].copy();p=anchor_B[3].copy()
            r0=ranks(w0);coh=cohorts(r0);gini0=2*np.dot(np.arange(1,n+1,dtype=float),np.sort(w0))/(n*w0.sum())-(n+1)/n
            ss=stream_seed(seed,ki,si);engine.reseed(ss)
            anchor_hash=array_hash(w0,a,u,p)
            anchor_file=ROOT/'anchors'/root_split/f's{seed}_g{int(target*100):02d}_{stage}.npz'
            meta=dict(root_seed=seed,target_G=target,stage=stage,stream_seed=ss,prep_rounds=prepround,actual_preparation_G=prepg,
              config_sha256=cfgsha,source_sha256=codesha,state_hash=anchor_hash,
              parent_A_lambda4_endpoint_hash=parent_hash,round=0,elapsed_time=0.,
              pre_preparation_rng=preprng if stage=='selection' else None,
              cohort_definition='rank0<n/2; n/2<=rank0<n-n/100; rank0>=n-n/100',
              rank_tie_policy='stable sort by wealth then original ID',
              random_stream_policy='independent stage seed; shared future draws among all parameter branches')
            write_anchor(anchor_file,w0,a,u,p,r0,meta);checkpoint_paths.append(str(anchor_file.relative_to(ROOT)))
            w=np.repeat(w0[:,None],len(params),axis=1)
            assert all(np.array_equal(w[:,b],w0) for b in range(len(params)))
            qpred=np.repeat(float(np.dot(w0-1.,u)/n),len(params))
            vpred=np.repeat(float(np.var(w0)),len(params))
            rep=CFG['record']['representative_lambdas' if stage=='selection' else 'representative_chis']
            repids=[i for i,v in enumerate(params) if v in rep]
            prevrnd=0
            for elapsed in CFG['times']:
                rnd=int(round(elapsed*r))
                if rnd>prevrnd:
                    budgets=engine.advance_chunk(w,a,u,p,la,ch,rnd-prevrnd,*args)
                    for off in range(len(budgets)):
                        for bi,pv in enumerate(params):
                            rho=float(np.exp(-ch[bi]/r))
                            vpred[bi]=rho*rho*(vpred[bi]+budgets[off,bi,0])
                            qpred[bi]=rho*(qpred[bi]+budgets[off,bi,3]+budgets[off,bi,5])
                            held=pv in CFG['heldout_parameters']['lambdas' if stage=='selection' else 'chis']
                            split='holdout_locked' if not dev or held else 'development'
                            br=dict(root_seed=seed,stage=stage,target_G=target,branch_index=bi,lambda_value=float(la[bi]),chi=float(ch[bi]),
                               round=prevrnd+off+1,elapsed_time=(prevrnd+off+1)/r,
                               stream_seed=ss,parameter_heldout=held)
                            br.update({key:float(budgets[off,bi,j]) for j,key in enumerate(engine.BUDGET_COLUMNS)})
                            br['Q_reconstructed']=float(qpred[bi]);br['variance_reconstructed']=float(vpred[bi])
                            budgetrows[split].append(br)
                ranku=ranks(u)
                for bi,pv in enumerate(params):
                    held=pv in CFG['heldout_parameters']['lambdas' if stage=='selection' else 'chis']
                    split='holdout_locked' if not dev or held else 'development'
                    row=dict(seed=seed,stage=stage,target_G=target,prep_time=prepround/r,lambda_value=float(la[bi]),chi=float(ch[bi]),
                          elapsed_time=float(elapsed),stream_seed=ss,branch_index=bi,parameter_heldout=held,
                          trajectory_id=f's{seed}_g{int(target*100):02d}_{stage}_b{bi:02d}',anchor_state_hash=anchor_hash)
                    met=metrics(w0,w[:,bi],u,r0,gini0,coh,ranku)
                    met['budget_Q_residual']=float(met['Q']-qpred[bi]);met['budget_variance_residual']=float(met['variance']-vpred[bi])
                    row.update(met);rows[split].append(row)
                    maxmass=max(maxmass,met['mass_rel_error']);max_qerr=max(max_qerr,abs(met['budget_Q_residual']))
                    max_verr=max(max_verr,abs(met['budget_variance_residual']))
                    neg+=met['neg_count'];zero+=met['zero_count'];ties+=met['cutoff_tie_count']
                    if dev and seed in CFG['record']['density_seeds'] and bi in repids:
                        dp=ROOT/'distributions/development'/f's{seed}_g{int(target*100):02d}_{stage}_b{bi:02d}_r{rnd:04d}.npz'
                        vals=density_arrays(w[:,bi],a,r0)
                        assert int(vals['counts'].sum())==n
                        assert abs(vals['wealth_mass'].sum()/n-1)<1e-10
                        vals['metadata_json']=np.array(json.dumps(dict(root_seed=seed,target_G=target,stage=stage,lambda_value=float(la[bi]),
                          chi=float(ch[bi]),round=rnd,elapsed_time=elapsed,anchor_state_hash=anchor_hash,
                          description='Empirical histogram counts, not a solved density equation; cohort conditional densities normalize by cohort_sizes.'),sort_keys=True))
                        savez(dp,**vals);density_paths.append(str(dp.relative_to(ROOT)))
                take_micro=(dev and seed in CFG['record']['microstate_seeds'] and rnd>0 and
                     ((target==CFG['record']['microstate_target'] and elapsed in CFG['record']['microstate_times']) or
                      (target!=CFG['record']['microstate_target'] and rnd==300)))
                if take_micro:
                    sp=ROOT/'snapshots/development'/f's{seed}_g{int(target*100):02d}_{stage}_r{rnd:04d}.npz'
                    savez(sp,w=w[:,repids],a=a,u=u,permutation=p.astype(np.int32),branch_indices=np.array(repids),
                         lambdas=la[repids],chis=ch[repids],rng_state_json=np.array(json.dumps(engine.rng_state(),sort_keys=True)),
                         metadata_json=np.array(json.dumps(dict(root_seed=seed,target_G=target,stage=stage,round=rnd,elapsed_time=elapsed,
                            anchor_file=str(anchor_file.relative_to(ROOT)),anchor_state_hash=anchor_hash,
                            config_sha256=cfgsha,source_sha256=codesha),sort_keys=True)))
                    micro_paths.append(str(sp.relative_to(ROOT)))
                prevrnd=rnd
            if stage=='selection':
                bi=params.index(4.)
                anchor_B=(w[:,bi].copy(),a.copy(),u.copy(),p.copy())
                parent_hash=array_hash(*anchor_B)
            elif anchor_hash!=parent_hash:
                raise AssertionError('B anchor is not full A lambda=4 endpoint')
            stage_audits.append(dict(stage=stage,target_G=target,branches=len(params),snapshot_records=len(params)*len(CFG['times']),
                       state_hash=anchor_hash,stream_seed=ss,start_clone_exact=True,
                       parent_A_endpoint_matches=(stage=='tax' and anchor_hash==parent_hash)))
    if maxmass>1e-10 or neg or zero or ties or max_qerr>1e-9 or max_verr>1e-8:
        raise AssertionError(f'Numerical QC failed: mass={maxmass}; neg={neg};zero={zero};ties={ties};Q={max_qerr};V={max_verr}')
    datafiles=[]
    for split in ('development','holdout_locked'):
        if rows[split]:
            rp=ROOT/'data'/split/f'population_{seed}.csv.gz';write_gzip_csv(rp,rows[split]);datafiles.append(str(rp.relative_to(ROOT)))
        if budgetrows[split]:
            bp=ROOT/'data'/split/f'budgets_{seed}.csv.gz';write_gzip_csv(bp,budgetrows[split]);datafiles.append(str(bp.relative_to(ROOT)))
    pr=ROOT/'data'/root_split/f'preparation_{seed}.csv';write_csv(pr,prep_rows);datafiles.append(str(pr.relative_to(ROOT)))
    pt=ROOT/'data'/root_split/f'preparation_trace_{seed}.csv.gz';write_gzip_csv(pt,prep_trace);datafiles.append(str(pt.relative_to(ROOT)))
    rec=dict(root_seed=seed,root_split=root_split,seconds=time.perf_counter()-clock,
        completed_trajectories=3*(len(CFG['lambdas'])+len(CFG['chis'])),
        snapshot_records=sum(len(v) for v in rows.values()),budget_records=sum(len(v) for v in budgetrows.values()),
        common_anchors=len(checkpoint_paths),density_files=len(density_paths),microstate_files=len(micro_paths),
        max_mass_relative_error=maxmass,max_Q_budget_error=max_qerr,max_variance_budget_error=max_verr,
        negative_observations=neg,zero_observations=zero,cutoff_ties=ties,
        config_sha256=cfgsha,source_sha256=codesha,stages=stage_audits,
        files=datafiles+checkpoint_paths+density_paths+micro_paths,
        complete_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    dest=ROOT/'logs'/f'completed_{seed}.json';dest.write_text(json.dumps(rec,indent=2)+'\n')
    return {k:v for k,v in rec.items() if k not in ('files','stages','config_sha256','source_sha256')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    requested=args.seeds or CFG['developer_seeds']+CFG['holdout_seeds'];todo=[]
    for seed in requested:
        f=ROOT/'logs'/f'completed_{seed}.json'
        if args.resume and f.exists():
            rec=json.loads(f.read_text())
            if rec['source_sha256']!=source_hash() or rec['config_sha256']!=digest(CONFIG_PATH):
                raise RuntimeError('Cannot resume with changed code/configuration')
            if not all((ROOT/x).exists() for x in rec['files']):raise RuntimeError('Completed root has missing files')
            print(json.dumps(dict(skipped_completed_seed=seed)),flush=True)
        else:todo.append(seed)
    (ROOT/'logs/environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,numba=numba.__version__,platform=platform.platform(),
        config_sha256=digest(CONFIG_PATH),source_sha256=source_hash()),indent=2)+'\n')
    # Prewarm kernels before creating processes; no production root uses this RNG.
    w,a,u,perm=engine.initialise(100,42,2)
    engine.advance_chunk(w,a,u,perm,np.array([0.,4.]),np.array([0.,.01]),1,.14,.3,1.,10)
    r0=ranks(w[:,0]); metrics(w[:,0],w[:,0],u,r0,engine.gini_numba(w[:,0]),cohorts(r0))
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(compute_root,s):s for s in todo}
        for fut in as_completed(futures):
            print(json.dumps(fut.result()),flush=True)
    print(json.dumps(dict(status='requested_roots_complete',root_count=len(requested),wall_seconds=time.perf_counter()-start)),flush=True)
if __name__=='__main__':main()
