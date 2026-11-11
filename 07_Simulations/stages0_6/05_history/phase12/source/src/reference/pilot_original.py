#!/usr/bin/env python3
"""Common-state Yard-Sale intervention pilot.

Run: python run_pilot.py --workers 3
Reuses the mathematical loops of the supplied original kernels, with a new
saved-state interface. Untouched originals remain in source/ for audit.
No statistics in this file are imported from earlier simulation outputs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Avoid BLAS oversubscription inside independent population workers.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/wealth_common_state_mpl')
import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)
TARGETS = np.array([0.50, 0.65, 0.80], dtype=np.float64)
LAMBDAS = np.array([0.0, 4.0, 16.0], dtype=np.float64)
CHIS = np.array([0.0, 0.01, 0.03], dtype=np.float64)
TIMES = np.array([0.0, 1.0, 5.0, 10.0, 20.0, 30.0])
SNAPROUNDS = np.rint(TIMES * 10).astype(np.int64)
SEEDS = list(range(2026101201, 2026101221))
CFG = dict(N=100000, epsilon=0.14, h=0.30, mu=1.0, r=10)

@njit(cache=True)
def logistic(z):
    if z >= 0.0:
        return 1.0/(1.0+math.exp(-z))
    ez=math.exp(z)
    return ez/(1.0+ez)

@njit(cache=True)
def initialise(N, seed, branches=1):
    np.random.seed(seed)
    wealth=np.ones((N, branches), dtype=np.float64)
    latent=np.random.normal(0.0, 1.0, N)
    ability=np.empty(N, dtype=np.float64)
    for i in range(N):
        ability[i]=math.erf(latent[i]/math.sqrt(2.0))
    perm=np.arange(N)
    return wealth,latent,ability,perm

@njit(cache=True)
def reseed(seed):
    np.random.seed(seed)

@njit(cache=True)
def advance_one_round(w,a,u,perm,lambdas,chis,epsilon,h,mu,r):
    """Ordering is exactly shuffle/trade, Poisson renewal, then tax."""
    N=w.shape[0]
    nb=w.shape[1]
    dt=1.0/r
    innovation=math.sqrt(max(0.0,1.0-h*h))
    np.random.shuffle(perm)
    for p in range(0,N,2):
        i=perm[p]; j=perm[p+1]
        du=u[i]-u[j]
        draw=np.random.random()
        for b in range(nb):
            iwin=draw < logistic(lambdas[b]*du)
            wi=w[i,b]; wj=w[j,b]
            stake=epsilon*(wi if wi<wj else wj)
            if iwin:
                w[i,b]=wi+stake; w[j,b]=wj-stake
            else:
                w[i,b]=wi-stake; w[j,b]=wj+stake
    if mu>0:
        nev=np.random.poisson(N*mu*dt)
        for _ in range(nev):
            i=np.random.randint(0,N)
            a[i]=h*a[i]+innovation*np.random.normal()
            u[i]=math.erf(a[i]/math.sqrt(2.0))
    for b in range(nb):
        if chis[b]>0:
            rho=math.exp(-chis[b]*dt)
            rebate=1.0-rho
            for i in range(N):
                w[i,b]=rho*w[i,b]+rebate

@njit(cache=True)
def gini_numba(w):
    ordered=np.sort(w)
    total=0.0; weighted=0.0
    for i in range(w.size):
        total += ordered[i]
        weighted += (i+1)*ordered[i]
    return 2.0*weighted/(w.size*total)-(w.size+1.0)/w.size

@njit(cache=True)
def prepare_neutral(N,seed,targets,epsilon,h,mu,r,max_rounds=20000):
    w,a,u,perm=initialise(N,seed,1)
    nt=targets.size
    ws=np.empty((nt,N)); aa=np.empty((nt,N)); uu=np.empty((nt,N))
    pp=np.empty((nt,N),dtype=np.int64)
    rr=np.zeros(nt,dtype=np.int64); gg=np.zeros(nt)
    z=np.zeros(1)
    k=0
    for rnd in range(1,max_rounds+1):
        advance_one_round(w,a,u,perm,z,z,epsilon,h,mu,r)
        g=gini_numba(w[:,0])
        while k<nt and g>=targets[k]:
            ws[k]=w[:,0]; aa[k]=a; uu[k]=u; pp[k]=perm
            rr[k]=rnd; gg[k]=g
            k+=1
        if k==nt:
            break
    if k<nt:
        raise RuntimeError('Neutral preparation did not reach all target Gini levels')
    return ws,aa,uu,pp,rr,gg

@njit(cache=True)
def continue_states(w,a,u,perm,lambdas,chis,snapshot_rounds,epsilon,h,mu,r):
    """Mutates complete state; RNG is seeded separately to allow exact replay tests."""
    ns=snapshot_rounds.size
    N,nb=w.shape
    sw=np.empty((ns,N,nb))
    su=np.empty((ns,N))
    cursor=0
    for rnd in range(snapshot_rounds[-1]+1):
        while cursor<ns and rnd==snapshot_rounds[cursor]:
            sw[cursor]=w; su[cursor]=u
            cursor+=1
        if rnd==snapshot_rounds[-1]:
            break
        advance_one_round(w,a,u,perm,lambdas,chis,epsilon,h,mu,r)
    return sw,su


def ranks(x):
    out=np.empty(x.size,dtype=np.int64)
    out[np.argsort(x,kind='mergesort')]=np.arange(x.size)
    return out


def spearman_ranks(rx,ry):
    n=rx.size
    # There are no wealth ties in recorded cohorts; stable ranks handle exact identity.
    d=rx.astype(float)-ry.astype(float)
    return float(1.0-6.0*np.dot(d,d)/(n*(n*n-1.0)))


def state_stats(w,u):
    n=w.size
    ix=np.argsort(w,kind='mergesort'); y=w[ix]
    rw=np.empty(n,dtype=np.int64); rw[ix]=np.arange(n)
    total=float(np.sum(w)); rr=np.arange(1,n+1,dtype=float)
    return dict(gini=float(2*np.dot(rr,y)/(n*total)-(n+1)/n),
                median_wealth=float(np.median(y)),variance=float(np.var(w)),
                bottom50_share=float(y[:n//2].sum()/total),
                top1_share=float(y[-n//100:].sum()/total),
                top10_share=float(y[-n//10:].sum()/total),
                rho_s=spearman_ranks(rw,ranks(u)),
                mean_ability=float(u.mean()),min_wealth=float(y[0]),
                mass_rel_error=abs(total/n-1.0),zero_count=int(np.count_nonzero(w==0)),
                neg_count=int(np.count_nonzero(w<0))),rw


def metrics(w0,w,u,rank0,stats0):
    n=w.size
    out,rw=state_stats(w,u)
    b50=rank0<n//2
    top10=rw>=n-n//10
    out['gini0']=stats0['gini']
    out['B50_T10']=float(np.mean(top10[b50]))
    out['B50_T10_count']=int(np.count_nonzero(top10[b50]))
    # Wealth of the fixed pre-intervention bottom half.
    out['B50_mean_w0']=float(w0[b50].mean())
    out['B50_mean_wt']=float(w[b50].mean())
    out['B50_wealth_change']=float(w[b50].sum()/w0[b50].sum()-1)
    ratio_b=w[b50]/w0[b50]-1
    out['B50_individual_change_median']=float(np.median(ratio_b))
    out['B50_loss_fraction']=float(np.mean(w[b50]<w0[b50]))
    out['B50_share_t']=float(w[b50].sum()/w.sum())
    for q in range(1,11):
        cohort=rank0 >= n-n*q//100
        k=int(np.count_nonzero(cohort))
        out[f'Etop{q}']=float(np.mean(rw[cohort]<n-n*q//100))
        out[f'Etop{q}_count']=int(np.count_nonzero(rw[cohort]<n-n*q//100))
        if q in (1,5,10):
            w0c=w0[cohort]; wc=w[cohort]
            out[f'T{q}_mean_w0']=float(w0c.mean())
            out[f'T{q}_mean_wt']=float(wc.mean())
            out[f'T{q}_wealth_change']=float(wc.sum()/w0c.sum()-1)
            out[f'T{q}_individual_change_median']=float(np.median(wc/w0c-1))
            out[f'T{q}_loss_fraction']=float(np.mean(wc<w0c))
            out[f'T{q}_remain_T10']=float(np.mean(top10[cohort]))
            out[f'T{q}_fall_B50']=float(np.mean(rw[cohort]<n//2))
    # No ambiguity in wealth-rank cutoffs for these saved states.
    y=np.sort(w)
    cutoffs=[n//2]+[n-n*q//100 for q in range(1,11)]
    out['cutoff_tie_count']=sum(int(y[c]==y[c-1]) for c in cutoffs)
    return out


def stream_seed(seed,target_index,stage):
    return int(np.random.SeedSequence([seed,target_index,stage]).generate_state(1)[0])


def save_csv(path,rows):
    if not rows:
        return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)


def compute_population(seed):
    st=time.perf_counter()
    n=CFG['N']; args=(CFG['epsilon'],CFG['h'],CFG['mu'],CFG['r'])
    ws,aa,uu,pp,rr,gg=prepare_neutral(n,seed,TARGETS,*args)
    rows=[]; prep=[]; traces={}
    for ki,target in enumerate(TARGETS):
        w0=ws[ki]; a0=aa[ki]; u0=uu[ki]; p0=pp[ki]
        base,rank0=state_stats(w0,u0)
        prep.append(dict(seed=seed,target_G=float(target),prep_rounds=int(rr[ki]),
                         prep_time=float(rr[ki]/CFG['r']),actual_G=float(gg[ki]),
                         **base))
        # Every branch begins with identical identities, wealth and ability.
        w=np.repeat(w0[:,None],len(LAMBDAS),axis=1)
        a=a0.copy(); u=u0.copy(); p=p0.copy()
        reseed(stream_seed(seed,ki,1))
        aw,au=continue_states(w,a,u,p,LAMBDAS,np.zeros(len(LAMBDAS)),SNAPROUNDS,*args)
        assert np.array_equal(aw[0],np.repeat(w0[:,None],len(LAMBDAS),axis=1))
        for si,t in enumerate(TIMES):
            for ell,lam in enumerate(LAMBDAS):
                row=dict(seed=seed,stage='selection',target_G=float(target),
                         prep_time=float(rr[ki]/CFG['r']),lambda_value=float(lam),
                         chi=0.0,elapsed_time=float(t),
                         stream_seed=stream_seed(seed,ki,1))
                row.update(metrics(w0,aw[si,:,ell],au[si],rank0,base)); rows.append(row)
        # Take the lambda=4 endpoint, not the original neutral starting population.
        ell4=int(np.flatnonzero(LAMBDAS==4)[0])
        bw0=w[:,ell4].copy(); ba0=a.copy(); bu0=u.copy(); bp0=p.copy()
        bbase,brank0=state_stats(bw0,bu0)
        bw=np.repeat(bw0[:,None],len(CHIS),axis=1)
        ba=ba0.copy(); bu=bu0.copy(); bp=bp0.copy()
        reseed(stream_seed(seed,ki,2))
        tw,tu=continue_states(bw,ba,bu,bp,np.full(len(CHIS),4.0),CHIS,SNAPROUNDS,*args)
        assert np.array_equal(tw[0],np.repeat(bw0[:,None],len(CHIS),axis=1))
        for si,t in enumerate(TIMES):
            for ci,chi in enumerate(CHIS):
                row=dict(seed=seed,stage='tax',target_G=float(target),
                         prep_time=float(rr[ki]/CFG['r']),lambda_value=4.0,
                         chi=float(chi),elapsed_time=float(t),
                         stream_seed=stream_seed(seed,ki,2))
                row.update(metrics(bw0,tw[si,:,ci],tu[si],brank0,bbase)); rows.append(row)
        if seed==SEEDS[0]:
            key=f'G{int(target*100)}'
            traces.update({key+'_neutral_wealth':w0,key+'_neutral_latent':a0,
                           key+'_neutral_ability':u0,key+'_neutral_permutation':p0,
                           key+'_selection_final_wealth':aw[-1],
                           key+'_selection_final_latent':ba0,
                           key+'_selection_final_ability':bu0,
                           key+'_selection_final_permutation':bp0,
                           key+'_tax_final_wealth':tw[-1],key+'_tax_final_ability':tu[-1]})
    if traces:
        np.savez_compressed(ROOT/'sample_microstates_seed_2026101201.npz',**traces)
    # Save each completed population independently; restart never loses other seeds.
    save_csv(DATA/f'population_{seed}.csv',rows)
    save_csv(DATA/f'preparation_{seed}.csv',prep)
    return dict(seed=seed,seconds=time.perf_counter()-st,rows=len(rows),
                max_mass_error=max(x['mass_rel_error'] for x in rows),
                min_wealth=min(x['min_wealth'] for x in rows),
                negatives=sum(x['neg_count'] for x in rows),
                zeros=sum(x['zero_count'] for x in rows),
                cutoff_ties=sum(x['cutoff_tie_count'] for x in rows),
                prep_times=[x['prep_time'] for x in prep])


def load_original(name):
    path=ROOT/'source'/name
    spec=importlib.util.spec_from_file_location(name[:-3],path)
    module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def validate():
    """Check the continuation wrapper against both untouched original protocols."""
    orig=load_original('compute_fig2_chi0_prl.py')
    taxorig=load_original('compute_fig3_redistribution_prl.py')
    errors=[]
    args=(CFG['epsilon'],CFG['h'],CFG['mu'],CFG['r'])
    # All elementary random draws and numerical updates should match bit for bit.
    for n,seed,lamset,chi in [(2000,777,[0.,4.,16.],0.),
                              (2000,778,[0.,4.],0.01),
                              (100000,779,[0.,4.,16.],0.)]:
        la=np.array(lamset)
        sr=np.array([0,7,20],dtype=np.int64)
        ow,ou0,ouf=orig.simulate_multi_lambda(n,2.,10,.14,chi,.3,1.,la,
                                             np.zeros(la.size,dtype=np.bool_),sr,seed)
        w,a,u,p=initialise(n,seed,la.size)
        nw,nu=continue_states(w,a,u,p,la,np.full(la.size,chi),sr,*args)
        errors.append(dict(test='original_fig2',N=n,seed=seed,
                           max_wealth_diff=float(np.max(np.abs(ow-nw))),
                           max_ability_diff=float(np.max(np.abs(ouf-nu[-1])))))
        assert np.array_equal(ow,nw) and np.array_equal(ouf,nu[-1])
    n=2000; seed=780
    chis=CHIS.copy()
    ow0,ou0,owf,ouf=taxorig.simulate_anchor_and_branches(n,1.,2.,10,.14,4.,.3,1.,chis,seed)
    w,a,u,p=initialise(n,seed,1)
    continue_states(w,a,u,p,np.array([4.]),np.array([0.]),np.array([0,10]),*args)
    nw0=w[:,0].copy(); nu0=u.copy()
    w=np.repeat(nw0[:,None],chis.size,axis=1)
    nw,nu=continue_states(w,a,u,p,np.full(chis.size,4.),chis,np.array([0,20]),*args)
    errors.append(dict(test='original_fig3',N=n,seed=seed,
                       max_wealth_diff=float(np.max(np.abs(owf-nw[-1]))),
                       max_ability_diff=float(np.max(np.abs(ouf-nu[-1])))))
    assert np.array_equal(ow0,nw0) and np.array_equal(ou0,nu0)
    assert np.array_equal(owf,nw[-1]) and np.array_equal(ouf,nu[-1])
    # Identical parameters yield identical populations, not merely similar moments.
    w,a,u,p=initialise(2000,781,2)
    nw,nu=continue_states(w,a,u,p,np.array([4.,4.]),np.array([.01,.01]),np.array([0,20]),*args)
    assert np.array_equal(nw[:,:,0],nw[:,:,1])
    # Intervention order must not change any branch (same RNG consumption).
    w0,a0,u0,p0=initialise(2000,782,1)
    outputs=[]
    for la in [np.array([0.,4.,16.]),np.array([16.,4.,0.])]:
        w=np.repeat(w0,3,axis=1); a=a0.copy();u=u0.copy();p=p0.copy()
        reseed(783)
        ww,uu=continue_states(w,a,u,p,la,np.zeros(3),np.array([0,20]),*args)
        outputs.append(ww)
    assert np.array_equal(outputs[0],outputs[1][:,:,::-1])
    # Hitting-time preparation should exactly replay the neutral original process.
    n=2000; seed=784; targets=np.array([.2,.3])
    ws,aa,uu,pp,rr,gg=prepare_neutral(n,seed,targets,*args)
    ow,ou0,ouf=orig.simulate_multi_lambda(n,float(rr[-1]/10),10,.14,0.,.3,1.,
                                         np.array([0.]),np.array([False]),rr,seed)
    assert np.array_equal(ws,ow[:,:,0]) and np.array_equal(uu[-1],ouf)
    errors.append(dict(test='neutral_hitting_state',N=n,seed=seed,
                       max_wealth_diff=float(np.max(np.abs(ws-ow[:,:,0]))),
                       max_ability_diff=float(np.max(np.abs(uu[-1]-ouf)))))
    record={'kernel_comparisons':errors,'duplicate_branches_equal':True,
            'branch_order_invariance':True,
            'original_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (ROOT/'source').glob('*.py')}}
    (ROOT/'VALIDATION_PRE.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return record


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--validate-only',action='store_true')
    ap.add_argument('--seeds',type=int,default=20)
    args=ap.parse_args()
    start=time.perf_counter()
    val=validate()
    print('Original-kernel, clone and neutral-preparation tests PASSED.',flush=True)
    if args.validate_only:
        return
    todo=SEEDS[:args.seeds]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(compute_population,s):s for s in todo}
        for fut in as_completed(futures):
            rec=fut.result();results.append(rec)
            print(json.dumps(rec),flush=True)
    meta={'config':CFG,'targets':TARGETS.tolist(),'lambdas':LAMBDAS.tolist(),
          'chis':CHIS.tolist(),'times':TIMES.tolist(),'seeds':todo,
          'wall_seconds':time.perf_counter()-start,'population_runs':results,
          'n_population_replicates':len(results),
          'completed_endpoint_branch_conditions':len(results)*len(TARGETS)*(len(LAMBDAS)+len(CHIS)),
          'max_mass_error':max(r['max_mass_error'] for r in results),
          'min_wealth':min(r['min_wealth'] for r in results),
          'negatives':sum(r['negatives'] for r in results),
          'zeros':sum(r['zeros'] for r in results),'cutoff_ties':sum(r['cutoff_ties'] for r in results),
          'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),
          'run_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'validation':val}
    (ROOT/'COMPLETED.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print('COMPLETE',json.dumps({k:v for k,v in meta.items() if k not in ('validation','population_runs')}),flush=True)

if __name__=='__main__':
    main()
