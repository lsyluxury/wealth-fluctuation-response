#!/usr/bin/env python3
"""Structural/numerical QC of all roots; outcome regression only on old pilot points.

No holdout outcome means, effects or parameter-response curves are produced.
Independent snapshot statistics do not call records.metrics.
"""
from __future__ import annotations
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
R=Path(__file__).resolve().parents[1]
CFG=json.loads((R/'config/experiment_config.json').read_text())

def independent_stats(w0,w,u,r0,g0):
    n=len(w);y=np.sort(w);total=float(w.sum())
    rw=rankdata(w,method='ordinal').astype(np.int64)-1
    ru=rankdata(u,method='ordinal').astype(np.int64)-1
    g=(n+1.0-2.0*float(np.cumsum(y).sum())/total)/n
    cuts=[n//2]+[n-n*j//100 for j in range(1,11)]
    d=dict(gini=g,gini0=g0,variance=float(np.dot(w-w.mean(),w-w.mean())/n),
       median_wealth=float(np.quantile(w,.5)),bottom50_share=float(np.sum(w[rw<n//2])/total),
       top1_share=float(np.sum(w[rw>=n-n//100])/total),top10_share=float(np.sum(w[rw>=n-n//10])/total),
       rho_s=float(np.corrcoef(rw,ru)[0,1]),Q=float(np.mean((w-1)*u)),
       mean_ability=float(np.mean(u)),min_wealth=float(np.min(w)),max_wealth=float(np.max(w)),
       mass_rel_error=abs(total/n-1),zero_count=int((w==0).sum()),neg_count=int((w<0).sum()),
       cutoff_tie_count=sum(int(y[c]==y[c-1]) for c in cuts),
       threshold_T10_inclusive=float(y[n-n//10]),threshold_T1_inclusive=float(y[n-n//100]),threshold_B50_upper=float(y[n//2-1]))
    masks={'B50':r0<n//2,'Middle49':(r0>=n//2)&(r0<n-n//100),
       'T1':r0>=n-n//100,'T5':r0>=n-n*5//100,'T10':r0>=n-n//10}
    b=masks['B50'];in10=rw>=n-n//10
    d['B50_T10_count']=int(np.logical_and(b,in10).sum());d['B50_T10']=d['B50_T10_count']/int(b.sum())
    for j in range(1,11):
        c=r0>=n-n*j//100;ex=int(np.logical_and(c,rw<n-n*j//100).sum())
        d[f'Etop{j}_count']=ex;d[f'Etop{j}']=ex/int(c.sum())
    for name,m in masks.items():
        x=w[m];b0=w0[m];rel=(x-b0)/b0  # alternate arithmetic for independent check
        d[f'{name}_mean_w0']=float(np.sum(b0)/m.sum());d[f'{name}_mean_wt']=float(np.sum(x)/m.sum())
        d[f'{name}_wealth_change']=float(np.sum(x)/np.sum(b0)-1)
        # This arithmetic may differ at rounding level, hence tolerance below.
        d[f'{name}_individual_change_median']=float(np.quantile(rel,.5))
        d[f'{name}_loss_fraction']=float(np.sum(x<b0)/m.sum())
        d[f'{name}_share_t']=float(np.sum(x)/total)
        for label,p in [('q05',.05),('q25',.25),('median',.5),('q75',.75),('q95',.95)]:
            d[f'{name}_{label}_wealth']=float(np.quantile(x,p))
        if name!='B50':
            d[f'{name}_remain_T10']=float(np.logical_and(m,in10).sum()/m.sum())
            d[f'{name}_fall_B50']=float(np.logical_and(m,rw<n//2).sum()/m.sum())
    between=sum(masks[k].sum()/n*(w[masks[k]].mean()-w.mean())**2 for k in ('B50','Middle49','T1'))
    within=sum(masks[k].sum()/n*np.var(w[masks[k]]) for k in ('B50','Middle49','T1'))
    d['fixed_partition_between_variance']=float(between);d['fixed_partition_within_variance']=float(within)
    return d

def audit_snapshots():
    frames={}
    checks=[]
    for file in sorted((R/'snapshots/development').glob('*.npz')):
        z=np.load(file,allow_pickle=False);meta=json.loads(str(z['metadata_json']))
        s=meta['root_seed']
        if s not in frames:frames[s]=pd.read_csv(R/f'data/development/population_{s}.csv.gz')
        an=np.load(R/meta['anchor_file'],allow_pickle=False);w0=an['w'];r0=an['rank0'];n=len(w0)
        ys=np.sort(w0);g0=(n+1-2*np.cumsum(ys).sum()/ys.sum())/n
        for bi in range(z['w'].shape[1]):
            row=frames[s].loc[(frames[s].stage==meta['stage'])&(frames[s].target_G==meta['target_G'])&
               (frames[s].elapsed_time==meta['elapsed_time'])&(frames[s].lambda_value==z['lambdas'][bi])&
               (frames[s].chi==z['chis'][bi])]
            assert len(row)==1
            row=row.iloc[0];ind=independent_stats(w0,z['w'][:,bi],z['u'],r0,float(g0))
            for key,v in ind.items():
                err=abs(v-row[key]);norm=err/(1+abs(v))
                assert err<1e-9 or norm<1e-12,(file,key,err,norm)
                checks.append(dict(snapshot=file.name,branch_index=int(z['branch_indices'][bi]),metric=key,
                                   absolute_error=err,normalized_error=norm))
    pd.DataFrame(checks).to_csv(R/'validation/INDEPENDENT_SNAPSHOT_CHECKS.csv',index=False)
    return dict(snapshot_files=len(list((R/'snapshots/development').glob('*.npz'))),
        branch_snapshots=len(checks)//len(ind),metric_checks=len(checks),
        max_absolute_error=max(c['absolute_error'] for c in checks),
        max_normalized_error=max(c['normalized_error'] for c in checks),passed=True)

def audit_original():
    ref=R/'validation/reference_new_pilot_statistics.csv.gz'
    if not ref.exists():raise FileNotFoundError('Reference new-protocol pilot missing')
    old=pd.read_csv(ref);new=pd.concat([pd.read_csv(R/f'data/development/population_{s}.csv.gz') for s in CFG['developer_seeds']],ignore_index=True)
    keys=['seed','stage','target_G','lambda_value','chi','elapsed_time']
    old=old.set_index(keys).sort_index();new=new.set_index(keys).sort_index().loc[old.index]
    common=[c for c in old.columns if c in new.columns and pd.api.types.is_numeric_dtype(old[c])]
    err=(new[common]-old[common]).abs()
    assert float(err.max().max())<1e-10
    refs=np.load(R/'validation/reference_new_pilot_microstates_seed2026101201.npz',allow_pickle=False)
    arrchecks=[]
    for g in CFG['targets']:
        tag=f'G{int(g*100)}';gstr=f'g{int(g*100)}'
        a=np.load(R/f'anchors/development/s2026101201_{gstr}_selection.npz',allow_pickle=False)
        b=np.load(R/f'anchors/development/s2026101201_{gstr}_tax.npz',allow_pickle=False)
        sa=np.load(R/f'snapshots/development/s2026101201_{gstr}_selection_r0300.npz',allow_pickle=False)
        sb=np.load(R/f'snapshots/development/s2026101201_{gstr}_tax_r0300.npz',allow_pickle=False)
        maps={'neutral_wealth':a['w'],'neutral_latent':a['a'],'neutral_ability':a['u'],'neutral_permutation':a['permutation'],
              'selection_final_wealth':sa['w'],'selection_final_latent':b['a'],'selection_final_ability':b['u'],
              'selection_final_permutation':b['permutation'],'tax_final_wealth':sb['w'],'tax_final_ability':sb['u']}
        for key,val in maps.items():
            exact=bool(np.array_equal(refs[tag+'_'+key],val));assert exact
            arrchecks.append(dict(key=tag+'_'+key,elements=int(val.size),array_equal=exact))
    return dict(old_snapshot_rows=len(old),numeric_fields=len(common),field_comparisons=len(old)*len(common),
                max_absolute_error=float(err.max().max()),microstate_arrays=arrchecks,passed=True)

def audit_checkpoint():
    import engine
    s=2026101201
    tests=[]
    for stage in ['selection','tax']:
        z=np.load(R/f'snapshots/development/s{s}_g65_{stage}_r0001.npz',allow_pickle=False)
        end=np.load(R/f'snapshots/development/s{s}_g65_{stage}_r0005.npz',allow_pickle=False)
        w=z['w'].copy();a=z['a'].copy();u=z['u'].copy();p=z['permutation'].astype(np.int64)
        engine.restore_rng(json.loads(str(z['rng_state_json'])))
        engine.advance_chunk(w,a,u,p,z['lambdas'],z['chis'],4,.14,.3,1.,10)
        for key,value in [('w',w),('a',a),('u',u),('permutation',p)]:assert np.array_equal(value,end[key])
        assert engine.rng_state()==json.loads(str(end['rng_state_json']))
        tests.append(dict(stage=stage,N=100000,resume_from_round=1,resume_to_round=5,exact_equal=True))
    return tests

def audit_all():
    roots=CFG['developer_seeds']+CFG['holdout_seeds']
    registry=pd.read_csv(R/'config/run_registry.csv')
    assert registry.trajectory_id.nunique()==4860 and len(registry)==4860
    streams=pd.read_csv(R/'config/seed_registry.csv')
    assert streams.stream_seed.nunique()==len(streams)==360
    found=[];struct=[];nrows=0;nbud=0;qa={'mass':0.,'Q_budget':0.,'variance_budget':0.,'partition_variance':0.}
    numerical_metric_count=None
    for seed in roots:
        log=json.loads((R/f'logs/completed_{seed}.json').read_text())
        assert log['completed_trajectories']==81 and log['snapshot_records']==1134 and log['budget_records']==24300
        assert all((R/name).exists() for name in log['files'])
        for split in ['development','holdout_locked']:
            f=R/f'data/{split}/population_{seed}.csv.gz'
            if not f.exists():continue
            a=pd.read_csv(f);nrows+=len(a)
            nums=a.select_dtypes(include=[np.number]);assert np.isfinite(nums.to_numpy()).all()
            numerical_metric_count=len(nums.columns)
            assert not a.duplicated(['trajectory_id','elapsed_time']).any()
            for trajectory,g in a.groupby('trajectory_id',sort=False):
                assert len(g)==14 and np.allclose(sorted(g.elapsed_time),CFG['times'],rtol=0,atol=1e-14)
                assert g.anchor_state_hash.nunique()==1 and g.stream_seed.nunique()==1
                found.append(trajectory)
            if split=='development':
                assert seed in CFG['developer_seeds'] and not a.parameter_heldout.any()
            else:assert seed in CFG['holdout_seeds'] or a.parameter_heldout.all()
            for metric in ['B50_T10']+[f'Etop{q}' for q in range(1,11)]:
                den=50000 if metric=='B50_T10' else 1000*int(metric[4:])
                assert a[metric].between(0,1).all()
                assert (a[metric+'_count']>=0).all() and (a[metric+'_count']<=den).all()
                assert np.max(np.abs(a[metric]*den-a[metric+'_count']))<1e-9
            assert a.neg_count.sum()==0 and a.zero_count.sum()==0 and a.cutoff_tie_count.sum()==0
            qa['mass']=max(qa['mass'],float(a.mass_rel_error.max()))
            qa['Q_budget']=max(qa['Q_budget'],float(a.budget_Q_residual.abs().max()))
            qa['variance_budget']=max(qa['variance_budget'],float(a.budget_variance_residual.abs().max()))
            qa['partition_variance']=max(qa['partition_variance'],float((a.variance-a.fixed_partition_between_variance-a.fixed_partition_within_variance).abs().max()))
            bf=R/f'data/{split}/budgets_{seed}.csv.gz';b=pd.read_csv(bf);nbud+=len(b)
            assert np.isfinite(b.select_dtypes(include=[np.number]).to_numpy()).all()
            for _,g in b.groupby(['stage','target_G','branch_index'],sort=False):
                assert len(g)==300 and np.array_equal(g['round'].to_numpy(),np.arange(1,301))
            # Per-round exogenous update count is identical among branches.
            assert b.groupby(['stage','target_G','round']).renewal_event_count.nunique().max()==1
            struct.append(dict(root_seed=seed,split=split,snapshot_rows=len(a),budget_rows=len(b),trajectories=a.trajectory_id.nunique()))
    assert len(found)==len(set(found))==4860 and set(found)==set(registry.trajectory_id)
    assert nrows==68040 and nbud==1458000
    assert qa['mass']<1e-10 and qa['Q_budget']<1e-9 and qa['variance_budget']<1e-8
    pd.DataFrame(struct).to_csv(R/'validation/MATRIX_COMPLETENESS.csv',index=False)
    # Check complete anchors and empirical distributions without reporting outcomes.
    from records import array_hash
    afiles=list((R/'anchors').rglob('*.npz'));assert len(afiles)==360
    for f in afiles:
        z=np.load(f,allow_pickle=False);md=json.loads(str(z['metadata_json']));n=len(z['w'])
        assert n==100000 and np.array_equal(z['id'],np.arange(n))
        assert np.array_equal(np.sort(z['permutation']),np.arange(n))
        assert array_hash(z['w'],z['a'],z['u'],z['permutation'].astype(np.int64))==md['state_hash']
        assert np.array_equal(z['rank0'],rankdata(z['w'],method='ordinal')-1)
        assert np.array_equal(np.bincount(z['cohort_code']),[50000,49000,1000])
        if md['stage']=='tax':assert md['parent_A_lambda4_endpoint_hash']==md['state_hash']
    dfiles=list((R/'distributions/development').glob('*.npz'));assert len(dfiles)==1512
    densmass=0.;outside_max_pop=0
    for f in dfiles:
        z=np.load(f,allow_pickle=False)
        assert np.array_equal(z['counts'].sum(axis=(1,2)),[50000,49000,1000])
        assert np.array_equal(z['counts'].sum(axis=0),z['total_counts'])
        assert np.array_equal(z['wealth_mass'].sum(axis=0),z['total_wealth_mass'])
        assert np.array_equal(z['cdf_counts'][:,-1],[50000,49000,1000])
        assert (z['counts']>=0).all() and (z['wealth_mass']>=0).all()
        assert (np.diff(z['wealth_quantiles_total_then_cohorts'],axis=1)>=0).all()
        densmass=max(densmass,abs(float(z['wealth_mass'].sum())/100000-1))
        outside_max_pop=max(outside_max_pop,int(z['counts'][:,[0,-1],:].sum()))
    assert densmass<1e-10
    result=dict(status='PASSED',independent_roots=60,development_roots=20,heldout_roots=40,
         phase_trajectories=4860,snapshot_rows=nrows,branch_round_budget_rows=nbud,
         common_anchors=len(afiles),empirical_density_files=len(dfiles),
         numerical_columns_per_row=numerical_metric_count,
         negative_observations=0,zero_observations=0,cutoff_ties=0,
         max_errors=qa,density_max_mass_relative_error=densmass,max_wealth_outer_bin_population=outside_max_pop,
         original_pilot_regression=audit_original(),independent_snapshots=audit_snapshots(),
         production_checkpoint_resume=audit_checkpoint(),
         holdout_policy='Only automatic structural/numerical QA. No holdout outcome means or effects computed here.',
         not_done=['independent_density_dynamics','analytic_prediction_validation','time_50_100_extensions','final_four_figures'])
    (R/'validation/FINAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshots-only',action='store_true');p.add_argument('--pilot-only',action='store_true')
    args=p.parse_args()
    if args.snapshots_only:
        print(json.dumps({'snapshot_stats':audit_snapshots(),'checkpoint':audit_checkpoint()},indent=2))
    elif args.pilot_only:print(json.dumps(audit_original(),indent=2))
    else:audit_all()
