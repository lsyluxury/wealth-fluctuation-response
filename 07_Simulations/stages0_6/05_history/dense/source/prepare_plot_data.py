#!/usr/bin/env python3
"""Recompute Figure 2/3 summaries from raw ABM observations; never reads theory."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(root:Path|None=None):
    r=Path(root) if root else ROOT
    fc=json.loads((r/'figure_config.json').read_text());cfg=json.loads((r/'simulation_config.json').read_text())
    stage=fc['stage'];param='lambda_value' if stage=='A' else 'chi';params=cfg['lambdas' if stage=='A' else 'chis']
    frames=[];logs=[]
    for seed in cfg['seeds']:
        fp=r/f'runs/s{seed}.csv';jp=fp.with_suffix('.json')
        if not fp.exists() or not jp.exists():raise FileNotFoundError(f'Incomplete registered root: {seed}')
        rec=json.loads(jp.read_text())
        if rec['raw_sha256']!=sha(fp):raise ValueError(f'Raw file hash mismatch {seed}')
        if rec['engine_sha256']!=sha(r/'src/engine.py'):raise ValueError('Microengine does not match raw-run provenance')
        if rec['config_sha256']!=sha(r/'simulation_config.json'):raise ValueError('Simulation configuration mismatch')
        d=pd.read_csv(fp,float_precision='round_trip');frames.append(d);logs.append(rec)
    df=pd.concat(frames,ignore_index=True)
    df['param_idx']=df[param].map({float(v):i for i,v in enumerate(params)})
    if df.param_idx.isna().any():raise AssertionError('Unexpected parameter')
    df['param_idx']=df.param_idx.astype(int)
    if df.duplicated(['seed','param_idx','round']).any():raise AssertionError('Repeated seed/parameter/time')
    expected={(j,n) for j in range(len(params)) for n in cfg['rounds']}
    for seed,d in df.groupby('seed'):
        if set(zip(d.param_idx,d['round']))!=expected:raise AssertionError(f'Incomplete grid for {seed}')
    N=cfg['model']['N'];nb=N//2;nt=N//10
    if not (df.M_count+df.I_count==df.E_count).all():raise AssertionError('Cohort accounting violated')
    if not np.allclose(df.M,df.M_count/nb,atol=1e-16,rtol=0):raise AssertionError('Wrong bottom-group denominator')
    if not np.allclose(df.E,df.E_count/nt,atol=1e-16,rtol=0):raise AssertionError('Wrong top-group denominator')
    if not np.isfinite(df[['G','M','E']].values).all():raise AssertionError('Nonfinite observations')
    if ((df[['G','M','E']]<0)|(df[['G','M','E']]>1)).any().any():raise AssertionError('Invalid probability/Gini')
    start=df[df['round']==0]
    for seed,d in start.groupby('seed'):
        if d.G.nunique()!=1 or np.any(d.M) or np.any(d.E):raise AssertionError('Branches lack identical stage start')
    # The overlap with the prior 40-seed test is validated, not spliced into results.
    old=pd.read_csv(r/f'raw_data/previous_40_seed_{stage}_ABM.csv',float_precision='round_trip')
    old['param_idx']=old[param].map({float(v):i for i,v in enumerate(params)})
    overlap=old.merge(df,on=['seed','param_idx','round'],validate='one_to_one',suffixes=('_old','_new'))
    errors={k:float(np.max(abs(overlap[k]-overlap[k+'_abm']))) for k in ['G','M','E']}
    eq={k:bool(np.array_equal(overlap[k+'_old'],overlap[k+'_new'])) for k in ['M_count','E_count']}
    if len(overlap)!=2800 or errors['G']>1e-10 or not all(eq.values()):raise AssertionError(f'Original-condition mismatch: {errors}, {eq}')
    rows=[]
    for (j,rr),d in df.groupby(['param_idx','round'],sort=True):
        for k,mult in [('G',1.),('M',100.),('E',100.)]:
            x=d[k].to_numpy()*mult;n=len(x);mean=float(x.mean());sd=float(x.std(ddof=1));se=sd/math.sqrt(n);hw=float(student_t.ppf(.975,n-1))*se
            rows.append(dict(parameter=float(params[j]),round=int(rr),time=rr/cfg['model']['r'],metric=k,n=n,mean=mean,sd=sd,se=se,ci95_low=mean-hw,ci95_high=mean+hw))
    ss=pd.DataFrame(rows)
    origins=[]
    for j,d in df[df['round']==cfg['rounds'][-1]].groupby('param_idx',sort=True):
        if (d.E_count<=0).any():raise AssertionError('Undefined share of newcomers; never replace 0/0 with zero')
        x=100*d.M_count.to_numpy()/d.E_count.to_numpy();n=len(x);mu=float(x.mean());sd=float(x.std(ddof=1));se=sd/math.sqrt(n);hw=float(student_t.ppf(.975,n-1))*se
        row=dict(parameter=float(params[j]),n=n,bottom50_share_mean=mu,bottom50_share_sd=sd,bottom50_share_se=se,bottom50_ci95_low=mu-hw,bottom50_ci95_high=mu+hw,middle40_share_mean=100-mu,bottom50_pooled_share=float(100*d.M_count.sum()/d.E_count.sum()))
        for k in ['G','M','E']:
            u=ss[(ss.metric==k)&ss.parameter.eq(float(params[j]))&ss['round'].eq(cfg['rounds'][-1])].iloc[0]
            row.update({k+'_mean':u['mean'],k+'_low':u.ci95_low,k+'_high':u.ci95_high})
        origins.append(row)
    od=pd.DataFrame(origins)
    out=r/'source_data';out.mkdir(exist_ok=True)
    df=df.sort_values(['seed','param_idx','round']).reset_index(drop=True)
    df.to_csv(out/'ABM_per_seed.csv',index=False,float_format='%.17g')
    ss.to_csv(out/'ABM_mean_ci95.csv',index=False,float_format='%.17g')
    od.to_csv(out/'ABM_endpoint_origins.csv',index=False,float_format='%.17g')
    for k,l in zip(['G','M','E'],'abc'):
        ss[ss.metric==k].to_csv(out/f'panel_{l}.csv',index=False,float_format='%.17g')
    od[['parameter','n','G_mean','G_low','G_high','M_mean','M_low','M_high']].to_csv(out/'panel_d.csv',index=False,float_format='%.17g')
    od[['parameter','n','G_mean','G_low','G_high','E_mean','E_low','E_high']].to_csv(out/'panel_e.csv',index=False,float_format='%.17g')
    od[[c for c in od.columns if c.startswith(('parameter','n','bottom','middle'))]].to_csv(out/'panel_f.csv',index=False,float_format='%.17g')
    audit=dict(stage=stage,n_seeds=len(cfg['seeds']),n_agents=N,n_parameters=len(params),parameters=params,n_times=len(cfg['rounds']),rows=len(df),stage_trajectories=len(cfg['seeds'])*len(params),recording_rounds=cfg['rounds'],original_overlap_rows=len(overlap),original_max_difference=errors,integer_counts_equal=eq,cohort_accounting_exact=True,stage_start_identical_within_root=True,max_wealth_conservation_error=max(l['max_mass_error'] for l in logs),min_final_wealth=min(l['min_final_w'] for l in logs),engine_sha256=sha(r/'src/engine.py'),simulation_config_sha256=sha(r/'simulation_config.json'),CI='two-sided pointwise Student t 95% for the mean across 40 root seeds; no inflation; not simultaneous',panel_f='mean of within-root newcomer fractions; denominator is NEW top-decile entrants, not all B50',theory_used=False,numerical_interpolation=False,scope='post-unblinding descriptive densification of the same 40 roots; no new independent validation')
    (r/'validation/data_audit.json').write_text(json.dumps(audit,indent=2))
    print(f"Stage {stage}: {len(df)} validated observations, {len(ss)} mean/CI rows.")
    return audit
if __name__=='__main__':main()
