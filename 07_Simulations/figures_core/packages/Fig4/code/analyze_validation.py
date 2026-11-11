#!/usr/bin/env python3
"""Reproduce all acceptance decisions; no condition or outlier is omitted."""
from pathlib import Path
import argparse,json,math
import numpy as np,pandas as pd
from scipy.stats import t
R=Path(__file__).resolve().parents[1];D=R/'data/derived';D.mkdir(exist_ok=True)
KEY=['root','lambda_value','chi','round']
GROUP=['lambda_value','chi','round']
TOLS={'G':.005,'M':.002,'E':.002}

def load(path):
    df=pd.read_csv(path)
    for key in ['lambda_value','chi','dx']:
        if key in df:df[key]=df[key].round(8)
    return df

def stats(v):
    v=np.asarray(v,dtype=float);n=len(v);m=float(v.mean());sd=float(v.std(ddof=1));h=float(t.ppf(.975,n-1)*sd/np.sqrt(n))
    return dict(mean=m,ci_lo=m-h,ci_hi=m+h,sd=sd,n_roots=n)

def main(partial=False):
    files=list((R/'data/raw/theory').glob('*.csv'));thy=pd.concat([load(f) for f in files],ignore_index=True)
    fut=pd.concat([load(f) for f in (R/'data/raw/futures').glob('*.csv')],ignore_index=True)
    if fut.duplicated(KEY+['future']).any():raise ValueError('Duplicated future observation')
    counts=fut.groupby(KEY).future.nunique()
    complete_roots=counts[counts==16].reset_index()[KEY]
    abm=fut.groupby(KEY)[['G','M','E','Ruw']].mean().reset_index().merge(complete_roots,on=KEY)
    med=thy[np.isclose(thy.dx,.025)&(thy.na==48)];fine=thy[np.isclose(thy.dx,.01)&(thy.na==96)]
    if not partial:
        assert len(med)==len(fine)==42*6*5,(len(med),len(fine))
        assert len(fut)==42*6*5*16 and len(abm)==42*6*5,(len(fut),len(abm))
    abm.to_csv(D/'abm_conditional_means_by_root.csv',index=False)
    thy.to_csv(D/'theory_all_roots_grids.csv',index=False)
    joined=fine.merge(abm,on=KEY,suffixes=('_theory','_abm'),validate='1:1')
    joined.to_csv(D/'matched_theory_ABM_by_root.csv',index=False)
    rows=[];endpoints=[]
    for keys,g in joined.groupby(GROUP):
        lam,chi,n=keys
        if len(g)!=6:continue
        for m,tol in TOLS.items():
            delta=g[m+'_theory']-g[m+'_abm'];s=stats(delta)
            status='supported' if s['ci_lo']>=-tol and s['ci_hi']<=tol else ('outside' if s['ci_lo']>tol or s['ci_hi']<-tol else 'inconclusive')
            rows.append(dict(lambda_value=lam,chi=chi,round=n,time=n/10,metric=m,tolerance=tol,status=status,
                             theory_mean=g[m+'_theory'].mean(),abm_mean=g[m+'_abm'].mean(),**s))
    res=pd.DataFrame(rows,columns=['lambda_value','chi','round','time','metric','tolerance','status','theory_mean','abm_mean','mean','ci_lo','ci_hi','sd','n_roots']);res.to_csv(D/'acceptance_all_conditions_times.csv',index=False)
    gr=med.merge(fine,on=KEY,suffixes=('_medium','_fine'),validate='1:1');gr.to_csv(D/'grid_comparison_by_root.csv',index=False)
    checks=[]
    for keys,g in gr.groupby(GROUP):
        if len(g)!=6:continue
        for m,tol in TOLS.items():
            dif=g[m+'_fine']-g[m+'_medium'];avg=float(dif.mean())
            checks.append(dict(lambda_value=keys[0],chi=keys[1],round=keys[2],metric=m,
                               fine_minus_medium=avg,max_root_abs_difference=float(abs(dif).max()),
                               convergence_diagnostic_budget=tol/4,within_budget=abs(avg)<=tol/4))
    gc=pd.DataFrame(checks);gc.to_csv(D/'grid_convergence_summary.csv',index=False)
    final=res[(res['round']==300)];final.to_csv(D/'acceptance_endpoints42.csv',index=False)
    aux=[]
    for keys,g in joined.groupby(GROUP):
        if len(g)!=6:continue
        aux.append(dict(lambda_value=keys[0],chi=keys[1],round=keys[2],**stats(g['Ruw_grid']-g['Ruw'])))
    pd.DataFrame(aux).to_csv(D/'ancillary_rank_correlation_error.csv',index=False)
    dat=res[res['round']>0]
    summary={'partial':partial,'theory_trajectories':len(files),'ABM_futures_completed':len(fut)//210,
             'test_counts':dat.groupby(['metric','status']).size().unstack(fill_value=0).to_dict('index') if len(dat) else {},
             'endpoint_counts':final.groupby(['metric','status']).size().unstack(fill_value=0).to_dict('index') if len(final) else {},
             'max_absolute_paired_mean_errors':{m:float(abs(dat[dat.metric==m]['mean']).max()) for m in TOLS} if len(dat) else {},
             'max_grid_shifts':{m:float(abs(gc[(gc.metric==m)&(gc['round']>0)].fine_minus_medium).max()) for m in TOLS} if len(gc) else {},
             'convergence_flags':int((~gc.within_budget).sum()) if len(gc) else 0}
    (R/'validation/ACCEPTANCE_SUMMARY.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    if len(dat):print(dat[dat.status!='supported'][['lambda_value','chi','round','metric','mean','ci_lo','ci_hi','status']].to_string(index=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.partial)
