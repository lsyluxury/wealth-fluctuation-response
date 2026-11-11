#!/usr/bin/env python3
"""Aggregate long-time baseline Yard-Sale Monte Carlo runs without modifying them.
Independent replicate unit is the initial neutral preparation root (N=10000).
For Fig.3d, reuse Stage4 q=0.1, tau=30, n=80 root-level summary.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data'/'raw'; DER=ROOT/'data'/'derived'; DER.mkdir(exist_ok=True,parents=True)
files=sorted(RAW.glob('extended_root_*.csv'))
if len(files)<2: raise RuntimeError('Need at least two independent preparations')
root_data=pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
if root_data['root'].nunique()!=len(files): raise ValueError('Duplicate/missing root identifiers')
assert root_data.N.nunique()==1 and root_data.N.iloc[0]==10000
assert set(np.round(root_data['lambda'].unique(),6))=={0,.5,1,1.5,2,3,4,6,8,12}
assert np.all(root_data['wealth_sum'].gt(0)) and np.all(root_data['min_wealth'].gt(0))
assert np.max(np.abs(root_data.wealth_sum-root_data.N))<1e-8
n=len(files)
rows=[]
for (lam,tm),d in root_data.groupby(['lambda','time'],sort=True):
    assert len(d)==n
    for key in ['G','M_pct','E_pct']:
        mean=float(d[key].mean());sd=float(d[key].std(ddof=1));half=t.ppf(.975,n-1)*sd/np.sqrt(n)
        rows.append(dict(lambda_value=lam,time=tm,metric=key,mean=mean,sd=sd,ci95_low=mean-half,ci95_high=mean+half,n_roots=n,N=10000))
summary=pd.DataFrame(rows)
summary.to_csv(DER/'fig3_abc_longtime_20roots.csv',index=False)
root_data[['root','N','preparation_round','lambda','time','G','M_pct','E_pct']].to_csv(DER/'fig3_abc_root_level.csv',index=False)

s4=pd.read_csv(RAW/'stage4_summary80.csv')
s4=s4[(s4['q']==.1)&(s4.tau==30)&(s4.metric=='delta_Cq')].copy()
labels={'redraw':'Full redraw','baseline':'Finite persistence','strong':'Strong persistence','fixed':'Fixed advantage'}
s4['process']=s4['mode'].map(labels)
assert s4.process.notna().all() and s4['n_roots'].eq(80).all()
s4[['mode','process','tw','mean','ci_lo','ci_hi','n_roots']].sort_values(['mode','tw']).to_csv(DER/'fig3_d_waiting_overlap_80roots.csv',index=False)
print(f'Prepared Fig 3 data: {n} independent roots; {len(summary)} long-run summary rows; {len(s4)} waiting-time rows.')
