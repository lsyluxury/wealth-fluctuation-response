#!/usr/bin/env python3
"""Independent-root paired analysis of Stage4 ABM and late-state restarts."""
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t
BASE=Path(__file__).resolve().parents[1]

def stat(v):
 v=np.array(v,float);n=len(v);mean=v.mean();s=v.std(ddof=1);se=s/(n**.5);q=t.ppf(.975,n-1)
 return {'n_roots':n,'mean':mean,'sd':s,'se':se,'ci_lo':mean-q*se,'ci_hi':mean+q*se}

def effects(df):
 idx=['root','N','mode','tw','tau','q']; vals=['Cq','E','M_B50_to_Tq','G']
 p=df.pivot_table(index=idx,columns='lambda',values=vals).dropna()
 p.columns=[f'{metric}_lam{lam:g}' for metric,lam in p.columns]
 p=p.reset_index()
 for k in vals:p[f'delta_{k}']=p[f'{k}_lam4']-p[f'{k}_lam0']
 return p

def summarize_paired(p):
 z=[]
 for key,f in p.groupby(['N','mode','q','tw','tau']):
  for col in ('Cq_lam0','Cq_lam4','delta_Cq','E_lam0','E_lam4','delta_E','M_B50_to_Tq_lam0','M_B50_to_Tq_lam4','delta_M_B50_to_Tq','G_lam0','G_lam4','delta_G'):
   s=stat(f[col]);z.append(dict(zip(['N','mode','q','tw','tau'],key),metric=col,**s))
 return pd.DataFrame(z)

if __name__=='__main__':
 df=pd.read_csv(BASE/'results/stage4_main80.csv')
 p=effects(df)
 p.to_csv(BASE/'results/stage4_effects_by_root80.csv',index=False)
 s=summarize_paired(p);s.to_csv(BASE/'results/stage4_summary80.csv',index=False)
 show=s[(s.q==.1)&(s.tau==30)&(s.tw.isin([0.,30.,50.]))&(s.metric.isin(['Cq_lam4','delta_Cq','delta_G','delta_M_B50_to_Tq']))]
 print('PRIMARY q=.1 τ=30')
 print(show.pivot_table(index=['mode','tw'],columns='metric',values='mean').round(5))
 print('\nDELTA CQ 95% CI')
 print(show[show.metric=='delta_Cq'][['mode','tw','n_roots','mean','ci_lo','ci_hi']].round(5).to_string(index=False))
 print('\nFULL REDRAW delta max')
 print(p[p['mode']=='redraw'].groupby(['q','tw','tau'])['delta_Cq'].mean().abs().max())
 print('\nPERCENTILES q=1% λ4 Cq τ30')
 print(s[(s.q==.01)&(s.tau==30)&(s.metric=='Cq_lam4')&(s.tw.isin([0.,30.,50.]))][['mode','tw','mean','ci_lo','ci_hi']].round(5).to_string(index=False))
 print('\nAUDIT')
 z=pd.read_csv(BASE/'results/stage4_main80_audit.csv');print('wealth mean error',abs(z.mean_w-1).max(),'min wealth',z.min_w.min())
 try:
  rests=pd.read_csv(BASE/'results/stage4_late_state_restarts40.csv')
  r=rests.pivot_table(index=['root','mode','early_lambda','tw','tau'],columns='late_lambda',values=['Cq','E','M','G'])
  r.columns=[f'{a}_{b:g}' for a,b in r.columns];r=r.reset_index()
  for k in ['Cq','E','M','G']:r[f'delta_{k}']=r[f'{k}_4']-r[f'{k}_0']
  r.to_csv(BASE/'results/stage4_late_state_restarts_effects.csv',index=False)
  print('\nRESTART: λearly4, λfuture4 minus future0')
  for (m,tw),g in r[r.early_lambda==4].groupby(['mode','tw']):
   print(m,tw, 'delta_Cq', {k:round(v,5) for k,v in stat(g.delta_Cq).items() if k in ('mean','ci_lo','ci_hi')}, 'deltaG',round(g.delta_G.mean(),5))
 except FileNotFoundError:print('restart pending')
