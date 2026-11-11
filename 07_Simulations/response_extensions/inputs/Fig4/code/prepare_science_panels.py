#!/usr/bin/env python3
"""Derive Fig.4 plot tables exclusively from the archived Stage 5 ABM root data.
All means/CI are 40 independent prepared roots, no artificial interpolated data.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
R=Path(__file__).resolve().parents[1]
RAW=R/'data/raw';DER=R/'data/derived';DER.mkdir(parents=True,exist_ok=True)

def save(df,name):df.to_csv(DER/name,index=False,float_format='%.15g')
def ci(x):
    v=np.asarray(x,float);n=len(v);avg=v.mean();hw=t.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n)
    return dict(mean=avg,ci95_low=avg-hw,ci95_high=avg+hw,n_roots=n)

# a. analytic contraction of a representative pair by redistribution alone.
x=0.1;eps=.14;dt=10.0
rows=[]
for chi in [0,.01,.03,.06]:
    rho=np.exp(-chi*dt)
    for ratio in np.linspace(1,3,401):
        y=ratio*x;xp=rho*x+(1-rho);yp=rho*y+(1-rho)
        rows.append(dict(chi=chi,Delta_t=dt,rho=rho,x_initial=x,y_initial=y,ratio_before=ratio,ratio_after=yp/xp,overturn=yp/xp<1+2*eps))
save(pd.DataFrame(rows),'a_analytic_ratio_contraction.csv')

# b/c: late common-state restart at tw=30, tau 0,5,10,20,30.
late=pd.read_csv(RAW/'late_retag_summary_40roots.csv')
l=late[(late.tw==30)&(late.metric.isin(['G','M']))].copy()
l['value']=np.where(l.metric=='M',100*l['mean'],l['mean'])
l['lo']=np.where(l.metric=='M',100*l.ci95_lo,l.ci95_lo)
l['hi']=np.where(l.metric=='M',100*l.ci95_hi,l.ci95_hi)
l=l[['tw','tau','chi','metric','value','lo','hi','n']].sort_values(['metric','chi','tau'])
save(l,'bc_late_common_state_timeseries.csv')

# d/e: full 42-condition endpoint grid at t=30; rank correlation is Spearman Ruw.
s=pd.read_csv(RAW/'summary_42conditions_40roots_time.csv')
ss=s[(np.isclose(s.time,30))&s.metric.isin(['G','M','Ruw'])].copy()
wide=ss.pivot(index=['lam','chi'],columns='metric',values='mean').reset_index().rename(columns={'M':'M_fraction','G':'Gini','Ruw':'rank_corr'})
wide['M_pct']=100*wide['M_fraction']
wide=wide.sort_values(['lam','chi'])
save(wide,'de_joint_endpoint_42.csv')

# f: root-wise difference-in-differences interaction at each chi; compare lambda 4 vs 0.
root=pd.read_csv(RAW/'full40_N10000.csv')
r=root[(np.isclose(root.time,30))&root.lambda_value.isin([0,4])].copy()
assert r.root.nunique()==40
index=['root','lambda_value','chi']
assert not r.duplicated(index).any()
rows=[]
for chi in sorted(r.chi.unique()):
    for met in ['G','M']:
        piv=r.pivot_table(index='root',columns=['lambda_value','chi'],values=met)
        v=((piv[(4.,chi)]-piv[(4.,0.)])-(piv[(0.,chi)]-piv[(0.,0.)])).to_numpy()
        d=ci(v)
        factor=100 if met=='M' else 1
        rows.append(dict(chi=chi,metric=met,mean=d['mean']*factor,ci95_low=d['ci95_low']*factor,ci95_high=d['ci95_high']*factor,n_roots=d['n_roots'],units='percentage points' if met=='M' else 'Gini index'))
save(pd.DataFrame(rows),'f_nonadditivity_paired_40roots.csv')

# summary of representative comparisons and alternative late waiting time.
late_end=late[np.isclose(late.tau,30)&late.tw.isin([30,50])&late.chi.isin([0,.03])&late.metric.isin(['G','M','Ruw'])].copy()
save(late_end,'SI_late_tw30_tw50.csv')

# Consistency checks with independent archived summary for specific contrasts.
pub=pd.read_csv(RAW/'rootpaired_contrasts_40roots.csv')
diffs=pd.read_csv(DER/'f_nonadditivity_paired_40roots.csv')
for met,target in [('G',.067145),('M',2.288)]:
    val=diffs[(diffs.metric==met)&np.isclose(diffs.chi,.03)]['mean'].iloc[0]
    assert abs(val-target)<.001,(met,val,target)
print('Plot tables prepared:',[f.name for f in sorted(DER.glob('*.csv'))])
