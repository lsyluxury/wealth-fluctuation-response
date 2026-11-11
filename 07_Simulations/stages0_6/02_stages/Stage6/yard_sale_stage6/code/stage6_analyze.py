"""Stage6 statistics: root-block paired differences, conditional futures and robustness."""
from pathlib import Path
import glob,json
import numpy as np,pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
def ci(a):
 a=np.asarray(a,dtype=float);a=a[np.isfinite(a)];n=len(a)
 if n<2:return (float(a.mean()) if n else np.nan,np.nan,np.nan,n)
 m=a.mean();half=t.ppf(.975,n-1)*a.std(ddof=1)/np.sqrt(n)
 return m,m-half,m+half,n
# Sensitivity
files=[p for p in R.glob('robust_N*_r*_n*.csv') if '_audit' not in p.name and 'N2000' not in p.name]
print('ROBUST_FILES',[(p.name,len(pd.read_csv(p))) for p in files])
rob=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
rob=rob.drop_duplicates(subset=['N','root','Gstar','round','lambda_value','chi','q'])
rob.to_csv(R/'stage6_robustness_all_root.csv',index=False)
rows=[]
for (n,gstar,q),gg in rob[rob['round']==300].groupby(['N','Gstar','q']):
 pv=gg.pivot(index='root',columns=['lambda_value','chi'],values=['G','M','E','Cq','Ruw'])
 for metric in ['G','M','E','Cq','Ruw']:
  for name,c1,c0 in [('selection',(4.,0.),(0.,0.)),('redistribution_given_selection',(4.,.03),(4.,0.)),('joint_vs_fair',(4.,.03),(0.,0.))]:
   vals=pv[(metric,*c1)]-pv[(metric,*c0)]
   m,l,h,num=ci(vals)
   rows.append(dict(N=n,Gstar=gstar,q=q,metric=metric,contrast=name,mean=m,ci_lo=l,ci_hi=h,n_roots=num))
sm=pd.DataFrame(rows);sm.to_csv(R/'stage6_robustness_contrasts.csv',index=False)
print('ROBUST main contrasts G*=.65 q=.10:')
print(sm[(sm.Gstar==.65)&(sm.q==.1)&(sm.metric.isin(['G','M','E']))][['N','metric','contrast','mean','ci_lo','ci_hi','n_roots']].to_string(index=False,float_format=lambda x:f'{x:.5f}'))
print('ROBUST initial backgrounds N=100k q=.10')
print(sm[(sm.N==100000)&(sm.q==.1)&(sm.contrast=='redistribution_given_selection')&sm.metric.isin(['G','M','E'])][['Gstar','metric','mean','ci_lo','ci_hi']].to_string(index=False,float_format=lambda x:f'{x:.5f}'))
print('ROBUST top cutoffs N=100k G*=.65')
print(sm[(sm.N==100000)&(sm.Gstar==.65)&(sm.contrast=='redistribution_given_selection')&sm.metric.isin(['M','E'])][['q','metric','mean','ci_lo','ci_hi']].to_string(index=False,float_format=lambda x:f'{x:.5f}'))

# Kinetic vs single-future six-root stage5, and conditional 8 futures
files=sorted(R.glob('trunc14_s*dx0.025_na48.csv'));kt=pd.concat([pd.read_csv(f) for f in files]);kt=kt.rename(columns={'seed':'root'})
cond_files=sorted(R.glob('conditional_root*_f8.csv'))
cf=pd.concat([pd.read_csv(f) for f in cond_files]).drop_duplicates(['root','future_seed','round']);cf.to_csv(R/'stage6_conditional_futures_all.csv',index=False)
cm=cf.groupby(['root','round'],as_index=False)[['G','M','E']].mean()
zz=kt.merge(cm,on=['root','round'],suffixes=('_theory','_futureMean'))
rws=[]
for step,gr in zz.groupby('round'):
 for metric in ['G','M','E']:
  d=gr[metric+'_theory']-gr[metric+'_futureMean'];m,l,h,num=ci(d)
  rws.append(dict(round=step,metric=metric,mean_bias=m,ci_lo=l,ci_hi=h,n_roots=num))
mm=pd.DataFrame(rws);mm.to_csv(R/'stage6_kinetic_conditional_precision.csv',index=False)
print('KINETIC vs 8 conditional ABM futures/root:')
print(mm[mm['round'].isin([50,100,200,300])].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
print('Condition futures root endpoint E variance:')
print(cf[cf['round']==300].groupby('root').E.agg(['mean','std']).to_string())

# compare 0.025 & 0.01 for root1
v=kt[(kt.root==2026101201)&(kt['round']==300)].iloc[0]
hi=pd.read_csv(R/'trunc14_s2026101201_l4_c0.03_dx0.01_na96.csv').iloc[-1]
print('GRID',[(m,v[m],hi[m],hi[m]-v[m]) for m in ['G','M','E']])

report={'N_roots':rob.groupby('N').root.nunique().to_dict(),'Gstar_targets':list(sorted(rob.Gstar.unique())),'q_targets':list(sorted(rob.q.unique())),
        'conditional_futures_per_root':cf[cf['round']==300].groupby('root').future_seed.nunique().to_dict()}
(R/'stage6_sample_overview.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
