#!/usr/bin/env python3
"""Handoff addition: reconstruct final Fig3d theory; preserve approved data by default.
The supplied v8 archive omitted its recomputation script. This implementation
uses the included original DensitySolver and the same archived initial state.
"""
from pathlib import Path
import argparse,json,os,sys,time
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
ROOT=Path(__file__).resolve().parents[1];P4=ROOT/'packages/Fig4';P3=ROOT/'packages/Fig3';sys.path.insert(0,str(P4/'code'))
import numpy as np,pandas as pd
from density_solver import DensitySolver,Settings
from common import make_initial,measure
TIMES=np.array([0,.1,.2,.5,1,2,5,10,15,20,25,30,40,50,60,80,100])
def main():
 p=argparse.ArgumentParser();p.add_argument('--lambdas',default='0,1,2,4,12');p.add_argument('--until',type=float,default=100);p.add_argument('--write-plot-table',action='store_true');a=p.parse_args()
 times=TIMES[TIMES<=a.until];obs=set(np.rint(10*times).astype(int));lams=list(map(float,a.lambdas.split(',')))
 if not len(times):raise ValueError('No observations requested')
 rows=[];audits=[]
 for lam in lams:
  cfg=Settings(dx=.05,na=24,threads=1,win_quadrature=6,poisson_max=12);sol=DensitySolver(cfg)
  state=make_initial(sol,seed=2026101201,factorize=True);prob,idx,al=sol.prepare(lam,0.);start=time.perf_counter()
  for n in range(max(obs)+1):
   if n in obs:
    z=measure(state,sol.w,sol.u)
    for met,mult in [('G',1),('M',100),('E',100)]:rows.append(dict(lambda_value=lam,time=n/10,metric=met,theory_mean=mult*z[met]))
   if n<max(obs):state,_=sol.step(state,prob,idx,al,0.)
  audits.append(dict(lambda_value=lam,seconds=time.perf_counter()-start,settings=cfg.__dict__,initial_seed=2026101201,initial_factorized=True,kernel=sol.kernel_info))
  pd.DataFrame([v for v in rows if v['lambda_value']==lam]).to_csv(ROOT/f'validation/Fig3d_theory_lambda{lam:g}.csv',index=False)
  print('Completed',lam,flush=True)
 new=pd.DataFrame(rows);new.to_csv(ROOT/'validation/Fig3d_recomputed_theory.csv',index=False)
 old=pd.read_csv(P3/'data/derived/d_theory_vs_ABM_5lambda_to100_mode0.csv')
 comp=new.merge(old,on=['lambda_value','time','metric'],suffixes=('_new','_supplied'),validate='1:1')
 error=float(abs(comp.theory_mean_new-comp.theory_mean_supplied).max())
 record=dict(rows_checked=len(comp),maximum_absolute_theory_difference=error,scope='G dimensionless; M/E percent; d only',runs=audits)
 (ROOT/'validation/Fig3d_recompute_check.json').write_text(json.dumps(record,indent=2))
 if a.write_plot_table:
  if len(lams)!=5 or max(obs)!=1000:raise ValueError('Full grid required to replace plot table')
  new.merge(old.drop(columns='theory_mean'),on=['lambda_value','time','metric'],validate='1:1').to_csv(P3/'data/derived/d_theory_vs_ABM_5lambda_to100_mode0.csv',index=False)
 print(json.dumps({'max_error':error,'rows':len(comp)}))
if __name__=='__main__':main()
