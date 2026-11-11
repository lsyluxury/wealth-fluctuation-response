#!/usr/bin/env python3
"""Independent finite-stake tagged Boltzmann solver for Stage5 joint lambda,chi initial-state branches.
Uses ONLY initial archived N=100000 joint state. Never consumes ABM future states.
Projection and quadrature are numerical approximations; tax is post-trade and post-renewal.
"""
import os
os.environ['NUMBA_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time,json
from pathlib import Path
import numpy as np,pandas as pd
from density_solver import DensitySolver,Settings,top_fractions
from common import make_initial, measure
ROOT=Path(__file__).resolve().parents[1]

def kinetic_one(seed,lam,chi,dx,na,steps=300):
 stamp=time.perf_counter()
 st=Settings(dx=dx,na=na,h=.3,mu=1.,r=10.,epsilon=.14,threads=1,win_quadrature=6)
 sol=DensitySolver(st);G=make_initial(sol,seed=seed,factorize=False)
 P,inds,alph=sol.prepare(lam,chi)
 rec=[];merr=0.;werr=0.;emin=0.;coh=0.;obs={0,50,100,200,300}
 for rr in range(steps+1):
  if rr in obs:
   z=measure(G,sol.w,sol.u)
   cm=np.max(abs(G.sum(axis=(1,2))-np.array([.5,.4,.1])))
   rec.append(dict(seed=seed,lambda_value=lam,chi=chi,dx=dx,na=na,round=rr,time=rr/10.,
                   G=z['G'],M=z['M'],E=z['E'],Q=z['Q'],mass=z['mass'],mean=z['mean'],
                   min_mass=z['min_mass'],count_identity=z['count_identity'],cohort_error=cm))
   merr=max(merr,abs(z['mass']-1));werr=max(werr,abs(z['mean']-1));emin=min(emin,z['min_mass']);coh=max(coh,cm)
  if rr<steps:G,_=sol.step(G,P,inds,alph,chi)
 info=dict(seed=seed,lam=lam,chi=chi,dx=dx,na=na,steps=steps,elapsed=time.perf_counter()-stamp,
           max_mass_error=merr,max_mean_error=werr,min_mass=emin,max_cohort_error=coh,
           grid_nodes=len(sol.w),ability_bins=na)
 return rec,info

if __name__=='__main__':
 ap=argparse.ArgumentParser()
 ap.add_argument('--seed',type=int,default=2026101201)
 ap.add_argument('--lambda-value',type=float,default=4.)
 ap.add_argument('--chi',type=float,default=.03)
 ap.add_argument('--dx',type=float,default=.05)
 ap.add_argument('--na',type=int,default=24)
 ap.add_argument('--out',default=str(ROOT/'results'/'kinetic_demo.csv'))
 args=ap.parse_args();r,info=kinetic_one(args.seed,args.lambda_value,args.chi,args.dx,args.na)
 out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(r).to_csv(out,index=False)
 out.with_suffix('.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
 print('RESULT',info,flush=True)
 print('ENDPOINT',r[-1],flush=True)
