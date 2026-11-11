#!/usr/bin/env python3
"""Full nonlinear (not Taylor truncated) finite-stake tagged kinetic solver.
Exact discrete two-state advantage and quadrature-integrated uniform advantage.
Uses original Phase4 density_solver trade operator on same empirical wealth initial.
"""
from __future__ import annotations
import os,sys,math,time,argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['NUMBA_NUM_THREADS']='1'
ORIG=Path(__file__).resolve().parents[1]/'phase4'/'src'
sys.path.insert(0,str(ORIG))
from density_solver import DensitySolver,Settings
from common import make_initial,measure

CHECKPOINTS={0,1,2,5,10,20,50,100,150,200,250,300}

def simulate(mode,lam,dx,na,seed,rounds=300):
    if mode == 2:na=2
    cfg=Settings(dx=dx,na=na,threads=1,win_quadrature=6,poisson_max=12)
    t0=time.time();sol=DensitySolver(cfg)
    if mode==1:
        p=.1
        sol.R=(1-p)*np.eye(na)+(p/na)*np.ones((na,na))
    elif mode==7:
        p_redraw=.05;p_flip=.025
        reverse=np.eye(na)[:,::-1]
        sol.R=(1-p_redraw-p_flip)*np.eye(na)+(p_redraw/na)*np.ones((na,na))+p_flip*reverse
    elif mode==2:
        p=.1
        sol.u=np.array([-1/math.sqrt(3),1/math.sqrt(3)])
        sol.R=np.array([[1-p/2,p/2],[p/2,1-p/2]])
        U=sol.u
        sol.win[lam]=expit(lam*(U[:,None]-U[None,:]))
    else:raise ValueError('mode 1, 2 or 7')
    G=make_initial(sol,seed=seed,factorize=True)
    P,idx,al=sol.prepare(lam,0.)
    assert G.shape[1]==na
    rows=[];max_mass=0;max_mean=0;max_tag=0;min_density=1.;statmax=[0.,0.,0.,0.]
    at0=measure(G,sol.w,sol.u)
    for rr in range(rounds+1):
        if rr in CHECKPOINTS or rr==rounds:
            z=measure(G,sol.w,sol.u)
            rows.append({'seed':seed,'mode':mode,'lambda_value':lam,'round':rr,'time':rr/10,
                         'dx':dx,'na':na,'n_wealth':len(sol.w),**z})
        if rr==rounds:break
        G,st=sol.step(G,P,idx,al,0.)
        statmax=np.maximum(statmax,abs(st))
        max_mass=max(max_mass,abs(G.sum()-1))
        max_mean=max(max_mean,abs((G*sol.w).sum()-1))
        max_tag=max(max_tag,float(np.max(np.abs(G.sum(axis=(1,2))-[.5,.4,.1]))))
        min_density=min(min_density,float(G.min()))
        if (rr+1)%50==0:print('KINETIC_PROGRESS',mode,lam,seed,rr+1,round(time.time()-t0,2),flush=True)
    return rows,{'mode':mode,'lambda_value':lam,'seed':seed,'dx':dx,'na':na,'initial_metrics':at0,
                 'max_mass_error':float(max_mass),'max_mean_error':float(max_mean),
                 'max_tag_mass_error':float(max_tag),'min_density':min_density,
                 'max_numeric_corrections':list(map(float,statmax)),
                 'elapsed_sec':float(time.time()-t0),'settings':cfg.__dict__}

if __name__=='__main__':
 ap=argparse.ArgumentParser()
 ap.add_argument('--modes',default='1,2');ap.add_argument('--lambda-values',default='0,4')
 ap.add_argument('--na',type=int,default=24);ap.add_argument('--dx',type=float,default=.05)
 ap.add_argument('--seed',type=int,default=2026101201);ap.add_argument('--rounds',type=int,default=300)
 ap.add_argument('--out',default=str(Path(__file__).resolve().parents[1]/'results'/'stage3_kinetic_new.csv'))
 args=ap.parse_args();out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
 rows=[];aud={}
 for mode in list(map(int,args.modes.split(','))):
  for lam in list(map(float,args.lambda_values.split(','))):
   z,a=simulate(mode,lam,args.dx,args.na,args.seed,args.rounds);rows.extend(z)
   aud[f'{mode}_lam{lam:g}']=a
   pd.DataFrame(rows).to_csv(out,index=False)
   out.with_suffix('.json').write_text(json.dumps(aud,indent=2,ensure_ascii=False))
   print('KINETIC_COMPLETED',mode,lam,'GME',tuple(round(z[-1][v],6) for v in ('G','M','E')),'secs',round(a['elapsed_sec'],2),flush=True)
