#!/usr/bin/env python3
"""Stage 6: independent finite-stake Boltzmann theoretical precision audit.
This uses a Stage-5 solver copy with a documented tiny-density
optimization (<1e-14 per wealth–ability node).  Starts from archived
intervention-time microstates, not from future ABM outcomes.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4';os.environ['NUMBA_NUM_THREADS']='4'
import sys, time, argparse, json
from pathlib import Path
import numpy as np, pandas as pd
# Load the Stage-6 copy of the solver (tiny-density shortcut is audited).
sys.path.insert(0,str(Path(__file__).resolve().parent))
from density_solver import DensitySolver,Settings
from common import make_initial,measure

OUT=Path(__file__).resolve().parents[1]/'results'
def run(seed,lam,chi,dx,na,steps=300,save_every=50):
    key=f's{seed}_l{lam:g}_c{chi:g}_dx{dx:g}_na{na}'
    key='trunc14_'+key+(f'_n{steps}' if steps != 300 else '')
    out=OUT/f'{key}.csv'
    if out.exists():
        print('SKIP existing',out,flush=True)
        return pd.read_csv(out)
    t=time.perf_counter()
    s=Settings(dx=dx,na=na,threads=4,epsilon=.14,r=10.,h=.3,mu=1.,win_quadrature=6)
    sol=DensitySolver(s)
    print('INIT',key,'sec',round(time.perf_counter()-t,1),'wn',len(sol.w),flush=True)
    f=make_initial(sol,seed=seed,factorize=False)
    P,inds,alphas=sol.prepare(lam,chi)
    print('INIT2',key,'sec',round(time.perf_counter()-t,1),'wn',len(sol.w),flush=True)
    ob={0,10,50,100,200,300}
    rows=[]
    skipped_mass=0.
    for n in range(steps+1):
        if n in ob:
            z=measure(f,sol.w,sol.u)
            r=dict(seed=seed,lam=lam,chi=chi,dx=dx,na=na,round=n,**z,sec=time.perf_counter()-t,skipped_mass_cumulative=skipped_mass)
            rows.append(r)
            print('SNAP',key,n,'sec',round(r['sec'],1),'G',round(r['G'],6),'M',round(r['M'],6),'E',round(r['E'],6),flush=True)
        if n<steps:
            f,st=sol.step(f,P,inds,alphas,chi)
            skipped_mass+=float(st[1])
    df=pd.DataFrame(rows)
    df.to_csv(out,index=False)
    with open(OUT/f'{key}_audit.json','w') as fp:
        json.dump(dict(settings=s.__dict__,kernel=sol.kernel_info,elapsed=time.perf_counter()-t,
             max_mass_error=float(np.max(abs(df['mass']-1))),max_wealth_error=float(np.max(abs(df['mean']-1))),
             max_tagging_error=float(np.max(abs(df['count_identity']))),min_density=float(df['min_mass'].min()),skipped_mass_total=skipped_mass),fp,indent=2)
    return df

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=2026101201)
 p.add_argument('--lam',type=float,default=4.);p.add_argument('--chi',type=float,default=.03)
 p.add_argument('--dx',type=float,default=.01);p.add_argument('--na',type=int,default=96)
 p.add_argument('--steps',type=int,default=300)
 a=p.parse_args();run(a.seed,a.lam,a.chi,a.dx,a.na,a.steps)
