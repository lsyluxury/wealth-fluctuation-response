#!/usr/bin/env python3
"""Extend the EXACT baseline Yard-Sale ABM for Figure 3; no invented curves.

Only the trajectory length and selection grid change versus stage 4.
N=10000; every individual trades once per full matching round;
epsilon=.14, r=10; Poisson Gaussian advantage renewal (mu=1,h=.3), tax=0.
Common random draws across all lambda branches within each independent neutral-prepared root.
The preparation and seed convention match stage4_main80.csv exactly for the
lambda=0,4 branches over their 0..80 overlap.

Experimental purpose: display long-time *approach to the finite-N Gini upper bound*,
not a stationary phase or thermodynamic-limit saturation.
"""
from pathlib import Path
import argparse, sys, time
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
from stage2_abm import prepare
from engine_stage3 import advance_trace
from legacy_engine import reseed, gini_numba

LAMBDA = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 8, 12], dtype=np.float64)
N_ROUNDS = np.array([0,1,2,3,5,10,20,30,50,100,150,200,250,300,400,500,600,800,1000,1200,1500,2000,2500,3000,4000,5000,6000,8000,10000,12000], dtype=int)

def run_one(seed, n=10000):
    w0,a,u,prep = prepare(n, seed+101000,.65)
    wealth = np.repeat(w0[:,None], len(LAMBDA), axis=1)
    perm = np.arange(n,dtype=np.int64)
    rank0=np.argsort(w0,kind='mergesort')
    bottom=np.zeros(n,dtype=bool);bottom[rank0[:n//2]]=True
    top=np.zeros(n,dtype=bool);top[rank0[-n//10:]]=True
    reseed(seed+810000)
    result=[];previous=0
    for rr in N_ROUNDS:
        if rr>previous:
            advance_trace(wealth,a,u,perm,LAMBDA,np.zeros_like(LAMBDA),int(rr-previous),.14,.3,1.,10.,0,.1,False)
        previous=rr
        for j,lam in enumerate(LAMBDA):
            wr=wealth[:,j]
            G=gini_numba(wr)
            current_order=np.argsort(wr,kind='mergesort')
            curr_top=np.zeros(n,dtype=bool);curr_top[current_order[-n//10:]]=True
            M=100.0*np.mean(curr_top[bottom])
            E=100.0*(1.0-np.mean(curr_top[top]))
            result.append((seed,n,int(prep),float(lam),int(rr),rr/10.,float(G),float(M),float(E),float(np.sum(wr)),float(np.min(wr))))
    return result

if __name__=='__main__':
    pa=argparse.ArgumentParser()
    pa.add_argument('--start',type=int,default=0)
    pa.add_argument('--roots',type=int,default=1)
    pa.add_argument('--N',type=int,default=10000)
    pa.add_argument('--outdir',type=Path,default=ROOT/'data'/'raw')
    args=pa.parse_args()
    args.outdir.mkdir(parents=True,exist_ok=True)
    for i in range(args.start,args.start+args.roots):
        seed=20000000+i
        outfile=args.outdir/f'extended_root_{seed}.csv'
        if outfile.exists():
            print('SKIP',outfile,flush=True);continue
        t0=time.time()
        rows=run_one(seed,args.N)
        df=pd.DataFrame(rows,columns=['root','N','preparation_round','lambda','round','time','G','M_pct','E_pct','wealth_sum','min_wealth'])
        tmp=outfile.with_suffix('.tmp');df.to_csv(tmp,index=False);tmp.replace(outfile)
        print('FINISHED',seed,'s',round(time.time()-t0,2),'G_range',round(float(df[df.time==1200].G.min()),5),round(float(df[df.time==1200].G.max()),5),flush=True)
