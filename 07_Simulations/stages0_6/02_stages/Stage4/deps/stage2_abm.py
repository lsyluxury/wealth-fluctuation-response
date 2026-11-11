#!/usr/bin/env python3
"""Finite-N weak-selection process comparisons against the archived Yard-Sale rules.

Independent neutral prepared wealth roots; each process uses stationary scores;
seven coupled lambda branches: 0, +/-0.1, +/-0.25, +/-0.5.
NB: Repeated *futures* from one same root are not counted as independent preps.
"""
from __future__ import annotations
import os,json,argparse,time,math,sys
from pathlib import Path
import numpy as np
import pandas as pd
from legacy_engine import initialise, neutral_round, reseed, gini_numba
from engine_stage01 import advance_trace, initialize_process_score

LAM=np.array([0.,.1,-.1,.25,-.25,.5,-.5])
CHI=np.zeros_like(LAM)

def summary_all(w,bottom,top):
    n,b=w.shape
    X=np.empty((b,3),dtype=float)
    for z in range(b):
        s=np.argsort(w[:,z],kind='mergesort')
        total=w[:,z].sum()
        G=2*np.dot(np.arange(1,n+1,dtype=float),w[s,z])/(n*total)-(n+1)/n
        curr=np.zeros(n,dtype=bool);curr[s[-n//10:]]=True
        M=curr[bottom].mean();E=1-curr[top].mean()
        X[z]=G,M,E
    return X

def prepare(n,root_seed,gini_target=.65):
    w,a,u,perm=initialise(n,int(root_seed),1)
    for k in range(1,1500):
        neutral_round(w[:,0],a,u,perm,.14,.3,1.,10.)
        if gini_numba(w[:,0])>=gini_target:
            return w[:,0].copy(),a.copy(),u.copy(),k
    raise RuntimeError('Target not reached')

def one(root,n,nrounds,p,p1_rho,modes):
    w0,a0,u0,prep=prepare(n,root+101000, .65)
    idx=np.argsort(w0,kind='mergesort');bottom=idx[:n//2];top=idx[-n//10:]
    rows=[]
    # hetero: half fixed plus half fast resample, C1 matches Gaussian-Poisson precisely.
    phet=2*(1-p1_rho)
    for mode in modes:
        w=np.repeat(w0[:,None],len(LAM),axis=1)
        a=a0.copy() if mode==0 else np.zeros(n,dtype=float)
        u=u0.copy() if mode==0 else np.empty(n,dtype=float)
        perm=np.arange(n,dtype=np.int64)
        reseed(root+710000+mode*100000)
        if mode!=0:initialize_process_score(u,mode)
        if mode==5:
            rng=np.random.default_rng(root+10200000)
            mask=np.zeros(n,dtype=float)
            mask[rng.permutation(n)[:n//2]]=1.
            a=mask
        # u stationary independent from neutral wealth by preparation construction.
        reseed(root+810000)
        pp=phet if mode==5 else p
        advance_trace(w,a,u,perm,LAM,CHI,nrounds,.14,.3,1.,10.,mode,pp,False)
        X=summary_all(w,bottom,top)
        for z,lam in enumerate(LAM):
            rows.append(dict(root=int(root),N=n,prepare_rounds=int(prep),mode=int(mode),p=float(pp),round=nrounds,lambda_value=float(lam),G=X[z,0],M=X[z,1],E=X[z,2]))
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--roots',type=int,default=16);ap.add_argument('--N',type=int,default=10000);ap.add_argument('--rounds',type=int,default=300);ap.add_argument('--p',type=float,default=.1);ap.add_argument('--output',default='../results/stage2_abm_pilot.csv');ap.add_argument('--start',type=int,default=0);ap.add_argument('--modes',default='0,1,2,5');args=ap.parse_args()
    # finite round latent score C(1)=2/pi sum_k Poi(0.1)*asin(h^k/2)
    wt=math.exp(-.1);c=0.
    for k in range(100):
        c+=wt*(2/math.pi)*math.asin((.3**k)/2)
        wt*=.1/(k+1)
    rho1=3*c
    modes=[int(i) for i in args.modes.split(',')]
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    ti=time.time();rows=[]
    for j in range(args.start,args.start+args.roots):
        v=one(20000000+j,args.N,args.rounds,args.p,rho1,modes);rows.extend(v)
        if (j-args.start+1)%5==0:
            pd.DataFrame(rows).to_csv(out,index=False)
            print(f'PROGRESS {j-args.start+1}/{args.roots} elapsed={time.time()-ti:.1f}s rows={len(rows)}',flush=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print('FINISHED',out,'rho1=',rho1,'time=',time.time()-ti,flush=True)
