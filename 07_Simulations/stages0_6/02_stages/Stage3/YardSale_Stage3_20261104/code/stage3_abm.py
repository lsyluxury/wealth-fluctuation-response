#!/usr/bin/env python3
"""Strong-bias Yard-Sale process comparison; reuses validated Stage2 engine verbatim.

Each root is an independent neutral preparation at G*=0.65. Within a root,
all lambda branches of a mode use the same score history, pairings and win uniforms.
Model time t = complete rounds/10. Does not change historical archived engine.
"""
from __future__ import annotations
import os,sys,math,argparse,time
from pathlib import Path
import numpy as np,pandas as pd
STAGE2=Path(__file__).resolve().parent
sys.path.insert(0,str(STAGE2))
from stage2_abm import prepare
from legacy_engine import reseed
from engine_stage01 import advance_trace,initialize_process_score

ALL_LAM = np.array([0., .25,-.25,.5,-.5,1.,-1.,2.,-2.,4.,-4.,8.,-8.])
CORE_LAM = np.array([0.,4.,-4.])
CHECKPOINTS = [0,10,20,50,100,200,300]

def metric_rows(w,u,bottom,top,root,mode,lams,t,prep):
    n,b=w.shape
    idx=np.arange(1,n+1,dtype=float)
    rows=[]
    for j in range(b):
        y=w[:,j];order=np.argsort(y,kind='mergesort')
        g=2*np.dot(idx,y[order])/(n*y.sum())-(n+1)/n
        T=np.zeros(n,dtype=bool);T[order[-n//10:]]=True
        m=float(T[bottom].mean());e=float(1-T[top].mean())
        Q=float(np.mean((y-1.)*u))
        rows.append({'root':root,'mode':mode,'N':n,'lambda_value':lams[j],
            'round':t,'time':t/10,'G':g,'M':m,'E':e,'Q':Q,
            'sum_w':float(y.sum()),'min_w':float(y.min()),'prep_rounds':prep})
    return rows

def simulate_one(seed,N,modes,checkpoints):
    w0,a0,u0,prep=prepare(N,int(seed)+101000,.65)
    order=np.argsort(w0,kind='mergesort');bottom=order[:N//2];top=order[-N//10:]
    rho1=.9312501788413264;pslow=.01;pfast=2*(1-rho1)-pslow
    rows=[]
    for mode in modes:
        lams=ALL_LAM if mode in (1,2) else CORE_LAM
        b=len(lams)
        w=np.repeat(w0[:,None],b,axis=1)
        a=a0.copy() if mode==0 else np.zeros(N,dtype=float)
        u=u0.copy() if mode==0 else np.empty(N,dtype=float)
        perm=np.arange(N,dtype=np.int64)
        reseed(int(seed)+710000+mode*100000)
        if mode!=0:initialize_process_score(u,mode)
        if mode==6:
            rng=np.random.default_rng(int(seed)+10200000)
            a[rng.permutation(N)[:N//2]]=1.
        reseed(int(seed)+810000)
        prev=0
        rows+=metric_rows(w,u,bottom,top,seed,mode,lams,0,prep)
        for checkpoint in checkpoints[1:]:
            advance_trace(w,a,u,perm,lams,np.zeros(b),checkpoint-prev,
                          .14,.3,pslow if mode==6 else 1.,10.,mode,
                          pfast if mode==6 else .1,False)
            rows+=metric_rows(w,u,bottom,top,seed,mode,lams,checkpoint,prep)
            prev=checkpoint
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--start',type=int,default=0)
    ap.add_argument('--roots',type=int,default=10)
    ap.add_argument('--N',type=int,default=10000)
    ap.add_argument('--modes',default='0,1,2,6')
    ap.add_argument('--out',default=str(Path(__file__).resolve().parents[1]/'results'/'stage3_abm_new.csv'))
    args=ap.parse_args()
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    modes=list(map(int,args.modes.split(',')))
    t0=time.time(); rows=[]
    for j in range(args.start,args.start+args.roots):
        seed=20000000+j
        rows.extend(simulate_one(seed,args.N,modes,CHECKPOINTS))
        if (j-args.start+1)%5==0:
            pd.DataFrame(rows).to_csv(out,index=False)
            print('PROGRESS',j-args.start+1,args.roots,'secs',round(time.time()-t0,2),'rows',len(rows),flush=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print('FINISHED',out,'time',round(time.time()-t0,3),flush=True)
