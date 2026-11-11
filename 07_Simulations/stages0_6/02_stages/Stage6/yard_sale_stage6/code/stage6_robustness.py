#!/usr/bin/env python3
"""Stage 6: independent ABM sensitivity in initial inequality, N and top-q.
Sources: original finite-stake engine + stage0 extension, unchanged source copies.
Every G* is reached on one neutral root history. Four branches share the SAME
prepared joint state and post-branch random input stream, treatment differences
are computed within independent population roots.
"""
from __future__ import annotations
import argparse, math, time, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from legacy_engine import initialise,neutral_round,reseed,gini_numba
from engine_stage3 import advance_trace

OUT=Path(__file__).resolve().parents[1]/'results'
EPS=.14; H=.3; MU=1.; R=10.; TARGETS=(.50,.65,.80)
CONDS=((0.,0.),(4.,0.),(0.,.03),(4.,.03))
TOPS=(.01,.05,.10)

def prepared_targets(N,root):
    w,a,u,perm=initialise(N,root+101000,1)
    result={}
    for rr in range(1,1500):
        neutral_round(w[:,0],a,u,perm,EPS,H,MU,R)
        g=gini_numba(w[:,0])
        for target in TARGETS:
            if g>=target and target not in result:
                result[target]=(w[:,0].copy(),a.copy(),u.copy(),rr,float(g))
        if len(result)==len(TARGETS):return result
    raise RuntimeError('neutral prep did not reach all targets')

def one_root(N,root):
    state=prepared_targets(N,root)
    rows=[];aud=[]
    for it,gstar in enumerate(TARGETS):
        w0,a0,u0,prep,actual=state[gstar]
        la=np.array([k[0] for k in CONDS]);ch=np.array([k[1] for k in CONDS]); B=len(la)
        order0=np.argsort(w0,kind='mergesort')
        bot=np.zeros(N,dtype=bool);bot[order0[:N//2]]=True
        initial_top={q: np.zeros(N,dtype=bool) for q in TOPS}
        for q in TOPS:initial_top[q][order0[-int(round(N*q)):]]=True
        w=np.repeat(w0[:,None],B,axis=1);a=a0.copy();u=u0.copy();perm=np.arange(N,dtype=np.int64)
        reseed(int(root+810000+it*10000))
        for obs in (0,100,300):
            if obs==0:pass
            elif obs==100:advance_trace(w,a,u,perm,la,ch,100,EPS,H,MU,R,0,.1,False)
            elif obs==300:advance_trace(w,a,u,perm,la,ch,200,EPS,H,MU,R,0,.1,False)
            us=np.argsort(np.argsort(u,kind='mergesort'),kind='mergesort').astype(np.float64)
            us-= (N-1)/2
            varu=float(np.dot(us,us))
            for j,(lam,chi) in enumerate(CONDS):
                ww=w[:,j];sr=np.argsort(ww,kind='mergesort');total=float(ww.sum())
                gini=2*np.dot(np.arange(1,N+1,dtype=float),ww[sr])/(N*total)-(N+1)/N
                wr=np.empty(N,dtype=np.float64);wr[sr]=np.arange(N,dtype=float)-(N-1)/2
                spearman=float(np.dot(wr,us)/varu)
                Q=float(np.mean((ww-1)*u))
                for q in TOPS:
                    k=int(round(q*N));tm=np.zeros(N,dtype=bool);tm[sr[-k:]]=True
                    m=float(tm[bot].mean());e=1-float(tm[initial_top[q]].mean());c=(1-e-q)/(1-q)
                    rows.append(dict(N=N,root=root,Gstar=gstar,G0=actual,prep_round=prep,
                       lambda_value=lam,chi=chi,round=obs,time=obs/10,q=q,G=gini,M=m,E=e,Cq=c,Ruw=spearman,Q=Q,
                       total_error=total-N,min_w=float(ww.min()),
                       count_error=float(e-(N/2*m+np.count_nonzero(tm & (~bot) & (~initial_top[q])))/k)))
                aud.append(dict(N=N,root=root,Gstar=gstar,round=obs,lam=lam,chi=chi,G=gini,
                         total_error=total-N,min_w=float(ww.min()),Ruw=spearman))
    return rows,aud

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--N',type=int,default=25000);p.add_argument('--roots',type=int,default=6)
    p.add_argument('--root-start',type=int,default=0);p.add_argument('--prefix',default='robust')
    p.add_argument('--targets',default='.50,.65,.80',help='comma-separated preparation Gini targets')
    a=p.parse_args();TARGETS=tuple(float(x) for x in a.targets.split(','));allr=[];alla=[];t=time.perf_counter()
    out=OUT/f'{a.prefix}_N{a.N}_r{a.root_start}_n{a.roots}.csv'
    outa=OUT/f'{a.prefix}_N{a.N}_r{a.root_start}_n{a.roots}_audit.csv'
    for k in range(a.roots):
        root=30000000+int(a.root_start)+k
        rec,au=one_root(a.N,root);allr+=rec;alla+=au
        pd.DataFrame(allr).to_csv(out,index=False)
        pd.DataFrame(alla).to_csv(outa,index=False)
        print('FINISHED',k+1,'/',a.roots,'N',a.N,'root',root,'runtime',round(time.perf_counter()-t,1),'G0',[(x,round(rec[i*36]['G0'],5)) for i,x in enumerate(TARGETS)],flush=True)
    print('RESULT',out,flush=True)
