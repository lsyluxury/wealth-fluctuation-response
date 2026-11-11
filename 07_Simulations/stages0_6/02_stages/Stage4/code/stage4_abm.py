#!/usr/bin/env python3
"""Stage 4: two-time rank overlap in original Yard-Sale engine.
All observations are endpoint overlap after retagging at each tw; NOT first passage.
Stage A common neutral-prepared G*=0.65; no redistribution, full pairing each round.
Four conditions: full redraw (mode4), baseline Gaussian-Poisson (h=.3),
strong persistence Gaussian-Poisson (h=.9), fixed score (mode3).
Each root and process shares matched lambda=0 vs lambda=4 trade uniforms and pairing.
All starts/observations are model time (10 full rounds = 1 time unit).
"""
import os,sys,time,argparse,math
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deps'))
from engine_stage3 import advance_trace
from stage2_abm import prepare
from legacy_engine import reseed

WAIT=(0,5,10,20,30,50)
TAUS=(0,0.1,0.5,1,2,3,5,10,15,20,30)
QS=(.01,.1)
LAM=np.array([0.,4.],dtype=float)
CHI=np.zeros(2)
MODES=('redraw','baseline','strong','fixed')


def metrics(y,rank_order=None):
    n=len(y);s=np.argsort(y,kind='mergesort') if rank_order is None else rank_order
    g=2*np.dot(np.arange(1,n+1,dtype=float),y[s])/(n*np.sum(y))-(n+1)/n
    return float(g),s


def one_process(root,w0,a0,u0,mode,tag_records=False):
    n=len(w0)
    ws=np.repeat(w0[:,None],2,axis=1)
    al=a0.copy();sc=u0.copy();perm=np.arange(n,dtype=np.int64)
    mu=1.;h=.3;kernelmode=0
    if mode=='redraw':kernelmode=4
    if mode=='baseline':kernelmode=0;h=.3
    if mode=='strong':kernelmode=0;h=.9
    if mode=='fixed':kernelmode=3
    # All control/selection branches of a process use the same source of ability and trade draws.
    reseed(root+810000)
    waits=[int(round(t*10)) for t in WAIT]
    tau_rounds=[int(round(t*10)) for t in TAUS]
    schedule={0}
    for tw in waits:
        schedule.add(tw)
        for lag in tau_rounds:schedule.add(tw+lag)
    checkpoint_list=sorted(schedule)
    tags={} # (tw,lambda_index,q)-> (boolean array top, boolean array bottom)
    rows=[];audit=[];prev=0
    for rr in checkpoint_list:
        if rr>prev:
            advance_trace(ws,al,sc,perm,LAM,CHI,rr-prev,.14,h,mu,10.,kernelmode,.1,False)
        prevr=prev;prev=rr
        current=[]
        for bi in range(2):
            y=ws[:,bi];g,rank=metrics(y)
            currtop={q:np.zeros(n,dtype=bool) for q in QS}
            for q in QS:currtop[q][rank[-int(round(n*q)):]]=True
            current.append((g,rank,currtop))
        if rr in waits:
            for bi in range(2):
                _,rank,topnow=current[bi]
                bot=np.zeros(n,dtype=bool);bot[rank[:n//2]]=True
                for q in QS:tags[(rr,bi,q)]=(topnow[q].copy(),bot.copy())
        for tw in waits:
            if rr<tw:continue
            if rr-tw not in tau_rounds:continue
            for bi in range(2):
                g,rank,ct=current[bi]
                for q in QS:
                    ref,bottom=tags[(tw,bi,q)]
                    frac=float(np.mean(ct[q][ref]))
                    C=(frac-q)/(1-q)
                    M=float(np.mean(ct[q][bottom]))
                    rows.append({'root':root,'N':n,'mode':mode,'lambda':LAM[bi],
                         'tw':tw/10.,'tau':(rr-tw)/10.,'time':rr/10.,'q':q,
                         'Cq':C,'stay':frac,'E':1-frac,'M_B50_to_Tq':M,'G':g})
        if rr in (0,300,500,800):
            for bi in range(2):
                g,rank,_=current[bi]
                audit.append({'root':root,'N':n,'mode':mode,'lambda':LAM[bi],
                         'time':rr/10.,'G':g,'mean_w':float(ws[:,bi].mean()),
                         'min_w':float(ws[:,bi].min()),
                         'Q':float(np.mean((ws[:,bi]-1)*sc))})
    return rows,audit


def run(root,N,kind):
    if kind=='archive':
        f=ROOT/'inputs'/f's{root}_g65_selection.npz'
        with np.load(f) as z:
            w0=z['w'].copy();a0=z['a'].copy();u0=z['u'].copy()
        preprounds=-1
        assert len(w0)==N
    else:
        w0,a0,u0,preprounds=prepare(N,root+101000,.65)
    rows=[];audit=[]
    for mode in MODES:
        z,zz=one_process(root,w0,a0,u0,mode)
        rows.extend(z);audit.extend(zz)
    return rows,audit,preprounds

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=0);ap.add_argument('--roots',type=int,default=10)
    ap.add_argument('--N',type=int,default=10000);ap.add_argument('--kind',choices=['independent','archive'],default='independent')
    ap.add_argument('--out',default=str(ROOT/'results'/'stage4_main.csv'));args=ap.parse_args()
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    tm=time.time();rows=[];au=[];started=0
    for j in range(args.start,args.start+args.roots):
        root=20000000+j if args.kind=='independent' else 2026101201+j
        z,aa,pp=run(root,args.N,args.kind);rows.extend(z);au.extend(aa)
        if (j-args.start+1)%2==0 or j==args.start+args.roots-1:
            pd.DataFrame(rows).to_csv(out,index=False)
            pd.DataFrame(au).to_csv(out.with_name(out.stem+'_audit.csv'),index=False)
            print('PROGRESS',j-args.start+1,args.roots,'seconds',round(time.time()-tm,2),'preparation',pp,flush=True)
    print('FINISHED',out,'seconds',round(time.time()-tm,2),flush=True)
