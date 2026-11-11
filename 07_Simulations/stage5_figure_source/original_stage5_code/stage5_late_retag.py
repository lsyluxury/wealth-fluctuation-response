#!/usr/bin/env python3
"""Stage 5 late tax policy on matched selected-developed w/a/u initial condition.
Prior evolution: lambda=4 chi=0 from neutral G*=.65 to tw=30 or 50;
at each waiting time clone the entire state, retag top decile and bottom half,
then continue lambda=4 with chi in [0,.01,.03,.06] for tau in [0,5,10,20,30].
Within-root tax contrasts are causal interventions on same full model state.
"""
import sys, argparse, time, math
from pathlib import Path
import numpy as np, pandas as pd
from stage5_grid_abm import make_prepared,ranks
from legacy_engine import reseed
from engine_stage3 import advance_trace

CHI=[0.,.01,.03,.06]
TW=[30,50]
TAU=[0,5,10,20,30]

def metricset(w,u,bottom,top,rr,tw,root,kind):
 N=w.shape[0];out=[];_,uRank=ranks(u);uRank=uRank.astype(float);d=uRank-(N-1)/2
 denom=np.sum(d*d)
 for j,chi in enumerate(CHI):
  v=w[:,j];s,wr=ranks(v)
  G=2*np.dot(np.arange(1,N+1,dtype=float),v[s])/(N*v.sum())-(N+1)/N
  curr=np.zeros(N,bool);curr[s[-N//10:]]=True
  M=float(curr[bottom].mean());E=float(1-curr[top].mean())
  R=float(np.dot(wr-(N-1)/2,d)/denom)
  out.append(dict(root=root,kind=kind,tw=tw,tau=rr/10.,time=tw+rr/10.,chi=chi,lambda_value=4.,G=G,M=M,E=E,Cq=1-E/.9,Ruw=R,Q=float(np.mean((v-1)*u)),
                  wealth_error=float(v.sum()-N),min_w=float(v.min())))
 return out

def run(root,N,kind,archive_dir):
 w0,a0,u0,pr=make_prepared(N,root,archive_dir if kind=='archive' else None)
 w=w0[:,None].copy();a=a0.copy();u=u0.copy();p=np.arange(N,dtype=np.int64)
 reseed(root+810000)
 pre={};last=0
 for tw in TW:
  advance_trace(w,a,u,p,np.array([4.]),np.array([0.]),(tw*10-last),.14,.3,1.,10.,0,.1,False)
  pre[tw]=(w[:,0].copy(),a.copy(),u.copy())
  last=tw*10
 rows=[]
 for tw,(ww,aa,uu) in pre.items():
  N=len(ww);s,_=ranks(ww);bottom=np.zeros(N,bool);bottom[s[:N//2]]=True
  top=np.zeros(N,bool);top[s[-N//10:]]=True
  ws=np.repeat(ww[:,None],len(CHI),axis=1);a=aa.copy();u=uu.copy();perm=np.arange(N,dtype=np.int64)
  la=np.full(len(CHI),4.,dtype=float);ch=np.array(CHI)
  reseed(root+101000000+tw*101)
  prev=0
  for tau in TAU:
   rr=tau*10
   if rr>prev:advance_trace(ws,a,u,perm,la,ch,rr-prev,.14,.3,1.,10.,0,.1,False)
   prev=rr
   rows.extend(metricset(ws,u,bottom,top,rr,tw,root,kind))
 return rows

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--N',type=int,default=10000);ap.add_argument('--roots',type=int,default=40);ap.add_argument('--start',type=int,default=0)
 ap.add_argument('--kind',choices=['independent','archive'],default='independent')
 ap.add_argument('--archive-dir',default=str(Path(__file__).resolve().parents[1]/'inputs'))
 ap.add_argument('--out',default=str(Path(__file__).resolve().parents[1]/'results'/'late40.csv'))
 args=ap.parse_args();stamp=time.perf_counter();out=Path(args.out);out.parent.mkdir(exist_ok=True,parents=True)
 rows=[]
 for j in range(args.start,args.start+args.roots):
  root=20000000+j if args.kind=='independent' else 2026101201+j
  rows+=run(root,args.N,args.kind,args.archive_dir)
  if (j-args.start+1)%5==0 or j==args.start+args.roots-1:
   pd.DataFrame(rows).to_csv(out,index=False)
   print('LATE',j-args.start+1,'/',args.roots,'sec',round(time.perf_counter()-stamp,1),flush=True)
 print('DONE')
