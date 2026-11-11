#!/usr/bin/env python3
"""Stage 5 simultaneous lambda x tax experiment with common neutral-prepared initial wealth AND abilities.
As in archived Yard-Sale: complete disjoint pairing, stake .14 * min(w_i,w_j), logistic win,
Gaussian/Poisson score renewal, per-round tax-rebate rho=exp(-chi/10).
A single latent ability trajectory, pairing and win uniform used across treatment branches.
Unit of inference: independent preparation root, NOT branch or observation.
"""
from __future__ import annotations
import argparse,time,os,sys,json,math
from pathlib import Path
import numpy as np,pandas as pd
from numba import njit
from legacy_engine import initialise,neutral_round,reseed,gini_numba
from engine_stage3 import advance_trace

LAMBDA_PILOT=[0.,4.,8.]
CHI_PILOT=[0.,.01,.03]
LAMBDA_FULL=[0.,.5,1.,2.,4.,8.,12.]
CHI_FULL=[0.,.005,.01,.02,.03,.06]
CHECKS=(0,50,100,200,300)

def make_prepared(N,root,archive_dir=None):
 if archive_dir is None:
  seed=root+101000
  w,a,u,perm=initialise(N,int(seed),1)
  for n in range(1,1600):
   neutral_round(w[:,0],a,u,perm,.14,.3,1.,10.)
   if gini_numba(w[:,0])>=.65: return w[:,0].copy(),a.copy(),u.copy(),n
  raise RuntimeError('neutral preparation failed')
 p=Path(archive_dir)/f's{root}_g65_selection.npz'
 with np.load(p,allow_pickle=False) as z:
  return z['w'].copy(),z['a'].copy(),z['u'].copy(),-1

def ranks(v):
 s=np.argsort(v,kind='mergesort')
 rk=np.empty(len(v),dtype=np.int64)
 rk[s]=np.arange(len(v))
 return s,rk

def one_root(root,N,ls,cs,kind,archive_dir,obs=CHECKS):
 w0,a0,u0,prep=make_prepared(N,root,archive_dir if kind=='archive' else None)
 combos=[(l,c) for l in ls for c in cs]
 B=len(combos)
 la=np.asarray([l for l,c in combos],dtype=np.float64)
 ch=np.asarray([c for l,c in combos],dtype=np.float64)
 w=np.repeat(w0[:,None],B,axis=1);a=a0.copy();u=u0.copy();perm=np.arange(N,dtype=np.int64)
 origs, orig_rank=ranks(w0)
 bot=np.zeros(N,dtype=bool);bot[origs[:N//2]]=True
 top=np.zeros(N,dtype=bool);top[origs[-N//10:]]=True
 rows=[];au=[]
 reseed(root+810000)
 prev=0
 for rr in obs:
  if rr>prev:
   advance_trace(w,a,u,perm,la,ch,int(rr-prev),.14,.3,1.,10.,0,.1,False)
  prev=rr
  us,urank=ranks(u)
  urank=urank.astype(float)
  U2=float(np.sum((urank-(N-1)/2)**2))
  for k,(lam,chi) in enumerate(combos):
   v=w[:,k]
   sr,wr=ranks(v)
   total=float(np.sum(v))
   g=2*np.dot(np.arange(1,N+1,dtype=np.float64),v[sr])/(N*total)-(N+1)/N
   newtop=np.zeros(N,dtype=bool);newtop[sr[-N//10:]]=True
   m=float(np.mean(newtop[bot])); e=1.-float(np.mean(newtop[top]))
   cr=float(np.sum((wr-(N-1)/2)*(urank-(N-1)/2))/U2)
   q=float(np.mean((v-1)*u))
   count=float(np.count_nonzero(newtop & (~bot) & (~top))) / (N*.4)
   rows.append(dict(root=root,N=N,kind=kind,prep_rounds=prep,lambda_value=lam,chi=chi,
                    round=int(rr),time=rr/10.,G=g,M=m,E=e,Cq=(1-e-.1)/.9,
                    Ruw=cr,Q=q,Mmiddle=count,source_B50=float(5*m/e) if e>0 else np.nan))
   if (rr in (0,300)):
    au.append(dict(root=root,lambda_value=lam,chi=chi,round=rr,
                   err_total=total-N,min_w=float(v.min()),identity_err=e-5*m-4*count,
                   rho=math.exp(-chi/10)))
 return rows,au

if __name__=='__main__':
 ap=argparse.ArgumentParser()
 ap.add_argument('--grid',choices=['pilot','full'],default='pilot')
 ap.add_argument('--roots',type=int,default=4)
 ap.add_argument('--start',type=int,default=0)
 ap.add_argument('--N',type=int,default=10000)
 ap.add_argument('--kind',choices=['independent','archive'],default='independent')
 ap.add_argument('--archive-dir',default=str(Path(__file__).resolve().parents[1]/'inputs'))
 ap.add_argument('--out',default=str(Path(__file__).resolve().parents[1]/'results'/'stage5_pilot.csv'))
 args=ap.parse_args()
 ls=LAMBDA_PILOT if args.grid=='pilot' else LAMBDA_FULL
 cs=CHI_PILOT if args.grid=='pilot' else CHI_FULL
 out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
 stamp=time.perf_counter();rows=[];audit=[]
 for j in range(args.start,args.start+args.roots):
  root=(20000000+j) if args.kind=='independent' else (2026101201+j)
  r,a=one_root(root,args.N,ls,cs,args.kind,args.archive_dir)
  rows.extend(r);audit.extend(a)
  pd.DataFrame(rows).to_csv(out,index=False)
  pd.DataFrame(audit).to_csv(out.with_name(out.stem+'_audit.csv'),index=False)
  print(f'PROGRESS {j+1-args.start}/{args.roots} root={root} branches={len(ls)*len(cs)} secs={time.perf_counter()-stamp:.1f}',flush=True)
 print('DONE',out,'secs',time.perf_counter()-stamp)
