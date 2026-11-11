#!/usr/bin/env python3
"""Conditional ABM futures for fixed archived 100k starting states.
Future random streams independently reseeded for each root/future.
Replications within the same root are averaged; they are NOT new root units.
"""
import argparse,time,json
from pathlib import Path
import numpy as np,pandas as pd
from legacy_engine import reseed
from engine_stage3 import advance_trace
ROOT=Path(__file__).resolve().parents[1]

def one(root,seed,lam,chi):
 z=np.load(ROOT/'inputs'/f's{root}_g65_selection.npz',allow_pickle=False)
 v=z['w'].copy();a=z['a'].copy();u=z['u'].copy();n=len(v)
 original=np.argsort(v,kind='mergesort');top=np.zeros(n,dtype=bool);top[original[-n//10:]]=True
 bottom=np.zeros(n,dtype=bool);bottom[original[:n//2]]=True
 v=v.reshape(-1,1);perm=np.arange(n,dtype=np.int64)
 reseed(seed)
 rows=[]
 prev=0
 for step in (0,50,100,200,300):
  if step>prev:advance_trace(v,a,u,perm,np.array([lam]),np.array([chi]),step-prev,.14,.3,1.,10.,0,.1,False)
  prev=step
  p=v[:,0];ids=np.argsort(p,kind='mergesort');N=len(p);T=float(p.sum());
  g=2*np.dot(np.arange(1,N+1,dtype=float),p[ids])/(N*T)-(N+1)/N
  current=np.zeros(N,dtype=bool);current[ids[-N//10:]]=True
  rows.append(dict(root=root,future_seed=seed,lam=lam,chi=chi,round=step,G=float(g),M=float(current[bottom].mean()),E=float(1-current[top].mean()),
              total_error=T-N,min_w=float(p.min())))
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--roots',default='2026101201,2026101202')
 p.add_argument('--futures',type=int,default=8);p.add_argument('--lam',type=float,default=4.);p.add_argument('--chi',type=float,default=.03)
 p.add_argument('--out',default=str(ROOT/'results'/'conditional_futures.csv'))
 a=p.parse_args();t=time.perf_counter()
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 rows=pd.read_csv(out).to_dict('records') if out.exists() else []
 existing={int(x['future_seed']) for x in rows if int(x['round'])==300}
 for root in [int(k) for k in a.roots.split(',')]:
  for j in range(a.futures):
   seed=int(root+810000+310007*j)
   if seed not in existing:
    rec=one(root,seed,a.lam,a.chi)
    rows.extend(rec);existing.add(seed)
    pd.DataFrame(rows).to_csv(out,index=False)
    print('FUTURE',root,j+1,'/',a.futures,'elapsed',round(time.perf_counter()-t,1),flush=True)
  print('DONE_ROOT',root,'futures',a.futures,'elapsed',round(time.perf_counter()-t,1),flush=True)
 print('FILE',out)
