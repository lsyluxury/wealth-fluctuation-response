#!/usr/bin/env python3
"""P7 alternative: same U(-1,1) and full score C_u as P2 but different 4-point structure.
P2: redraw p=.1, P7: redraw p=.05, sign-flip p=.025, hold p=.925.
For both: E[u(k+d)u(k)] = (0.9)^d/3. For u^2, lag decay differs.
"""
from __future__ import annotations
import sys,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd
from stage2_abm import prepare
from legacy_engine import reseed
from engine_stage3 import advance_trace,initialize_process_score
from stage3_abm import metric_rows
LAM=np.array([0.,.25,-.25,.5,-.5,1.,-1.,2.,-2.,4.,-4.,8.,-8.])
OBS=[0,10,20,50,100,200,300]

def one(root,n,mode):
 w0,a0,u0,prep=prepare(n,int(root)+101000,.65)
 order=np.argsort(w0,kind='mergesort');bottom=order[:n//2];top=order[-n//10:]
 w=np.repeat(w0[:,None],len(LAM),axis=1)
 a=np.zeros(n)
 u=np.empty(n)
 perm=np.arange(n,dtype=np.int64)
 # Same initial scores and same exchange RNG as P2 in stage3_abm.
 reseed(root+710000+100000);initialize_process_score(u,1)
 reseed(root+810000)
 rows=metric_rows(w,u,bottom,top,root,mode,LAM,0,prep)
 prev=0
 for step in OBS[1:]:
  advance_trace(w,a,u,perm,LAM,np.zeros(len(LAM)),step-prev,.14,.3,1.,10.,7,.05,False)
  rows+=metric_rows(w,u,bottom,top,root,mode,LAM,step,prep)
  prev=step
 return rows

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=0);ap.add_argument('--roots',type=int,default=10);ap.add_argument('--N',type=int,default=10000);ap.add_argument('--out',default=str(Path(__file__).resolve().parents[1]/'results'/'stage3_P7_new.csv'));args=ap.parse_args()
 rows=[];t0=time.time();out=Path(args.out)
 for j in range(args.start,args.start+args.roots):
  rows+=one(20000000+j,args.N,7)
  if (j-args.start+1)%5==0:
   pd.DataFrame(rows).to_csv(out,index=False)
   print('P7_PROGRESS',j-args.start+1,args.roots,'secs',round(time.time()-t0,2),flush=True)
 pd.DataFrame(rows).to_csv(out,index=False)
 print('P7_FINISHED',out,'time',round(time.time()-t0,2),flush=True)
