#!/usr/bin/env python3
"""Six real archived N=100000 stage-A states; P2/P3/P7, 300 rounds.
No new neutral preparation; source anchor has w,a,u, id arrays; initialise P2/P7 scores
from same stationary U(-1,1) state and P3 from same-score-variance binary law.
"""
from __future__ import annotations
import sys,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np,pandas as pd
from legacy_engine import reseed
from engine_stage3 import advance_trace,initialize_process_score
from stage3_abm import metric_rows
AR=Path(__file__).resolve().parents[1]/'phase4'/'inputs'
LAM=np.array([0,4,-4,8,-8],dtype=float)
OBS=[0,10,20,50,100,200,300]

def one(root,modes):
  with np.load(AR/f's{root}_g65_selection.npz') as z:w0=z['w'].copy()
  n=len(w0);inds=np.argsort(w0,kind='mergesort');bottom=inds[:n//2];top=inds[-n//10:]
  rows=[]
  for mode in modes:
    w=np.repeat(w0[:,None],len(LAM),axis=1)
    a=np.zeros(n);u=np.empty(n);perm=np.arange(n,dtype=np.int64)
    reseed(root+710000+(100000 if mode==7 else mode*100000));initialize_process_score(u,1 if mode==7 else mode)
    reseed(root+810000)
    rows+=metric_rows(w,u,bottom,top,root,mode,LAM,0,-1)
    prev=0
    for checkpoint in OBS[1:]:
      advance_trace(w,a,u,perm,LAM,np.zeros(len(LAM)),checkpoint-prev,
                    .14,.3,1.,10.,mode,.05 if mode==7 else .1,False)
      rows+=metric_rows(w,u,bottom,top,root,mode,LAM,checkpoint,-1)
      prev=checkpoint
  return rows

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=1);ap.add_argument('--roots',type=int,default=6);ap.add_argument('--modes',default='1,2,7');ap.add_argument('--out',default=str(Path(__file__).resolve().parents[1]/'results'/'stage3_anchors_new.csv'));args=ap.parse_args()
 rows=[];tm=time.time();modes=list(map(int,args.modes.split(',')))
 for j in range(args.start,args.start+args.roots):
  root=2026101200+j
  rows+=one(root,modes)
  pd.DataFrame(rows).to_csv(args.out,index=False)
  print('ANCHOR_COMPLETED',root,'secs',round(time.time()-tm,1),flush=True)
 print('DONE total time',round(time.time()-tm,1),flush=True)
