#!/usr/bin/env python3
"""Stage4 paired late-state intervention, conditional on the same full w,a,u at t_w.
Create pre-intervention trajectories lambda=0,4 through tw=50; at tw=30/50,
clone each selected early state (or neutral early state) into lambda future=0,4.
This is additional to within-trajectory two-time overlap, not a replacement.
"""
import sys,time,argparse
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'deps'))
from stage4_abm import metrics
from stage2_abm import prepare
from legacy_engine import reseed
from engine_stage3 import advance_trace
LAM=np.array([0.,4.]);CHI=np.zeros(2)
ORIGINS=(300,500)
MODES=('baseline','strong','fixed')

def one(root,n):
 w0,a0,u0,_=prepare(n,int(root)+101000,.65)
 rows=[]
 for mode in MODES:
  h=.9 if mode=='strong' else .3;kernelmode=3 if mode=='fixed' else 0
  w=np.repeat(w0[:,None],2,axis=1);a=a0.copy();u=u0.copy();perm=np.arange(n,dtype=np.int64)
  reseed(root+810000)
  prev=0;origins={}
  for tw in ORIGINS:
   advance_trace(w,a,u,perm,LAM,CHI,tw-prev,.14,h,1.,10.,kernelmode,.1,False)
   origins[tw]=(w.copy(),a.copy(),u.copy());prev=tw
  for tw,(ws,aas,uus) in origins.items():
   for early in (0.,4.):
    w_init=ws[:,0 if early==0. else 1]
    ords=np.argsort(w_init,kind='mergesort');tag=np.zeros(n,dtype=bool);tag[ords[-n//10:]]=True
    bot=np.zeros(n,dtype=bool);bot[ords[:n//2]]=True
    wt=np.repeat(w_init[:,None],2,axis=1);aa=aas.copy();uu=uus.copy();per=np.arange(n,dtype=np.int64)
    reseed(root+101000000+int(tw)*100+int(early)*1000)
    advance_trace(wt,aa,uu,per,LAM,CHI,300,.14,h,1.,10.,kernelmode,.1,False)
    for ix,late in enumerate(LAM):
     y=wt[:,ix];G,order=metrics(y)
     tn=np.zeros(n,dtype=bool);tn[order[-n//10:]]=True
     stay=float(np.mean(tn[tag]));C=(stay-.1)/.9;M=float(np.mean(tn[bot]))
     rows.append({'root':root,'N':n,'mode':mode,'early_lambda':early,'late_lambda':late,
           'tw':tw/10.,'tau':30.,'Cq':C,'stay':stay,'E':1-stay,'M':M,'G':G,
           'Gorigin':metrics(w_init)[0]})
 return rows

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--roots',type=int,default=30);ap.add_argument('--start',type=int,default=0)
 ap.add_argument('--N',type=int,default=10000);ap.add_argument('--out',default=str(ROOT/'results'/'stage4_late_state_restarts.csv'));args=ap.parse_args()
 out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
 st=time.time();rows=[]
 for j in range(args.start,args.start+args.roots):
  root=20000000+j;rows+=one(root,args.N)
  if (j+1)%5==0:
   pd.DataFrame(rows).to_csv(out,index=False)
   print('PROGRESS',j+1,args.roots,'seconds',round(time.time()-st,1),flush=True)
 pd.DataFrame(rows).to_csv(out,index=False)
 print('DONE',round(time.time()-st,1),flush=True)
