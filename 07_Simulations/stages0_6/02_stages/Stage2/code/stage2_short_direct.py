"""Direct weak-selection ABM from same fixed wealth initial condition as online kernel MC."""
import time,math,argparse
from pathlib import Path
import numpy as np,pandas as pd
from legacy_engine import reseed,gini_numba
from engine_stage01 import advance_trace,initialize_process_score
from stage2_abm import LAM,CHI
ap=argparse.ArgumentParser();ap.add_argument('--reps',type=int,default=4000);ap.add_argument('--rounds',type=int,default=40);ap.add_argument('--N',type=int,default=1000);ap.add_argument('--out',default='../results/stage2_short_direct.csv');args=ap.parse_args()
with np.load(str(Path(__file__).resolve().parents[3] / '05_history/phase12/source/anchors/development/s2026101201_g65_selection.npz'),allow_pickle=False) as z:full=z['w'].copy()
idx=np.random.default_rng(20001).permutation(len(full))[:args.N]
w0=full[idx];w0*=len(w0)/w0.sum()
rows=[];tm=time.time()
for root in range(args.reps):
 for mode in [0,1,2]:
  reseed(root+719210+mode*100000)
  if mode==0:
   from scipy.special import erf
   a=np.random.default_rng(root+719210).normal(size=args.N).astype(float)
   u=erf(a/math.sqrt(2))
  else:
   a=np.zeros(args.N);u=np.empty(args.N)
   initialize_process_score(u,mode)
  reseed(root+719210+mode*100000+12345)
  w=np.repeat(w0[:,None],len(LAM),axis=1);perm=np.arange(args.N)
  advance_trace(w,a,u,perm,LAM,CHI,args.rounds,.14,.3,1.,10.,mode,.1,False)
  for j,l in enumerate(LAM):rows.append(dict(root=root,mode=mode,lambda_value=l,G=gini_numba(w[:,j])))
 if (root+1)%1000==0:print('SHORT_DIRECT',root+1,'sec',round(time.time()-tm,1),flush=True)
 if (root+1)%5000==0:pd.DataFrame(rows).to_csv(args.out,index=False)
pp=Path(args.out);pd.DataFrame(rows).to_csv(pp,index=False);print('DONE',pp,round(time.time()-tm,1),flush=True)
