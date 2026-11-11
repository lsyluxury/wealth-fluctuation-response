"""Full N=100000 microscopic stage2 check using six original Stage-A anchors."""
import time,math,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import erf
from legacy_engine import reseed
from engine_stage01 import advance_trace,initialize_process_score
from stage2_abm import summary_all,LAM,CHI
AR=Path(str(Path(__file__).resolve().parents[3] / '05_history/phase12/source/anchors/development'))
wt=math.exp(-.1);c=0.
for k in range(100):
 c+=wt*(2/math.pi)*math.asin(.3**k/2);wt*=.1/(k+1)
phet=2*(1-3*c)
rows=[];t0=time.time()
for root in range(2026101201,2026101207):
 with np.load(AR/f's{root}_g65_selection.npz',allow_pickle=False) as z:
  w0=z['w'].copy();a0=z['a'].copy();u0=z['u'].copy()
 n=len(w0);ind=np.argsort(w0,kind='mergesort');bottom=ind[:n//2];top=ind[-n//10:]
 for mode in [0,1,2,5]:
  w=np.repeat(w0[:,None],len(LAM),axis=1)
  a=a0.copy() if mode==0 else np.zeros(n)
  u=u0.copy() if mode==0 else np.empty(n)
  perm=np.arange(n,dtype=np.int64)
  reseed(root+710000+mode*100000)
  if mode!=0:initialize_process_score(u,mode)
  if mode==5:
   mask=np.zeros(n);rng=np.random.default_rng(root+10200000)
   mask[rng.permutation(n)[:n//2]]=1.;a=mask
  reseed(root+810000)
  advance_trace(w,a,u,perm,LAM,CHI,300,.14,.3,1.,10.,mode,phet if mode==5 else .1,False)
  X=summary_all(w,bottom,top)
  for j,lam in enumerate(LAM):
   rows.append(dict(root=root,N=n,mode=mode,round=300,lambda_value=lam,G=X[j,0],M=X[j,1],E=X[j,2]))
 print('ANCHOR_DONE',root,'elapsed',round(time.time()-t0,1),flush=True)
 pd.DataFrame(rows).to_csv('../results/stage2_real_anchors_100k.csv',index=False)
print('ANCHOR_COMPLETE',time.time()-t0,flush=True)
