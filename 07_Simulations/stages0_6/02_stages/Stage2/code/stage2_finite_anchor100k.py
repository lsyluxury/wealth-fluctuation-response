"""Six archived N=100000 prepared states, finite slow/fast renewal (P5)."""
import math,time
from pathlib import Path
import numpy as np,pandas as pd
from legacy_engine import reseed
from engine_stage01 import advance_trace,initialize_process_score
from stage2_abm import summary_all,LAM,CHI
wt=math.exp(-.1);c=0
for k in range(100):c+=wt*(2/math.pi)*math.asin(.3**k/2);wt*=.1/(k+1)
pslow=.01;pfast=2*(1-3*c)-pslow
rows=[];tm=time.time();rootfolder=Path(str(Path(__file__).resolve().parents[3] / '05_history/phase12/source/anchors/development'))
for root in range(2026101201,2026101207):
 with np.load(rootfolder/f's{root}_g65_selection.npz') as z:w0=z['w'].copy()
 n=len(w0);ix=np.argsort(w0,kind='mergesort');bottom=ix[:n//2];top=ix[-n//10:]
 w=np.repeat(w0[:,None],len(LAM),axis=1);a=np.zeros(n);u=np.empty(n);perm=np.arange(n)
 reseed(root+1310000);initialize_process_score(u,6)
 a[np.random.default_rng(root+10200000).permutation(n)[:n//2]]=1.
 reseed(root+810000);advance_trace(w,a,u,perm,LAM,CHI,300,.14,.3,pslow,10.,6,pfast,False)
 x=summary_all(w,bottom,top)
 for j,l in enumerate(LAM):rows.append({'root':root,'mode':6,'N':n,'lambda_value':l,'round':300,'G':x[j,0],'M':x[j,1],'E':x[j,2]})
 print('P5_100K',root,'elapsed',round(time.time()-tm,1),flush=True)
pd.DataFrame(rows).to_csv('../results/stage2_P5_anchors100k.csv',index=False)
