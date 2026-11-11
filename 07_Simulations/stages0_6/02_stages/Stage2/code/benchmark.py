from pathlib import Path
import time,sys,os,math
import numpy as np
from legacy_engine import reseed,gini_numba
from engine_stage01 import advance_trace,initialize_process_score
p0=str(Path(__file__).resolve().parents[3] / '05_history/phase12/source/anchors/development/s2026101201_g65_selection.npz')
with np.load(p0) as z:w0=z['w'][:10000].copy()
w0*=len(w0)/w0.sum()
for N in (2000,10000,100000):
 with np.load(p0) as z:w0=z['w'][:N].copy()
 w0*=len(w0)/w0.sum();w=np.column_stack([w0]*7);a=np.zeros(N);u=np.zeros(N);perm=np.arange(N)
 reseed(20199);initialize_process_score(u,1)
 lam=np.array([0,.1,-.1,.25,-.25,.5,-.5]);chi=np.zeros(7)
 t=time.perf_counter();advance_trace(w,a,u,perm,lam,chi,300,.14,.3,1.,10,1,.1,False)
 print('benchmark',N,round(time.perf_counter()-t,4),[round(gini_numba(w[:,k]),7) for k in range(7)],flush=True)
