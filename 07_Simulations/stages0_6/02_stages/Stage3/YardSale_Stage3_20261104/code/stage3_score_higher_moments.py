#!/usr/bin/env python3
"""Analytic and independent Monte Carlo check: same score C2, distinct squared-score memory."""
from __future__ import annotations
import numpy as np,pandas as pd
from pathlib import Path
rng=np.random.default_rng(2026110403)
N=250000;dlist=[1,2,5,10,20];rows=[]
for mode in ('P2','P7'):
 u0=rng.uniform(-1,1,N)
 u=u0.copy();u20=u0**2;sum0=float(np.mean(u0**2));sigma20=float(np.mean(u20))
 for d in range(1,max(dlist)+1):
  x=rng.random(N)
  if mode=='P2':
   mask=x<.1;u[mask]=rng.uniform(-1,1,mask.sum())
   alpha2=.9
  else:
   mask=x<.05;flip=(x>=.05)&(x<.075)
   u[mask]=rng.uniform(-1,1,mask.sum());u[flip]*=-1
   alpha2=.95
  if d not in dlist:continue
  c=float(np.mean(u0*u));k=float(np.mean((u0*u0)*(u*u))-sum0*float(np.mean(u*u)))
  rows.append({'mode':mode,'d':d,'C_theory':(.9**d)/3,'C_MC':c,
               'C_squared_theory':(4/45)*alpha2**d,'C_squared_MC':k,
               'N':N})
pd.DataFrame(rows).to_csv(str(Path(__file__).resolve().parents[1]/'results'/'score_higher_moments.csv'),index=False)
print(pd.DataFrame(rows).to_string(index=False))
