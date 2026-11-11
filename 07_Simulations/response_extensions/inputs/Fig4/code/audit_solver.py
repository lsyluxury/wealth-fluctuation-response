#!/usr/bin/env python3
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import numpy as np,json,time
from fast_solver import FastSolver
from density_solver import Settings
from common import make_initial,measure
ROOT=Path(__file__).resolve().parents[1]
checks=[]
for dx,na,steps in [(.025,48,300),(.01,96,100)]:
 s=FastSolver(Settings(dx=dx,na=na,threads=1,poisson_max=12));g=make_initial(s,2026101201,False);a=g.copy()
 P,idx,al=s.prepare(4.,.03);start=time.time()
 for n in range(steps):
  g,_=s.step(g,P,idx,al,.03);a,_=s.step_fast(a,P,idx,al,.03)
  if n+1 in [1,50,100,300]:
   mg=measure(g,s.w,s.u);ma=measure(a,s.w,s.u)
   checks.append(dict(dx=dx,na=na,round=n+1,max_density_difference=float(np.max(abs(g-a))),**{m+'_difference':ma[m]-mg[m] for m in ['G','M','E','mass','mean']}))
 print('OPERATOR_CHECK',dx,steps,time.time()-start,checks[-1],flush=True)
(ROOT/'validation/SOLVER_EQUIVALENCE.json').write_text(json.dumps(checks,indent=2))
assert max(c['max_density_difference'] for c in checks)<1e-12
assert max(abs(c[m+'_difference']) for c in checks for m in ['G','M','E'])<1e-10
