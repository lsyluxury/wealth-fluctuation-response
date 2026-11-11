#!/usr/bin/env python3
"""Grid halving for all positive-tax weak-selection conditions and the Stage-6 reference.
All six roots are used; no time point or metric is selected by its ABM agreement.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from run_theory_grid import run_root_grid,ROOT
if __name__=='__main__':
 jobs=[]
 for seed in range(2026101201,2026101207):
  todo=[]
  for lam,chi in [(.5,c) for c in [.005,.01,.02,.03,.06]]+[(4.,.03)]:
   key=f's{seed}_l{lam:g}_c{chi:g}_dx0.005_na96';d=ROOT/'data/raw/refinement'
   if not ((d/(key+'.csv')).exists() and (d/(key+'.json')).exists()):todo.append((lam,chi))
  if todo:jobs.append((seed,.005,96,todo,'refinement'))
 with ProcessPoolExecutor(max_workers=3) as pool:
  fs=[pool.submit(run_root_grid,*j) for j in jobs]
  for f in as_completed(fs):f.result()
 print('SELECTED_REFINEMENT_COMPLETE',flush=True)
