#!/usr/bin/env python3
"""Exact score integration at lambda=0, wealth-grid refinement, bounded memory."""
import gc
from run_theory_grid import run_root_grid,CHIS,ROOT,BASE_CACHE,CROP_CACHE
if __name__=='__main__':
 for dx in [.01,.005,.0025]:
  for root in range(2026101201,2026101207):
   todo=[]
   for chi in CHIS:
    key=f's{root}_l0_c{chi:g}_dx{dx:g}_na1'
    d=ROOT/'data/raw/refinement'
    if not ((d/(key+'.csv')).exists() and (d/(key+'.json')).exists()):todo.append((0.,chi))
   if todo:run_root_grid(root,dx,1,todo,'refinement')
   BASE_CACHE.clear();CROP_CACHE.clear();gc.collect()
 print('NEUTRAL_REFINEMENT_COMPLETE',flush=True)
