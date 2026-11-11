#!/usr/bin/env python3
"""Run all archived six-root x sixteen-future redraw cases via the source engine.
This driver combines the original 24-future run and its 72-future extension,
without changing the engine, seed formulas, or root-level CI definition.
"""
from __future__ import annotations
import argparse,importlib.util,json,sys,time
from pathlib import Path
import pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--roots',type=int,default=6);p.add_argument('--futures',type=int,default=16);a=p.parse_args()
 if not 1<=a.roots<=6 or not 1<=a.futures<=16:raise ValueError('Archived protocol uses 6 roots and 16 futures.')
 sys.path.insert(0,str(a.source/'src'))
 from run_abm_mechanisms import one
 from extend_redraw_precision import ci_table
 a.out.mkdir(parents=True,exist_ok=True);rows=[];stamp=time.perf_counter()
 for seed in range(2026101201,2026101201+a.roots):
  for rep in range(1,a.futures+1):
   rr,_=one(seed,rep,'full_redraw',1.,.3,True);rows.extend(rr)
  pd.DataFrame(rows).to_csv(a.out/'full_redraw_combined_raw.csv',index=False)
  print('COMPLETED_ROOT',seed,flush=True)
 df=pd.DataFrame(rows);ci_table(df,a.futures,'combined96' if (a.roots,a.futures)==(6,16) else 'subset').to_csv(a.out/'full_redraw_equivalence.csv',index=False)
 (a.out/'run_metadata.json').write_text(json.dumps({'roots':a.roots,'futures_per_root':a.futures,'root_start':2026101201,'source_engine':'run_abm_mechanisms.one','seconds':time.perf_counter()-stamp},indent=2))
if __name__=='__main__':main()
