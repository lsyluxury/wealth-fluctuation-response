#!/usr/bin/env python3
"""Six archived A states, full nonlinear kinetic theory, matched against ABM; no fitting."""
from __future__ import annotations
import os,sys,json,time,argparse
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage3_full_kinetic import simulate
ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=1);ap.add_argument('--roots',type=int,default=6);ap.add_argument('--dx',type=float,default=.025);ap.add_argument('--na',type=int,default=48);ap.add_argument('--out',default=str(Path(__file__).resolve().parents[1]/'results'/'stage3_kinetic_6roots_new.csv'));args=ap.parse_args()
out=Path(args.out);allrows=[];audit={};t0=time.time()
for i in range(args.start,args.start+args.roots):
 seed=2026101200+i
 for mode,lam in [(1,0.),(1,4.),(2,4.),(7,4.)]:
  rows,info=simulate(mode,lam,args.dx,args.na,seed);allrows.extend(rows);audit[f'{seed}:{mode}:{lam}']=info
  pd.DataFrame(allrows).to_csv(out,index=False)
  out.with_suffix('.json').write_text(json.dumps(audit,indent=2))
 print('ROOT_COMPLETE',seed,'elapsed',round(time.time()-t0,2),flush=True)
print('DONE',out,'elapsed',round(time.time()-t0,2),flush=True)
