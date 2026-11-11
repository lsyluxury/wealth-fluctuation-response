#!/usr/bin/env python3
"""Batch runner: Stage5 independent finite-stake theory with checkpointed CSV."""
import sys,argparse, time, json
from pathlib import Path
import numpy as np, pandas as pd
from stage5_kinetic_validation import kinetic_one
DEFAULT=[(0.,0.),(4.,0.),(4.,.01),(4.,.03),(4.,.06),(0.,.03),(8.,.03)]
if __name__=='__main__':
 ap=argparse.ArgumentParser()
 ap.add_argument('--seed',type=int,default=2026101201);ap.add_argument('--dx',type=float,default=.05);ap.add_argument('--na',type=int,default=24)
 ap.add_argument('--conditions',default=';'.join(f'{x},{y}' for x,y in DEFAULT))
 ap.add_argument('--out',default=str(Path(__file__).resolve().parents[1]/'results'/'kinetic_batch.csv'))
 args=ap.parse_args();c=[tuple(map(float,x.split(','))) for x in args.conditions.split(';')]
 out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
 rows=[];audit=[];t0=time.perf_counter()
 for j,(la,ch) in enumerate(c):
  z,info=kinetic_one(args.seed,la,ch,args.dx,args.na)
  rows.extend(z);audit.append(info)
  pd.DataFrame(rows).to_csv(out,index=False)
  Path(str(out)+'.audit.json').write_text(json.dumps(audit,default=float,ensure_ascii=False,indent=2))
  print('KINETIC PROGRESS',j+1,'/',len(c),'seed',args.seed,'lambda',la,'chi',ch,'G',round(z[-1]['G'],6),'M',round(z[-1]['M'],6),'E',round(z[-1]['E'],6),'sec',round(time.perf_counter()-t0,1),flush=True)
 print('DONE',out,flush=True)
