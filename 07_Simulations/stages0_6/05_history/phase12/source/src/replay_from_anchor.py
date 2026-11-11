#!/usr/bin/env python3
"""Recompute one registered branch from its saved stage anchor.

Example:
 python src/replay_from_anchor.py --seed 2026101201 --target 0.65 --stage selection --parameter 0.25 --time 30 --output replay.npz

This command generates data. It does not summarize a holdout branch. Use holdout
replay only after recording an unblinding decision, or for authorized numerical QA.
"""
from __future__ import annotations
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json
from pathlib import Path
import numpy as np
import engine
from records import savez
R=Path(__file__).resolve().parents[1];cfg=json.loads((R/'config/experiment_config.json').read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--target',type=float,required=True)
    p.add_argument('--stage',choices=['selection','tax'],required=True);p.add_argument('--parameter',type=float,required=True)
    p.add_argument('--time',type=float,default=30);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-holdout',action='store_true');a=p.parse_args()
    if a.seed not in cfg['developer_seeds']+cfg['holdout_seeds'] or a.target not in cfg['targets']:raise ValueError('Unregistered root/target')
    key='lambdas' if a.stage=='selection' else 'chis'
    if a.parameter not in cfg[key]:raise ValueError('Parameter is not in frozen main grid')
    if (a.seed in cfg['holdout_seeds'] or a.parameter in cfg['heldout_parameters'][key]) and not a.allow_holdout:
        raise RuntimeError('Holdout replay blocked by default. Use --allow-holdout only after authorized unblinding/QA.')
    split='development' if a.seed in cfg['developer_seeds'] else 'holdout_locked'
    anchor=R/f'anchors/{split}/s{a.seed}_g{int(a.target*100)}_{a.stage}.npz'
    z=np.load(anchor,allow_pickle=False);w=z['w'][:,None].copy();latent=z['a'].copy();u=z['u'].copy();perm=z['permutation'].astype(np.int64)
    engine.restore_rng(json.loads(str(z['rng_state_json'])))
    la=np.array([a.parameter if a.stage=='selection' else 4.]);ch=np.array([0. if a.stage=='selection' else a.parameter])
    rounds=int(round(a.time*cfg['model']['r']))
    if a.time<0 or a.time>30 or abs(rounds/cfg['model']['r']-a.time)>1e-12:raise ValueError('Time must be a registered-round time within [0,30]')
    m=cfg['model'];budgets=engine.advance_chunk(w,latent,u,perm,la,ch,rounds,m['epsilon'],m['h'],m['mu'],m['r'])
    savez(a.output,w=w[:,0],a=latent,u=u,permutation=perm.astype(np.int32),budgets=budgets,
       metadata_json=np.array(json.dumps(dict(seed=a.seed,target_G=a.target,stage=a.stage,parameter=a.parameter,time=a.time,anchor_file=str(anchor.relative_to(R))))),
       rng_state_json=np.array(json.dumps(engine.rng_state())))
    print('Wrote',a.output)
if __name__=='__main__':main()
