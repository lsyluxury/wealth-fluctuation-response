#!/usr/bin/env python3
"""Post-holdout descriptive sampling only. Does NOT compute/change theory.

Uses the byte-for-byte original finite-stake ABM engine. Parameter branches
share exactly the same random matching, draws, and ability renewal stream.
Sparse and dense observations are passive. Run --seeds for a subset, --workers
for process parallelism, --resume to keep completed roots. Output includes one
row per root/lambda/time and exact entry/exit counts. No smoothing/interpolation.
"""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
import sys,json,time,hashlib,argparse,csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from numba import njit
import engine
ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'simulation_config.json').read_text())

@njit(cache=True)
def measure(w,r0):
    n, nb=w.shape
    out=np.zeros((nb,8))
    for j in range(nb):
        order=np.argsort(w[:,j])
        total=0.; weighted=0.;mb=0;mi=0;remain=0
        for k in range(n):
            i=order[k]; wi=w[i,j]
            total+=wi;weighted+=(k+1)*wi
            if k>=9*n//10:
                if r0[i]<n//2:mb+=1
                elif r0[i]<9*n//10:mi+=1
                else:remain+=1
        ec=n//10-remain
        ties=(w[order[9*n//10-1],j]==w[order[9*n//10],j])
        out[j,0]=2.*weighted/(n*total)-(n+1.)/n
        out[j,1]=mb/(n/2.)
        out[j,2]=ec/(n/10.)
        out[j,3]=mb;out[j,4]=mi;out[j,5]=ec
        out[j,6]=abs(total/n-1.)
        out[j,7]=ties
    return out

def hasharr(*arrays):
    h=hashlib.sha256()
    for a in arrays: h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def compute(seed):
    start=time.perf_counter();m=CFG['model'];n=m['N']
    if seed not in CFG['seeds']:raise ValueError('Seed not in prewritten visualization plan')
    ww,a,u,p=engine.initialise(n,seed,1);w=ww[:,0].copy()
    prep=0
    while True:
        engine.neutral_round(w,a,u,p,m['epsilon'],m['h'],m['mu'],m['r']);prep+=1
        g=engine.gini_numba(w)
        if g>=m['G_target']:break
        if prep>=20000:raise RuntimeError('Preparation not completed')
    ahash=hasharr(w,a,u,p)
    # Target .65 is index 1 in the original [.50,.65,.80] protocol.
    a_seed=int(np.random.SeedSequence([seed,1,1]).generate_state(1)[0])
    order=np.argsort(w,kind='stable');r0=np.empty(n,np.int64);r0[order]=np.arange(n)
    la=np.array(CFG['lambdas']);chi=np.zeros(la.size)
    w=np.repeat(w[:,None],la.size,axis=1)
    engine.reseed(a_seed)
    records=[];last=0;maxmass=0.;ties=0
    for rnd in CFG['rounds']:
        if rnd>last:
            engine.advance_chunk(w,a,u,p,la,chi,rnd-last,m['epsilon'],m['h'],m['mu'],m['r'])
        vals=measure(w,r0)
        maxmass=max(maxmass,float(vals[:,6].max()));ties+=int(vals[:,7].sum())
        for j,value in enumerate(la):
            v=vals[j]
            if int(v[3]+v[4])!=int(v[5]):raise AssertionError('Exact cohort count identity failed')
            records.append(dict(seed=seed,lambda_value=float(value),chi=0.0,round=int(rnd),time=rnd/m['r'],G=float(v[0]),M=float(v[1]),E=float(v[2]),M_count=int(v[3]),I_count=int(v[4]),E_count=int(v[5])))
        last=rnd
    if ties or maxmass>1e-10 or w.min()<=0:raise AssertionError(f'Numerical failure {ties=} {maxmass=}')
    base=ROOT/'runs';base.mkdir(exist_ok=True)
    dest=base/f's{seed}.csv';tmp=dest.with_suffix('.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
    os.replace(tmp,dest)
    report=dict(seed=seed,seconds=time.perf_counter()-start,records=len(records),A_trajectories=len(la),A_rounds=len(la)*300,prep_round=prep,actual_prep_G=g,A_state_sha256=ahash,A_stream_seed=a_seed,max_mass_error=maxmass,cutoff_ties=ties,min_final_w=float(w.min()),max_final_w=float(w.max()),engine_sha256=hashlib.sha256((ROOT/'src/engine.py').read_bytes()).hexdigest(),config_sha256=hashlib.sha256((ROOT/'simulation_config.json').read_bytes()).hexdigest(),raw_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    dest.with_suffix('.json').write_text(json.dumps(report,indent=2))
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--seeds',type=int,nargs='+');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    seeds=args.seeds or CFG['seeds'];todo=[]
    for s in seeds:
        p=ROOT/f'runs/s{s}.json'
        if args.resume and p.exists():
            rec=json.loads(p.read_text())
            if rec['config_sha256']!=hashlib.sha256((ROOT/'simulation_config.json').read_bytes()).hexdigest():raise RuntimeError('Config changed since saved run')
            if rec['raw_sha256']!=hashlib.sha256(p.with_suffix('.csv').read_bytes()).hexdigest():raise RuntimeError('Raw output hash mismatch')
            if rec['engine_sha256']!=hashlib.sha256((ROOT/'src/engine.py').read_bytes()).hexdigest():raise RuntimeError('Engine changed since saved run')
            print('Skip',s,flush=True)
        else:todo.append(s)
    # compile before worker launch without consuming any production seed stream
    w,a,u,p=engine.initialise(100,13,2)
    engine.advance_chunk(w,a,u,p,np.full(2,4.),np.array([0.,.03]),1,.14,.3,1.,10.)
    measure(w,np.arange(100))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs={ex.submit(compute,s):s for s in todo}
        for f in as_completed(futs):print(json.dumps(f.result()),flush=True)
if __name__=='__main__':main()
