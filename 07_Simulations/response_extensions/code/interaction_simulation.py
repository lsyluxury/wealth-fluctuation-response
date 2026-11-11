#!/usr/bin/env python3
"""Paired neutral-score prediction and independent direct redistribution response.

Portable use, from any current directory:
  python code/interaction_simulation.py --mode neutral --reps 16000 --workers 2
  python code/interaction_simulation.py --mode direct --reps 6000 --workers 2

Each row in raw NPZ files is one independent future of the SAME supplied initial
wealth vector. Columns/checkpoints/branches are deliberately retained. No biased
endpoint is used to construct the neutral-score prediction. At each round all
agents are randomly matched, trade once, advantage renews, then wealth is taxed.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
import argparse, hashlib, json, math, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from numba import njit

ROOT=Path(__file__).resolve().parents[1]
EPS=0.14; H=0.3; MU=1.0; RATE=10.0
CHECKS=np.array([80,160,300],dtype=np.int64)
CHIS=np.array([0.0,0.03])
LAMBDAS=np.array([0.0,0.1,-0.1,0.25,-0.25,0.5,-0.5])
ODD=np.arange(1,24,2,dtype=np.int64)
GAMMAS=np.exp(-MU/RATE*(1.0-H**ODD))
WEIGHTS=np.array([math.exp(math.lgamma(int(j)+1)-math.log(math.pi)
                 -2*((int(j)-1)//2)*math.log(4)
                 -2*math.lgamma((int(j)-1)//2+1)-2*math.log(int(j))) for j in ODD])
BASE_SEEDS={'neutral':661071000,'direct':771071000}

@njit(cache=True)
def observe(w, bottom, top):
    n=len(w)
    idx=np.argsort(w,kind='mergesort')
    sw=0.0; siw=0.0
    for j in range(n):
        x=w[idx[j]]; sw+=x; siw+=(j+1)*x
    gout=2.0*siw/(n*sw)-(n+1.0)/n
    mb=0; et=0
    for j in range(n-n//10,n):
        i=idx[j]
        mb+=bottom[i]
        et+=top[i]
    return gout, mb/(n//2), 1.0-et/(n//10)

@njit(cache=True)
def neutral_one(w0, bottom, top, seed, checks, eps, gammas, weights, chis, rate):
    np.random.seed(seed)
    n=len(w0); nc=len(chis)
    wealth=np.empty((n,nc))
    for c in range(nc): wealth[:,c]=w0
    # Histories are per agent and Hermite mode. This layout makes the inner
    # mode loop contiguous while retaining exactly the analytic covariance.
    past=np.zeros((n,len(gammas)))
    a=0.0; perm=np.arange(n); ck=0
    out=np.empty((len(checks),nc,3)); scores=np.empty(len(checks))
    rho=np.exp(-chis/rate)
    for round_no in range(1,checks[-1]+1):
        np.random.shuffle(perm)
        for q in range(0,n,2):
            i=perm[q]; j=perm[q+1]
            sig=1 if np.random.random()<0.5 else -1
            for c in range(nc):
                wi=wealth[i,c]; wj=wealth[j,c]
                stake=eps*min(wi,wj)
                wealth[i,c]=wi+sig*stake
                wealth[j,c]=wj-sig*stake
            for m in range(len(gammas)):
                a += 0.25*weights[m]*sig*(past[i,m]-past[j,m])
                past[i,m]=gammas[m]*(past[i,m]+sig)
                past[j,m]=gammas[m]*(past[j,m]-sig)
        for i in range(n):
            for c in range(nc):
                wealth[i,c]=rho[c]*wealth[i,c]+1.0-rho[c]
        if round_no==checks[ck]:
            for c in range(nc):
                out[ck,c,:]=observe(wealth[:,c],bottom,top)
            scores[ck]=a
            ck+=1
            if ck==len(checks): break
    return out,scores

@njit(cache=True)
def direct_one(w0, bottom, top, seed, checks, eps, h, mu, rate, lambdas, chis):
    np.random.seed(seed)
    n=len(w0); nl=len(lambdas); nc=len(chis)
    wealth=np.empty((n,nl,nc))
    for l in range(nl):
        for c in range(nc): wealth[:,l,c]=w0
    av=np.random.normal(0.0,1.0,n)
    u=np.empty(n)
    for i in range(n): u[i]=math.erf(av[i]/math.sqrt(2.0))
    rho=np.exp(-chis/rate); perm=np.arange(n); ck=0
    out=np.empty((len(checks),nl,nc,3))
    renewsd=math.sqrt(1.0-h*h)
    for round_no in range(1,checks[-1]+1):
        np.random.shuffle(perm)
        for q in range(0,n,2):
            i=perm[q]; j=perm[q+1]
            draw=np.random.random(); du=u[i]-u[j]
            for l in range(nl):
                pp=1.0/(1.0+math.exp(-lambdas[l]*du))
                sig=1 if draw<pp else -1
                for c in range(nc):
                    wi=wealth[i,l,c]; wj=wealth[j,l,c]
                    stake=eps*min(wi,wj)
                    wealth[i,l,c]=wi+sig*stake
                    wealth[j,l,c]=wj-sig*stake
        # Global Poisson superposition gives independent Poisson(mu/rate)
        # event counts for each agent, including repeated renewals per round.
        nev=np.random.poisson(n*mu/rate)
        for event in range(nev):
            i=np.random.randint(n)
            av[i]=h*av[i]+renewsd*np.random.normal()
            u[i]=math.erf(av[i]/math.sqrt(2.0))
        for i in range(n):
            for l in range(nl):
                for c in range(nc):
                    wealth[i,l,c]=rho[c]*wealth[i,l,c]+1.0-rho[c]
        if round_no==checks[ck]:
            for l in range(nl):
                for c in range(nc):
                    out[ck,l,c,:]=observe(wealth[:,l,c],bottom,top)
            ck+=1
            if ck==len(checks): break
    return out


def load_initial():
    f=ROOT/'inputs'/'w0_fixed.npy'
    w0=np.load(f,allow_pickle=False).astype(np.float64)
    if len(w0)!=1000 or np.any(w0<0) or not np.isclose(w0.mean(),1,atol=1e-12):
        raise ValueError('Expected supplied N=1000 initial state with mean wealth 1.')
    idx=np.argsort(w0,kind='stable')
    bottom=np.zeros(len(w0),dtype=np.int64); top=bottom.copy()
    bottom[idx[:len(w0)//2]]=1; top[idx[-len(w0)//10:]]=1
    return w0,bottom,top,hashlib.sha256(f.read_bytes()).hexdigest()


def block_job(mode,start,stop):
    w0,bottom,top,_=load_initial()
    seeds=np.arange(BASE_SEEDS[mode]+start,BASE_SEEDS[mode]+stop,dtype=np.int64)
    if mode=='neutral':
        raw=np.empty((len(seeds),len(CHECKS),len(CHIS),3))
        score=np.empty((len(seeds),len(CHECKS)))
        for k,seed in enumerate(seeds):
            raw[k],score[k]=neutral_one(w0,bottom,top,int(seed),CHECKS,EPS,GAMMAS,WEIGHTS,CHIS,RATE)
        return start,raw,score,seeds
    raw=np.empty((len(seeds),len(CHECKS),len(LAMBDAS),len(CHIS),3))
    for k,seed in enumerate(seeds):
        raw[k]=direct_one(w0,bottom,top,int(seed),CHECKS,EPS,H,MU,RATE,LAMBDAS,CHIS)
    return start,raw,None,seeds


def metadata(mode,count,elapsed,digest):
    return dict(mode=mode,n_futures=count,N=1000,checkpoints_rounds=CHECKS.tolist(),
       times=(CHECKS/RATE).tolist(),epsilon=EPS,h=H,mu=MU,rounds_per_time=RATE,
       chi=CHIS.tolist(),lambdas=LAMBDAS.tolist(),observables=['G','M','E'],
       bottom_initial_fraction=.5,top_initial_fraction=.1,
       initial_state='inputs/w0_fixed.npy',initial_sha256=digest,
       condition='one fixed initial wealth vector, independent stationary Gaussian advantages',
       branch_randomness='matching, uniform winner draw and external advantage history shared within future',
       future_unit='independent future seed; not an independently prepared initial state',
       independent_prediction_and_validation_seeds=True,
       seed_base=BASE_SEEDS[mode],seed_end_exclusive=BASE_SEEDS[mode]+count,
       hermite_orders=ODD.tolist(),hermite_weights=WEIGHTS.tolist(),
       covariance_truncation_tail_at_zero=float(1/3-WEIGHTS.sum()),
       score_formula='A_n=(1/4) sum_{d=1}^{n-1} C_u(d) sum_i sum_{k=1}^{n-d} sigma[i,k] sigma[i,k+d]',
       score_is_path_quantity=True,elapsed_seconds=elapsed,
       branch_order='raw[future,checkpoint,chi,observable]' if mode=='neutral' else
                    'raw[future,checkpoint,lambda,chi,observable]',
       rounding='no winsorization, clipping, smoothing or endpoint fitting')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['neutral','direct'],required=True)
    parser.add_argument('--reps',type=int,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--block-size',type=int,default=100)
    parser.add_argument('--tag',default='')
    args=parser.parse_args()
    if not 1<=args.workers<=2: raise ValueError('Use one or two workers.')
    outdir=ROOT/'data'; outdir.mkdir(exist_ok=True)
    logdir=ROOT/'logs'; logdir.mkdir(exist_ok=True)
    w0,bottom,top,digest=load_initial()
    start_time=time.time()
    print(json.dumps({'mode':args.mode,'reps':args.reps,'workers':args.workers,
                      'initial_sha256':digest,'covariance_tail':float(1/3-WEIGHTS.sum())}),flush=True)
    # Compile in the parent once so both fork workers load identical cached code.
    block_job(args.mode,0,1)
    compiled_time=time.time()
    print('JIT warmup seconds %.3f'%(compiled_time-start_time),flush=True)
    shape=(args.reps,len(CHECKS),len(CHIS),3) if args.mode=='neutral' else (
        args.reps,len(CHECKS),len(LAMBDAS),len(CHIS),3)
    all_raw=np.empty(shape)
    all_scores=np.empty((args.reps,len(CHECKS))) if args.mode=='neutral' else None
    all_seeds=np.empty(args.reps,dtype=np.int64)
    done=0
    suffix=('_'+args.tag) if args.tag else ''
    target=outdir/('interaction_'+args.mode+'_raw'+suffix+'.npz')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(block_job,args.mode,s,min(s+args.block_size,args.reps))
              for s in range(0,args.reps,args.block_size)]
        for job in as_completed(jobs):
            idx,raw,scores,seeds=job.result(); stop=idx+len(raw)
            all_raw[idx:stop]=raw; all_seeds[idx:stop]=seeds
            if all_scores is not None: all_scores[idx:stop]=scores
            done+=len(raw)
            if done%500==0 or done==args.reps:
                print('%s completed %d/%d; sampling %.1fs'%(args.mode,done,args.reps,time.time()-compiled_time),flush=True)
    meta=metadata(args.mode,args.reps,time.time()-start_time,digest)
    payload=dict(raw=all_raw,seeds=all_seeds,checkpoints=CHECKS,chis=CHIS,lambdas=LAMBDAS,
                 observables=np.array(['G','M','E']),metadata=np.array(json.dumps(meta)))
    if all_scores is not None: payload['A']=all_scores
    np.savez_compressed(target,**payload)
    (logdir/('interaction_'+args.mode+'_metadata'+suffix+'.json')).write_text(json.dumps(meta,indent=2))
    print('SAVED '+str(target),flush=True)
    print('TOTAL_SECONDS %.3f; SAMPLING_SECONDS %.3f'%(time.time()-start_time,time.time()-compiled_time),flush=True)

if __name__=='__main__': main()
