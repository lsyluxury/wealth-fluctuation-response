#!/usr/bin/env python3
"""Fixed archived initial states; multiple independent conditional futures.
All 42 branches share matching, Bernoulli uniforms and latent-score history.
The first future replays the original Stage-5 seed. Do not count futures as
independent preparations: average within each root before computing paired CI.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import math,time,json,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from numba import njit
from legacy_engine import logistic,reseed
ROOT=Path(__file__).resolve().parents[1]
LS=np.array([0.,.5,1.,2.,4.,8.,12.]);CS=np.array([0.,.005,.01,.02,.03,.06]);OBS=[0,50,100,200,300]

@njit(cache=True)
def advance_grid(w,a,u,perm,steps,ls,cs):
    n=w.shape[0];nch=len(cs);rho=np.exp(-cs*.1);rebate=1-rho
    for step in range(steps):
        np.random.shuffle(perm)
        for ii in range(0,n,2):
            i=perm[ii];j=perm[ii+1];du=u[i]-u[j];draw=np.random.random()
            for il in range(len(ls)):
                win=draw<logistic(ls[il]*du)
                for ic in range(nch):
                    b=il*nch+ic;wi=w[i,b];wj=w[j,b];stake=.14*(wi if wi<wj else wj)
                    if win:w[i,b]=wi+stake;w[j,b]=wj-stake
                    else:w[i,b]=wi-stake;w[j,b]=wj+stake
        nev=np.random.poisson(n*.1)
        for _ in range(nev):
            i=np.random.randint(0,n);a[i]=.3*a[i]+math.sqrt(1-.3**2)*np.random.normal();u[i]=math.erf(a[i]/math.sqrt(2))
        for i in range(n):
            for il in range(len(ls)):
                for ic in range(nch):
                    if cs[ic]>0:
                        b=il*nch+ic;w[i,b]=rho[ic]*w[i,b]+rebate[ic]

def future(root,fid):
    folder=ROOT/'data/raw/futures';folder.mkdir(parents=True,exist_ok=True)
    out=folder/f's{root}_future{fid:02d}.csv'
    if out.exists():return str(out)
    begin=time.time();z=np.load(ROOT/'inputs'/f's{root}_g65_selection.npz')
    w0=z['w'].copy();a=z['a'].copy();u=z['u'].copy();N=len(w0)
    w=np.repeat(w0[:,None],42,axis=1);perm=np.arange(N,dtype=np.int64)
    original=np.argsort(w0,kind='mergesort');bottom=original[:N//2];top=original[-N//10:]
    seed=int(root+810000+310007*fid);reseed(seed);rows=[];prev=0
    for n in OBS:
        if n>prev:advance_grid(w,a,u,perm,n-prev,LS,CS)
        prev=n;rr_u=np.empty(N);rr_u[np.argsort(u,kind='mergesort')]=np.arange(N)
        norm_rank=float(np.sum((rr_u-(N-1)/2)**2))
        for il,lam in enumerate(LS):
            for ic,chi in enumerate(CS):
                v=w[:,il*6+ic];order=np.argsort(v,kind='mergesort');total=float(v.sum())
                g=2*np.dot(np.arange(1,N+1,dtype=float),v[order])/(N*total)-(N+1)/N
                is_top=np.zeros(N,dtype=bool);is_top[order[-N//10:]]=True
                m=float(is_top[bottom].mean());e=float(1-is_top[top].mean())
                rr_w=np.empty(N);rr_w[order]=np.arange(N)
                corr=float(np.dot(rr_w-(N-1)/2,rr_u-(N-1)/2)/norm_rank)
                rows.append(dict(root=root,future=fid,future_seed=seed,N=N,lambda_value=float(lam),chi=float(chi),
                                 round=n,time=n/10,G=g,M=m,E=e,Ruw=corr,total_error=total-N,min_wealth=float(v.min())))
    pd.DataFrame(rows).to_csv(out,index=False,float_format='%.17g')
    # All entries and independent root/future identifiers are retained.
    print('ABM_FUTURE_DONE',root,fid,'seconds',round(time.time()-begin,2),flush=True)
    return str(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',default='2026101201,2026101202,2026101203,2026101204,2026101205,2026101206')
    p.add_argument('--futures',type=int,default=16);p.add_argument('--workers',type=int,default=1)
    a=p.parse_args();jobs=[(int(r),j) for r in a.roots.split(',') for j in range(a.futures)]
    if a.workers==1:
        for job in jobs:future(*job)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            fs=[pool.submit(future,*j) for j in jobs]
            for f in as_completed(fs):f.result()
    print('ABM_COMPLETE',flush=True)
if __name__=='__main__':main()
