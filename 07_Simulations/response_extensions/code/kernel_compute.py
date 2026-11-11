#!/usr/bin/env python3
"""Exact neutral Yard-Sale paths and all-lag response sufficient statistics.

Run from any directory. Requires Python3/numpy/scipy and g++ (C++17).
For example:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/kernel_compute.py --reps 16000 --workers 2
The fixed initial wealth is inputs/w0_fixed.npy; path seeds are independent.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse, ctypes, hashlib, json, subprocess, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len

ROOT=Path(__file__).resolve().parents[1]
NATIVE=r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>
struct RNG {
    uint64_t s[4];
    static uint64_t sm(uint64_t &x) {
        uint64_t z=(x+=0x9e3779b97f4a7c15ULL);
        z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
        z=(z^(z>>27))*0x94d049bb133111ebULL;
        return z^(z>>31);
    }
    explicit RNG(uint64_t seed) {for(int j=0;j<4;j++)s[j]=sm(seed);}
    static uint64_t rot(uint64_t x,int k){return (x<<k)|(x>>(64-k));}
    uint64_t next(){
        const uint64_t z=rot(s[0]+s[3],23)+s[0];
        const uint64_t t=s[1]<<17;
        s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=t;s[3]=rot(s[3],45);
        return z;
    }
    uint64_t bounded(uint64_t b){uint64_t x,th=(-b)%b;do{x=next();}while(x<th);return x%b;}
};
extern "C" void neutral_path(const double *w0,int N,const int *cp,int nc,double eps,
 uint64_t seed,int8_t *sgn,double *out,double *diag){
    int T=cp[nc-1]; RNG rng(seed);
    std::vector<double>w(w0,w0+N);std::vector<int>p(N),ix(N);
    std::iota(p.begin(),p.end(),0);std::iota(ix.begin(),ix.end(),0);
    std::sort(ix.begin(),ix.end(),[&](int a,int b){return w0[a]<w0[b]||(w0[a]==w0[b]&&a<b);});
    std::vector<uint8_t>bottom(N,0),top(N,0);
    for(int k=0;k<N/2;k++)bottom[ix[k]]=1;
    int ntop=N/10;for(int k=N-ntop;k<N;k++)top[ix[k]]=1;
    double sum0=std::accumulate(w.begin(),w.end(),0.0),maxerr=0.0,minw=1e300;
    int cc=0;
    for(int t=0;t<T;t++){
        for(int j=N-1;j>0;j--)std::swap(p[j],p[rng.bounded(j+1)]);
        for(int k=0;k<N;k+=2){
            int a=p[k],b=p[k+1];int sig=(rng.next()>>63)?1:-1;
            double stake=eps*std::min(w[a],w[b]);w[a]+=sig*stake;w[b]-=sig*stake;
            sgn[a*T+t]=sig;sgn[b*T+t]=-sig;
        }
        if(t+1==cp[cc]){
            std::sort(ix.begin(),ix.end(),[&](int a,int b){return w[a]<w[b]||(w[a]==w[b]&&a<b);});
            double sum=0.,wg=0.;int nb=0,nkeep=0;
            for(int k=0;k<N;k++){sum+=w[ix[k]];wg+=(2.*(k+1)-N-1.)*w[ix[k]];}
            for(int k=N-ntop;k<N;k++){nb+=bottom[ix[k]];nkeep+=top[ix[k]];}
            out[cc*3]=wg/(N*sum);
            out[cc*3+1]=double(nb)/(N/2);
            out[cc*3+2]=1.-double(nkeep)/ntop;
            maxerr=std::max(maxerr,std::abs(sum-sum0));minw=std::min(minw,w[ix[0]]);
            cc++;
        }
    }
    diag[0]=maxerr;diag[1]=minw;
}
'''

def compile_native():
    native=ROOT/'logs'/'kernel_native.cpp';lib=ROOT/'logs'/'kernel_native.so'
    native.write_text(NATIVE)
    cmd=['g++','-O3','-std=c++17','-shared','-fPIC',str(native),'-o',str(lib)]
    subprocess.run(cmd,check=True,capture_output=True)
    return str(lib),cmd

def loadlib(path):
    lib=ctypes.CDLL(path)
    lib.neutral_path.argtypes=[np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS'),ctypes.c_int,
      np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS'),ctypes.c_int,ctypes.c_double,ctypes.c_uint64,
      np.ctypeslib.ndpointer(np.int8,flags='C_CONTIGUOUS'),np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS'),
      np.ctypeslib.ndpointer(np.float64,flags='C_CONTIGUOUS')]
    lib.neutral_path.restype=None
    return lib

def autocorrelation(sign,n):
    z=rfft(sign[:,:n],n=next_fast_len(2*n-1),axis=1,workers=1)
    power=(z.real*z.real+z.imag*z.imag).sum(axis=0)
    a=irfft(power,n=next_fast_len(2*n-1),workers=1)[:n]
    residual=float(np.max(np.abs(a-np.rint(a))))
    return np.rint(a).astype(np.int32),residual

def batch_run(args):
    start,stop,cp,eps,seedbase,libpath,outdir=args
    t0=time.time();lib=loadlib(libpath)
    w0=np.ascontiguousarray(np.load(ROOT/'inputs'/'w0_fixed.npy'),dtype=np.float64)
    cp=np.asarray(cp,dtype=np.int32);N=len(w0);T=int(cp[-1]);C=len(cp)
    ids=np.arange(start,stop,dtype=np.int64)
    xs=np.empty((len(ids),C,3),np.float64)
    sd=[np.empty((len(ids),int(n)),np.int32) for n in cp]
    sign=np.empty((N,T),np.int8);diag=np.empty(2)
    maxres=0.;maxdrift=0.;minwealth=np.inf
    for q,pathid in enumerate(ids):
        x=np.empty((C,3),np.float64)
        lib.neutral_path(w0,N,cp,C,eps,int(seedbase+int(pathid)),sign,x,diag)
        xs[q]=x;maxdrift=max(maxdrift,float(diag[0]));minwealth=min(minwealth,float(diag[1]))
        if not np.all(sign.sum(axis=0)==0):raise RuntimeError('Pair sign balance violated')
        for j,n in enumerate(cp):
            s,rr=autocorrelation(sign,int(n));sd[j][q]=s;maxres=max(maxres,rr)
            if s[0]!=N*n:raise RuntimeError('S0 != N*n')
            if pathid==0:
                # Independent non-circular integer direct convolution on first path.
                tmp=sign[:,:n].astype(np.int32)
                direct=np.array([(tmp[:,:n-d]*tmp[:,d:n]).sum(dtype=np.int64) for d in range(n)])
                if not np.array_equal(s,direct):raise RuntimeError('FFT != direct lag sum')
        if maxres>1e-6:raise RuntimeError('FFT integer reconstruction failed')
    output=Path(outdir)/f'kernel_raw_{start:06d}_{stop:06d}.npz'
    payload={'path_id':ids,'seed':np.uint64(seedbase)+ids.astype(np.uint64),'checkpoints':cp,'X_GME':xs}
    for j,n in enumerate(cp):payload[f'S_{n}']=sd[j]
    np.savez_compressed(output,**payload)
    return {'start':start,'stop':stop,'paths':len(ids),'seconds':time.time()-t0,'file':str(output),
            'max_fft_rounding_error':maxres,'max_total_wealth_drift':maxdrift,'min_wealth':minwealth}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reps',type=int,default=16000)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--batch-size',type=int,default=250)
    p.add_argument('--checkpoints',type=int,nargs='+',default=[40,80,160,300])
    p.add_argument('--seed-base',type=int,default=202611070910000)
    p.add_argument('--output-tag',default='main')
    p.add_argument('--eps',type=float,default=.14)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();assert 1<=a.workers<=2
    assert a.checkpoints==sorted(set(a.checkpoints)) and min(a.checkpoints)>1
    outdir=ROOT/'data'/f'kernel_{a.output_tag}_raw';outdir.mkdir(parents=True,exist_ok=True)
    lib,cmd=compile_native();t0=time.time()
    batches=[(i,min(i+a.batch_size,a.reps),a.checkpoints,a.eps,a.seed_base,lib,str(outdir)) for i in range(0,a.reps,a.batch_size)]
    if a.resume:batches=[b for b in batches if not (outdir/f'kernel_raw_{b[0]:06d}_{b[1]:06d}.npz').exists()]
    meta={'model':'neutral fair-coin pairwise Yard-Sale','chi':0.,'N':1000,'epsilon':a.eps,'rounds_per_time':10.,
      'checkpoints':a.checkpoints,'reps':a.reps,'workers':a.workers,'seed_base':a.seed_base,'rng':'xoshiro256++ seeded by splitmix64; seed=seed_base+path_id',
      'matching':'unbiased Fisher-Yates random permutation, sequential disjoint pairs each round','initial_file':'inputs/w0_fixed.npy',
      'initial_sha256':hashlib.sha256((ROOT/'inputs'/'w0_fixed.npy').read_bytes()).hexdigest(),
      'X_order':['G','M','E'],'M_definition':'initial bottom50% entering current top10%, divided by bottom cohort size',
      'E_definition':'fraction of initial top10% absent from current top10%',
      'S_definition':'S_d=sum_i sum_{k=0}^{n-1-d} sigma[i,k]*sigma[i,k+d]; includes S0=N*n',
      'fft':'real FFT per individual with zero padding to next_fast_len(2*n-1), sum power before inverse, rounded to exact integer',
      'compile':cmd,'path_blocks':[],'status':'running','start_unix':t0}
    mpath=ROOT/'data'/f'kernel_{a.output_tag}_meta.json';mpath.write_text(json.dumps(meta,indent=2))
    done=0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs=[ex.submit(batch_run,b) for b in batches]
        for f in as_completed(futs):
            result=f.result();done+=result['paths'];meta['path_blocks'].append(result)
            meta['elapsed_sec']=time.time()-t0;mpath.write_text(json.dumps(meta,indent=2))
            print(json.dumps({'done_this_run':done,'target':a.reps,'elapsed_sec':round(meta['elapsed_sec'],2),**result}),flush=True)
    meta['status']='complete';meta['elapsed_sec']=time.time()-t0;mpath.write_text(json.dumps(meta,indent=2))
    print('COMPLETE',json.dumps({'seconds':meta['elapsed_sec'],'meta':str(mpath)}),flush=True)
if __name__=='__main__':main()
