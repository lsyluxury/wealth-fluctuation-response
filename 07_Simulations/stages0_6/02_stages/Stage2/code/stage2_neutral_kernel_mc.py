#!/usr/bin/env python3
"""Independent direct neutral-path covariance estimator; paired with exact N correction."""
import argparse,json,time,sys,math
from pathlib import Path
import numpy as np
from scipy.stats import t
from legacy_engine import reseed, gini_numba
from engine_stage01 import advance_trace
from response_kernel import S_by_lag_direct

def kernel_two_round_equal(N,R,seed):
    vals=np.empty((R,2),dtype=float)
    for k in range(R):
        w=np.ones((N,1),dtype=float);a=np.zeros(N);u=np.zeros(N);perm=np.arange(N)
        reseed(seed+k)
        ss=advance_trace(w,a,u,perm,np.array([0.]),np.array([0.]),2,.14,.3,0.,10.,3,0.,True)
        s=np.dot(ss[0,:,0].astype(float),ss[1,:,0].astype(float))
        vals[k,0]=gini_numba(w[:,0]);vals[k,1]=s
    X=vals[:,0];S=vals[:,1]
    estimator=(X-X.mean())*(S-S.mean())/4
    ktrue=(N/(N-1))*.14*(1-.14)/8
    kavg=np.mean(estimator);se=np.std(estimator,ddof=1)/np.sqrt(R)
    return dict(N=N,R=R,kernel_estimate=kavg,kernel_SE=se,CI95=[kavg-t.ppf(.975,R-1)*se,kavg+t.ppf(.975,R-1)*se],exact=ktrue,mean_S=S.mean(),error=kavg-ktrue,seconds=None)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--N',type=int,default=600);ap.add_argument('--reps',type=int,default=10000);ap.add_argument('--out',default='../results/neutral_kernel_2round.json');args=ap.parse_args()
    tm=time.time();z=kernel_two_round_equal(args.N,args.reps,430000000);z['seconds']=time.time()-tm
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(z,indent=2));print(json.dumps(z,indent=2),flush=True)
