#!/usr/bin/env python3
"""Late-lag response fractions with joint path-level delta-method uncertainty."""
import os
for v in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[v]='1'
from pathlib import Path
import json,math
import numpy as np,pandas as pd
from scipy.stats import t
from kernel_analyze import all_covariances,cov_stats
ROOT=Path(__file__).resolve().parents[1]
def main():
    analysis=json.loads((ROOT/'data'/'kernel_analysis_meta.json').read_text())
    pieces=[np.load(ROOT/f) for f in analysis['raw_data_files']]
    X=np.concatenate([p['X_GME'] for p in pieces]);cp=pieces[0]['checkpoints'];R=len(X)
    covs,_,_=all_covariances(np.arange(cp[-1]))
    covs['Mixed-Gaussian']=covs['Mixed']-covs['Gaussian']
    out=[];rows=[]
    for j,n in enumerate(cp):
        S=np.concatenate([p[f'S_{n}'] for p in pieces])[:,1:].astype(float)/4
        for xi,xname in enumerate(['G','M','E']):
            x=X[:,j,xi];xc=x-x.mean()
            for process,c in covs.items():
                weights=S*c[1:n]
                full=weights.sum(axis=1);cf,sf,lf,hf=cov_stats(x,full)
                zf=xc*(full-full.mean())*R/(R-1)
                for cut in [5,10,20,40,80,160]:
                    if cut>=n-1:continue
                    tail=weights[:,cut:].sum(axis=1);ct,st,lt,ht=cov_stats(x,tail)
                    stable=bool(abs(cf)>3*sf and np.count_nonzero(x)>=25)
                    ratio=ct/cf if cf else np.nan
                    zt=xc*(tail-tail.mean())*R/(R-1)
                    sr=(zt-ratio*zf).std(ddof=1)/math.sqrt(R)/abs(cf) if cf else np.nan
                    h=t.ppf(.975,R-1)*sr
                    out.append(dict(rounds=int(n),X=xname,process=process,d_gt=cut,
                      total=cf,tail=ct,tail_se=st,tail_ci_low=lt,tail_ci_high=ht,
                      tail_fraction=ratio,fraction_se=sr,fraction_ci_low=ratio-h,fraction_ci_high=ratio+h,
                      fraction_signal_stable=stable,n_paths=R))
                if process=='Mixed-Gaussian':
                    C=np.cumsum(weights,axis=1);cs,ss,ll,hh=cov_stats(x,C)
                    for k in range(n-1):rows.append(dict(rounds=int(n),X=xname,process=process,D=k+1,
                      cumulative=cs[k],se=ss[k],ci_low=ll[k],ci_high=hh[k],n_paths=R))
    pd.DataFrame(out).to_csv(ROOT/'data'/'kernel_lag_shares.csv',index=False)
    pd.DataFrame(rows).to_csv(ROOT/'data'/'kernel_cumulative_difference.csv',index=False)
    print(pd.DataFrame(out).query("X=='G' and rounds==80 and d_gt==20").to_string(index=False))
if __name__=='__main__':main()
