#!/usr/bin/env python3
"""Analyze retained independent neutral paths; retain covariance between lags.

Usage: python code/kernel_analyze.py --tag main
All covariance estimates use divisor R-1. CI is an asymptotic 95% t interval
from the path-level influence product, after centering BOTH endpoint and score.
For cumulative/bin quantities the score is aggregated within each path BEFORE
estimating its covariance and standard error; lag errors are never added.
"""
import os
for _v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_v]='1'
import argparse,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]
PROCESSES=['Gaussian','Uniform','Two-state','Mixed']
X_NAMES=['G','M','E']

def gaussian_covariance(d):
    d=np.asarray(d,dtype=float);theta=.1*d
    weight=np.exp(-theta);out=weight*((2/math.pi)*math.asin(.5))
    for k in range(1,201):
        weight=weight*theta/k
        out+=weight*((2/math.pi)*math.asin(.3**k/2))
    return out

def all_covariances(d):
    d=np.asarray(d,dtype=int);rho1=3*float(gaussian_covariance([1])[0])
    pfast=2*(1-rho1)-.01
    return {'Gaussian':gaussian_covariance(d),'Uniform':.9**d/3,
            'Two-state':.9**d/3,'Mixed':(.99**d+(1-pfast)**d)/6},rho1,pfast

def cov_stats(x,a):
    x=np.asarray(x,np.float64);a=np.asarray(a,np.float64)
    scalar=a.ndim==1
    if scalar:a=a[:,None]
    R=len(x);xc=x-x.mean();ac=a-a.mean(axis=0)
    z=xc[:,None]*ac
    c=z.sum(axis=0)/(R-1)
    se=z.std(axis=0,ddof=1)/np.sqrt(R)*R/(R-1)
    half=t.ppf(.975,R-1)*se
    return [v[0] if scalar else v for v in [c,se,c-half,c+half]]

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',default='main');a=p.parse_args()
    folder=ROOT/'data'/f'kernel_{a.tag}_raw'
    source_meta=json.loads((ROOT/'data'/f'kernel_{a.tag}_meta.json').read_text())
    assert source_meta['status']=='complete','Simulation has not completed; analyze a completed tag'
    files=sorted(folder.glob('kernel_raw_*.npz'));assert files,folder
    pieces=[np.load(f,allow_pickle=False) for f in files]
    cp=pieces[0]['checkpoints'].astype(int)
    ids=np.concatenate([b['path_id'] for b in pieces]);assert len(np.unique(ids))==len(ids)
    X=np.concatenate([b['X_GME'] for b in pieces]);R=len(X)
    assert R==source_meta['reps'],(R,source_meta['reps'])
    meta=json.loads((ROOT/'data'/f'kernel_{a.tag}_meta.json').read_text())
    dmax=np.arange(int(cp[-1]));covs,rho1,pfast=all_covariances(dmax)
    pd.DataFrame([{'process':p,'d':int(d),'C':float(c[d]),'rho':float(3*c[d])}
                  for p,c in covs.items() for d in dmax]).to_csv(ROOT/'data'/'kernel_advantage_covariance.csv',index=False)
    rows_raw=[];rows_bin=[];rows_cum=[];rows_coef=[];rows_diag=[];rows_x=[];rows_tail=[]
    for j,n in enumerate(cp):
        S=np.concatenate([b[f'S_{n}'] for b in pieces]);assert np.all(S[:,0]==meta['N']*n)
        d=np.arange(1,n);score=S[:,1:].astype(float)/4
        # Exact second moment follows independent round sign vectors and
        # E(sigma_i sigma_j)=-1/(N-1) for i!=j due to perfect matching.
        expected_sd=np.sqrt((n-d)*(meta['N']**2/(meta['N']-1)))
        for k,lag in enumerate(d):
            rows_diag.append({'rounds':n,'d':lag,'S_mean':S[:,k+1].mean(),
              'S_sd':S[:,k+1].std(ddof=1),'S_exact_sd':expected_sd[k],
              'S_mean_z':S[:,k+1].mean()/(expected_sd[k]/math.sqrt(R)),'n_paths':R})
        for xi,xn in enumerate(X_NAMES):
            x=X[:,j,xi]
            rows_x.append({'rounds':n,'X':xn,'mean':x.mean(),'sd':x.std(ddof=1),
                           'nonzero_paths':int(np.count_nonzero(x)),'n_paths':R})
            c,se,lo,hi=cov_stats(x,score)
            for k,lag in enumerate(d):
                rows_raw.append({'rounds':n,'X':xn,'d':lag,'K':c[k],'se':se[k],
                                 'ci_low':lo[k],'ci_high':hi[k],'n_paths':R})
            width=5 if n<=80 else 10
            for start in range(1,n,width):
                end=min(start+width-1,n-1);w=end-start+1
                c1,s1,l1,h1=cov_stats(x,score[:,start-1:end].sum(axis=1))
                rows_bin.append({'rounds':n,'X':xn,'bin_start':start,'bin_end':end,
                  'bin_center':(start+end)/2,'width':w,'K_mean':c1/w,'se_mean':s1/w,
                  'ci_low_mean':l1/w,'ci_high_mean':h1/w,'K_sum':c1,'se_sum':s1,
                  'ci_low_sum':l1,'ci_high_sum':h1,'n_paths':R})
            for process,cv in covs.items():
                weighted=score*cv[1:n]
                accum=np.cumsum(weighted,axis=1)
                cc,ss,ll,hh=cov_stats(x,accum)
                rows_cum.append({'rounds':n,'X':xn,'process':process,'D':0,'cumulative':0.,
                                 'se':0.,'ci_low':0.,'ci_high':0.,'n_paths':R})
                for k,lag in enumerate(d):
                    rows_cum.append({'rounds':n,'X':xn,'process':process,'D':lag,'cumulative':cc[k],
                                     'se':ss[k],'ci_low':ll[k],'ci_high':hh[k],'n_paths':R})
                rows_coef.append({'rounds':n,'X':xn,'process':process,'c':cc[-1],
                                  'se':ss[-1],'ci_low':ll[-1],'ci_high':hh[-1],'n_paths':R})
                for cut in [5,10,20,40,80,160]:
                    if cut>=n-1:continue
                    vals=cov_stats(x,weighted[:,cut:].sum(axis=1))
                    rows_tail.append({'rounds':n,'X':xn,'process':process,'d_gt':cut,'tail':vals[0],
                                      'se':vals[1],'ci_low':vals[2],'ci_high':vals[3],'n_paths':R})
            # Paired differences across process weights must use same-path score.
            for p1,p0 in [('Mixed','Gaussian'),('Gaussian','Uniform')]:
                cc,ss,ll,hh=cov_stats(x,score@(covs[p1][1:n]-covs[p0][1:n]))
                rows_coef.append({'rounds':n,'X':xn,'process':f'{p1}-{p0}','c':cc,'se':ss,'ci_low':ll,'ci_high':hh,'n_paths':R})
    for name,rows in [('raw_lags',rows_raw),('binned',rows_bin),('cumulative',rows_cum),
                      ('coefficients',rows_coef),('null_diagnostics',rows_diag),('observables',rows_x),('tails',rows_tail)]:
        pd.DataFrame(rows).to_csv(ROOT/'data'/f'kernel_{name}.csv',index=False)
    # Reuse an already recorded independent biased simulation; no new ABM values invented.
    directfile=ROOT/'inputs'/'Fig2'/'data'/'direct_multitime.csv'
    if directfile.exists():
        direct=pd.read_csv(directfile);predict=pd.DataFrame(rows_coef)
        ds=[];mode_to_name={0:'Gaussian',1:'Uniform',2:'Two-state',6:'Mixed'}
        for (mode,n),g in direct.groupby(['mode','rounds']):
            if n not in cp:continue
            y=g.response.to_numpy();se=y.std(ddof=1)/math.sqrt(len(y));half=t.ppf(.975,len(y)-1)*se
            pr=predict[(predict['rounds']==n)&(predict['X']=='G')&(predict['process']==mode_to_name[mode])].iloc[0]
            ds.append({'rounds':n,'process':mode_to_name[mode],'prediction':pr.c,'pred_se':pr.se,
              'pred_low':pr.ci_low,'pred_high':pr.ci_high,'direct':y.mean(),'direct_se':se,
              'direct_low':y.mean()-half,'direct_high':y.mean()+half,
              'difference':y.mean()-pr.c,'difference_se':math.sqrt(se*se+pr.se*pr.se),
              'n_pred':R,'n_direct':len(y),'direct_lambda':.25,
              'direct_source':'inputs/Fig2/data/direct_multitime.csv; reused independent biased futures'})
        pd.DataFrame(ds).to_csv(ROOT/'data'/'kernel_validation.csv',index=False)
    diagnostics=pd.DataFrame(rows_diag)
    analysis={'input_tag':a.tag,'n_paths':R,'checkpoints':[int(n) for n in cp],
      'estimator':'unbiased sample covariance, divisor R-1; K=Cov(X,S_d)/4',
      'ci':'95% asymptotic Student-t interval from centered path-level product influence values; pointwise, not simultaneous',
      'lag_correlation':'all cumulative, grouped, process-difference and tail SEs computed after per-path score aggregation',
      'bins':'prespecified width5 at n40,80; width10 at n160,300; final bin may be shorter',
      'gaussian_C':'Poisson sum k=0..200 of (2/pi)asin(h^k/2), h=.3, rate per round .1',
      'rho1':rho1,'mixed_p_slow':.01,'mixed_p_fast':pfast,
      'uniform_two_state':'exactly equal Cu(d)=0.9^d/3 at every lag and exactly equal second-order weighted coefficients',
      'kernel_process_independence':'one neutral wealth kernel shared across all four processes',
      'max_abs_null_mean_z':float(diagnostics.S_mean_z.abs().max()),
      'n_null_checks':len(diagnostics),'warning':'pointwise null z extremes are multiple correlated checks; not a standalone failure test',
      'raw_data_files':[str(f.relative_to(ROOT)) for f in files],
      'created_unix':time.time()}
    (ROOT/'data'/'kernel_analysis_meta.json').write_text(json.dumps(analysis,indent=2))
    print(pd.DataFrame(rows_coef).to_string(index=False))
    print('\nOBSERVABLES\n'+pd.DataFrame(rows_x).to_string(index=False))
    print('\nDIAGNOSTICS\n'+json.dumps(analysis,indent=2))
if __name__=='__main__':main()
