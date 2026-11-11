#!/usr/bin/env python3
"""Summarize neutral covariance and paired finite-selection interaction.

The independent statistical unit is an independent future conditional on the
single supplied initial state. Every branch of a future stays together.
"""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[_k]='1'
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student
ROOT=Path(__file__).resolve().parents[1]
OBS=['G','M','E']


def mean_ci(y):
    n=len(y); mean=y.mean(axis=0); se=y.std(axis=0,ddof=1)/math.sqrt(n)
    half=student.ppf(.975,n-1)*se
    return mean,se,mean-half,mean+half


def covariance_ci(x,a):
    n=len(a)
    # Sample covariance; influence-function standard error retains per-future
    # product and includes centering of both factors.
    z=(x-x.mean(axis=0))*(a-a.mean())[:,None]
    estimate=z.sum(axis=0)/(n-1)
    influence=z*n/(n-1)
    se=influence.std(axis=0,ddof=1)/math.sqrt(n)
    half=student.ppf(.975,n-1)*se
    return estimate,se,estimate-half,estimate+half,influence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag',default=''); p.add_argument('--bootstrap',type=int,default=0)
    args=p.parse_args(); suffix=('_'+args.tag) if args.tag else ''
    nf=ROOT/'data'/('interaction_neutral_raw'+suffix+'.npz')
    df=ROOT/'data'/('interaction_direct_raw'+suffix+'.npz')
    n=np.load(nf,allow_pickle=False)
    d=np.load(df,allow_pickle=False) if df.exists() else None
    nr=n['raw']; a=n['A']; checks=n['checkpoints']; chis=n['chis']
    delta=nr[:,:,1,:]-nr[:,:,0,:]
    rng=np.random.default_rng(813503)
    preds=[]; rows=[]; finite=[]; baseline=[]; symmetry=[]; baseline_check=[]
    for ck,rounds in enumerate(checks):
        pred,se,low,high,inf=covariance_ci(delta[:,ck],a[:,ck])
        blo=low.copy(); bhi=high.copy()
        if args.bootstrap:
            # Resample complete independent futures; all observables and both
            # tax branches remain paired. Used as a percentile sensitivity CI.
            boot=np.empty((args.bootstrap,3))
            for b in range(args.bootstrap):
                ind=rng.integers(0,len(nr),len(nr)); xb=delta[ind,ck]; ab=a[ind,ck]
                boot[b]=np.sum((xb-xb.mean(axis=0))*(ab-ab.mean())[:,None],axis=0)/(len(ind)-1)
            blo,bhi=np.quantile(boot,[.025,.975],axis=0)
        for j,obs in enumerate(OBS):
            preds.append(dict(rounds=int(rounds),time=float(rounds/10),observable=obs,
              chi=float(chis[1]),prediction=float(pred[j]),prediction_se=float(se[j]),
              prediction_low=float(low[j]),prediction_high=float(high[j]),
              prediction_boot_low=float(blo[j]),prediction_boot_high=float(bhi[j]),
              n_neutral=len(nr),mean_A=float(a[:,ck].mean()),se_mean_A=float(a[:,ck].std(ddof=1)/math.sqrt(len(nr))),
              ci_unit='independent future conditional on fixed initial wealth'))
        for c,chi in enumerate(chis):
            bm,bs,bl,bh=mean_ci(nr[:,ck,c])
            for j,obs in enumerate(OBS):
                baseline.append(dict(rounds=int(rounds),time=rounds/10,chi=float(chi),observable=obs,
                  mean=float(bm[j]),se=float(bs[j]),ci_low=float(bl[j]),ci_high=float(bh[j]),n=len(nr)))
        if d is None: continue
        dr=d['raw']; lv=d['lambdas']
        delta_d=dr[:,ck,:,1,:]-dr[:,ck,:,0,:]
        for c,chi in enumerate(chis):
            nm,ns,_,_=mean_ci(nr[:,ck,c])
            dm,ds,_,_=mean_ci(dr[:,ck,0,c])
            for j,obs in enumerate(OBS):
                combined=math.sqrt(ns[j]**2+ds[j]**2)
                baseline_check.append(dict(rounds=int(rounds),observable=obs,chi=float(chi),
                    neutral_mean=float(nm[j]),direct_zero_mean=float(dm[j]),
                    difference=float(dm[j]-nm[j]),difference_se=float(combined),
                    standardized_difference=float((dm[j]-nm[j])/combined) if combined else 0.0))
        qs=[]; ls=[]
        for l in lv[lv>0]:
            pos=int(np.flatnonzero(lv==l)[0]); neg=int(np.flatnonzero(lv==-l)[0])
            direct=(0.5*(delta_d[:,pos]+delta_d[:,neg])-delta_d[:,0])/(l*l)
            qs.append(direct); ls.append(l)
            odd=(delta_d[:,pos]-delta_d[:,neg])/(2*l)
            om,os,ol,oh=mean_ci(odd)
            for j,obs in enumerate(OBS):
                symmetry.append(dict(rounds=int(rounds),observable=obs,lambda_abs=float(l),
                    odd_interaction_mean=float(om[j]),odd_interaction_se=float(os[j]),
                    odd_interaction_low=float(ol[j]),odd_interaction_high=float(oh[j]),
                    theory_expectation=0.0))
            dm,ds,dl,dh=mean_ci(direct)
            for j,obs in enumerate(OBS):
                diff=dm[j]-pred[j]; combined_se=math.sqrt(ds[j]**2+se[j]**2)
                # Independent prediction and validation simulations.
                rows.append(dict(rounds=int(rounds),time=float(rounds/10),observable=obs,
                    chi=float(chis[1]),lambda_abs=float(l),prediction=float(pred[j]),
                    prediction_se=float(se[j]),prediction_low=float(low[j]),prediction_high=float(high[j]),
                    prediction_boot_low=float(blo[j]),prediction_boot_high=float(bhi[j]),
                    direct=float(dm[j]),direct_se=float(ds[j]),direct_low=float(dl[j]),direct_high=float(dh[j]),
                    direct_minus_prediction=float(diff),difference_se=float(combined_se),
                    difference_low=float(diff-1.96*combined_se),difference_high=float(diff+1.96*combined_se),
                    standardized_discrepancy=float(diff/combined_se) if combined_se else 0.0,
                    n_neutral=len(nr),n_direct=len(dr),
                    response_definition='[0.5*(DeltaX(+lambda)+DeltaX(-lambda))-DeltaX(0)]/lambda^2',
                    ci_unit='independent future conditional on fixed initial wealth'))
        q=np.stack(qs,axis=1); lam2=np.array(ls)**2
        design=np.column_stack((np.ones(len(ls)),lam2))
        for j,obs in enumerate(OBS):
            cov=np.cov(q[:,:,j],rowvar=False,ddof=1)/len(dr)
            wt=np.linalg.pinv(cov,rcond=1e-12)
            transform=np.linalg.pinv(design.T@wt@design)@design.T@wt
            coef=transform@q[:,:,j].mean(axis=0)
            ccov=transform@cov@transform.T
            cs=np.sqrt(np.diag(ccov)); resid=q[:,:,j].mean(axis=0)-design@coef
            finite.append(dict(rounds=int(rounds),time=rounds/10,observable=obs,chi=float(chis[1]),
                fitted_c2=float(coef[0]),fitted_c2_se=float(cs[0]),fitted_c2_low=float(coef[0]-1.96*cs[0]),
                fitted_c2_high=float(coef[0]+1.96*cs[0]),fitted_c4=float(coef[1]),fitted_c4_se=float(cs[1]),
                fitted_c4_low=float(coef[1]-1.96*cs[1]),fitted_c4_high=float(coef[1]+1.96*cs[1]),
                chi_square_one_df=float(resid@wt@resid),prediction=float(pred[j]),prediction_se=float(se[j]),
                fit_model='Q(lambda)=c2+c4*lambda^2, diagnostic fit over 0.1,0.25,0.5',
                caution='intercept retains possible O(lambda^4) bias; not the neutral prediction'))
    pd.DataFrame(preds).to_csv(ROOT/'data'/('interaction_prediction'+suffix+'.csv'),index=False)
    pd.DataFrame(baseline).to_csv(ROOT/'data'/('interaction_neutral_baseline'+suffix+'.csv'),index=False)
    if rows:
        table=pd.DataFrame(rows)
        table.to_csv(ROOT/'data'/('interaction_summary'+suffix+'.csv'),index=False)
        pd.DataFrame(finite).to_csv(ROOT/'data'/('interaction_finite_lambda'+suffix+'.csv'),index=False)
        pd.DataFrame(symmetry).to_csv(ROOT/'data'/('interaction_symmetry'+suffix+'.csv'),index=False)
        pd.DataFrame(baseline_check).to_csv(ROOT/'data'/('interaction_baseline_check'+suffix+'.csv'),index=False)
        print(table[['rounds','observable','lambda_abs','prediction','prediction_se','direct','direct_se','standardized_discrepancy']].to_string(index=False))
    else:
        print(pd.DataFrame(preds).to_string(index=False))
    report=dict(neutral_file=nf.name,direct_file=df.name if d is not None else None,
        bootstrap_replicates=args.bootstrap,bootstrap_seed=813503,
        prediction_ci='95% Student-t influence-function covariance CI',
        direct_ci='95% Student-t CI of within-future even interaction quotient',
        prediction_and_direct_are_independent=True,
        fixed_initial_condition=True,not_uncertainty_over_initial_wealth=True,
        neutral_estimated_mean_A=float(a.mean()),
        neutral_no_redistribution_interaction_exactly_zero=True,
        no_sign_selection=True)
    (ROOT/'logs'/('interaction_analysis_metadata'+suffix+'.json')).write_text(json.dumps(report,indent=2))

if __name__=='__main__': main()
