"""Fixed precision extension, declared after the first 24 development futures.
Adds 72 new futures, not new preparation roots; no repeat-until-significant loop.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import time
import numpy as np,pandas as pd
from scipy.stats import t
from common import ROOT,dump_json
from run_abm_mechanisms import one

def ci_table(df,perroot,label):
    rows=[]
    for (n,lam),g in df[df.lam>0].groupby(['round','lam']):
        ctl=df[(df['round']==n)&(df.lam==0)];z=g.merge(ctl,on=['root','future_rep'],suffixes=('','_control'),validate='one_to_one')
        for m,b in [('G',.001),('M',.001),('E',.002)]:
            rd=(z[m]-z[m+'_control']).groupby(z.root).mean();mean=float(rd.mean());se=float(rd.std(ddof=1)/np.sqrt(len(rd)));half90=float(t.ppf(.95,len(rd)-1)*se);half95=float(t.ppf(.975,len(rd)-1)*se)
            rows.append(dict(sample=label,lam=lam,round=n,time=n/10,metric=m,mean_difference=mean,ci90_half=half90,ci95_half=half95,equivalence_bound=b,equivalence_pass=abs(mean)+half90<b,n_roots=len(rd),futures_per_root=perroot))
    return pd.DataFrame(rows)

def main():
    out=ROOT/'results/abm_mechanisms';tic=time.time();rows=[]
    for seed in range(2026101201,2026101207):
        for rep in range(5,17):
            rr,_=one(seed,rep,'full_redraw',1.,.3,True);rows+=rr
        print('EXTENSION root',seed,'seconds',round(time.time()-tic,1),flush=True)
    new=pd.DataFrame(rows);new.to_csv(out/'full_redraw_precision_extension_raw.csv',index=False)
    original=pd.read_csv(out/'full_redraw_raw.csv');combined=pd.concat([original,new],ignore_index=True)
    ci_table(new,12,'new72').to_csv(out/'full_redraw_new72_equivalence.csv',index=False)
    ci_table(combined,16,'combined96').to_csv(out/'full_redraw_combined96_equivalence.csv',index=False)
    dump_json(out/'precision_extension_audit.json',dict(added_futures=72,added_population_branches=216,rounds_per_branch=300,combined_futures=96,combined_branches=288,preparation_roots=6,new_metric_rows=len(new),max_mean_error=float((new['mean']-1).abs().max()),max_count_error=float(new.count_error.abs().max()),minimum_wealth=float(new.minimum.min()),holdout_read=False,seconds=time.time()-tic))
    print('PRECISION_EXTENSION_COMPLETE',flush=True)
if __name__=='__main__':main()
