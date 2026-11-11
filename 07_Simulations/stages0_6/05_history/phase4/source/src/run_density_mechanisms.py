"""Density predictions for predeclared capability-persistence interventions."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import time
import numpy as np,pandas as pd
from common import *

def trajectory(mu,h,name,lam,reset=False):
    sol=DensitySolver(Settings(dx=.01,na=96,mu=mu,h=h,threads=1,poisson_max=18))
    G=make_initial(sol,factorize=reset)
    if reset:sol.R=np.ones((sol.s.na,sol.s.na))/sol.s.na
    P,ti,ta=sol.prepare(lam,0);rr=[];audit=dict(max_mass_error=0.,max_wealth_error=0.);tic=time.time()
    for n in range(301):
        if n in OBS:rr.append(dict(mode=name,lam=lam,mu=mu,h=h,lag_score_corr=0. if reset else corr_u(mu,h),round=n,time=n/10,**measure(G,sol.w,sol.u)))
        if n==300:break
        G,_=sol.step(G,P,ti,ta,0)
        audit['max_mass_error']=max(audit['max_mass_error'],float(abs(G.sum()-1)));audit['max_wealth_error']=max(audit['max_wealth_error'],float(abs(np.sum(G*sol.w)-1)))
    audit['seconds']=time.time()-tic;audit['kernel']=sol.kernel_info
    print(name,lam,'completed seconds',round(audit['seconds'],1),flush=True)
    return rr,audit

def main():
    out=ROOT/'results/density_mechanisms';out.mkdir(parents=True,exist_ok=True);rows=[];audit={}
    for name,mu,h in [('sparse_h0',1.,0.),('persistent_h09',1.,.9),('frozen_mu0',0.,.3),('fast_mu5',5.,.3)]:
        rr,aa=trajectory(mu,h,name,4.);rows+=rr;audit[name]=aa
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    for lam in (0.,4.,16.):
        rr,aa=trajectory(1.,.3,'full_redraw',lam,True);rows+=rr;audit[f'full_redraw_{lam}']=aa
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    d=pd.DataFrame(rows);reset=d[d['mode']=='full_redraw'];ref=reset[reset.lam==0]
    z=reset[reset.lam>0].merge(ref,on=['round','time'],suffixes=('','_control'))
    audit['full_redraw_max_diff']={m:float((z[m]-z[m+'_control']).abs().max()) for m in ['G','M','E']}
    dump_json(out/'audit.json',audit);print('DENSITY_MECHANISMS_COMPLETE',flush=True)
if __name__=='__main__':main()
