"""General non-equal-wealth initial state: early second-order coefficients vs memory."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,pandas as pd
from common import *
from weak_response import jet_step,jet_metrics

def main():
    rows=[]
    for mu,h in [(1.,0.),(1.,.3),(1.,.9),(0.,.3),(5.,.3)]:
        sol=DensitySolver(Settings(dx=.01,na=96,mu=mu,h=h,threads=1,poisson_max=18));G=make_initial(sol,factorize=True);J=np.zeros((3,)+G.shape);J[0]=G
        rho_discrete=float(np.dot(sol.u,sol.R@sol.u)/np.dot(sol.u,sol.u))
        for n in range(1,3):
            J=jet_step(sol,J);rows.append(dict(mu=mu,h=h,round=n,rho_exact=corr_u(mu,h),rho_discrete=rho_discrete,**jet_metrics(J,sol.w,sol.u)))
    df=pd.DataFrame(rows);df.to_csv(ROOT/'results/weak_response/two_round_persistence_coefficients.csv',index=False)
    d=df[df['round']==2];audit={}
    for m in ['G2','M2','E2']:
        v=d[m]/d.rho_discrete;audit[m+'_ratio_range']=[float(v.min()),float(v.max())]
    dump_json(ROOT/'validation/two_round_memory_scaling.json',audit)
    print(d[['mu','h','rho_exact','G2','M2','E2']].to_string(index=False));print(audit)
if __name__=='__main__':main()
