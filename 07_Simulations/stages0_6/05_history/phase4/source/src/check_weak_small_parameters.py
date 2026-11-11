"""Independent near-zero full solves to test propagated coefficients, no fitting."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import time
import numpy as np,pandas as pd
from common import *

def main():
    folder=ROOT/'results/weak_response';sol=DensitySolver(Settings(dx=.01,na=96,threads=1,poisson_max=18));G0=make_initial(sol,factorize=True);rows=[]
    for lam in (.01,.02):
        G=G0.copy();P,ti,ta=sol.prepare(lam,0);tic=time.time()
        for n in range(301):
            if n in OBS:rows.append(dict(lam=lam,round=n,time=n/10,**measure(G,sol.w,sol.u)))
            if n==300:break
            G,_=sol.step(G,P,ti,ta,0)
        print('SMALL',lam,round(time.time()-tic,1),flush=True)
    pd.DataFrame(rows).to_csv(folder/'small_parameter_full_solutions.csv',index=False)
    while not (folder/'response_coefficients.csv').exists():time.sleep(2)
    d=pd.DataFrame(rows).merge(pd.read_csv(folder/'response_coefficients.csv'),on=['round','time'],validate='many_to_one')
    for m in ['G','M','E']:
        d[m+'_quotient']=(d[m]-d[m+'0'])/d.lam**2;d[m+'_coefficient_error']=d[m+'_quotient']-d[m+'2']
    d['Q_quotient']=d.Q/d.lam;d['Q_coefficient_error']=d.Q_quotient-d.Q1
    d.to_csv(folder/'small_parameter_coefficient_check.csv',index=False);print('SMALL_CHECK_COMPLETE',flush=True)
if __name__=='__main__':main()
