"""Check analytic signed operator derivatives with an independent difference test."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from common import *
from weak_response import jet_step

def main():
    sol=DensitySolver(Settings(dx=.03,na=16,threads=1,poisson_max=18));G=make_initial(sol,factorize=True)
    J=np.zeros((3,)+G.shape);J[0]=G;J[1]=G*(.04+.02*sol.u[None,:,None]);J[2]=G*(.02+.01*np.sin(sol.w))[None,None,:]
    analytic=jet_step(sol,J);checks=[]
    for d in [.01,.005,.0025]:
        zero,_=sol.step(G,*sol.prepare(0.,0.),0.)
        plus,_=sol.step(G+d*J[1]+d*d*J[2],*sol.prepare(d,0.),0.)
        minus,_=sol.step(G-d*J[1]+d*d*J[2],*sol.prepare(-d,0.),0.)
        c1=(plus-minus)/(2*d);c2=(plus+minus-2*zero)/(2*d*d)
        checks.append(dict(delta=d,order0_max_abs=float(np.max(abs(zero-analytic[0]))),order1_max_abs=float(np.max(abs(c1-analytic[1]))),order2_max_abs=float(np.max(abs(c2-analytic[2]))),relative_L1_order2=float(np.sum(abs(c2-analytic[2]))/np.sum(abs(analytic[2])))))
    assert checks[-1]['relative_L1_order2']<1e-6
    dump_json(ROOT/'validation/response_operator_check.json',dict(purpose='Signed Taylor recurrence checked with nonzero derivative masses, including intrinsic partner normalization.',checks=checks))
if __name__=='__main__':main()
