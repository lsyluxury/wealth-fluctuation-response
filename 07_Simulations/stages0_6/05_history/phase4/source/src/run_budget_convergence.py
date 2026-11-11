"""New B50/M40/T10 density, endogenous A->B; no future ABM inputs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,time
import numpy as np,pandas as pd
from common import *
def advance(sol,G,stage,lam,chi,label,budgets):
    P,ti,ta=sol.prepare(lam,chi);rows=[];br=[];audit={'max_mass_error':0.,'max_wealth_error':0.,'min_mass':0.,'max_count_error':0.}
    rho=np.exp(-chi/sol.s.r);modes=mode_vectors(sol.s.na,5);gamma=np.exp(-sol.s.mu/sol.s.r*(1-sol.s.h**np.arange(1,6)));tic=time.time()
    for n in range(301):
        if n in OBS:rows.append(dict(stage=stage,lam=lam,chi=chi,round=n,time=n/10,**measure(G,sol.w,sol.u)))
        if n==300:break
        if budgets:
            f=G.sum(axis=0);mass=f.sum();H=np.stack([P@f,(1-P)@f])/mass;T,stats=trade(G,H,sol.w,*sol.plan)
            H0=np.broadcast_to(f.sum(axis=0)[None,None,:]/mass/2,(2,sol.s.na,len(sol.w))).copy();T0,_=trade(G,H0,sol.w,*sol.plan)
            Rg=np.matmul(sol.R,T);out=tax_map(Rg,ti,ta) if chi>0 else Rg
            mm=G.sum(axis=1);mt=T.sum(axis=1);mn=out.sum(axis=1);oldp=mm.sum(axis=0);newp=mn.sum(axis=0)
            tailold=top_fractions(oldp,.1);tailnew=top_fractions(newp,.1)
            g0=gini(oldp,sol.w);gt=gini(mt.sum(axis=0),sol.w);gn=gini(T0.sum(axis=(0,1)),sol.w);gout=gini(newp,sol.w)
            rec=dict(stage=stage,lam=lam,chi=chi,round=n+1,time=(n+1)/10,G_start=g0,G_trade=gt,G_fair_current_state=gn,G_end=gout,
              G_neutral_contribution=gn-g0,G_selection_extra=gt-gn,G_tax_exact=-(1-rho)*gt,G_tax_remap_error=gout-rho*gt,
              M_fixed_threshold=float(np.dot(mn[0]-mm[0],tailold)/.5),M_threshold_move=float(np.dot(mn[0],tailnew-tailold)/.5),M_change=float((np.dot(mn[0],tailnew)-np.dot(mm[0],tailold))/.5),
              E_fixed_threshold=float(-np.dot(mn[2]-mm[2],tailold)/.1),E_threshold_move=float(-np.dot(mn[2],tailnew-tailold)/.1),E_change=float((np.dot(mm[2],tailold)-np.dot(mn[2],tailnew))/.1),
              Q_start=float(np.sum(f*(sol.w-1)*sol.u[:,None])),Q_trade=float(np.sum(T.sum(axis=0)*(sol.w-1)*sol.u[:,None])),Q_end=float(np.sum(out.sum(axis=0)*(sol.w-1)*sol.u[:,None])))
            for k in range(5):
                m0=float(np.sum(f*(sol.w-1)*modes[k,:,None]));mT=float(np.sum(T.sum(axis=0)*(sol.w-1)*modes[k,:,None]));mN=float(np.sum(out.sum(axis=0)*(sol.w-1)*modes[k,:,None]));mKernel=float(rho*np.sum(T.sum(axis=0)*(sol.w-1)*(sol.R.T@modes[k])[:,None]))
                rec[f'mode{k+1}_start']=m0;rec[f'mode{k+1}_trade_injection']=mT-m0;rec[f'mode{k+1}_end']=mN;rec[f'mode{k+1}_scalar_residual']=mN-rho*gamma[k]*mT;rec[f'mode{k+1}_kernel_residual']=mN-mKernel
            br.append(rec);G=out
        else:G,_=sol.step(G,P,ti,ta,chi)
        a=measure(G,sol.w,sol.u)
        for key,val in [('max_mass_error',abs(a['mass']-1)),('max_wealth_error',abs(a['mean']-1)),('max_count_error',abs(a['count_identity']))]:audit[key]=max(audit[key],float(val))
        audit['min_mass']=min(audit['min_mass'],a['min_mass'])
        if a['min_mass'] < -1e-13 or abs(a['mean']-1)>1e-7:raise ArithmeticError('density failed')
        if (n+1)%100==0:print(label,stage,lam,chi,n+1,round(time.time()-tic,1),flush=True)
    audit['seconds']=time.time()-tic
    return G,rows,br,audit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dx',type=float,default=.01);ap.add_argument('--na',type=int,default=96);ap.add_argument('--budget',action='store_true');args=ap.parse_args()
    label=f'dx{args.dx:g}_na{args.na}';folder=ROOT/'results'/'convergence'/label;folder.mkdir(parents=True,exist_ok=True)
    settings=Settings(dx=args.dx,na=args.na,threads=1,poisson_max=18);sol=DensitySolver(settings);G0=make_initial(sol);allrows=[];allb=[];aud={}
    for lam in (0.,4.):
        end,rows,b,a=advance(sol,G0.copy(),'A',lam,0.,label,args.budget);allrows+=rows;allb+=b;aud[f'A{lam}']=a
        if lam==4.:A4=end
    B0,wn,info=relabel_t10(A4,sol.w);aud['relabel']=info;sol.set_grid(wn)
    for chi in (0.,.03):
        end,rows,b,a=advance(sol,B0.copy(),'B',4.,chi,label,args.budget);allrows+=rows;allb+=b;aud[f'B{chi}']=a
    pd.DataFrame(allrows).to_csv(folder/'metrics.csv',index=False)
    if allb:pd.DataFrame(allb).to_csv(folder/'budgets.csv',index=False)
    aud['settings']=__import__('dataclasses').asdict(settings);aud['kernel']=sol.kernel_info;dump_json(folder/'audit.json',aud);print('COMPLETE',label,flush=True)
if __name__=='__main__':main()
