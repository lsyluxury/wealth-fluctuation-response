"""Analytic Taylor-coefficient propagation (not fitted or finite-differenced).
Taylor convention: G(lambda)=G0+lambda G1+lambda^2 G2+... .
The finite-difference comparisons are independent checks, not the coefficient source.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import time
import numpy as np,pandas as pd
from numba import njit
from common import *

@njit(cache=True)
def signed_trade(G,H,w,nn,idx,coeff,tails):
    """Bilinear signed extension of the conservative trade remap.
    No positivity clipping: derivative densities legitimately change sign.
    """
    nc,na,nw=G.shape;out=np.zeros_like(G)
    for a in range(na):
        hmass=np.sum(H[0,a])+np.sum(H[1,a])
        for c in range(nc):out[c,a,0]=G[c,a,0]*hmass
        for sg in range(2):
            p=H[sg,a];pref=np.zeros(nw+1);suff=np.zeros(nw+1);pm=np.zeros(nw+1);sm=np.zeros(nw+1)
            for j in range(nw):pref[j+1]=pref[j]+p[j];pm[j+1]=pm[j]+p[j]*w[j]
            for j in range(nw-1,-1,-1):suff[j]=suff[j+1]+p[j];sm[j]=sm[j+1]+p[j]*w[j]
            for i in range(1,nw):
                absg=0.
                for c in range(nc):absg+=abs(G[c,a,i])
                if absg==0:continue
                for c in range(nc):out[c,a,i]+=G[c,a,i]*p[0]
                k=int(tails[sg,i,0]);alpha=tails[sg,i,1];mass=suff[i]
                # Same domain convention as base operator. Actual boundary mass
                # is separately zero/negligible in all evaluated conditions.
                alpha=min(alpha,1.)
                for c in range(nc):
                    val=G[c,a,i]*mass;out[c,a,k]+=val*(1-alpha);out[c,a,k+1]+=val*alpha
                ov=int(tails[sg,i,3])
                if ov<i:
                    v=pref[i]-pref[ov]
                    for c in range(nc):out[c,a,nw-1]+=G[c,a,i]*v
                for e in range(nn[sg,i]):
                    k=idx[sg,i,e,0];l=idx[sg,i,e,1];r=idx[sg,i,e,2]
                    # Choose the less cancellation-prone side by absolute size.
                    mass=pref[r]-pref[l] if abs(pref[r])+abs(pref[l])<abs(suff[l])+abs(suff[r]) else suff[l]-suff[r]
                    mom=pm[r]-pm[l] if abs(pm[r])+abs(pm[l])<abs(sm[l])+abs(sm[r]) else sm[l]-sm[r]
                    high=coeff[sg,i,e,0]*mass+coeff[sg,i,e,1]*mom;low=mass-high
                    for c in range(nc):out[c,a,k]+=G[c,a,i]*low;out[c,a,k+1]+=G[c,a,i]*high
    return out

def jet_step(sol,J):
    f=J.sum(axis=1);m=f.sum(axis=(1,2));mass0=m[0];P1=(sol.u[:,None]-sol.u[None,:])/4
    normalized=np.empty_like(f)
    normalized[0]=f[0]/mass0
    normalized[1]=(f[1]-m[1]*normalized[0])/mass0
    normalized[2]=(f[2]-m[1]*normalized[1]-m[2]*normalized[0])/mass0
    H=[]
    for k in range(3):
        neutral=np.broadcast_to(normalized[k].sum(axis=0)[None,:]/2,(sol.s.na,len(sol.w)))
        selection=P1@normalized[k-1] if k else np.zeros_like(neutral)
        H.append(np.stack([neutral+selection,neutral-selection]))
    out=np.zeros_like(J)
    for k in range(3):
        z=np.zeros_like(J[0])
        for j in range(k+1):z+=signed_trade(J[j],H[k-j],sol.w,*sol.plan)
        out[k]=np.matmul(sol.R,z)
    return out

def absdiff_potential(p,w):
    return w*(2*np.cumsum(p)-p.sum())+np.dot(p,w)-2*np.cumsum(p*w)

def ratio_coeff(n,d):
    out=np.zeros(3)
    for k in range(3):out[k]=(n[k]-sum(d[j]*out[k-j] for j in range(1,k+1)))/d[0]
    return out

def jet_metrics(J,w,u):
    m=J.sum(axis=2);p=m.sum(axis=1);mass=p.sum(axis=1);wealth=p@w
    den=np.array([mass[0]*wealth[0],mass[0]*wealth[1]+mass[1]*wealth[0],mass[0]*wealth[2]+mass[1]*wealth[1]+mass[2]*wealth[0]])
    num=np.array([.5*np.dot(p[0],absdiff_potential(p[0],w)),np.dot(p[1],absdiff_potential(p[0],w)),np.dot(p[2],absdiff_potential(p[0],w))+.5*np.dot(p[1],absdiff_potential(p[1],w))])
    g=ratio_coeff(num,den)
    k=int(np.searchsorted(np.cumsum(p[0]),.9*mass[0],side='left'))
    theta=ratio_coeff(.1*mass-p[:,k+1:].sum(axis=1),p[:,k]);ret={}
    for cohort,name in [(0,'M'),(1,'I'),(2,'retention')]:
        numerator=m[:,cohort,k+1:].sum(axis=1).copy()
        for order in range(3):numerator[order]+=sum(theta[j]*m[order-j,cohort,k] for j in range(order+1))
        coeff=ratio_coeff(numerator,m[:,cohort].sum(axis=1))
        if cohort==2:coeff=np.array([1.,0.,0.])-coeff;name='E'
        for order in range(3):ret[f'{name}{order}']=float(coeff[order])
    Q=np.sum(J.sum(axis=1)*(w[None,None,:]-1)*u[None,:,None],axis=(1,2))
    for order in range(3):ret[f'G{order}']=float(g[order]);ret[f'Q{order}']=float(Q[order]);ret[f'mass{order}']=float(mass[order]);ret[f'wealth{order}']=float(wealth[order])
    ret['cut_node']=k;ret['cut_fraction0']=float(theta[0]);return ret

def main():
    out=ROOT/'results'/'weak_response';out.mkdir(parents=True,exist_ok=True)
    sol=DensitySolver(Settings(dx=.01,na=96,threads=1,poisson_max=18));G0=make_initial(sol,factorize=True)
    # Direct signed extension vs base operator on positive inputs.
    P,ti,ta=sol.prepare(.25,0);f=G0.sum(axis=0);H=np.stack([P@f,(1-P)@f])/f.sum()
    positive,_=trade(G0,H,sol.w,*sol.plan);signed=signed_trade(G0,H,sol.w,*sol.plan)
    parity=[];audit={'positive_kernel_max_abs_difference':float(np.max(abs(positive-signed))),'initial_factorization':True,'lambda_coefficients_fitted':False};J=np.zeros((3,)+G0.shape);J[0]=G0;rows=[];tic=time.time()
    for n in range(301):
        if n in OBS:rows.append(dict(round=n,time=n/10,**jet_metrics(J,sol.w,sol.u)))
        if n==300:break
        J=jet_step(sol,J)
        if (n+1)%50==0:print('JET',n+1,round(time.time()-tic,1),flush=True)
    df=pd.DataFrame(rows);df.to_csv(out/'response_coefficients.csv',index=False)
    audit['jet_seconds']=time.time()-tic;audit['first_round_wealth_response_max']=float(df[df['round']==1][['G1','G2','M1','M2','E1','E2']].abs().to_numpy().max())
    audit['parity_errors']={c:float(df[c].abs().max()) for c in ['G1','M1','E1','Q0','Q2']}
    audit['max_derivative_mass_error']=float(df[['mass1','mass2']].abs().to_numpy().max());audit['max_derivative_wealth_error']=float(df[['wealth1','wealth2']].abs().to_numpy().max())
    full=[]
    for lam in (.1,.25,.5):
        G=G0.copy();P,ti,ta=sol.prepare(lam,0);tic=time.time()
        for n in range(301):
            if n in OBS:full.append(dict(lam=lam,round=n,time=n/10,**measure(G,sol.w,sol.u)))
            if n==300:break
            G,_=sol.step(G,P,ti,ta,0)
        print('FULL',lam,round(time.time()-tic,1),flush=True)
    fd=pd.DataFrame(full);fd.to_csv(out/'full_density_comparison.csv',index=False)
    comp=fd.merge(df,on=['round','time'],validate='many_to_one')
    for m in ['G','M','E']:
        comp[m+'_pred2']=comp[m+'0']+comp.lam*comp[m+'1']+comp.lam**2*comp[m+'2'];comp[m+'_residual']=comp[m]-comp[m+'_pred2']
    comp['Q_pred1']=comp.Q0+comp.lam*comp.Q1;comp['Q_residual']=comp.Q-comp.Q_pred1
    comp.to_csv(out/'prediction_residuals.csv',index=False)
    dump_json(out/'audit.json',audit);print('WEAK_COMPLETE',flush=True)
if __name__=='__main__':main()
