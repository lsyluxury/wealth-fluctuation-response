"""Accelerated, numerically audited original finite-stake operator.
No fitted coefficients or post-intervention ABM data. Small nodes below a
logged 1e-18 threshold are retained exactly once per round; smooth renewal
operator low-rank error is separately measured. Input solver is unmodified.
"""
import numpy as np
from numba import njit
from density_solver import DensitySolver, Settings, tax_map, interval_sum
@njit(cache=True)
def trade_fast(G,H,w,nn,idx,coeff,tails,cutoff):
    nc,na,nw=G.shape
    out=np.zeros_like(G);stats=np.zeros((na,4))
    for a in range(na):
        for c in range(nc):out[c,a,0]=G[c,a,0]
        for sg in range(2):
            p=H[sg,a]
            pref=np.zeros(nw+1);suff=np.zeros(nw+1)
            pm=np.zeros(nw+1);sm=np.zeros(nw+1)
            for j in range(nw):
                pref[j+1]=pref[j]+p[j];pm[j+1]=pm[j]+p[j]*w[j]
            for j in range(nw-1,-1,-1):
                suff[j]=suff[j+1]+p[j];sm[j]=sm[j+1]+p[j]*w[j]
            for i in range(1,nw):
                totalg=0.
                for c in range(nc):totalg+=G[c,a,i]
                if totalg==0.:continue
                if totalg<cutoff:
                    # Half in each sign loop: retain once, NOT twice per round.
                    stats[a,1]+=.5*totalg
                    for c in range(nc):out[c,a,i]+=.5*G[c,a,i]
                    continue
                for c in range(nc):out[c,a,i]+=G[c,a,i]*p[0]
                # Richer/equal partners yield the single full multiplicative move.
                k=int(tails[sg,i,0]);alpha=tails[sg,i,1];mass=suff[i]
                if alpha>1.:
                    stats[a,2]+=totalg*mass
                    stats[a,3]+=totalg*mass*tails[sg,i,2]
                    alpha=1.
                for c in range(nc):
                    val=G[c,a,i]*mass
                    out[c,a,k]+=val*(1-alpha);out[c,a,k+1]+=val*alpha
                ov=int(tails[sg,i,3])
                if ov<i:
                    omass=interval_sum(pref,suff,ov,i)
                    omom=interval_sum(pm,sm,ov,i)
                    stats[a,2]+=totalg*omass
                    # Recover epsilon from the multiplicative tail destination.
                    eps_here=((w[k]+tails[sg,i,1]*(w[k+1]-w[k]))/w[i]-1.)
                    stats[a,3]+=totalg*((w[i]-w[-1])*omass+eps_here*omom)
                    for c in range(nc):out[c,a,nw-1]+=G[c,a,i]*omass
                for e in range(nn[sg,i]):
                    k=idx[sg,i,e,0];l=idx[sg,i,e,1];r=idx[sg,i,e,2]
                    mass=interval_sum(pref,suff,l,r)
                    mom=interval_sum(pm,sm,l,r)
                    high=coeff[sg,i,e,0]*mass+coeff[sg,i,e,1]*mom
                    # Only roundoff stabilization; record actual mass affected.
                    if high<0.:
                        stats[a,0]+=totalg*(-high);high=0.
                    if high>mass:
                        stats[a,0]+=totalg*(high-mass);high=mass
                    low=mass-high
                    for c in range(nc):
                        v=G[c,a,i]
                        out[c,a,k]+=v*low;out[c,a,k+1]+=v*high
    return out,stats.sum(axis=0)


class FastSolver(DensitySolver):
    def __init__(self,s:Settings,cutoff:float=1e-18):
        super().__init__(s)
        self.cutoff=cutoff;self.keep=np.exp(-s.mu/s.r)
        A=self.R-self.keep*np.eye(s.na)
        U,sig,Vt=np.linalg.svd(A,full_matrices=False)
        k=max(1,int(np.count_nonzero(sig>1e-15)))
        self.Rleft=U[:,:k]*sig[:k];self.Rright=Vt[:k,:]
        self.kernel_info['renewal_lowrank']=k
        self.kernel_info['renewal_reconstruction_max_error']=float(np.max(abs(A-self.Rleft@self.Rright)))
        self.kernel_info['cutoff']=cutoff
    def step_fast(self,G,P,idx,al,chi):
        f=G.sum(axis=0);mass=f.sum()
        support=np.flatnonzero(np.any(f!=0,axis=0));lo,hi=int(support[0]),int(support[-1])+1
        fs=f[:,lo:hi];Hp=P@fs
        H=np.zeros((2,self.s.na,len(self.w)))
        H[0,:,lo:hi]=Hp/mass
        H[1,:,lo:hi]=(fs.sum(axis=0)[None,:]-Hp)/mass
        Tg,stats=trade_fast(G,H,self.w,*self.plan,self.cutoff)
        support=np.flatnonzero(np.any(Tg!=0,axis=(0,1)));lo,hi=int(support[0]),int(support[-1])+1
        packed=np.concatenate([Tg[c,:,lo:hi] for c in range(G.shape[0])],axis=1)
        updated=self.keep*packed+self.Rleft@(self.Rright@packed)
        Rg=np.zeros_like(G);length=hi-lo
        for c in range(G.shape[0]):Rg[c,:,lo:hi]=updated[:,c*length:(c+1)*length]
        return (tax_map(Rg,idx,al) if chi>0 else Rg),stats
