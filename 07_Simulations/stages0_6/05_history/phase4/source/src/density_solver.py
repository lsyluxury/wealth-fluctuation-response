"""Independent, deterministic, finite-stake density solver.

No agent simulation, random pairing, Gaussian draws, or ABM import is used.
G[c,a,i] is population probability mass of fixed cohort c in ability bin a
and wealth node i. Sum_c G is the endogenous partner distribution.
"""
from __future__ import annotations
import math, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from scipy.special import ndtr, ndtri, expit
from scipy.stats import poisson
from numpy.polynomial.legendre import leggauss
from numba import njit, prange, set_num_threads

@dataclass
class Settings:
    dx: float = 0.02
    na: int = 48
    xmin: float = -55.0
    xmax: float = 8.0
    epsilon: float = .14
    h: float = .3
    mu: float = 1.0
    r: float = 10.0
    ability_quadrature: int = 32
    win_quadrature: int = 6
    poisson_max: int = 10
    threads: int = 4


def wealth_grid(s: Settings):
    # Node at zero permits conservative interpolation without negative wealth.
    n = int(np.ceil((s.xmax-s.xmin)/s.dx))
    return np.r_[0.,np.exp(s.xmin+np.arange(n+1)*s.dx)]


def probability_kernel(na, lam, q=6):
    z,wt=leggauss(q); wt=wt/2
    centers=-1+(np.arange(na)+.5)*2/na
    uu=centers[:,None]+z[None,:]/na
    P=np.zeros((na,na))
    for i in range(q):
        for j in range(q):
            P+=wt[i]*wt[j]*expit(lam*(uu[:,i,None]-uu[None,:,j]))
    # Enforce exact mathematical reciprocity up to floating-point roundoff.
    symerr=np.max(np.abs(P+P.T-1))
    P=.5*(P+1-P.T)
    return P,{'p_reciprocity_before':float(symerr)}


def ability_kernel(s: Settings):
    na=s.na;nu=s.mu/s.r
    if nu==0 or s.h==1:
        return np.eye(na),dict(poisson_tail=0.,ability_column_error=0.,ability_stationary_error=0.,ability_min_entry=0.,identity=True)
    ue=np.linspace(0,1,na+1); ae=ndtri(ue)
    z,wt=leggauss(s.ability_quadrature)
    # Integrate in latent Gaussian a, not in inverse-CDF u at +/-1.
    # Finite endpoints +/-10 omit <1.6e-23 stationary Gaussian mass.
    finite_edges=np.clip(ae,-10.,10.)
    half=(finite_edges[1:]-finite_edges[:-1])/2
    a=(finite_edges[1:,None]+finite_edges[:-1,None])/2+half[:,None]*z
    aw=half[:,None]*wt*na*np.exp(-a*a/2)/np.sqrt(2*np.pi)
    R=np.exp(-nu)*np.eye(na)
    max_corr=0.;max_sym=0.
    for k in range(1,s.poisson_max+1):
        pk=float(poisson.pmf(k,nu));corr=s.h**k;sd=np.sqrt(1-corr*corr)
        # K[dest,source] integrated over uniform stationary u in source bin.
        K=np.zeros((na,na))
        for j in range(na):
            cc=ndtr((ae[:,None]-corr*a[j][None,:])/sd)
            K[:,j]=(np.diff(cc,axis=0)*aw[j]).sum(axis=1)
        max_sym=max(max_sym,float(np.max(abs(K-K.T))))
        K=.5*(K+K.T)
        corrdiag=1-K.sum(axis=0)
        max_corr=max(max_corr,float(np.max(abs(corrdiag))))
        # Tiny, explicitly logged quadrature correction, not evolution rescaling.
        K[np.diag_indices(na)]+=corrdiag
        if np.min(K)<0:raise ArithmeticError('negative ability transition')
        R+=pk*K
    # Omitted probability is logged; here <1e-18 for nu=.1,max=10.
    tail=float(poisson.sf(s.poisson_max,nu))
    return R,dict(poisson_tail=tail,ability_quadrature=s.ability_quadrature,latent_gaussian_truncation=2*float(ndtr(-10.)),
        ability_max_symmetrization=max_sym,ability_max_diagonal_correction=max_corr,
        ability_column_error=float(np.max(abs(R.sum(axis=0)-1))),
        ability_stationary_error=float(np.max(abs(R.sum(axis=1)-1))),
        ability_min_entry=float(R.min()))


def make_trade_plan(w,eps):
    """Group poorer partners by their common destination interpolation interval.
    Since the map w -> w +/- eps*v is linear in v in each such interval,
    its sum is evaluated EXACTLY from partner mass and first-moment prefixes.
    This is not a small-stake/Fokker-Planck approximation.
    """
    nw=len(w); allplans=[[],[]];tails=np.zeros((2,nw,4))
    if eps==0:
        return np.zeros((2,nw),np.int32),np.zeros((2,nw,1,3),np.int32),np.zeros((2,nw,1,2)),tails
    maxn=1
    for sg,sign in enumerate((1.,-1.)):
        for i,wi in enumerate(w):
            if i==0:allplans[sg].append([]);continue
            dest=wi*(1+sign*eps)
            j=int(np.searchsorted(w,dest,side='right')-1)
            if j>=nw-1:
                j=nw-2;alpha=(dest-w[j])/(w[j+1]-w[j])
            else:alpha=(dest-w[j])/(w[j+1]-w[j])
            overstart=max(1,min(i,int(np.searchsorted(w,(w[-1]-wi)/eps,side='left')))) if sign>0 and dest>w[-1] else i
            tails[sg,i]=[j,alpha,max(0.,dest-w[-1]),overstart]
            entries=[]
            lo=max(0,min(i,j));hi=min(nw-2,max(i,j))
            for k in range(lo,hi+1):
                if sign>0:
                    l=np.searchsorted(w,(w[k]-wi)/eps,side='left')
                    r=np.searchsorted(w,(w[k+1]-wi)/eps,side='left')
                else:
                    l=np.searchsorted(w,(wi-w[k+1])/eps,side='right')
                    r=np.searchsorted(w,(wi-w[k])/eps,side='right')
                l=max(1,min(int(l),i));r=max(1,min(int(r),i))
                if r>l:
                    gap=w[k+1]-w[k]
                    entries.append((k,l,r,(wi-w[k])/gap,sign*eps/gap))
            allplans[sg].append(entries);maxn=max(maxn,len(entries))
    nn=np.zeros((2,nw),np.int32);idx=np.zeros((2,nw,maxn,3),np.int32)
    coeff=np.zeros((2,nw,maxn,2))
    for sg in range(2):
        for i,entries in enumerate(allplans[sg]):
            nn[sg,i]=len(entries)
            for k,e in enumerate(entries):idx[sg,i,k]=e[:3];coeff[sg,i,k]=e[3:]
    return nn,idx,coeff,tails


@njit(cache=True,inline='always')
def interval_sum(pref,suff,l,r):
    if pref[r]<=suff[l]:return pref[r]-pref[l]
    return suff[l]-suff[r]


@njit(cache=True,parallel=True)
def trade(G,H,w,nn,idx,coeff,tails):
    nc,na,nw=G.shape
    out=np.zeros_like(G);stats=np.zeros((na,4))
    for a in prange(na):
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


@njit(cache=True,parallel=True)
def tax_map(G,inds,alphas):
    nc,na,nw=G.shape;out=np.zeros_like(G)
    for a in prange(na):
        for c in range(nc):
            for i in range(nw):
                k=inds[i];alpha=alphas[i];v=G[c,a,i]
                out[c,a,k]+=v*(1-alpha);out[c,a,k+1]+=v*alpha
    return out


def projected_initial(paths,w,na):
    G=np.zeros((3,na,len(w)));N=0;raw_ginis=[];source=[]
    means=[]
    for path in paths:
        z=np.load(path,allow_pickle=False)
        ww=z['w'];u=z['u'];tags=z['cohort_code'];n=len(ww);N+=n
        ai=np.minimum(na-1,np.maximum(0,((u+1)*na/2).astype(int)))
        wi=np.searchsorted(w,ww,side='right')-1
        if wi.min()<0 or wi.max()>=len(w)-1:raise ValueError('initial wealth out of grid')
        al=(ww-w[wi])/(w[wi+1]-w[wi])
        np.add.at(G,(tags,ai,wi),1-al);np.add.at(G,(tags,ai,wi+1),al)
        sy=np.sort(ww);rg=2*np.dot(np.arange(1,n+1),sy)/(n*sy.sum())-(n+1)/n
        raw_ginis.append(float(rg));means.append(float(ww.mean()))
        source.append(dict(path=Path(path).name,N=n,mean=float(ww.mean()),gini=float(rg)))
    G/=N
    return G,dict(number_of_anchors=len(paths),total_agents=N,sources=source,
       source_mean=float(np.mean(means)),source_mean_gini=float(np.mean(raw_ginis)))


def top_fractions(p,top):
    # Fractions of nodes admitted to a population tail. No ABM cutoffs.
    above=np.cumsum(p[::-1])[::-1]-p
    return np.divide(np.clip(top*p.sum()-above,0,p),p,out=np.zeros_like(p),where=p>0)


def relabel(G):
    f=G.sum(axis=0);p=f.sum(axis=0)
    b=1-top_fractions(p,.5);t=top_fractions(p,.01);m=1-b-t
    return np.stack([f*b,f*m,f*t])


def observables(G,w,u):
    marg=G.sum(axis=1);p=marg.sum(axis=0);mass=p.sum();mean=np.dot(p,w)/mass
    cm=np.cumsum(p*w)-p*w;cp=np.cumsum(p)-p
    gini=np.sum(p*(w*cp-cm))/(mass*mass*mean)
    t10=top_fractions(p,.1);t1=top_fractions(p,.01)
    p50=1-top_fractions(p,.5)
    cmass=G.sum(axis=(1,2))
    def quantile(q):
        # Continuous quantile from linearly interpolated CDF at geometric nodes.
        c=np.cumsum(p)/mass
        return float(np.interp(q,c,w))
    Q=float(np.sum(G.sum(axis=0)*(w[None,:]-1)*u[:,None]))/mass
    return dict(gini=float(gini),B50_T10=float(np.dot(marg[0],t10)/cmass[0]),
        Etop1=float(1-np.dot(marg[2],t1)/cmass[2]),Q=Q,
        mean_wealth=float(mean),probability_mass=float(mass),variance=float(np.dot(p,(w-mean)**2)/mass),
        B50_mass=float(cmass[0]),Middle49_mass=float(cmass[1]),T1_mass=float(cmass[2]),
        bottom50_share=float(np.dot(p*w,p50)/(mean*mass)),top1_share=float(np.dot(p*w,t1)/(mean*mass)),
        top10_share=float(np.dot(p*w,t10)/(mean*mass)),
        B50_mean_wt=float(np.dot(marg[0],w)/cmass[0]),T1_mean_wt=float(np.dot(marg[2],w)/cmass[2]),
        threshold_T10=quantile(.9),threshold_T1=quantile(.99),threshold_B50=quantile(.5),
        min_density=float(G.min()),low_boundary_mass=float(p[:3].sum()),
        high_boundary_mass=float(p[-3:].sum()),high_boundary_wealth=float(np.dot(p[-3:],w[-3:])))


class DensitySolver:
    def __init__(self,s:Settings):
        self.s=s;set_num_threads(s.threads)
        self.w=wealth_grid(s);self.u=-1+(np.arange(s.na)+.5)*2/s.na
        t=time.time();self.last_budget={};self.plan=make_trade_plan(self.w,s.epsilon)
        self.R,self.kernel_info=ability_kernel(s)
        self.kernel_info['setup_seconds']=time.time()-t
        self.win={}
    def set_grid(self,w):
        self.w=np.asarray(w);self.plan=make_trade_plan(self.w,self.s.epsilon)
    def prepare(self,lam,chi):
        if lam not in self.win:self.win[lam]=probability_kernel(self.s.na,lam,self.s.win_quadrature)[0]
        rho=np.exp(-chi/self.s.r);dest=rho*self.w+(1-rho)
        idx=np.clip(np.searchsorted(self.w,dest,side='right')-1,0,len(self.w)-2)
        al=(dest-self.w[idx])/(self.w[idx+1]-self.w[idx])
        return self.win[lam],idx,al
    def step(self,G,P,taxidx,taxal,chi):
        f=G.sum(axis=0);mass=f.sum()
        # Normalization is intrinsic to partner selection. We never rescale G.
        H=np.stack([P@f,(1-P)@f])/mass
        if self.s.epsilon==0:Tg,stats=G.copy(),np.zeros(4)
        else:Tg,stats=trade(G,H,self.w,*self.plan)
        Rg=np.matmul(self.R,Tg)
        if chi>0:out=tax_map(Rg,taxidx,taxal)
        else:out=Rg
        # Exact finite-node physical jump moments, distinct from interpolation noise.
        w=self.w
        D=H[0]-H[1]
        lo1=np.cumsum(D*w[None,:],axis=1)-D*w[None,:]
        hi0=np.cumsum(D[:,::-1],axis=1)[:,::-1]
        drift=self.s.epsilon*(lo1+w[None,:]*hi0)
        p=f.sum(axis=0)/mass
        lo2=np.cumsum(p*w*w)-p*w*w
        hi=np.cumsum(p[::-1])[::-1]
        jump2=self.s.epsilon**2*(lo2+w*w*hi)
        qtrade=float(np.sum(f*self.u[:,None]*drift))
        vtrade=float(np.sum(f*(2*(w[None,:]-1)*drift+jump2[None,:])))
        vin=float(np.sum(G*(w-1)**2));vt=float(np.sum(Tg*(w-1)**2));vr=float(np.sum(Rg*(w-1)**2))
        vout=float(np.sum(out*(w-1)**2));rho=np.exp(-chi/self.s.r)
        qin=float(np.sum(f*self.u[:,None]*(w-1)))
        qt=float(np.sum(Tg.sum(axis=0)*self.u[:,None]*(w-1)))
        self.last_budget=dict(physical_trade_dV=vtrade,wealth_remap_added_dV=vt-vin-vtrade,
          physical_trade_dQ=qtrade,Q_trade_discretization_residual=qt-qin-qtrade,
          renewal_wealth_variance_residual=vr-vt,tax_remap_added_dV=vout-rho*rho*vr,
          total_partner_normalization=mass,ability_mean=float(np.sum(out*self.u[None,:,None])),
          cohort_B50_mass=float(out[0].sum()),cohort_Middle49_mass=float(out[1].sum()),cohort_T1_mass=float(out[2].sum()))
        return out,stats


def cohort_aligned_grid(w,paths):
    """Insert bounding observed wealth values at initial group boundaries.
    Uses ONLY intervention-time inputs, never post-intervention ABM values.
    Best used separately for each seed, not for pooled cohort inference.
    """
    values=[]
    for path in paths:
        z=np.load(path);a=np.sort(z['w']);n=len(a)
        for cut in (n//2,n-n//100):values.extend([a[cut-1],a[cut]])
    return np.unique(np.r_[w,values])


def relabel_resolved(G,w):
    """Resolve theoretical percentile cuts within a boundary node.
    Reconstruct only a cut node as uniform on a symmetric local interval;
    store means of its two pieces at distinct nodes. This preserves its mass,
    mean wealth and mean ability-wealth product. No ABM endpoint is read.
    Reconstruction perturbations are reported as discretization error.
    """
    f=G.sum(axis=0);p=f.sum(axis=0);mass=p.sum();cuts=[.5*mass,.99*mass]
    columns=[];dest=[];tags=[];before=0.;newvalues=[];split_records=[]
    for i in range(len(w)):
        pi=p[i]
        if pi==0:continue
        fractions=[0.]+[(x-before)/pi for x in cuts if before+1e-16 < x < before+pi-1e-16]+[1.]
        width=min(w[i]-w[i-1],w[min(i+1,len(w)-1)]-w[i]) if i>0 else 0.
        for low,high in zip(fractions[:-1],fractions[1:]):
            mid=before+.5*(low+high)*pi
            c=0 if mid<cuts[0] else (1 if mid<cuts[1] else 2)
            target=w[i]+.5*(low+high-1)*width if len(fractions)>2 else w[i]
            columns.append(f[:,i]*(high-low));dest.append(target);tags.append(c);newvalues.append(target)
        if len(fractions)>2:split_records.append(dict(node=i,wealth=float(w[i]),mass=float(pi),fractions=fractions,width=float(width)))
        before+=pi
    wn=np.unique(np.r_[w,newvalues]);out=np.zeros((3,G.shape[1],len(wn)))
    inds=np.searchsorted(wn,dest)
    for c,j,v in zip(tags,inds,columns):out[c,:,j]+=v
    oldmean=np.sum(G*w);newmean=np.sum(out*wn)
    info=dict(split_nodes=split_records,mass_change=float(out.sum()-G.sum()),mean_change=float(newmean-oldmean),
        added_variance=float(np.sum(out*wn**2)-np.sum(G*w**2)),old_nodes=len(w),new_nodes=len(wn))
    return out,wn,info
