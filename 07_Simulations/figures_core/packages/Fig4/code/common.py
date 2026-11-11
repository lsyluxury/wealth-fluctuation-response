"""Stage-four cohorts and diagnostics; development inputs only."""
from pathlib import Path
import json, math
import numpy as np
from scipy.special import eval_hermitenorm, ndtri
from scipy.stats import poisson
from density_solver import DensitySolver, Settings, top_fractions, trade, tax_map
ROOT=Path(__file__).resolve().parents[1]
OBS=[0,1,2,3,5,10,20,30,50,100,150,200,250,300]
CUTS=np.array([.5,.9]);ALPHAS=np.array([.5,.4,.1])
def make_initial(sol,seed=2026101201,factorize=False):
    z=np.load(ROOT/'inputs'/f's{seed}_g65_selection.npz',allow_pickle=False)
    ww=z['w'];u=z['u'];rk=z['rank0'];n=len(ww);sw=np.sort(ww);extras=[]
    for q in CUTS:
        k=int(n*q);extras.extend([sw[k-1],sw[k]])
    sol.set_grid(np.unique(np.r_[sol.w,extras]));tag=np.searchsorted((CUTS*n).astype(int),rk,side='right')
    ai=np.clip(((u+1)*sol.s.na/2).astype(int),0,sol.s.na-1);ix=np.searchsorted(sol.w,ww,side='right')-1
    al=(ww-sol.w[ix])/(sol.w[ix+1]-sol.w[ix]);G=np.zeros((3,sol.s.na,len(sol.w)))
    np.add.at(G,(tag,ai,ix),(1-al)/n);np.add.at(G,(tag,ai,ix+1),al/n)
    if factorize:G=np.repeat((G.sum(axis=1)/sol.s.na)[:,None,:],sol.s.na,axis=1)
    assert np.max(abs(G.sum(axis=(1,2))-ALPHAS))<1e-12
    return G

def relabel_t10(G,w):
    f=G.sum(axis=0);p=f.sum(axis=0);cuts=CUTS*p.sum();cols=[];vals=[];tags=[];before=0.;splits=[]
    for i,pi in enumerate(p):
        if pi==0:continue
        fs=[0.]+[(c-before)/pi for c in cuts if before+1e-16<c<before+pi-1e-16]+[1.]
        width=min(w[i]-w[i-1],w[min(i+1,len(w)-1)]-w[i]) if i>0 else 0.
        if len(fs)>2:splits.append(dict(wealth=float(w[i]),mass=float(pi),fractions=fs))
        for lo,hi in zip(fs[:-1],fs[1:]):
            cat=int(np.searchsorted(cuts,before+(lo+hi)*pi/2,side='right'))
            dest=w[i]+.5*(lo+hi-1)*width if len(fs)>2 else w[i]
            cols.append(f[:,i]*(hi-lo));vals.append(dest);tags.append(cat)
        before+=pi
    wn=np.unique(np.r_[w,vals]);out=np.zeros((3,G.shape[1],len(wn)))
    for c,i,v in zip(tags,np.searchsorted(wn,vals),cols):out[c,:,i]+=v
    info=dict(mass_error=float(out.sum()-G.sum()),wealth_error=float(np.sum(out*wn)-np.sum(G*w)),added_variance=float(np.sum(out*wn**2)-np.sum(G*w**2)),splits=splits)
    assert np.max(abs(out.sum(axis=(1,2))-ALPHAS))<1e-10
    return out,wn,info

def gini(p,w):
    cp=np.cumsum(p)-p;cm=np.cumsum(p*w)-p*w
    return float(np.sum(p*(w*cp-cm))/(p.sum()*np.dot(p,w)))

def measure(G,w,u):
    m=G.sum(axis=1);p=m.sum(axis=0);a=m.sum(axis=1);tail=top_fractions(p,.1)
    M=float(np.dot(m[0],tail)/a[0]);I=float(np.dot(m[1],tail)/a[1]);E=float(1-np.dot(m[2],tail)/a[2])
    return dict(G=gini(p,w),M=M,E=E,I=I,Q=float(np.sum(G.sum(axis=0)*(w-1)*u[:,None])),mass=float(p.sum()),mean=float(np.dot(p,w)),min_mass=float(G.min()),count_identity=float(E-5*M-4*I),B_share_of_newcomers=float(5*M/E) if E>1e-12 else np.nan)

def corr_u(mu,h,r=10):
    k=np.arange(100);return float(np.dot(poisson.pmf(k,mu/r),6/np.pi*np.arcsin(h**k/2)))

def mode_vectors(na,K=5):
    edges=ndtri(np.linspace(0,1,na+1));finite=np.where(np.isfinite(edges),edges,0.)
    phi=np.exp(-finite**2/2)/np.sqrt(2*np.pi);phi[[0,-1]]=0;out=[]
    for k in range(1,K+1):
        v=phi*eval_hermitenorm(k-1,finite);out.append(na*(v[:-1]-v[1:])/np.sqrt(math.factorial(k)))
    return np.array(out)

def dump_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
