#!/usr/bin/env python3
"""Stage4 predictive tagged kinetic, retag cohort at 0,30,50 model-time units.
Each simultaneous tag family carries 1/3 population mass, so the total
focal density remains mass one; each family follows the SAME self-consistent
partner field. Retagging by top_fractions(p,q) splits each wealth node's mass
fractionally, retaining wealth and ability composition. NO ABM future used.
"""
from __future__ import annotations
import sys,os,time,math,argparse
from pathlib import Path
os.environ['NUMBA_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deps'))
import numpy as np,pandas as pd
from scipy.stats import poisson
from density_solver import DensitySolver,Settings,top_fractions
from common import make_initial
WAIT=(0,30,50);TAUS=(0,1,2,5,10,15,20,30)
Q=.10

def split_cutoff_node(G,w,quantile):
    """Resolve a partially occupied wealth node into two nearby wealth nodes.

    Applies the split to ALL existing tagged groups and abilities, so that
    each group's mass and first wealth moment are preserved. Quantile label
    selection is exact at the instant of repartition; the added second moment
    is audited. This mirrors the archived Stage-B boundary reconstruction.
    """
    p=G.sum(axis=(0,1));mass=p.sum();cum=np.cumsum(p)
    target=quantile*mass
    i=int(np.searchsorted(cum,target,side='left'))
    if i>=len(w):raise RuntimeError('cutoff outside wealth grid')
    before=cum[i]-p[i]
    frac=float((target-before)/p[i])
    if frac<=1e-9 or frac>=1-1e-9:
        return G,w,{'quantile':quantile,'split':False,'fraction':frac,'added_variance':0.}
    assert i>0 and i<len(w)-1 and w[i]>0
    width=min(w[i]-w[i-1],w[i+1]-w[i]);delta=0.4*width
    lo=w[i]-(1-frac)*delta;hi=w[i]+frac*delta
    assert w[i-1]<lo<w[i]<hi<w[i+1]
    ww=np.concatenate((w[:i],np.array([lo,hi]),w[i+1:]))
    out=np.zeros((G.shape[0],G.shape[1],len(ww)),dtype=np.float64)
    out[:,:,:i]=G[:,:,:i]
    out[:,:,i]=frac*G[:,:,i]
    out[:,:,i+1]=(1-frac)*G[:,:,i]
    out[:,:,i+2:]=G[:,:,i+1:]
    merr=float(out.sum()-G.sum())
    werr=float(np.sum(out*ww)-np.sum(G*w))
    variance=float(np.sum(out*ww*ww)-np.sum(G*w*w))
    if abs(merr)>1e-11 or abs(werr)>1e-11:raise AssertionError((merr,werr))
    return out,ww,{'quantile':quantile,'split':True,'fraction':frac,'added_variance':variance,
                    'wealth':float(w[i]),'delta':float(delta),'mass_error':merr,'wealth_error':werr}


def repartition_family(G,ww,family):
    # all branches have exact same partner distribution f=sum groups.
    p=G.sum(axis=(0,1))
    bot=1-top_fractions(p,.5)
    top=top_fractions(p,Q)
    mid=1-bot-top
    sub=G[3*family:3*family+3].sum(axis=0)
    for j,mask in enumerate((bot,mid,top)):
        G[3*family+j]=sub*mask[None,:]


def observables(G,ww,family):
    p=G.sum(axis=(0,1));tail=top_fractions(p,Q)
    if family not in (0,1,2):raise ValueError
    F=G[3*family:3*family+3]
    m=F.sum(axis=1)
    mt=m.sum(axis=1)
    M=float((m[0]*tail).sum()/mt[0]);E=float(1-(m[2]*tail).sum()/mt[2]);
    C=float((1-E-Q)/(1-Q))
    cp=np.cumsum(p)-p;cm=np.cumsum(p*ww)-p*ww
    g=float(np.sum(p*(ww*cp-cm)) / (p.sum()*np.sum(p*ww)))
    return dict(Cq=C,stay=1-E,E=E,M_B50_to_Tq=M,G=g)


def simulate(seed=2026101201,lam=4.,mode='baseline',dx=.05,na=24,rounds=800):
    if mode in ('baseline','strong','fixed','redraw'):
        h=.9 if mode=='strong' else .3;mu=0. if mode=='fixed' else 1.
    else:raise ValueError(mode)
    cfg=Settings(dx=dx,na=na,h=h,mu=mu,r=10.,threads=1,win_quadrature=6,poisson_max=12)
    tm=time.time();sol=DensitySolver(cfg)
    if mode=='redraw':sol.R[:]=1./na
    G0=make_initial(sol,seed=seed,factorize=True)
    # Tag family0 at t0, other families unassigned until their waiting times.
    nc=3*len(WAIT);G=np.zeros((nc,na,len(sol.w)))
    G[:3]=G0/len(WAIT)
    for fam in range(1,len(WAIT)):G[3*fam]=G0.sum(axis=0)/len(WAIT)
    assert abs(G.sum()-1)<1e-11
    P,idx,al=sol.prepare(lam,0.)
    waits=[t*10 for t in WAIT]
    taurs=[t*10 for t in TAUS]
    checkpoints=sorted(set([tw+tau for tw in waits for tau in taurs]+waits))
    rows=[];split_log=[];masserr=0.;wealtherr=0.;tagerr=0.;mindens=1.;prev=0
    for rr in checkpoints:
        if rr>prev:
            for r in range(rr-prev):
                G,_=sol.step(G,P,idx,al,0.)
                masserr=max(masserr,abs(G.sum()-1))
                wealtherr=max(wealtherr,abs(np.sum(G*sol.w)-1))
                mindens=min(mindens,float(G.min()))
        prev=rr
        if rr in waits[1:]:
            # Preserve each prior tag's joint density; split cutoff nodes globally.
            for qq in (.5,.9):
                G,ww,inf=split_cutoff_node(G,sol.w,qq)
                if inf['split']:sol.set_grid(ww)
                split_log.append({'at_round':rr,**inf})
            repartition_family(G,sol.w,waits.index(rr))
            tagerr=max(tagerr,float(max(abs(G[3*waits.index(rr)+j].sum()-v/len(WAIT))
                      for j,v in enumerate((.5,.4,.1)))))
        for fam,tw in enumerate(waits):
            if rr>=tw and rr-tw in taurs:
                z=observables(G,sol.w,fam)
                rows.append(dict(seed=seed,lambda_value=lam,mode=mode,N='kinetic',tw=tw/10,
                         tau=(rr-tw)/10,time=rr/10,q=Q,dx=dx,na=na,**z))
        if rr in (0,300,500,800):print('KINETIC',mode,lam,'round',rr,'sec',round(time.time()-tm,1),flush=True)
    info=dict(mode=mode,lam=lam,seed=seed,dx=dx,na=na,rounds=rounds,
         mass_error=masserr,mean_wealth_error=wealtherr,tag_error=tagerr,min_density=mindens,
         seconds=time.time()-tm,quantile_splits=split_log)
    return rows,info

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',default='baseline');ap.add_argument('--lambda-value',type=float,default=4.);ap.add_argument('--seed',type=int,default=2026101201);ap.add_argument('--dx',type=float,default=.05);ap.add_argument('--na',type=int,default=24);ap.add_argument('--out',default=str(ROOT/'results'/'kinetic_pilot.csv'));args=ap.parse_args()
    import json
    rows,info=simulate(args.seed,args.lambda_value,args.mode,args.dx,args.na)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    out.with_suffix('.json').write_text(json.dumps(info,indent=2,ensure_ascii=False))
    print('DONE',info,flush=True)
