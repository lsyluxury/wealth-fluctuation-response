"""Additional DEVELOPMENT mechanism experiments at N=100000.
Paired transaction randomness is separated from capability-update randomness.
24 full-redraw futures = six preparation roots x four future streams; not 24 roots.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import time,math,json
import numpy as np,pandas as pd
from numba import njit
from scipy.stats import t
from common import ROOT,OBS,corr_u,dump_json

@njit(cache=True)
def reset_all(a,u,seed):
    np.random.seed(seed)
    for i in range(len(a)):
        a[i]=np.random.normal();u[i]=math.erf(a[i]/math.sqrt(2.))

@njit(cache=True)
def chunk(w,a,u,perm,lams,n0,n1,trade_seed,renew_seed,mu,h,fullreset):
    n,nb=w.shape
    for rr in range(n0+1,n1+1):
        np.random.seed(trade_seed+rr);np.random.shuffle(perm)
        for j in range(0,n,2):
            i=perm[j];k=perm[j+1];draw=np.random.random();du=u[i]-u[k]
            for b in range(nb):
                wi=w[i,b];wk=w[k,b];stake=.14*min(wi,wk);p=1/(1+math.exp(-lams[b]*du));s=1. if draw<p else -1.
                w[i,b]=wi+s*stake;w[k,b]=wk-s*stake
        np.random.seed(renew_seed+rr)
        if fullreset:
            for i in range(n):
                a[i]=np.random.normal();u[i]=math.erf(a[i]/math.sqrt(2.))
        else:
            nev=np.random.poisson(n*mu/10);innov=math.sqrt(1-h*h)
            for e in range(nev):
                i=np.random.randint(n);a[i]=h*a[i]+innov*np.random.normal();u[i]=math.erf(a[i]/math.sqrt(2.))

def metric(w,u,tags):
    n=len(w);ix=np.argsort(w);rank=np.empty(n,np.int32);rank[ix]=np.arange(n);ws=w[ix]
    M=float(np.mean(rank[tags==0]>=.9*n));I=float(np.mean(rank[tags==1]>=.9*n));E=float(np.mean(rank[tags==2]<.9*n))
    return dict(G=float(2*np.dot(np.arange(1,n+1),ws)/(n*ws.sum())-(n+1)/n),M=M,E=E,I=I,Q=float(np.mean((w-1)*u)),mean=float(w.mean()),minimum=float(w.min()),count_error=E-5*M-4*I)

def one(seed,rep,mode,mu,h,fullreset):
    z=np.load(ROOT/'inputs'/f's{seed}_g65_selection.npz');a=z['a'].copy();u=z['u'].copy();tags=np.searchsorted([50000,90000],z['rank0'],side='right');lam=np.array([0.,4.,16.]) if fullreset else np.array([0.,4.])
    w=np.repeat(z['w'][:,None],len(lam),axis=1);perm=np.arange(len(a));rseed=91000000+(seed-2026101200)*10000+rep*500;aseed=92000000+(seed-2026101200)*10000+rep*500
    if fullreset:reset_all(a,u,aseed)
    rows=[];prev=0
    for n in OBS:
        if n>prev:chunk(w,a,u,perm,lam,prev,n,rseed,aseed,mu,h,fullreset)
        prev=n
        for b,l in enumerate(lam):rows.append(dict(root=seed,future_rep=rep,mode=mode,mu=mu,h=h,full_redraw=fullreset,lag_score_corr=0. if fullreset else corr_u(mu,h),lam=l,round=n,time=n/10,**metric(w[:,b],u,tags)))
    return rows,w[:,0].copy()

def main():
    out=ROOT/'results'/'abm_mechanisms';out.mkdir(parents=True,exist_ok=True);tic=time.time();rows=[];neutral_invariance=[]
    for seed in range(2026101201,2026101207):
        for rep in range(1,5):
            rr,_=one(seed,rep,'full_redraw',1.,.3,True);rows+=rr
        print('REDRAW root',seed,'seconds',round(time.time()-tic,1),flush=True)
    pd.DataFrame(rows).to_csv(out/'full_redraw_raw.csv',index=False)
    persistence=[];modes=[('base',1.,.3),('sparse_h0',1.,0.),('persistent_h09',1.,.9),('frozen_mu0',0.,.3),('fast_mu5',5.,.3)]
    for seed in range(2026101201,2026101207):
        reference=None
        for name,mu,h in modes:
            rr,end=one(seed,1,name,mu,h,False);persistence+=rr
            if reference is None:reference=end
            neutral_invariance.append(dict(root=seed,mode=name,neutral_max_abs_diff=float(np.max(abs(end-reference)))))
        print('PERSISTENCE root',seed,'seconds',round(time.time()-tic,1),flush=True)
    pd.DataFrame(persistence).to_csv(out/'persistence_raw.csv',index=False)
    pd.DataFrame(neutral_invariance).to_csv(out/'neutral_path_invariance.csv',index=False)
    df=pd.DataFrame(rows);ci=[]
    # Replicate histories are averaged within root before t inference.
    for (n,lam),g in df[df.lam>0].groupby(['round','lam']):
        ctl=df[(df['round']==n)&(df.lam==0)]
        pair=g.merge(ctl,on=['root','future_rep'],suffixes=('','_control'),validate='one_to_one')
        for metric_name,bound in [('G',.001),('M',.001),('E',.002)]:
            d=pair[metric_name]-pair[metric_name+'_control'];rootd=d.groupby(pair.root).mean();mean=float(rootd.mean());se=float(rootd.std(ddof=1)/np.sqrt(len(rootd)));h90=float(t.ppf(.95,len(rootd)-1)*se);h95=float(t.ppf(.975,len(rootd)-1)*se)
            ci.append(dict(round=n,time=n/10,lam=lam,metric=metric_name,mean_difference=mean,ci90_half=h90,ci95_half=h95,equivalence_bound=bound,equivalence_pass=abs(mean)+h90<bound,n_roots=len(rootd),futures_per_root=4))
    pd.DataFrame(ci).to_csv(out/'full_redraw_equivalence.csv',index=False)
    ps=pd.DataFrame(persistence);pp=[]
    for (name,n),g in ps[ps.lam==4].groupby(['mode','round']):
        ctl=ps[(ps.lam==0)&(ps['round']==n)&(ps['mode']==name)];pair=g.merge(ctl,on=['root','future_rep'],suffixes=('','_control'),validate='one_to_one')
        for m in ['G','M','E','Q']:
            d=pair[m]-pair[m+'_control'];half=float(t.ppf(.975,5)*d.std(ddof=1)/np.sqrt(6));pp.append(dict(mode=name,round=n,time=n/10,metric=m,mean_effect=float(d.mean()),ci95_half=half,n_roots=6,lag_score_corr=float(g.lag_score_corr.iloc[0])))
    pd.DataFrame(pp).to_csv(out/'persistence_effects.csv',index=False)
    both=pd.concat([df,ps],ignore_index=True);dump_json(out/'audit.json',dict(N=100000,full_redraw_trajectories=72,persistence_trajectories=60,rounds_per_trajectory=300,distinct_preparation_roots=6,new_independent_future_streams_per_root=4,raw_metric_rows=len(both),max_mean_wealth_error=float((both['mean']-1).abs().max()),minimum_wealth=float(both.minimum.min()),max_count_identity_error=float(both.count_error.abs().max()),neutral_path_max_abs_difference=max(r['neutral_max_abs_diff'] for r in neutral_invariance),seconds=time.time()-tic,holdout_read=False))
    print('ABM_COMPLETE',flush=True)
if __name__=='__main__':main()
