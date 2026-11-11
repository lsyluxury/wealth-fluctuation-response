"""Independent finite-population enumeration and analytic-control checks.
No fit, ABM holdout, or intervention endpoint is used.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import itertools,math
import numpy as np,pandas as pd
from scipy.special import expit,eval_hermitenorm
from scipy.stats import poisson
from numpy.polynomial.hermite import hermgauss
from common import *
from density_solver import probability_kernel,ability_kernel
PAIRS=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
def discrete_gini(w):
    w=np.sort(w);n=len(w);return float(2*np.dot(np.arange(1,n+1),w)/(n*w.sum())-(n+1)/n)
def enumerate_trade(w,u,lam):
    states=[]
    for pairing in PAIRS:
        for signs in itertools.product((1,-1),repeat=2):
            v=np.array(w,dtype=float);prob=1/3
            for (i,j),sign in zip(pairing,signs):
                p=expit(lam*(u[i]-u[j]));prob*=p if sign==1 else 1-p;d=.14*min(w[i],w[j]);v[i]+=sign*d;v[j]-=sign*d
            states.append((v,prob))
    return states

def first_and_second_law(lam,keep):
    w0=np.array([.4,.8,1.2,1.6]);law={}
    for uv in itertools.product((-.7,.7),repeat=4):
        u=np.array(uv)
        for w,p in enumerate_trade(w0,u,lam):
            if keep:sub=enumerate_trade(w,u,lam)
            else:sub=enumerate_trade(w,np.zeros(4),0)
            for v,q in sub:
                key=tuple(np.round(v,12));law[key]=law.get(key,0.)+p*q/16
    return law

def atomic_two_round(na,lam,mu=1.,h=.3):
    s=Settings(na=na,mu=mu,h=h,poisson_max=18);R,_=ability_kernel(s);P,_=probability_kernel(na,lam)
    uniform=np.ones(na)/na;win=P@uniform
    f=np.stack([uniform*(1-win),uniform*win],axis=1);f=R@f;w=np.array([1-.14,1+.14]);dist={}
    for i in range(2):
        for j in range(2):
            plus=float(np.dot(f[:,i],P@f[:,j]));pair=float(f[:,i].sum()*f[:,j].sum());stake=.14*min(w[i],w[j])
            for ww,pp in [(w[i]+stake,plus),(w[i]-stake,pair-plus)]:
                key=round(float(ww),12);dist[key]=dist.get(key,0.)+pp
    xx=np.array(sorted(dist));pp=np.array([dist[x] for x in xx]);return gini(pp,xx)

def main():
    out=ROOT/'results'/'exact_controls';out.mkdir(parents=True,exist_ok=True);audit={}
    # Four-agent counterexample, all three matchings and four outcome patterns.
    w=np.array([.01,.02,1.9,2.07]);u=np.array([-.8,-.8,.8,0]);lam=math.log(3)/.8
    eg0=sum(p*discrete_gini(v) for v,p in enumerate_trade(w,u,0));eg=sum(p*discrete_gini(v) for v,p in enumerate_trade(w,u,lam));Q=float(np.mean((w-1)*u))
    audit['positive_Q_counterexample']=dict(Q=Q,lambda_value=lam,G_before=discrete_gini(w),G_fair_expected=eg0,G_selected_expected=eg,selection_extra=eg-eg0,outcomes=12)
    # Finite N first-round mean formula with binary iid ability law.
    w=np.array([.4,.8,1.2,1.6]);enum=[]
    for lam in (0.,.1,4.,16.):
        dQ=0.;first={}
        for uv in itertools.product((-.7,.7),repeat=4):
            u=np.array(uv)
            for v,p in enumerate_trade(w,u,lam):
                dQ+=p*np.mean((v-w)*u)/16;key=tuple(np.round(v,12));first[key]=first.get(key,0)+p/16
        A=.5*1.4*np.tanh(lam*1.4/2);pred=.14/2*(1-4*discrete_gini(w)/3)*A
        if lam==0:base=first
        err=max(abs(first.get(k,0)-base.get(k,0)) for k in set(first)|set(base))
        enum.append(dict(lam=lam,first_round_law_max_error=err,deltaQ_enumerated=dQ,deltaQ_analytic=pred,error=dQ-pred))
    pd.DataFrame(enum).to_csv(out/'finite_population_first_round.csv',index=False)
    base=first_and_second_law(0,False);tw=[]
    for lam in (4.,16.):
        for keep in (False,True):
            law=first_and_second_law(lam,keep);tv=.5*sum(abs(law.get(k,0)-base.get(k,0)) for k in set(law)|set(base));tw.append(dict(lam=lam,keep_ability=keep,total_variation_from_fair=tv,probability_error=sum(law.values())-1))
    pd.DataFrame(tw).to_csv(out/'finite_population_two_round.csv',index=False)
    # Equal-wealth analytic coefficient, using exact atom locations (no wealth mesh).
    atom=[];truecoef=.14*(1-.14)/24*corr_u(1,.3)
    for na in (48,96,192):
        gg0=atomic_two_round(na,0)
        for lam in (.05,.1,.2):
            gg=atomic_two_round(na,lam);atom.append(dict(ability_bins=na,lam=lam,G0=gg0,G=gg,coefficient_analytic=truecoef,coefficient_difference_quotient=(gg-gg0)/lam**2,predicted_G=gg0+truecoef*lam**2,residual=gg-gg0-truecoef*lam**2))
    pd.DataFrame(atom).to_csv(out/'equal_wealth_two_round.csv',index=False)
    # Pure tax: direct rank and Gini/variance/Q checks at full N.
    z=np.load(ROOT/'inputs/s2026101201_g65_selection.npz');w0=z['w'];u=z['u'];g0=discrete_gini(w0);v0=np.mean((w0-1)**2);q0=np.mean((w0-1)*u);order0=np.argsort(w0);tax=[]
    for chi in (.001,.01,.03):
        rho=np.exp(-chi/10);w=w0.copy()
        for n in range(1,301):
            w=rho*w+1-rho
            if n in OBS:
                tax.append(dict(chi=chi,round=n,G_error=discrete_gini(w)-rho**n*g0,V_error=np.mean((w-1)**2)-rho**(2*n)*v0,Q_error=np.mean((w-1)*u)-rho**n*q0,changed_order_positions=int(np.count_nonzero(np.argsort(w)!=order0)),direct_affine_max_error=float(np.max(abs(w-(rho**n*w0+1-rho**n)))), floating_ties=int(len(w)-len(np.unique(w))), top10_membership_changes=int(len(set(np.argsort(w)[90000:]) ^ set(order0[90000:])))))
    pd.DataFrame(tax).to_csv(out/'pure_tax_full_population.csv',index=False)
    # Worst-case recurrence reaches the bound exactly; paired overtaking condition.
    bounds=[]
    for chi in (.01,.03):
        rho=np.exp(-chi/10);b=rho*(1-.14);ell=(1-rho)/(1-b);w=.0001;err=0
        for n in range(1,301):
            w=rho*(1-.14)*w+(1-rho);closed=b**n*.0001+ell*(1-b**n);err=max(err,abs(w-closed))
        bounds.append(dict(chi=chi,limiting_floor=ell,max_recurrence_error=err))
    pd.DataFrame(bounds).to_csv(out/'tax_wealth_lower_bound.csv',index=False)
    # Hermite conditional expectation under the actual Poisson Gaussian renewal.
    x,wt=hermgauss(12);wt=wt/np.sqrt(np.pi);herm=[]
    for mu,h in [(1.,0.),(1.,.3),(1.,.9),(0.,.3),(5.,.3)]:
        for k in range(1,6):
            gamma=np.exp(-mu/10*(1-h**k));maxerr=0.
            for a in [-2.,-.3,.4,2.4]:
                val=0.
                for nn in range(30):
                    c=h**nn;val+=poisson.pmf(nn,mu/10)*np.dot(wt,eval_hermitenorm(k,c*a+np.sqrt(2*(1-c*c))*x))
                maxerr=max(maxerr,abs(val-gamma*eval_hermitenorm(k,a)))
            herm.append(dict(mu=mu,h=h,k=k,gamma=gamma,max_absolute_error=maxerr,score_lag_correlation=corr_u(mu,h)))
    pd.DataFrame(herm).to_csv(out/'hermite_renewal_quadrature.csv',index=False)
    audit['first_round_max_law_error']=max(r['first_round_law_max_error'] for r in enum);audit['first_round_max_Q_error']=max(abs(r['error']) for r in enum)
    audit['two_round_full_redraw_max_TV']=max(r['total_variation_from_fair'] for r in tw if not r['keep_ability']);audit['pure_tax_max_G_error']=max(abs(r['G_error']) for r in tax);audit['pure_tax_changed_order_positions']=max(r['changed_order_positions'] for r in tax);audit['pure_tax_max_top10_membership_changes']=max(r['top10_membership_changes'] for r in tax);audit['pure_tax_max_floating_ties']=max(r['floating_ties'] for r in tax);audit['pure_tax_rank_note']='Rounding can merge extremely close low-wealth values; entire population argsort need not be bitwise invariant. Top10 membership and the mathematical monotonic mapping are checked separately.'
    audit['mode_quadrature_max_error']=max(r['max_absolute_error'] for r in herm);audit['equal_wealth_two_round_coefficient']=truecoef
    dump_json(out/'audit.json',audit);print(audit,flush=True)
if __name__=='__main__':main()
