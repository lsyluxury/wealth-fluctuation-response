"""Independent exhaustive finite-N checks of path-weight memory-response theorem.

Enumerates all pairings/outcomes for N=4 and n<=4.
For finite lambda, integrates out 16 binary advantage configurations via a
16x16 Markov transition matrix (not Monte Carlo, no mean-field closure).
The path kernel is computed only from the unbiased (lambda=0) paths.
"""
from __future__ import annotations
import itertools,functools,json,math,sys,time
from pathlib import Path
import numpy as np
from scipy.special import expit

@functools.lru_cache(None)
def matchings(inds):
    if not inds:return ((),)
    i,*rest=inds; ans=[]
    for j in rest:
        rr=tuple(k for k in rest if k!=j)
        for tail in matchings(rr): ans.append(((i,j),)+tail)
    return tuple(ans)

def formula_eq(N,eps,rho):
    return N/(N-1)*eps*(1-eps)*rho/24.

def finite_N_equal_first(n,eps=.14):
    # By permutation invariance, first-round signs can be fixed (+,-,+,-,...)
    signs0=np.array([1,-1]*(n//2))
    before=1+eps*signs0
    signs=np.array(list(itertools.product((1.,-1.),repeat=n//2)))
    total=0.;paths=0;E_S=0.
    for pairs in matchings(tuple(range(n))):
        a=np.array([p[0] for p in pairs]);b=np.array([p[1] for p in pairs]);s=eps*np.minimum(before[a],before[b])
        wealth=np.broadcast_to(before,(len(signs),n)).copy()
        wealth[:,a]+=signs*s; wealth[:,b]-=signs*s
        ordw=np.sort(wealth,axis=1)
        G=2*np.sum(ordw*np.arange(1,n+1),axis=1)/(n*n)-(n+1)/n
        S=np.sum(signs*(signs0[a]-signs0[b]),axis=1)
        total+=np.sum(G*S); E_S+=np.sum(S); paths+=len(signs)
    c_div_C=total/paths/4
    return dict(N=n,paths=paths,coef_div_C=float(c_div_C),formula_div_C=float(formula_eq(n,eps,1.)*3),max_diff=float(c_div_C-formula_eq(n,eps,1.)*3),mean_S=float(E_S/paths))

def simple_gini(w):
    z=np.sort(w);n=len(w)
    return 2*np.dot(z,np.arange(1,n+1))/(n*np.sum(z))-(n+1)/n

def rank_metrics(w, top, bottom):
    n=len(w); order=np.argsort(w,kind='mergesort')
    pos=np.empty(n,dtype=int);pos[order]=np.arange(n)
    is_top=pos>=n-len(top)
    return np.array([simple_gini(w),float(np.mean(is_top[bottom])),float(1-np.mean(is_top[top]))])

def path_enumerate(nrounds,w0,eps,chi,flip_p,lam):
    n=len(w0); assert n==4
    pairs=matchings(tuple(range(n)))
    outcomes=list(itertools.product((1,-1), repeat=n//2))
    configs=np.array(list(itertools.product((-1.,1.),repeat=n)))/math.sqrt(3)
    nstates=len(configs)
    # For individual flip prob p/2, lag correlation = (1-p)^d
    f=flip_p/2
    T=np.ones((nstates,nstates))
    for a in range(nstates):
        for b in range(nstates):
            flips=np.count_nonzero(configs[a]!=configs[b])
            T[a,b]=(f**flips)*((1-f)**(n-flips))
    # Each round one of 3 matchings, each pairing has 4 possible outcomes.
    histories=[]
    for pp in pairs:
        for ss in outcomes:
            sig=np.zeros(n,dtype=np.int8)
            for (i,j),val in zip(pp,ss):sig[i]=val;sig[j]=-val
            histories.append((pp,ss,sig))
    weights=[]
    for pp,ss,_ in histories:
        v=np.ones(nstates)
        for (i,j),val in zip(pp,ss):
            d=configs[:,i]-configs[:,j]
            pwin=expit(lam*d)
            v*=pwin if val==1 else 1-pwin
        weights.append(v/len(pairs))
    top=np.argsort(w0,kind='mergesort')[-1:]
    bot=np.argsort(w0,kind='mergesort')[:2]
    sumX=np.zeros(3)
    # Keep neutral path cumulants S_d (to verify covariance directly)
    S_means=np.zeros(nrounds-1)
    XS_means=np.zeros((nrounds-1,3))
    count=0
    # Path recursion: probability vector over the 16 ability states.
    def walk(depth,w,pi,sigs,neutral_weight):
        nonlocal count,sumX,S_means,XS_means
        if depth==nrounds:
            X=rank_metrics(w,top,bot)
            sumX[:]+=pi.sum()*X
            if lam==0.:
                count+=1
                for d in range(1,nrounds):
                    sd=sum(int(np.dot(sigs[k],sigs[k+d])) for k in range(nrounds-d))
                    S_means[d-1]+=neutral_weight*sd
                    XS_means[d-1,:]+=neutral_weight*sd*X
            return
        for t,((pp,ss,sg),vv) in enumerate(zip(histories,weights)):
            v=w.copy()
            for (i,j),sgn in zip(pp,ss):
                s=eps*min(w[i],w[j]);v[i]+=sgn*s;v[j]-=sgn*s
            if chi:
                rho=math.exp(-chi/10.);v=rho*v+(1-rho)
            next_pi=pi*vv
            if depth < nrounds-1:next_pi=next_pi@T
            walk(depth+1,v,next_pi,sigs+[sg],neutral_weight/len(histories))
    walk(0,np.asarray(w0,dtype=float),np.full(nstates,1/nstates),[],1.)
    return dict(meanX=sumX,meansS=S_means,meansXS=XS_means,leaf_paths=count)

def main():
  out={'general_finiteN_two_round': [finite_N_equal_first(n) for n in (2,4,6,8,10,12)]}
  for n,chi in [(3,0.),(4,.03)]:
    w0=[.4,.8,1.2,1.6];eps=.14;flip_p=.4
    baseline=path_enumerate(n,w0,eps,chi,flip_p,0.)
    kernels=(baseline['meansXS']-baseline['meansS'][:,None]*baseline['meanX'])/4
    # C_u(d)= (1-p)^d/3
    C=np.array([(1-flip_p)**d/3 for d in range(1,n)])
    pred=C@kernels
    tests=[]
    for lam in (.04,.08,.16):
      got=path_enumerate(n,w0,eps,chi,flip_p,lam)
      gotCoef=(got['meanX']-baseline['meanX'])/lam**2
      tests.append({'lambda':lam,'direct_response_div_lambda2':gotCoef.tolist(), 'prediction':pred.tolist(), 'error':(gotCoef-pred).tolist()})
    out[f'non_equal_n{n}_chi{chi}']={'num_neutral_histories':baseline['leaf_paths'], 'E0_S':baseline['meansS'].tolist(), 'K_G_M_E_by_lag':kernels.tolist(), 'score_C_by_lag':C.tolist(),'coefficients':pred.tolist(), 'tests':tests}
  dest=Path(__file__).resolve().parents[1]/'results/stage1_exact_enumeration.json';dest.write_text(json.dumps(out,ensure_ascii=False,indent=2))
  for item in out['general_finiteN_two_round']:
    print('finite N',item)
  for name,grp in out.items():
    if name.startswith('non_equal'):
      print(name,'paths=',grp['num_neutral_histories'],'prediction=',grp['coefficients'])
      for t in grp['tests']:print('  lambda=',t['lambda'],'coefficient=',t['direct_response_div_lambda2'],'error=',t['error'])
  return out
if __name__=='__main__':main()
