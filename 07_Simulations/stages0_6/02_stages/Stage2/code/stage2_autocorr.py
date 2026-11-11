"""Empirical score covariance vs analytic formula for all four stationary processes."""
import math, json, numpy as np,sys,time
from pathlib import Path
from legacy_engine import reseed
from engine_stage01 import advance_trace,initialize_process_score
N=100000;T=60;p=.1;lags=[1,2,5,10,20,40,60]
# Baseline Gaussian-Poisson C(t) from sum over total Poisson events at t/r.
def p1_C(d):
  mass=math.exp(-d/10);res=0.
  for k in range(200):
    res += mass*(2/math.pi)*math.asin((.3**k)/2)
    mass*= (d/10)/(k+1)
  return res
rho1=3*p1_C(1);phet=2*(1-rho1)
out={'N':N,'rounds':T,'lags':lags,'p':p,'P1_corr1':rho1,'P4_p_fast':phet,'modes':{}}
for mode in [0,1,2,5]:
  reseed(60700+mode)
  w=np.ones((N,1));a=np.zeros(N);u=np.zeros(N);perm=np.arange(N)
  if mode==0:
    a=np.random.default_rng(90700).normal(size=N).astype(float)
    from scipy.special import erf
    u=erf(a/math.sqrt(2))
  else:
    initialize_process_score(u,mode)
    if mode==5:
      mask=np.zeros(N)
      rng=np.random.default_rng(9004)
      mask[rng.permutation(N)[:N//2]]=1.
      a=mask
  arr=np.empty((T+1,N),dtype=np.float64);arr[0]=u
  for t in range(T):
    advance_trace(w,a,u,perm,np.array([0.]),np.array([0.]),1,0.,.3,1.,10.,mode,phet if mode==5 else p,False)
    arr[t+1]=u
  measured=np.array([np.mean(arr[:-d]*arr[d:]) for d in lags]);variance=float((arr*arr).mean())
  if mode==0:expected=np.array([p1_C(d) for d in lags])
  elif mode in [1,2]:expected=np.array([(1-p)**d/3 for d in lags])
  else:expected=np.array([(.5+.5*(1-phet)**d)/3 for d in lags])
  out['modes'][str(mode)]={'measured':measured.tolist(),'theory':expected.tolist(),'variance':variance,'max_abs_error':float(np.max(abs(measured-expected)))}
  print('mode',mode,'maxerr=',float(np.max(abs(measured-expected))),flush=True)
  del arr
out['P2_P3_max_empirical_C_difference']=float(np.max(abs(np.array(out['modes']['1']['measured'])-np.array(out['modes']['2']['measured']))))
path=Path(__file__).resolve().parents[1]/'results/stage2_score_correlations.json'
path.write_text(json.dumps(out,indent=2));print('OUTPUT',path)
