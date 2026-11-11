"""Validate the stationary and finite-memory covariance of slow/fast renewals."""
import math,json
from pathlib import Path
import numpy as np
from legacy_engine import reseed
from engine_stage01 import advance_trace,initialize_process_score
N=100000;T=60;pslow=.01
wt=math.exp(-.1);s=0.
for k in range(100):s+=wt*(2/math.pi)*math.asin(.3**k/2);wt*=.1/(k+1)
rho1=3*s;pfast=2*(1-rho1)-pslow
reseed(70001);a=np.zeros(N);u=np.zeros(N);w=np.ones((N,1));perm=np.arange(N)
initialize_process_score(u,6);a[np.random.default_rng(70505).permutation(N)[:N//2]]=1.
arr=np.empty((T+1,N));arr[0]=u
for t in range(T):
 advance_trace(w,a,u,perm,np.array([0.]),np.array([0.]),1,0.,.3,pslow,10.,6,pfast,False)
 arr[t+1]=u
lags=[1,2,5,10,20,40,60];obs=[float(np.mean(arr[:-d]*arr[d:])) for d in lags]
theory=[((1-pslow)**d+(1-pfast)**d)/6 for d in lags]
res={'mode':6,'N':N,'rounds':T,'p_slow':pslow,'p_fast':pfast,'rho_one_lag':rho1,'lags':lags,'observed':obs,'expected':theory,'max_abs_error':float(np.max(abs(np.array(obs)-theory)))}
path=Path(__file__).resolve().parents[1]/'results/stage2_P5_finite_autocorr.json';path.write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
