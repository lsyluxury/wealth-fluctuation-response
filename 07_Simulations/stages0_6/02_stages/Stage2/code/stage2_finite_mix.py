"""Fully renewing heterogeneous memory P5 with same C(1) as P1.
Half of agents refresh per round p_slow=0.01, half at p_fast=2*(1-rho1)-p_slow.
Both rates >0: full advantage memory eventually decays to zero.
"""
import time,math,argparse
from pathlib import Path
import numpy as np,pandas as pd
from legacy_engine import reseed,gini_numba
from engine_stage01 import advance_trace,initialize_process_score
from stage2_abm import LAM,CHI,prepare,summary_all
ap=argparse.ArgumentParser();ap.add_argument('--roots',type=int,default=130);ap.add_argument('--N',type=int,default=10000);ap.add_argument('--rounds',type=int,default=300);ap.add_argument('--out',default='../results/stage2_finite_mix.csv');args=ap.parse_args()
wt=math.exp(-.1);c=0.
for k in range(100):c+=wt*(2/math.pi)*math.asin(.3**k/2);wt*=.1/(k+1)
rho1=3*c;pslow=.01;pfast=2*(1-rho1)-pslow
assert 0<pslow<pfast<1
rows=[];tm=time.time();
for j in range(args.roots):
 root=20000000+j
 w0,a0,u0,prep=prepare(args.N,root+101000,.65)
 inds=np.argsort(w0,kind='mergesort');bottom=inds[:args.N//2];top=inds[-args.N//10:]
 w=np.repeat(w0[:,None],len(LAM),axis=1);a=np.zeros(args.N);u=np.empty(args.N);perm=np.arange(args.N)
 reseed(root+710000+600000);initialize_process_score(u,6)
 rng=np.random.default_rng(root+10200000)
 a[rng.permutation(args.N)[:args.N//2]]=1.
 reseed(root+810000)
 advance_trace(w,a,u,perm,LAM,CHI,args.rounds,.14,.3,pslow,10.,6,pfast,False)
 X=summary_all(w,bottom,top)
 for z,lam in enumerate(LAM):rows.append(dict(root=root,N=args.N,prepare_rounds=prep,mode=6,p=pfast,p_slow=pslow,round=args.rounds,lambda_value=lam,G=X[z,0],M=X[z,1],E=X[z,2]))
 if (j+1)%10==0:print('FINITE_MIX',j+1,'elapsed',round(time.time()-tm,1),flush=True)
 if (j+1)%50==0:pd.DataFrame(rows).to_csv(args.out,index=False)
pd.DataFrame(rows).to_csv(args.out,index=False)
print('DONE',args.out,'Pfast',pfast,'rho1',rho1,'elapsed',round(time.time()-tm,1),flush=True)
