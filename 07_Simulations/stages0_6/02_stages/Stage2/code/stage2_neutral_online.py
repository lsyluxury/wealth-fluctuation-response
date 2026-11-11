"""Online neutral-path importance-score estimator (lag-weighted covariance).

All winning signs are unbiased; scores are computed from sign histories without
simulating abilities. Baseline and alternative C(d) enter through exponential
recursions, matching the exact finite-N memory-response formula.
"""
import sys,math,time,json,argparse
from pathlib import Path
import numpy as np,pandas as pd
from numba import njit
from scipy.stats import t
from legacy_engine import reseed,gini_numba

# P1 Hermite coefficients: u=sum_j uj * ej, odd j only.
J=np.arange(1,24,2,dtype=np.int64)
gammas=np.exp(-.1*(1.-.3**J));weights=np.array([math.exp(math.lgamma(int(j)+1)-math.log(math.pi)-2*int((j-1)//2)*math.log(4)-2*math.lgamma(int((j-1)//2)+1)-2*math.log(int(j))) for j in J])
C0=weights.sum()
@njit(cache=True)
def neutral_with_online_scores(w0,nrounds,eps,q2,qhet,gammas,weights):
 N=len(w0);w=w0.copy();perm=np.arange(N)
 acc_p1=np.zeros((len(gammas),N),dtype=np.float64)
 acc_p2=np.zeros(N);acc_qhet=np.zeros(N);acc_static=np.zeros(N)
 a1=0.;a2=0.;a4=0.
 for k in range(nrounds):
  np.random.shuffle(perm)
  for j0 in range(0,N,2):
   i=perm[j0];j=perm[j0+1]
   coin=1 if np.random.random()<.5 else -1
   v=w[i];z=w[j];stake=eps*(v if v<z else z)
   if coin==1:w[i]=v+stake;w[j]=z-stake
   else:w[i]=v-stake;w[j]=z+stake
   # each identity updated exactly once per complete round
   for ind,sig in ((i,coin),(j,-coin)):
    a2+=sig*acc_p2[ind]/12.
    acc_p2[ind]=q2*(acc_p2[ind]+sig)
    a4+=sig*(acc_qhet[ind]+acc_static[ind])/24.
    acc_qhet[ind]=qhet*(acc_qhet[ind]+sig)
    acc_static[ind]+=sig
    for m in range(len(gammas)):
     a1+=weights[m]*sig*acc_p1[m,ind]/4.
     acc_p1[m,ind]=gammas[m]*(acc_p1[m,ind]+sig)
 return w,a1,a2,a4

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--N',type=int,default=1000);ap.add_argument('--rounds',type=int,default=40);ap.add_argument('--reps',type=int,default=5000);ap.add_argument('--out',default='../results/neutral_kernel_online.csv');args=ap.parse_args()
 f=str(Path(__file__).resolve().parents[3] / '05_history/phase12/source/anchors/development/s2026101201_g65_selection.npz')
 with np.load(f,allow_pickle=False) as z: w0=z['w'].copy()
 # independent identity subsampling from actual archived unequal start
 ix=np.random.default_rng(20001).permutation(len(w0))[:args.N]
 w0=w0[ix];w0*=len(w0)/w0.sum()
 ranks=np.argsort(w0,kind='mergesort');bottom=ranks[:args.N//2];top=ranks[-args.N//10:]
 rho1=3*sum(weights*gammas);qhet=1-2*(1-rho1)
 vals=[];tm=time.time()
 for root in range(args.reps):
  reseed(557900000+root)
  w,a1,a2,a4=neutral_with_online_scores(w0,args.rounds,.14,.9,qhet,gammas,weights)
  s=np.argsort(w,kind='mergesort');curr=np.zeros(args.N,dtype=bool);curr[s[-args.N//10:]]=True
  vals.append(dict(root=root,G=gini_numba(w),M=float(curr[bottom].mean()),E=float(1-curr[top].mean()),A_P1=a1,A_P2=a2,A_P4=a4))
  if (root+1)%1000==0:print('KERNEL',root+1,'elapsed',round(time.time()-tm,1),flush=True)
  if (root+1)%5000==0:pd.DataFrame(vals).to_csv(args.out,index=False)
 df=pd.DataFrame(vals);pp=Path(args.out);pp.parent.mkdir(parents=True,exist_ok=True);df.to_csv(pp,index=False)
 out={'N':args.N,'rounds':args.rounds,'R':args.reps,'source_initial_gini':float(gini_numba(w0)),'truncated_score_variance':float(C0),'P1_rho1_approx':rho1,'modes':{}}
 for mode in ('P1','P2','P4'):
  a=df[f'A_{mode}'].values
  out['modes'][mode]={}
  for x in ('G','M','E'):
   y=df[x].values
   cov=(y-y.mean())*(a-a.mean());mean=float(cov.sum()/(len(y)-1));se=float(cov.std(ddof=1)/np.sqrt(len(y)))
   out['modes'][mode][x]={'c':mean,'se_approx':se,'CI_approx':[mean-t.ppf(.975,len(y)-1)*se,mean+t.ppf(.975,len(y)-1)*se]}
  out['modes'][mode]['mean_score']=float(a.mean());out['modes'][mode]['std_score']=float(a.std(ddof=1))
 pp.with_suffix('.json').write_text(json.dumps(out,indent=2))
 print('DONE',pp,'sec',round(time.time()-tm,1),'Gini P1',out['modes']['P1']['G'],'Gini P2',out['modes']['P2']['G'],flush=True)
if __name__=='__main__':main()
