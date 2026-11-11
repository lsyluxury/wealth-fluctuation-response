#!/usr/bin/env python3
"""Independently re-enumerate the finite-N two-round Gini coefficient.
Fix first-round winners by permutation symmetry; enumerate every second-round
perfect matching and fair outcome. E_0[G2*S1]/12 equals c_G2/rho_u for Var(u)=1/3.
No simulations, fitted data, or theoretical coefficient enter the enumeration.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd
from numba import njit
R=Path(__file__).resolve().parents[1]
def matchings(ids):
 if not ids:
  yield []
  return
 i=ids[0]
 for k in range(1,len(ids)):
  j=ids[k]
  for tail in matchings(ids[1:k]+ids[k+1:]):yield [(i,j)]+tail
@njit(cache=True)
def integrate(mat,n,eps):
 total=0.;total_s=0.;total_g=0.;count=0
 first=np.ones(n);first[n//2:]=-1
 init=1+eps*first
 for m in range(len(mat)):
  for outcome in range(1<<(n//2)):
   w=init.copy();s1=0.
   for pair in range(n//2):
    i,j=mat[m,pair,0],mat[m,pair,1]
    sig=1. if outcome&(1<<pair) else -1.
    stake=eps*min(init[i],init[j]);w[i]+=sig*stake;w[j]-=sig*stake
    s1+=first[i]*sig-first[j]*sig
   ws=np.sort(w);g=0.
   for i in range(n):g+=(2*i-n+1)*ws[i]
   g/=n*ws.sum();total+=g*s1;total_s+=s1;total_g+=g;count+=1
 return (total/count-(total_g/count)*(total_s/count))/12.,count,total_s/count

def main():
 rows=[]
 for n in [2,4,6,8,10,12]:
  mat=np.array(list(matchings(list(range(n)))),dtype=np.int64)
  coef,paths,scoremean=integrate(mat,n,.14);exact=n/(n-1)*.14*.86/24
  rows.append(dict(N=n,enumerated_coefficient_per_rho=coef,formula_coefficient_per_rho=exact,paths=paths,neutral_score_mean=scoremean,error=coef-exact))
 d=pd.DataFrame(rows);p=R/'validation/fresh_two_round_enumeration.csv';p.parent.mkdir(exist_ok=True);d.to_csv(p,index=False)
 assert abs(d.error).max()<2e-12
 (R/'validation/fresh_two_round_enumeration.json').write_text(json.dumps({'status':'PASS','max_absolute_error':float(abs(d.error).max()),'total_enumerated_paths':int(d.paths.sum()),'source':'independent symmetry-reduced enumeration of matching and fair second-round outcomes'},indent=2))
 print(d.to_string(index=False))
if __name__=='__main__':main()
