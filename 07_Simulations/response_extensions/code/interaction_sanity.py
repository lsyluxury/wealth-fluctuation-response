#!/usr/bin/env python3
"""Independent algebraic checks that matter for the paired interaction driver."""
import json,math
import numpy as np
from numba import njit
from interaction_simulation import (ROOT,neutral_one,direct_one,load_initial,
    EPS,H,MU,RATE,LAMBDAS,CHIS,CHECKS,GAMMAS,WEIGHTS)

@njit(cache=True)
def fair_signs(seed,n,rounds):
    np.random.seed(seed); perm=np.arange(n)
    signs=np.empty((rounds,n),dtype=np.int64)
    for k in range(rounds):
        np.random.shuffle(perm)
        for q in range(0,n,2):
            sig=1 if np.random.random()<.5 else -1
            signs[k,perm[q]]=sig; signs[k,perm[q+1]]=-sig
    return signs

def main():
    w0,bottom,top,_=load_initial()
    seed=199913
    checks=np.array([1,8],dtype=np.int64)
    out,scores=neutral_one(w0,bottom,top,seed,checks,EPS,GAMMAS,WEIGHTS,CHIS,RATE)
    s=fair_signs(seed,len(w0),8)
    explicit=0.0
    for lag in range(1,8):
        c=float(np.sum(WEIGHTS*GAMMAS**lag))
        explicit+=.25*c*np.sum(s[:-lag]*s[lag:])
    recurrence_error=abs(explicit-scores[-1])
    g_affine_error=abs(out[0,1,0]-math.exp(-CHIS[1]/RATE)*out[0,0,0])
    rank_affine_error=float(np.max(np.abs(out[0,0,1:]-out[0,1,1:])))
    duplicate_chi=np.array([0.,0.])
    nd,ad=neutral_one(w0,bottom,top,seed,checks,EPS,GAMMAS,WEIGHTS,duplicate_chi,RATE)
    dd=direct_one(w0,bottom,top,seed,checks,EPS,H,MU,RATE,LAMBDAS,duplicate_chi)
    neutral_dup=float(np.max(np.abs(nd[:,0]-nd[:,1])))
    direct_dup=float(np.max(np.abs(dd[:,:,0]-dd[:,:,1])))
    nr=no_redistribution_interaction=neutral_dup==0.0 and direct_dup==0.0
    report=dict(score_explicit_lag=explicit,score_online=float(scores[-1]),
        score_absolute_error=recurrence_error,first_round_A=float(scores[0]),
        first_round_gini_affine_error=g_affine_error,
        first_round_rank_unchanged_error=rank_affine_error,
        duplicate_zero_tax_neutral_error=neutral_dup,duplicate_zero_tax_direct_error=direct_dup,
        chi_zero_interaction_exact_zero=nr,
        seed=seed,N=len(w0),rounds=8,
        passed=bool(recurrence_error<1e-10 and scores[0]==0 and g_affine_error<1e-12
                    and rank_affine_error==0 and nr))
    (ROOT/'logs'/'interaction_sanity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert report['passed']

if __name__=='__main__':main()
