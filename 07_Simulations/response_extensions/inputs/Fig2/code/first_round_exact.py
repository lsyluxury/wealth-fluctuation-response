#!/usr/bin/env python3
"""Standalone first-round exact control, matching archived exact_controls.py.

Four agents: w=(0.4,0.8,1.2,1.6); iid u=+/-0.7; epsilon=0.14.
Enumerates all 16 score vectors, 3 matchings and 4 outcome patterns.
This is an exact finite-state calculation, NOT an ABM sample or fitted curve.
"""
from itertools import product
import numpy as np
from scipy.special import expit

WEALTH=np.array([.4,.8,1.2,1.6])
EPSILON=.14
SCORE_AMPLITUDE=.7
MATCHINGS=(((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2)))

def gini(w: np.ndarray) -> float:
    return float(np.abs(w[:,None]-w[None,:]).sum()/(2*len(w)**2*w.mean()))

def analytic_delta_q(lam: float) -> float:
    n=len(WEALTH)
    amplitude=.5*(2*SCORE_AMPLITUDE)*np.tanh(lam*SCORE_AMPLITUDE)
    return float(EPSILON/2*(1-n/(n-1)*gini(WEALTH))*amplitude)

def enumerate_first_round(lam: float) -> tuple[dict,dict,float]:
    if not np.isfinite(lam): raise ValueError('lambda must be finite')
    law={};marginal={};delta_q=0.
    for values in product((-SCORE_AMPLITUDE,SCORE_AMPLITUDE),repeat=4):
        u=np.asarray(values)
        for matching in MATCHINGS:
            for signs in product((-1,1),repeat=2):
                v=WEALTH.copy();prob=1/48
                for (i,j),sign in zip(matching,signs):
                    prob*=float(expit(sign*lam*(u[i]-u[j])))
                    stake=EPSILON*min(WEALTH[i],WEALTH[j])
                    v[i]+=sign*stake;v[j]-=sign*stake
                state=tuple(np.round(v,12))
                law[state]=law.get(state,0.)+prob
                delta_q+=prob*np.mean((v-WEALTH)*u)
                for wealth in v:
                    w=round(float(wealth),12)
                    marginal[w]=marginal.get(w,0.)+prob/4
    return law,marginal,float(delta_q)
