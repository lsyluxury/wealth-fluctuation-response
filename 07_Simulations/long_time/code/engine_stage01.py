"""Stage 0 extension of the archived Numba Yard-Sale engine.

The original source code (legacy_engine.py) is unmodified.  In mode=0
(gaussian_poisson), this observationally instrumented kernel uses its exact
random draw schedule. It can replay original anchors bit for bit.

Modes:
  0  original Gaussian/Poisson latent update
  1  refresh U(-1,1) independently per agent with per-round probability p
  2  two-state (+/- 1/sqrt(3)) flip with per-round probability p/2
  3  fixed score u
  4  full U(-1,1) redraw for every individual BEFORE each round

For modes 1/2/4, a is an inactive latent auxiliary array and ONLY u defines
individual advantage; initialize u from the process's stationary marginal.
New modes do not claim bitwise identity with original Gaussian random streams.

Operation order: full-redraw control (if selected), pairing/trading, score renewal
(if selected), taxation/rebate. Each individual participates once per round.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit
from legacy_engine import logistic

@njit(cache=True)
def advance_trace(w, a, u, perm, lambdas, chis, nrounds,
                  epsilon, h, mu, r, mode, p, record_signs):
    """Evolve one or more lambda/chi branches sharing the same internal score.

    w shape (N,B), a/u/perm shape (N,), returns sigma as int8 array
    (nrounds,N,B) if requested; when not requested returns (0,0,0).
    This function does not record passive budgets. Original stage-A comparison
    is performed on all state and RNG arrays, not on budget output.
    """
    n, nb = w.shape
    dt = 1.0/r
    innovation = math.sqrt(max(0.0,1.0-h*h))
    sigma_trace = np.empty((nrounds,n,nb),np.int8) if record_signs else np.empty((0,0,0),np.int8)
    same_lambda = True
    for b in range(1, nb):
        if lambdas[b] != lambdas[0]: same_lambda = False
    for k in range(nrounds):
        if mode == 4:
            for i in range(n):
                u[i] = 2.0 * np.random.random() - 1.0
        np.random.shuffle(perm)
        for j0 in range(0,n,2):
            i=perm[j0];j=perm[j0+1]
            du=u[i]-u[j]
            draw=np.random.random()
            pshared=logistic(lambdas[0]*du) if same_lambda else 0.0
            for b in range(nb):
                prob=pshared if same_lambda else logistic(lambdas[b]*du)
                win=draw < prob
                wi=w[i,b];wj=w[j,b]
                stake=epsilon*(wi if wi<wj else wj)
                if record_signs:
                    sigma_trace[k,i,b]=1 if win else -1
                    sigma_trace[k,j,b]=-1 if win else 1
                if win:
                    w[i,b]=wi+stake;w[j,b]=wj-stake
                else:
                    w[i,b]=wi-stake;w[j,b]=wj+stake
        if mode == 0:
            if mu > 0:
                nev=np.random.poisson(n*mu*dt)
                for _ in range(nev):
                    i=np.random.randint(0,n)
                    a[i]=h*a[i]+innovation*np.random.normal()
                    u[i]=math.erf(a[i]/math.sqrt(2.0))
        elif mode == 1:
            for i in range(n):
                if np.random.random() < p:
                    u[i] = 2.0*np.random.random()-1.0
        elif mode == 2:
            for i in range(n):
                if np.random.random() < p/2.0:
                    u[i]=-u[i]
        elif mode == 5:
            # Latent a=1 marks fast-updating individuals; a=0 permanently fixed.
            # Both strata have the same stationary uniform score law.
            for i in range(n):
                if a[i]>0.5 and np.random.random()<p:
                    u[i]=2.0*np.random.random()-1.0
        elif mode == 6:
            # Both subpopulations renew at strictly positive rates.
            # a=0 (slow, probability mu per round), a=1 (fast, probability p).
            for i in range(n):
                probability=p if a[i]>0.5 else mu
                if np.random.random() < probability:
                    u[i] = 2.0*np.random.random()-1.0
        # mode 3: fixed. mode 4: already fully redrawn before matching
        for b in range(nb):
            if chis[b]>0:
                rho=math.exp(-chis[b]*dt);rebate=1.0-rho
                for i in range(n):
                    w[i,b]=rho*w[i,b]+rebate
    return sigma_trace

@njit(cache=True)
def initialize_process_score(u, mode):
    """Initialize a stationary advantage score for an alternative process.

    This consumes random draws and therefore must NOT be used in legacy-mode
    replay. Original Gaussian initialization is supplied by the stored anchor.
    """
    if mode == 1 or mode == 4 or mode == 5 or mode == 6:
        for i in range(len(u)):u[i]=2.0*np.random.random()-1.0
    elif mode == 2:
        c=1.0/math.sqrt(3.0)
        for i in range(len(u)):
            u[i]=c if np.random.random()<.5 else -c
    # mode0 and 3: keep supplied score


def top_membership(w:np.ndarray, q:float=.10)->np.ndarray:
    """Same tie breaking as archived legacy_records.ranks: stable argsort by wealth, then ID."""
    assert 0 < q < 1
    n=len(w); k=int(round(n*q))
    assert k>0 and abs(k/n-q)<1e-8
    order=np.argsort(w,kind='mergesort')
    ans=np.zeros(n,dtype=bool)
    ans[order[-k:]]=True
    return ans


def cohort_overlap(tag, wealth_current, q=.1):
    """Two-time top-q overlap, conditional membership and normalized C_q."""
    current=top_membership(wealth_current,q)
    fraction=np.count_nonzero(tag & current)/np.count_nonzero(tag)
    return {'stay':float(fraction),'exit':float(1-fraction), 'C_q':float((fraction-q)/(1-q))}
