"""Finite-stake, complete-pairing Yard-Sale engine with passive budget recording.

No fastmath; the sequence of random draws and wealth arithmetic follows the
supplied common-state pilot. Branches share exogenous streams. This is an ABM,
NOT an independent density solver or an analytic prediction.
"""
from __future__ import annotations
import ctypes
import json
import math
import numpy as np
import numba
from numba import njit, _helperlib

@njit(cache=True)
def logistic(z):
    if z >= 0.0:
        return 1.0/(1.0+math.exp(-z))
    ez=math.exp(z)
    return ez/(1.0+ez)

@njit(cache=True)
def initialise(N, seed, branches=1):
    np.random.seed(seed)
    wealth=np.ones((N,branches), dtype=np.float64)
    latent=np.random.normal(0.0,1.0,N)
    ability=np.empty(N,dtype=np.float64)
    for i in range(N):
        ability[i]=math.erf(latent[i]/math.sqrt(2.0))
    return wealth, latent, ability, np.arange(N)

@njit(cache=True)
def reseed(seed):
    np.random.seed(seed)

@njit(cache=True)
def neutral_round(w,a,u,perm,epsilon,h,mu,r):
    n=w.size
    np.random.shuffle(perm)
    for p in range(0,n,2):
        i=perm[p]; j=perm[p+1]
        draw=np.random.random()
        wi=w[i]; wj=w[j]
        stake=epsilon*(wi if wi<wj else wj)
        if draw<0.5:
            w[i]=wi+stake;w[j]=wj-stake
        else:
            w[i]=wi-stake;w[j]=wj+stake
    if mu>0:
        nev=np.random.poisson(n*mu*(1.0/r))
        innovation=math.sqrt(max(0.0,1.0-h*h))
        for _ in range(nev):
            i=np.random.randint(0,n)
            a[i]=h*a[i]+innovation*np.random.normal()
            u[i]=math.erf(a[i]/math.sqrt(2.0))

@njit(cache=True)
def gini_numba(w):
    ordered=np.sort(w)
    total=0.0; weighted=0.0
    for i in range(w.size):
        total+=ordered[i];weighted+=(i+1)*ordered[i]
    return 2.0*weighted/(w.size*total)-(w.size+1.0)/w.size

# Units for every budget term are change per person, per complete round.
BUDGET_COLUMNS = [
 'trade_dV_actual','trade_dV_selection_expected','trade_dV_diffusion',
 'trade_dQ_actual','trade_dQ_expected','renewal_dQ_actual',
 'mean_stake_per_person','tax_gross_per_person','renewal_event_count']

@njit(cache=True)
def advance_chunk(w,a,u,perm,lambdas,chis,nrounds,epsilon,h,mu,r):
    n,nb=w.shape
    budgets=np.zeros((nrounds,nb,9),dtype=np.float64)
    innovation=math.sqrt(max(0.0,1.0-h*h))
    dt=1.0/r
    same_lambda=True
    for b in range(1,nb):
        if lambdas[b]!=lambdas[0]: same_lambda=False
    for k in range(nrounds):
        np.random.shuffle(perm)
        for p in range(0,n,2):
            i=perm[p];j=perm[p+1]
            du=u[i]-u[j]
            draw=np.random.random()
            pshared=logistic(lambdas[0]*du) if same_lambda else 0.0
            for b in range(nb):
                prob=pshared if same_lambda else logistic(lambdas[b]*du)
                win=draw<prob
                wi=w[i,b];wj=w[j,b]
                stake=epsilon*(wi if wi<wj else wj)
                sigma=1.0 if win else -1.0
                # Passive observations: no random draws and no state writes.
                diff=wi-wj
                budgets[k,b,0]+=2.0*sigma*stake*diff+2.0*stake*stake
                budgets[k,b,1]+=2.0*(2.0*prob-1.0)*stake*diff
                budgets[k,b,2]+=2.0*stake*stake
                budgets[k,b,3]+=sigma*stake*du
                budgets[k,b,4]+=(2.0*prob-1.0)*stake*du
                budgets[k,b,6]+=2.0*stake
                if win:
                    w[i,b]=wi+stake;w[j,b]=wj-stake
                else:
                    w[i,b]=wi-stake;w[j,b]=wj+stake
        nev=0
        if mu>0:
            nev=np.random.poisson(n*mu*dt)
            for _ in range(nev):
                i=np.random.randint(0,n)
                oldu=u[i]
                a[i]=h*a[i]+innovation*np.random.normal()
                u[i]=math.erf(a[i]/math.sqrt(2.0))
                du=u[i]-oldu
                for b in range(nb):
                    budgets[k,b,5]+=(w[i,b]-1.0)*du
        for b in range(nb):
            if chis[b]>0:
                rho=math.exp(-chis[b]*dt);rebate=1.0-rho
                gross=0.0
                for i in range(n):
                    gross+=rebate*w[i,b]
                    w[i,b]=rho*w[i,b]+rebate
                budgets[k,b,7]=gross
            for c in range(8):
                budgets[k,b,c]/=n
            budgets[k,b,8]=nev
    return budgets

class _NumbaRNGState(ctypes.Structure):
    # Matches numba 0.65.1's installed _random.c rnd_state_t.
    _fields_=[('index',ctypes.c_int),('mt',ctypes.c_uint32*624),
              ('has_gauss',ctypes.c_int),('gauss',ctypes.c_double),
              ('is_initialized',ctypes.c_int)]

def rng_state() -> dict:
    """Full checkpoint includes the cached Gaussian, omitted by rnd_get_state."""
    if numba.__version__!='0.65.1':
        raise RuntimeError('Full RNG checkpoint ABI tested only with numba==0.65.1; use pinned environment.')
    ptr=_helperlib.rnd_get_np_state_ptr()
    s=_NumbaRNGState.from_address(ptr)
    index,mt=_helperlib.rnd_get_state(ptr)
    if s.index!=index or list(s.mt)!=mt or s.has_gauss not in (0,1) or s.is_initialized!=1:
        raise RuntimeError('Numba RNG ABI validation failed.')
    return dict(index=int(index),mt=mt,has_gauss=int(s.has_gauss),gauss=float(s.gauss),
                is_initialized=1,engine='numba.numpy.MT19937',numba_version=numba.__version__)

def restore_rng(state:dict) -> None:
    if state['numba_version']!=numba.__version__ or numba.__version__!='0.65.1':
        raise RuntimeError('Use numba==0.65.1 for exact checkpoint replay.')
    ptr=_helperlib.rnd_get_np_state_ptr()
    _helperlib.rnd_set_state(ptr,(int(state['index']),list(state['mt'])))
    s=_NumbaRNGState.from_address(ptr)
    s.has_gauss=int(state['has_gauss']);s.gauss=float(state['gauss'])
    s.is_initialized=1
