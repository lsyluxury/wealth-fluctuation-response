"""Unbiased path-covariance estimator for weak-selection memory response.

sigma[k,i]: +1 for win, -1 for loss; matching is complete each round.
S_d = sum_i sum_{k=0}^{n-1-d} sigma[k,i]*sigma[k+d,i].
K_X(n,d) = 1/4 Cov_0(X_n, S_d); C_u(d) is raw score covariance.
"""
from __future__ import annotations
import numpy as np

def S_by_lag_direct(signs, max_lag=None):
    signs=np.asarray(signs)
    if signs.ndim==3:
        if signs.shape[2]!=1:raise ValueError('Use one unbiased trajectory per call')
        signs=signs[:,:,0]
    rounds,n=signs.shape
    max_lag=rounds-1 if max_lag is None else int(max_lag)
    assert 0 <= max_lag < rounds
    return np.array([np.einsum('ij,ij->', signs[:-d].astype(np.float64), signs[d:].astype(np.float64), optimize=True) for d in range(1,max_lag+1)])

def S_by_lag_fft(signs, max_lag=None, block=4000):
    signs=np.asarray(signs)
    if signs.ndim==3:
        if signs.shape[2]!=1:raise ValueError('Use one unbiased trajectory per call')
        signs=signs[:,:,0]
    rounds,n=signs.shape
    max_lag=rounds-1 if max_lag is None else int(max_lag)
    assert 0<=max_lag<rounds
    nfft=1 << (2*rounds-2).bit_length()
    accum=np.zeros(nfft//2+1,dtype=np.float64)
    for a in range(0,n,block):
        f=np.fft.rfft(signs[:,a:min(n,a+block)].astype(np.float64),n=nfft,axis=0)
        accum+=np.sum(f.real*f.real+f.imag*f.imag,axis=1)
    result=np.fft.irfft(accum,n=nfft)
    return result[1:max_lag+1]

def response_kernel_from_roots(X:np.ndarray, S:np.ndarray):
    """Each row is an independent neutral root; returns covariance / 4."""
    X=np.asarray(X,dtype=float);S=np.asarray(S,dtype=float)
    if X.ndim==1:X=X[:,None]
    assert S.ndim==2 and X.shape[0]==S.shape[0] and X.shape[0]>=2
    return ((X-X.mean(0)).T @ (S-S.mean(0))/(X.shape[0]-1)/4).T

def predicted_coeff(K,C):
    C=np.asarray(C,dtype=float)
    return C @ np.asarray(K,dtype=float)
