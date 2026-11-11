#!/usr/bin/env python3
"""Rebuild the exact fixed initial wealth vector used for Fig. 2e."""
from pathlib import Path
import sys, numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'deps'))
from stage2_abm import prepare
from legacy_engine import gini_numba
ROOT=Path(__file__).resolve().parents[1]
full,_,_,prep_rounds=prepare(10000,20012345,.65)
idx=np.random.default_rng(20001).permutation(len(full))[:1000]
w0=full[idx].astype(float)
w0*=len(w0)/w0.sum()
np.save(ROOT/'data'/'w0_fixed.npy',w0)
np.savetxt(ROOT/'data'/'w0_fixed.csv',w0,delimiter=',',header='wealth',comments='')
print('prep_rounds',prep_rounds,'Gini',gini_numba(w0),'w0_mean',w0.mean())
