from __future__ import annotations
import sys,json
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from engine_stage01 import advance_trace as old
from engine_stage3 import advance_trace as new
from legacy_engine import reseed
ar=Path(__file__).resolve().parents[1]/'phase4'/'inputs'/'s2026101201_g65_selection.npz'
with np.load(ar) as z:w0=z['w'].copy();a0=z['a'].copy();u0=z['u'].copy()
records=[]
for mode in (0,1,2):
 n=len(w0); L=np.array([0.,4.,-4.]); C=np.zeros(len(L));w=np.repeat(w0[:,None],3,axis=1)
 ao=a0.copy() if mode==0 else np.zeros(n);uo=u0.copy() if mode==0 else np.full(n,0.577350269 if mode==2 else 0.21)
 perm=np.arange(n,dtype=np.int64)
 w_old=w.copy();w_new=w.copy();a_old=ao.copy();a_new=ao.copy();u_old=uo.copy();u_new=uo.copy();p_old=perm.copy();p_new=perm.copy()
 reseed(20261104+mode);o=old(w_old,a_old,u_old,p_old,L,C,20,.14,.3,1.,10.,mode,.1,False)
 reseed(20261104+mode);s=new(w_new,a_new,u_new,p_new,L,C,20,.14,.3,1.,10.,mode,.1,False)
 r={'mode':mode,'N':n,'rounds':20,'all_array_equal':all(np.array_equal(x,y) for x,y in ((w_old,w_new),(a_old,a_new),(u_old,u_new),(p_old,p_new))),'max_w_error':float(np.max(abs(w_old-w_new))),'max_u_error':float(np.max(abs(u_old-u_new)))}
 assert r['all_array_equal'],r
 records.append(r)
out=Path(__file__).resolve().parents[1]/'results'/'extension_regression_recheck.json';out.write_text(json.dumps(records,indent=2));print(records)
