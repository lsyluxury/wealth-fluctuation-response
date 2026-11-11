#!/usr/bin/env python3
"""Pre-production regression checks. No holdout outcomes are read."""
from __future__ import annotations
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,math,time
from pathlib import Path
import numpy as np
import engine
from records import ranks,cohorts,metrics,array_hash
from reference import pilot_original as old
R=Path(__file__).resolve().parents[1]

def main():
    rec=[]
    for n,seed,lam,chi in [(2000,301,[0.,.1,.25,.5,.75,1.,1.5,2.,3.,4.,6.,8.,12.,16.],[0.]*14),
        (2000,302,[4.]*13,[0.,.001,.002,.003,.005,.0075,.01,.015,.02,.03,.04,.06,.1]),
        (100000,303,[0.,4.,16.],[0.,0.,0.])]:
        la=np.array(lam);ch=np.array(chi)
        w0,a0,u0,p0=old.initialise(n,seed,len(la))
        ww,aa,uu,pp=engine.initialise(n,seed,len(la))
        assert all(np.array_equal(x,y) for x,y in zip((w0,a0,u0,p0),(ww,aa,uu,pp)))
        old.reseed(seed+100);old.continue_states(w0,a0,u0,p0,la,ch,np.array([0,1,3,20]),.14,.3,1.,10)
        oldrng=engine.rng_state()
        engine.reseed(seed+100);engine.advance_chunk(ww,aa,uu,pp,la,ch,20,.14,.3,1.,10)
        nrng=engine.rng_state()
        assert all(np.array_equal(x,y) for x,y in zip((w0,a0,u0,p0),(ww,aa,uu,pp)))
        assert oldrng==nrng
        rec.append(dict(test='enhanced_vs_original_final_full_state_and_rng',N=n,branches=len(la),rounds=20,exact_equal=True))
    # Chunked checkpoint: complete RNG restoration, including cached normal.
    w,a,u,p=engine.initialise(2000,987,3);la=np.array([0.,4.,16.]);ch=np.zeros(3)
    engine.advance_chunk(w,a,u,p,la,ch,7,.14,.3,1.,10)
    checkpoint=(w.copy(),a.copy(),u.copy(),p.copy());rng=engine.rng_state()
    engine.advance_chunk(w,a,u,p,la,ch,13,.14,.3,1.,10)
    fw,fa,fu,fp=[x.copy() for x in (w,a,u,p)]; finalrng=engine.rng_state()
    w,a,u,p=[x.copy() for x in checkpoint];engine.restore_rng(rng)
    engine.advance_chunk(w,a,u,p,la,ch,13,.14,.3,1.,10)
    assert all(np.array_equal(x,y) for x,y in zip((w,a,u,p),(fw,fa,fu,fp)))
    assert engine.rng_state()==finalrng
    rec.append(dict(test='checkpoint_resume_full_rng',exact_equal=True,cached_normal_flag_at_checkpoint=rng['has_gauss']))
    # Duplicated and reversed branches preserve every state.
    outputs=[]
    for la in (np.array([0.,4.,16.,4.]),np.array([4.,16.,4.,0.])):
        w,a,u,p=engine.initialise(2000,457,4)
        engine.advance_chunk(w,a,u,p,la,np.zeros(4),15,.14,.3,1.,10)
        outputs.append(w.copy())
    assert np.array_equal(outputs[0],outputs[1][:,::-1]);assert np.array_equal(outputs[0][:,1],outputs[0][:,3])
    rec.append(dict(test='duplicate_and_branch_order_invariance',exact_equal=True))
    # Neutral preparation remains exactly identical, including first crossing.
    n=2000;seed=2026101201;targets=np.array([.5,.65,.8])
    ow,oa,ou,op,orr,og=old.prepare_neutral(n,seed,targets,.14,.3,1.,10)
    w,a,u,p=engine.initialise(n,seed,1);w=w[:,0].copy();j=0
    for rnd in range(1,10001):
        engine.neutral_round(w,a,u,p,.14,.3,1.,10);g=engine.gini_numba(w)
        if g>=targets[j]:
            assert rnd==orr[j] and g==og[j]
            assert all(np.array_equal(x,y) for x,y in zip((w,a,u,p),(ow[j],oa[j],ou[j],op[j])))
            j+=1
        if j==3:break
    rec.append(dict(test='neutral_first_crossing_complete_state',N=n,targets=3,exact_equal=True))
    # Budgets reproduce actually observed Q and variance, not only expectations.
    la=np.array([0.,4.,16.]);ch=np.array([0.,.01,.1]);n=2000
    w,a,u,p=engine.initialise(n,1011,3);w0=w[:,0].copy();uinit=u.copy()
    q=np.array([np.dot(w[:,b]-1,u)/n for b in range(3)]);v=np.var(w,axis=0)
    bd=engine.advance_chunk(w,a,u,p,la,ch,70,.14,.3,1.,10)
    for br in bd:
        q=np.exp(-ch/10)*(q+br[:,3]+br[:,5]);v=np.exp(-2*ch/10)*(v+br[:,0])
    qerr=float(np.max(np.abs(q-np.array([np.dot(w[:,b]-1,u)/n for b in range(3)]))))
    verr=float(np.max(np.abs(v-np.var(w,axis=0))))
    assert qerr<1e-11 and verr<1e-10
    rec.append(dict(test='exact_round_budget_reconstruction',rounds=70,Q_max_error=qerr,variance_max_error=verr))
    # Reuse a non-tied common starting state for metric comparison.
    base,r0=old.state_stats(ow[1],ou[1]);c=cohorts(r0)
    d1=old.metrics(ow[1],ow[2],ou[2],r0,base)
    d2=metrics(ow[1],ow[2],ou[2],r0,base['gini'],c)
    errors={k:abs(d1[k]-d2[k]) for k in d1}
    assert max(errors.values())<1e-10
    rec.append(dict(test='all_existing_metric_definitions',fields=len(errors),max_abs_error=max(errors.values())))
    # Tax-only ranking and Gini contraction.
    n=2000;w0=ow[1];rho=math.exp(-.03/10);wt=rho*w0+1-rho
    assert np.array_equal(ranks(w0),ranks(wt))
    assert abs(engine.gini_numba(wt)-rho*engine.gini_numba(w0))<1e-12
    rec.append(dict(test='pure_tax_rank_invariance_gini_contraction',passed=True))
    (R/'validation/PRE_RUN_ENGINE_TESTS.json').write_text(json.dumps(dict(passed=True,tests=rec),indent=2)+'\n')
    print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
