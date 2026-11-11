"""Run small exact legacy compatibility, 100k 300-round anchor replay, new controls."""
import json,sys,time,os,hashlib,math
from pathlib import Path
import numpy as np
import legacy_engine as eng
from engine_stage01 import advance_trace,initialize_process_score,top_membership,cohort_overlap
from legacy_records import ranks, cohorts, metrics

def go(root=str(Path(__file__).resolve().parents[3] / '05_history/phase12/source')):
 root=Path(root);results={}
 s=root/'anchors/development/s2026101201_g65_selection.npz'
 tgt=root/'anchors/development/s2026101201_g65_tax.npz'
 with np.load(s,allow_pickle=False) as z:
  start={k:z[k].copy() for k in ['w','a','u','permutation','rank0']};rng=json.loads(str(z['rng_state_json']))
 with np.load(tgt,allow_pickle=False) as z:
  target={k:z[k].copy() for k in ['w','a','u']}
 n=len(start['w']);trace_time=time.perf_counter()
 w=start['w'][:,None].copy();a=start['a'].copy();u=start['u'].copy();perm=start['permutation'].astype(np.int64)
 eng.restore_rng(rng)
 sig=advance_trace(w,a,u,perm,np.array([4.]),np.array([0.]),300,.14,.3,1.,10,0,0.0,True)
 results['legacy_100k_300round_traced']={'n':n,'time_seconds':time.perf_counter()-trace_time,'wealth_bitwise':bool(np.array_equal(w[:,0],target['w'])),'latent_bitwise':bool(np.array_equal(a,target['a'])),'score_bitwise':bool(np.array_equal(u,target['u'])),'trace_shape':list(sig.shape),'signed_per_round_zero':bool(np.all(sig.sum(axis=1)==0)),'all_winners_losers':bool(np.all(abs(sig)==1)),'total_wealth_error':float(np.max(abs(w.sum(axis=0)-n))), 'minimum_wealth':float(w.min())}
 assert all(results['legacy_100k_300round_traced'][k] for k in ('wealth_bitwise','latent_bitwise','score_bitwise','signed_per_round_zero','all_winners_losers'))
 # 300-round outcome metrics vs archival CSV
 import pandas as pd
 df=pd.read_csv(root/'summaries/development/endpoint_runs.csv')
 r=df[(df.seed==2026101201)&(df.stage=='selection')&(df.target_G==.65)&(df.lambda_value==4)&(df.elapsed_time==30)].iloc[0]
 met=metrics(start['w'],w[:,0],u,start['rank0'],eng.gini_numba(start['w']),cohorts(start['rank0']))
 results['archived_metric_differences']={k:float(met[k]-r[k]) for k in ('gini','Q','B50_T10','Etop10','rho_s')}
 assert max(abs(t) for t in results['archived_metric_differences'].values())<1e-12
 # comparison under 14 legacy selection and tax rates
 for lam,chi in [(0.,0.),(4.,0.),(16.,0.),(4.,.03)]:
  w0=np.linspace(.01,2.0,2000); w0*=len(w0)/sum(w0)
  a0=np.linspace(-1.8,1.8,2000)
  u0=np.array([math.erf(v/math.sqrt(2)) for v in a0]);p0=np.arange(2000)
  w1=w0[:,None].copy();a1=a0.copy();u1=u0.copy();p1=p0.copy()
  w2=w1.copy();a2=a1.copy();u2=u1.copy();p2=p1.copy()
  eng.reseed(30001);eng.advance_chunk(w1,a1,u1,p1,np.array([lam]),np.array([chi]),20,.14,.3,1.,10);state1=eng.rng_state()
  eng.reseed(30001);advance_trace(w2,a2,u2,p2,np.array([lam]),np.array([chi]),20,.14,.3,1.,10,0,0.0,False);state2=eng.rng_state()
  same=all(np.array_equal(v,t) for v,t in zip((w1,a1,u1,p1),(w2,a2,u2,p2))) and state1==state2
  results[f'legacy_short_branch_{lam}_{chi}']={'all_state_and_rng_bitwise':bool(same)}
  assert same
 # all new process modes: strong selection and tax, positivity, normalized overlap
 for mode in (1,2,3,4):
  w0=np.linspace(.02,2.0,2000);w0*=2000/w0.sum()
  a0=np.zeros(2000);u0=np.zeros(2000); perm=np.arange(2000)
  eng.reseed(30101+mode);initialize_process_score(u0,mode)
  if mode==3:u0[:]=np.linspace(-1,1,len(u0)) # Fixed nontrivial advantage, not zero vector
  # two linked branches, with shared ability, random matching, and win uniforms
  w=np.column_stack((w0,w0));a=a0.copy();u=u0.copy()
  sig=advance_trace(w,a,u,perm,np.array([0.,4.]),np.array([.03,.03]),50,.14,.3,1.,10,mode,.2,True)
  c0=top_membership(w0,.1)
  result={'conservation':float(np.max(abs(w.sum(axis=0)-2000))),'min_wealth':float(w.min()),'trace_shape':list(sig.shape),'overlap':cohort_overlap(c0,w[:,1],.1)}
  assert result['conservation']<1e-9 and result['min_wealth']>0
  assert np.all(sig.sum(axis=1)==0) and np.all(abs(sig)==1)
  results[f'new_mode_{mode}']=result
 # exact tax identity: no exchange (manual map) and top identities unchanged
 ww=np.linspace(.01,2.,2000); ww*=2000/ww.sum(); rho=math.exp(-.03/10)
 tag=top_membership(ww,.1)
 after=rho*ww+(1-rho)
 assert np.array_equal(top_membership(after,.1),tag)
 results['tax_rank_preservation']=True
 return results

if __name__=='__main__':
 x=go(sys.argv[1] if len(sys.argv)>1 else str(Path(__file__).resolve().parents[3] / '05_history/phase12/source'))
 path=Path(__file__).resolve().parents[1]/'results/extension_smoke.json';path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(x,indent=2,ensure_ascii=False));print(json.dumps(x,indent=2,ensure_ascii=False))
