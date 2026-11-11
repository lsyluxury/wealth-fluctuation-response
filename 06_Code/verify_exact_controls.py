#!/usr/bin/env python3
"""Independent finite-state checks and algebraic/data consistency checks."""
from __future__ import annotations
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:os.environ[k]='1'
import argparse,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def load(path,name):
 sys.path.insert(0,str(path.parent));s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'10_Reproduced/tests');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();results={}
 m=load(ROOT/'07_Simulations/figures_core/packages/Fig2/code/first_round_exact.py','first_round_control')
 law0,_,_=m.enumerate_first_round(0);qrows=[]
 for lam in [0,.1,4,16]:
  law,mar,q=m.enumerate_first_round(lam);err=max(abs(law.get(k,0)-law0.get(k,0)) for k in set(law)|set(law0));qa=49/1500*np.tanh(.7*lam)
  assert err<1e-13 and abs(q-qa)<1e-13 and abs(sum(law.values())-1)<1e-13
  qrows.append(dict(lambda_value=lam,delta_Q_enumerated=q,delta_Q_analytic=float(qa),wealth_law_max_error=err))
 pd.DataFrame(qrows).to_csv(a.out/'first_round_exact.csv',index=False);results['first_round']={'status':'PASS','results':qrows}
 two=load(ROOT/'07_Simulations/figures_core/scripts/verify_two_round_exact.py','two_round_control');rows=[]
 for n in [2,4,6,8,10,12]:
  mats=np.array(list(two.matchings(list(range(n)))),dtype=np.int64);coef,count,score=two.integrate(mats,n,.14);exact=n/(n-1)*.14*.86/24
  assert abs(coef-exact)<2e-12 and abs(score)<1e-14
  rows.append(dict(N=n,enumerated_coefficient_per_rho=float(coef),formula=float(exact),enumerated_paths=int(count),absolute_error=float(abs(coef-exact))))
 pd.DataFrame(rows).to_csv(a.out/'two_round_exact.csv',index=False);results['two_round']={'status':'PASS','results':rows,'summed_paths':sum(x['enumerated_paths'] for x in rows)}
 data=ROOT/'04_Plot_Data/Main';curve=pd.read_csv(data/'Fig2_a_generated_Q_curve.csv');qerr=float(np.max(abs(curve.delta_Q_analytic-49/1500*np.tanh(.7*curve.lambda_value))));assert qerr<1e-13
 h=pd.read_csv(data/'Fig3_h_composition.csv');g=pd.read_csv(data/'Fig3_g_retagged_entry.csv');both=h.merge(g,on='tw');cerr=float(np.max(abs(both.new_bottom_pct-5*both.mean_pct)));assert cerr<1e-10
 sums=h[['old_top_retained_pct','new_middle_pct','new_bottom_pct']].sum(axis=1);assert abs(sums-100).max()<1e-10
 ratio=pd.read_csv(data/'Fig4_a_ratio_map.csv');err=np.max(abs(ratio.ratio_after-(ratio.rho*ratio.y_initial+1-ratio.rho)/(ratio.rho*ratio.x_initial+1-ratio.rho)));assert err<1e-12
 results['archived_table_identities']={'status':'PASS','Q_curve_max_abs_error':qerr,'composition_max_abs_error':cerr,'ratio_map_max_abs_error':float(err)}
 results.update(status='PASS',seconds=time.perf_counter()-start,scope='Exact finite-state controls and selected mathematical/data identities; not an ensemble rerun.')
 (a.out/'exact_controls.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
