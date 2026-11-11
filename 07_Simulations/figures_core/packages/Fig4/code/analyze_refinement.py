#!/usr/bin/env python3
"""Resolution-only selection: never select a theory by its distance to ABM.
All 42 conditions have a uniform .01/96 baseline. Neutral conditions use the
exact score-integrated .0025 wealth grid, and weak-selection/Stage-6 whole-trajectory controls
use .005/96. Both baseline and final acceptance tables are preserved.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd
from analyze_validation import load,stats,KEY,GROUP,TOLS
R=Path(__file__).resolve().parents[1];D=R/'data/derived'
base=pd.read_csv(D/'matched_theory_ABM_by_root.csv')
ref=pd.concat([load(p) for p in (R/'data/raw/refinement').glob('*.csv')],ignore_index=True)
ref.to_csv(D/'refinement_all_roots.csv',index=False)
fine=pd.read_csv(D/'theory_all_roots_grids.csv');fine=fine[np.isclose(fine.dx,.01)&(fine.na==96)]
for df in [base,fine,ref]:df['chi']=df.chi.round(8);df['lambda_value']=df.lambda_value.round(8)
scalar=ref[np.isclose(ref.dx,.01)&(ref.na==1)].merge(fine[fine.lambda_value==0],on=KEY,suffixes=('_scalar','_full'))
assert len(scalar)==180,len(scalar)
reduction_error={m:float(abs(scalar[m+'_scalar']-scalar[m+'_full']).max()) for m in TOLS}
assert max(reduction_error.values())<1e-10,reduction_error
selected=ref[((ref.lambda_value==0)&np.isclose(ref.dx,.0025)&(ref.na==1)) |
             ((ref.lambda_value>0)&np.isclose(ref.dx,.005)&(ref.na==96))].copy()
assert len(selected)==12*6*5,len(selected)
assert not selected.duplicated(KEY).any()
base=base.set_index(KEY).sort_index()
base['final_dx']=.01;base['final_score_bins']=96
for row in selected.itertuples(index=False):
 k=(row.root,row.lambda_value,row.chi,row.round)
 for m in TOLS:base.loc[k,m+'_theory']=getattr(row,m)
 base.loc[k,'final_dx']=row.dx;base.loc[k,'final_score_bins']=row.na
base=base.reset_index();base.to_csv(D/'final_matched_theory_ABM_by_root.csv',index=False)
rows=[]
for key,g in base.groupby(GROUP):
 assert len(g)==6
 for m,tol in TOLS.items():
  s=stats(g[m+'_theory']-g[m+'_abm'])
  status='supported' if s['ci_lo']>=-tol and s['ci_hi']<=tol else ('outside' if s['ci_lo']>tol or s['ci_hi']<-tol else 'inconclusive')
  rows.append(dict(lambda_value=key[0],chi=key[1],round=key[2],time=key[2]/10,metric=m,tolerance=tol,status=status,
                   theory_mean=g[m+'_theory'].mean(),abm_mean=g[m+'_abm'].mean(),
                   dx=g.final_dx.iloc[0],score_bins=g.final_score_bins.iloc[0],**s))
a=pd.DataFrame(rows);a.to_csv(D/'final_acceptance_all_conditions_times.csv',index=False)
a[a['round']==300].to_csv(D/'final_acceptance_endpoints42.csv',index=False)
# Additional halving shifts, reported without inflating root count.
shifts=[]
for label,coarse,fin in [
 ('neutral_.01_to_.005',ref[(ref.na==1)&np.isclose(ref.dx,.01)],ref[(ref.na==1)&np.isclose(ref.dx,.005)]),
 ('neutral_.005_to_.0025',ref[(ref.na==1)&np.isclose(ref.dx,.005)],ref[(ref.na==1)&np.isclose(ref.dx,.0025)]),
 ('selected_.01_to_.005',fine[fine.lambda_value>0],ref[(ref.na==96)&np.isclose(ref.dx,.005)])]:
 z=coarse.merge(fin,on=KEY,suffixes=('_coarse','_fine'))
 for key,g in z.groupby(GROUP):
  for m in TOLS:
   v=g[m+'_fine']-g[m+'_coarse']
   shifts.append(dict(refinement=label,lambda_value=key[0],chi=key[1],round=key[2],metric=m,
                      mean_shift=float(v.mean()),max_abs_root_shift=float(abs(v).max()),n_roots=len(g)))
pd.DataFrame(shifts).to_csv(D/'additional_mesh_halving.csv',index=False)
q=a[a['round']>0];end=a[a['round']==300]
summary=dict(uniform_fine_baseline_preserved=True,final_policy='finest computed spatial resolution per complete condition; no selection by metric/time agreement',
             scalar_reduction_max_error=reduction_error,
             all_time_counts=q.groupby(['metric','status']).size().unstack(fill_value=0).to_dict('index'),
             endpoint_counts=end.groupby(['metric','status']).size().unstack(fill_value=0).to_dict('index'),
             max_mean_abs_error={m:float(abs(q[q.metric==m]['mean']).max()) for m in TOLS},
             max_CI_extent={m:float(np.maximum(abs(q[q.metric==m].ci_lo),abs(q[q.metric==m].ci_hi)).max()) for m in TOLS},
             all_four_times_supported_conditions={m:int(q[q.metric==m].groupby(['lambda_value','chi']).status.apply(lambda x:(x=='supported').all()).sum()) for m in TOLS})
(R/'validation/FINAL_ACCEPTANCE_SUMMARY.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2));print(q[q.status!='supported'].to_string(index=False))
