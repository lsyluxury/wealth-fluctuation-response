#!/usr/bin/env python3
"""Audit current figure inputs and key mathematical/representation invariants.
No new large simulations; original figures and scientific data are not altered.
"""
from pathlib import Path
import json,hashlib,sys,ast,re,math
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];P=R/'packages';V=R/'validation';V.mkdir(exist_ok=True)
record={}
# all table inputs and their explicit row counts
idx=pd.read_csv(R/'tables/PLOT_DATA_INDEX.csv')
for row in idx.itertuples(index=False):
 p=R/row.table;d=pd.read_csv(p);assert len(d)==row.rows,(row.id,len(d))
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row.sha256,row.id
record['plot_table_count']=len(idx);record['listed_rows']=int(idx.rows.sum())
# first round law, numerical errors independently checkable from supplied table
q=pd.read_csv(P/'Fig2/data/a_first_round_Q_exact.csv')
err=float(abs(q.delta_Q_enumerated-q.delta_Q_analytic).max());assert err<1e-12
assert q.joint_law_max_error.max()<1e-12
record['first_round_Q_max_error']=err
finite=pd.read_csv(P/'Fig2/data/b_finiteN_exact.csv');anal=finite.N/(finite.N-1)*.14*.86/24
err=float(abs(finite.enumeration_cG_per_rho-anal).max());assert err<2e-12
record['two_round_finiteN_max_error']=err
# Fresh exact evaluation of first-round case using the shipped implementation.
sys.path.insert(0,str(P/'Fig2/code'))
from first_round_exact import enumerate_first_round,analytic_delta_q
base,_,_=enumerate_first_round(0.0);maxlaw=0
for lam in [0,.1,4,16]:
 law,_,val=enumerate_first_round(lam);maxlaw=max(maxlaw,max(abs(law.get(k,0)-base.get(k,0)) for k in set(law)|set(base)))
 assert abs(val-analytic_delta_q(lam))<1e-12
record['fresh_exact_joint_law_max_error']=maxlaw
# Tax: conservation, order, G scaling; analytic overtaking inequality.
w=np.array([.03,.12,.45,1.1,3.3]);w=w/w.mean();rho=math.exp(-.03/10);wp=rho*w+1-rho
g=lambda z:float(np.abs(z[:,None]-z).sum()/(2*len(z)**2*z.mean()))
assert abs(wp.mean()-1)<1e-14 and np.array_equal(np.argsort(w),np.argsort(wp))
assert abs(g(wp)-rho*g(w))<1e-14
ratios=np.linspace(1.001,3,101);x=.1;y=ratios*x
assert np.array_equal((1.14*x>y-.14*x),(ratios<1.28))
record['pure_redistribution_G_scale_error']=abs(g(wp)-rho*g(w))
# Fig2e recalculate summaries directly from path records using shipped code.
import importlib.util
sp=importlib.util.spec_from_file_location('handoff_e',P/'Fig2/code/recompute_panel_e.py');mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod)
direct=pd.read_csv(P/'Fig2/data/direct_multitime.csv');neutral=pd.read_csv(P/'Fig2/data/neutral_multitime.csv')
assert len(direct)==48000 and len(neutral)==64000
supplied=pd.read_csv(P/'Fig2/data/panel_e_points.csv');tmp=V/'e_summary_audit';tmp.mkdir(exist_ok=True)
original_D=mod.D;mod.D=tmp
recalc=mod.summarize(direct,neutral);mod.D=original_D
cols=['prediction','pred_low','pred_high','mean','ci_low','ci_high'];key=['mode','rounds']
x=supplied.merge(recalc,on=key,suffixes=('_a','_b'),validate='1:1')
err=max(float(abs(x[c+'_a']-x[c+'_b']).max()) for c in cols);assert err<1e-11
record['Fig2e_summary_recalculation_max_error']=err
p1=supplied[supplied['mode']==1].sort_values('rounds');p2=supplied[supplied['mode']==2].sort_values('rounds')
assert np.allclose(p1.prediction,p2.prediction,atol=0,rtol=0)
record['Fig2e_actual_comparisons']=len(supplied);record['Fig2e_distinct_prediction_abscissae']=supplied.prediction.nunique()
# Figure3: probabilities and composition denominators.
h=pd.read_csv(P/'Fig3/data/derived/h_top_decile_origins_by_root.csv')
cols=['old_top_retained_pct','new_middle_pct','new_bottom_pct']
err=float(abs(h[cols].sum(axis=1)-100).max());assert err<1e-9
assert np.allclose(h.new_bottom_pct,500*h.M_B50_to_Tq_lam4,atol=1e-10)
assert np.allclose(h.old_top_retained_pct,100*(1-h.E_lam4),atol=1e-10)
record['Fig3h_composition_sum_error']=err
# Complete Fig3d recalculation, when individual condition tests have been run.
parts=sorted(V.glob('Fig3d_theory_lambda*.csv'))
if len(parts)==5:
 d=pd.concat([pd.read_csv(p) for p in parts]);old=pd.read_csv(P/'Fig3/data/derived/d_theory_vs_ABM_5lambda_to100_mode0.csv')
 c=d.merge(old,on=['lambda_value','time','metric'],suffixes=('_new','_old'),validate='1:1');assert len(c)==255
 err=float(abs(c.theory_mean_new-c.theory_mean_old).max());assert err<1e-11
 record['Fig3d_recomputed_rows']=255;record['Fig3d_recomputed_max_error']=err
 d.to_csv(V/'Fig3d_recomputed_theory_all.csv',index=False)
else:record['Fig3d_recompute_scope']='Partial or not executed; inspect individual files'
# Figure4: exact counts and criterion, not merely non-significance.
a=pd.read_csv(P/'Fig4/data/derived/final_acceptance_all_conditions_times.csv');assert len(a)==630
positive=a[a['round']>0];assert len(positive)==504
inside=(positive.ci_lo>=-positive.tolerance)&(positive.ci_hi<=positive.tolerance)
assert inside.all() and positive.status.eq('supported').all()
record['Fig4_nonzero_time_tests']=504;record['Fig4_supported']=int(inside.sum())
record['Fig4_t30_points']=len(a[a['round']==300]);record['Fig4_statistics_unit']='6 initial states;16 futures averaged within initial state;pointwise df5 intervals'
# Current package files needed for plot reproduction exist; no font distribution.
for i in range(1,5):
 for suffix in ['_preview.png','_600dpi.png','.pdf','.svg']:assert (R/'figures'/f'Fig{i}{suffix}').is_file()
font_files=[str(p.relative_to(R)) for p in R.rglob('*') if p.suffix.lower() in ['.ttf','.otf','.woff','.woff2']]
assert not font_files,font_files
# All Python scripts parse; main has no unresolved generation placeholders.
py=list(R.rglob('*.py'))
for p in py:ast.parse(p.read_text())
main=(R/'01_MAIN_HANDOFF_CN.md').read_text();assert not re.findall(r'\{\{[A-Z_]+\}\}',main)
record['python_programs']=len(py);record['source_data_schema_entries']=len(json.loads((R/'tables/ALL_DATA_SCHEMAS.json').read_text()))
record['status']='PASS';record['scope']='Current numerical tables, exact control reruns, Fig2e statistical reconstruction, five-condition Fig3d when present, all-time Fig4 criterion; not a rerun of every historical trajectory'
(V/'HANDOFF_AUDIT.json').write_text(json.dumps(record,ensure_ascii=False,indent=2))
print(json.dumps(record,ensure_ascii=False,indent=2))
if __name__=='__main__':pass
