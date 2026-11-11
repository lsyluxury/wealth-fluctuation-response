#!/usr/bin/env python3
"""Audit multi-parameter extended simulations and duplicate stage4 checkpoints."""
from pathlib import Path
import glob,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
x=pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(ROOT/'data/raw/extended_root_*.csv')))],ignore_index=True)
agg=pd.read_csv(ROOT/'data/derived/fig3_abc_longtime_20roots.csv')
C=pd.read_csv(ROOT/'data/derived/fig3_d_waiting_overlap_80roots.csv')
assert len(x.root.unique())==20 and len(x['lambda'].unique())==10
assert len(x['time'].unique())==30
assert x.groupby(['root','time']).size().eq(10).all()
assert x['G'].between(0,1).all()
assert x['M_pct'].between(0,100).all() and x['E_pct'].between(0,100).all()
assert x['min_wealth'].gt(0).all()
mass_err=float(np.max(np.abs(x.wealth_sum-x.N)))
assert mass_err<1e-8
# Original stage4 run, shared first 20 prepared roots, one baseline process:
orig=ROOT/'data/raw/stage4_main80_reference.csv'
old=pd.read_csv(orig)
old=old[(old['mode']=='baseline')&(old['tw']==0)&(old['tau'].isin([0,10,30]))&(old['q']==.1)&(old.root.between(20000000,20000019))]
ss=x[x.time.isin([0,10,30])&x['lambda'].isin([0,4])]
r=ss.merge(old,on=['root','lambda','time'],how='left')
assert len(r)==120 and not r.Cq.isna().any()
diff_G=float(np.max(np.abs(r.G_x-r.G_y)))
diff_M=float(np.max(np.abs(r.M_pct-100*r.M_B50_to_Tq)))
diff_E=float(np.max(np.abs(r.E_pct-100*r.E)))
assert diff_G<1e-12 and diff_M<1e-10 and diff_E<1e-10
assert len(C)==24 and C.n_roots.eq(80).all()
# Compare t=30 to archived original 40-root means, avoiding false claims of identical ensembles.
orig_fig2=ROOT/'data/raw/original_Fig2_dense_40root.csv'
old2=pd.read_csv(orig_fig2)
# Independent ensemble of 40 original N=100000 roots; compare directions
# and representative means, not pathwise equality (N and seed sets differ).
old2 = old2[(old2['round']==300) & (old2['lambda_value'].isin([0,4]))]
comparison = {}
for lam in [0.,4.]:
    a=x[(x['time']==30)&(x['lambda']==lam)][['G','M_pct','E_pct']].mean()
    b=old2[old2['lambda_value']==lam][['G_abm','M_abm','E_abm']].mean()
    comparison[str(lam)]={'G_new':float(a['G']),'G_old':float(b['G_abm']),
        'M_pct_new':float(a['M_pct']),'M_pct_old':float(b['M_abm']*100),
        'E_pct_new':float(a['E_pct']),'E_pct_old':float(b['E_abm']*100)}
report={'n_roots':20,'N':10000,'n_lambda_values':10,'lambda_values':sorted(map(float,x['lambda'].unique())),
   'max_time':1200.,'max_relative_mass_error':mass_err/10000.,
   'original_stage4_replay_diff':{'G':diff_G,'M_percentage_points':diff_M,'E_percentage_points':diff_E},
   'gini_t30':x[x.time.eq(30)].groupby('lambda').G.mean().round(8).to_dict(),
   'gini_t1200':x[x.time.eq(1200)].groupby('lambda').G.mean().round(8).to_dict(),
   'min_wealth':float(x.min_wealth.min()),'comparison_to_original_N100000_40roots_t30':comparison}
(ROOT/'validation'/'audit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print('PASS: long-time and Stage4 replay checks:',report)
