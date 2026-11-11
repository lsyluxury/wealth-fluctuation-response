#!/usr/bin/env python3
"""Numerical audit for Fig4's geometry, root pairing, stage scopes and SI composition."""
from pathlib import Path
import json, hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];RAW=R/'data/raw';D=R/'data/derived';V=R/'validation';V.mkdir(exist_ok=True)
geo=pd.read_csv(D/'a_analytic_ratio_contraction.csv')
assert geo.chi.nunique()==4
assert np.max(geo.ratio_after-geo.ratio_before)<1e-12
assert np.min(geo.ratio_after)>0
assert np.all(geo.groupby('chi')['ratio_after'].apply(lambda x: (np.diff(x)>0).all()))
assert np.isclose(geo['ratio_after'][geo.chi==0].to_numpy(),geo['ratio_before'][geo.chi==0].to_numpy()).all()
assert (geo.ratio_after>=1-1e-12).all()
# fully independent late-stage retagging from same initial complete w,u state
late=pd.read_csv(D/'bc_late_common_state_timeseries.csv')
assert set(late.tw)=={30}
assert set(late.metric)=={'G','M'}
assert set(late.n)=={40}
for met in ['G','M']:
    qs=late[(late.metric==met)&(late.tau==0)]
    assert np.ptp(qs.value)<1e-12,met
lm=late[(late.metric=='M')&np.isclose(late.tau,30)]
late_m0=lm[np.isclose(lm.chi,0)].value.iloc[0];late_m3=lm[np.isclose(lm.chi,.03)].value.iloc[0]
assert abs(late_m0-0.1375)<.002 and abs(late_m3-7.225)<.002
lg=late[(late.metric=='G')&np.isclose(late.tau,30)]
assert abs(lg[np.isclose(lg.chi,0)].value.iloc[0]-.890526)<1e-5
assert abs(lg[np.isclose(lg.chi,.03)].value.iloc[0]-.670317)<1e-5
# joint grid correct definitions, no mixing late restart data
wide=pd.read_csv(D/'de_joint_endpoint_42.csv')
assert len(wide)==42 and wide.lam.nunique()==7 and wide.chi.nunique()==6
assert np.allclose(100*wide.M_fraction,wide.M_pct)
q0=wide[np.isclose(wide.lam,4)&np.isclose(wide.chi,0)].iloc[0]
q3=wide[np.isclose(wide.lam,4)&np.isclose(wide.chi,.03)].iloc[0]
assert abs(q0.Gini-.843480)<2e-6 and abs(q3.Gini-.665203)<2e-6
assert abs(q0.M_pct-3.3545)<.001 and abs(q3.M_pct-8.2835)<.001
assert abs(q3.rank_corr-.334913)<2e-6
# 40-root paired interaction = archived independent result
non=pd.read_csv(D/'f_nonadditivity_paired_40roots.csv')
pub=pd.read_csv(RAW/'rootpaired_contrasts_40roots.csv')
for metric in ['G','M']:
    a=non[(non.metric==metric)&np.isclose(non.chi,.03)].iloc[0]
    b=pub[(pub.type=='interaction')&(pub.metric==metric)].iloc[0]
    f=100 if metric=='M' else 1
    assert abs(a['mean']-b['mean']*f)<1e-11
    assert abs(a.ci95_low-b.ci95_lo*f)<1e-11
    assert abs(a.ci95_high-b.ci95_hi*f)<1e-11
# percentage composition among NEW entrants vs conditional entry M
si=pd.read_csv(D/'SI_new_entrant_composition_lam4.csv')
assert (si.bottom_origin_pct+si.middle_origin_pct).between(99.999999999,100.00000001).all()
checks={'status':'passed','inputs':{},'n_joint_grid':int(len(wide)),'n_roots':40,
'late_tw30_tau30':{'chi0_M_percent':float(late_m0),'chi003_M_percent':float(late_m3)},
'joint_lam4':{'chi0_G':float(q0.Gini),'chi003_G':float(q3.Gini),
             'chi0_M_percent':float(q0.M_pct),'chi003_M_percent':float(q3.M_pct),
             'chi003_Ruw':float(q3.rank_corr)},
'interaction_lam4_chi003':{met:float(non[(non.metric==met)&np.isclose(non.chi,.03)]['mean'].iloc[0]) for met in ['G','M']}}
for p in sorted(RAW.glob('*')):checks['inputs'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
(V/'AUDIT_RESULT.json').write_text(json.dumps(checks,indent=2),encoding='utf8')
print('Fig4 audit passed; source rows confirmed.')
