#!/usr/bin/env python3
"""Audit sources, exact controls, root statistics, control matches and outputs."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd

R=Path(__file__).resolve().parents[1];S=R/'source';D=R/'data';V=R/'validation';V.mkdir(exist_ok=True)
checks={}
def check(name,value,condition):
    checks[name]={'value':value,'passed':bool(condition)}
def main():
    manifest=json.loads((S/'INPUT_MANIFEST.json').read_text())
    intact=all(hashlib.sha256((R/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in manifest)
    check('input_files_unchanged',len(manifest),intact)
    q=pd.read_csv(D/'a_first_round_Q_exact.csv');oldq=pd.read_csv(S/'first_round_archived.csv')
    j=q.merge(oldq,left_on='lambda_value',right_on='lam')
    err=float(np.max(np.abs(j.delta_Q_enumerated-j.deltaQ_enumerated)))
    check('first_round_matches_archived_Q_max_error',err,err<1e-13)
    err=float(np.max(np.abs(q.delta_Q_enumerated-q.delta_Q_analytic)))
    check('first_round_Q_formula_max_error',err,err<1e-13)
    err=float(q.joint_law_max_error.max());check('first_round_full_joint_law_max_difference',err,err<1e-13)
    marginal=pd.read_csv(D/'a_first_round_marginal.csv')
    masserr=float((marginal.groupby('lambda_value').probability.sum()-1).abs().max())
    check('first_round_probability_mass_error',masserr,masserr<1e-13)
    wide=marginal.pivot(index='wealth',columns='lambda_value',values='cdf')
    err=float(np.max(np.abs(wide.sub(wide[0.0],axis=0).to_numpy())))
    check('first_round_marginal_cdf_difference',err,err<1e-13)
    fn=pd.read_csv(D/'b_finiteN_exact.csv');err=float((fn.enumeration_cG_per_rho-fn.analytic_cG_per_rho).abs().max())
    check('finiteN_enum_formula_max_coefficient_error',err,err<1e-13)
    memory=pd.read_csv(D/'cd_advantage_autocovariance.csv').pivot(index='lag',columns='mode',values='autocovariance')
    same=float((memory[1]-memory[2]).abs().max());one=float(abs(memory.loc[1,0]-memory.loc[1,6]))
    check('uniform_two_state_full_C_difference',same,same<1e-14)
    check('Gaussian_mixed_lag1_C_difference',one,one<1e-13)
    weak=pd.read_csv(D/'weak_response_all.csv');orig=pd.read_csv(S/'stage2_response_summary.csv')
    w=weak.merge(orig,on=['mode','metric','lambda_abs'],validate='one_to_one')
    err=float(max((w['mean']-w.response_quotient).abs().max(),(w.ci_low-w.CI95_lower).abs().max(),(w.ci_high-w.CI95_upper).abs().max(),(w.kinetic_c2-w.kinetic_c2_refined).abs().max()))
    check('weak_statistics_and_kinetic_match_archive',err,len(w)==36 and err<1e-11)
    check('weak_response_independent_roots',sorted(weak.n.unique().tolist()),(weak.n==130).all())
    check('weak_CI_ordering',len(weak),((weak.ci_low<=weak['mean'])&(weak['mean']<=weak.ci_high)).all())
    direct=pd.read_csv(D/'e_neutral_covariance_crosscheck.csv');nr=json.loads((S/'neutral_kernel_online_16000.json').read_text());dr=json.loads((S/'stage2_short_crosscheck_summary.json').read_text())
    errs=[]
    for row in direct.itertuples(index=False):
        key='P1' if row.mode==0 else 'P2'
        if row.method=='Neutral-path covariance':
            ref=nr['modes'][key]['G'];errs.extend([abs(row.mean-ref['c']),abs(row.ci_low-ref['CI_approx'][0]),abs(row.ci_high-ref['CI_approx'][1])])
        else:
            ref=dr[key]['0.25'];errs.extend([abs(row.mean-ref['coefficient']),abs(row.ci_low-ref['CI95'][0]),abs(row.ci_high-ref['CI95'][1])])
    err=float(max(errs));check('40round_covariance_and_ABM_match_archive',err,err<1e-11)
    check('40round_source_sample_counts',direct[['method','n']].to_dict('records'),set(direct.n)=={16000,6000})
    f=pd.read_csv(D/'f_higher_order_response.csv');old=pd.read_csv(S/'strong_bias_bootstrap_archived.csv')
    err=float(np.max(np.abs(f.select_dtypes('number').to_numpy()-old.select_dtypes('number').to_numpy())))
    check('strong_bias_statistics_unchanged',err,err<1e-11)
    if (V/'layout_axes.json').exists():
        layout=json.loads((V/'layout_axes.json').read_text())
        within=all(v['position'][0]>=0 and v['position'][1]>=0 and v['position'][0]+v['position'][2]<=1 and v['position'][1]+v['position'][3]<=1 for v in layout.values())
        check('all_numeric_axes_inside_canvas',len(layout),within)
    artifacts=['Fig2_aligned_v3_preview.png','Fig2_aligned_v3.pdf','Fig2_aligned_v3.svg']
    check('core_outputs_exist',artifacts,all((R/'figures'/name).is_file() for name in artifacts))
    report={'all_passed':all(item['passed'] for item in checks.values()),'checks':checks,
      'scope':'Data, exact-control reconstruction and plot input audit; no new simulations and no full revalidation of the general theorem.',
      'limits':['300-round kinetic coefficients are not direct measurements of all neutral-path lag kernels.','Neutral-path covariance check is 40 rounds and N=1000 with one fixed unequal-wealth initial state.','Shared full autocovariance implies equal quadratic response, not exact finite-bias equality.']}
    (V/'NUMERICAL_AUDIT.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['all_passed']:sys.exit(1)
if __name__=='__main__':main()
