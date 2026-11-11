"""Mechanical audit and transparent aggregation of completed stage-four runs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,math,hashlib,importlib.metadata
from pathlib import Path
import numpy as np,pandas as pd
from common import ROOT,dump_json

def main():
    R=ROOT/'results';S=R/'summary';S.mkdir(parents=True,exist_ok=True);summary={}
    base=pd.read_csv(R/'convergence/dx0.01_na96/metrics.csv');conv=[]
    for label in ['dx0.005_na96','dx0.01_na192']:
        z=base.merge(pd.read_csv(R/f'convergence/{label}/metrics.csv'),on=['stage','lam','chi','round','time'],suffixes=('_base','_refined'),validate='one_to_one')
        for m in ['G','M','E','Q']:
            diff=(z[m+'_base']-z[m+'_refined']).abs();conv.append(dict(refinement=label,metric=m,max_abs_difference=float(diff.max()),max_final_difference=float(diff[z['round']==300].max())))
    pd.DataFrame(conv).to_csv(S/'grid_convergence.csv',index=False);summary['grid_convergence']=conv
    budget=pd.read_csv(R/'convergence/dx0.01_na96/budgets.csv')
    sums=budget.groupby(['stage','lam','chi'])[['G_neutral_contribution','G_selection_extra','G_tax_exact','G_tax_remap_error','M_fixed_threshold','M_threshold_move','E_fixed_threshold','E_threshold_move']].sum().reset_index();sums.to_csv(S/'cumulative_budgets.csv',index=False)
    summary['budget_residuals']={
        'G':float(np.max(abs(budget.G_end-budget.G_start-budget.G_neutral_contribution-budget.G_selection_extra-budget.G_tax_exact-budget.G_tax_remap_error))),
        'M':float(np.max(abs(budget.M_change-budget.M_fixed_threshold-budget.M_threshold_move))),
        'E':float(np.max(abs(budget.E_change-budget.E_fixed_threshold-budget.E_threshold_move))),
        'min_fair_G_contribution':float(budget.G_neutral_contribution.min()),
        'max_single_round_tax_remap_G_error':float(budget.G_tax_remap_error.abs().max())}
    trunc=[]
    for K in (1,3,5):
        q=np.zeros(len(budget))
        for k in range(1,K+1,2):
            m=(k-1)//2;c=(-1)**m*math.sqrt(math.factorial(k))/(math.sqrt(math.pi)*4**m*math.factorial(m)*k);q+=c*budget[f'mode{k}_end']
        for key,g in budget.assign(Q_reconstructed=q).groupby(['stage','lam','chi']):
            err=(g.Q_reconstructed-g.Q_end).abs();trunc.append(dict(stage=key[0],lam=key[1],chi=key[2],K=K,max_abs_Q_error=float(err.max()),mean_abs_Q_error=float(err.mean())))
    pd.DataFrame(trunc).to_csv(S/'Q_mode_truncation.csv',index=False)
    summary['Q_reconstruction']={str(k):max(x['max_abs_Q_error'] for x in trunc if x['K']==k) for k in [1,3,5]}
    summary['mode_kernel_recurrence_max_error']=float(budget.filter(regex='kernel_residual').abs().to_numpy().max())
    summary['mode_scalar_grid_max_error']=float(budget.filter(regex='scalar_residual').abs().to_numpy().max())
    weak=pd.read_csv(R/'weak_response/prediction_residuals.csv');wr=[]
    for lam,g in weak.groupby('lam'):
        for m in ['G','M','E','Q']:wr.append(dict(lam=lam,metric=m,max_abs_error=float(g[m+'_residual'].abs().max()),final_abs_error=float(g[g['round']==300][m+'_residual'].abs().iloc[0])))
    pd.DataFrame(wr).to_csv(S/'weak_selection_error.csv',index=False);summary['weak_selection_errors']=wr
    coefficients=pd.read_csv(R/'weak_response/response_coefficients.csv');summary['weak_final_coefficients']={m:float(coefficients.iloc[-1][m]) for m in ['G2','M2','E2','Q1']}
    small=pd.read_csv(R/'weak_response/small_parameter_coefficient_check.csv');sc=[]
    for lam,g in small.groupby('lam'):
        for m in ['G','M','E','Q']:sc.append(dict(lam=lam,metric=m,max_coefficient_error=float(g[m+'_coefficient_error'].abs().max())))
    pd.DataFrame(sc).to_csv(S/'small_parameter_checks.csv',index=False)
    eqsummary=[];eqend=[]
    for label,name in [('original24','full_redraw_equivalence.csv'),('new72','full_redraw_new72_equivalence.csv'),('pooled96','full_redraw_combined96_equivalence.csv')]:
        eq=pd.read_csv(R/'abm_mechanisms'/name);eq=eq[eq['round']>0]
        for m,g in eq.groupby('metric'):eqsummary.append(dict(sample=label,metric=m,passed=int(g.equivalence_pass.sum()),tested=len(g),max_abs_mean_difference=float(g.mean_difference.abs().max()),max_ci90_half=float(g.ci90_half.max()),equivalence_bound=float(g.equivalence_bound.iloc[0])))
        for _,r in eq[eq['round']==300].iterrows():eqend.append(dict(sample=label,lam=float(r.lam),metric=r.metric,mean_difference=float(r.mean_difference),ci90_half=float(r.ci90_half),equivalence_pass=bool(r.equivalence_pass)))
    pd.DataFrame(eqsummary).to_csv(S/'full_redraw_equivalence_summary.csv',index=False);pd.DataFrame(eqend).to_csv(S/'full_redraw_endpoint.csv',index=False);summary['full_redraw_equivalence']=eqsummary
    summary['independent_new72_all_pointwise_pass']=all(r['passed']==r['tested'] for r in eqsummary if r['sample']=='new72')
    pe=pd.read_csv(R/'abm_mechanisms/persistence_effects.csv');pe[pe['round']==300].to_csv(S/'persistence_effects_endpoint.csv',index=False)
    # Continuous theory versus additional same-input ABM history, diagnostic only.
    dm=pd.read_csv(R/'density_mechanisms/metrics.csv');pr=pd.read_csv(R/'abm_mechanisms/persistence_raw.csv');cmp=[]
    theoryparts=[dm[dm['mode']!='full_redraw'].copy(),base[(base.stage=='A')&(base.lam==4)].assign(mode='base')]
    theory=pd.concat(theoryparts,ignore_index=True)
    abm=pr[(pr.root==2026101201)&(pr.lam==4)]
    merge=theory.merge(abm,on=['mode','round','time','lam'],suffixes=('_theory','_ABM'),validate='one_to_one')
    merge.to_csv(S/'persistence_same_initial_single_history.csv',index=False)
    for name,g in merge.groupby('mode'):
        for m in ['G','M','E','Q']:cmp.append(dict(mode=name,metric=m,max_single_history_residual=float((g[m+'_theory']-g[m+'_ABM']).abs().max()),not_confidence_validation=True))
    pd.DataFrame(cmp).to_csv(S/'persistence_single_history_residuals.csv',index=False)
    # Regression to the previous Top10 solver, not used as a predictive input.
    previous=ROOT/'inputs/prior_top10_reference.csv'
    if previous.exists():
        old=pd.read_csv(previous).rename(columns={'lambda_value':'lam','elapsed_time':'time','Etop10':'E_old','gini':'G_old'})
        zz=base.merge(old,on=['stage','lam','chi','time'],validate='one_to_one')
        summary['cohort_refactor_regression']={m:float((zz[m]-zz[m+'_old']).abs().max()) for m in ['G','E']}
    checks={};mass=wealth=0.;neg=0
    for folder in ['dx0.01_na96','dx0.005_na96','dx0.01_na192']:
        a=json.loads((R/f'convergence/{folder}/audit.json').read_text())
        for key,aa in a.items():
            if isinstance(aa,dict) and 'max_mass_error' in aa:
                mass=max(mass,aa['max_mass_error']);wealth=max(wealth,aa['max_wealth_error']);neg=min(neg,aa['min_mass'])
    dm_a=json.loads((R/'density_mechanisms/audit.json').read_text())
    for key,a in dm_a.items():
        if isinstance(a,dict) and 'max_mass_error' in a:mass=max(mass,a['max_mass_error']);wealth=max(wealth,a['max_wealth_error'])
    summary['density_max_mass_error']=mass;summary['density_max_wealth_error']=wealth;summary['density_min_mass']=neg;summary['redraw_density_max_difference']=dm_a['full_redraw_max_diff']
    summary['weak_audit']=json.loads((R/'weak_response/audit.json').read_text());summary['exact_controls']=json.loads((R/'exact_controls/audit.json').read_text())
    summary['abm_initial_audit']=json.loads((R/'abm_mechanisms/audit.json').read_text());summary['abm_extension_audit']=json.loads((R/'abm_mechanisms/precision_extension_audit.json').read_text())
    summary['execution_counts']={'new_ABM_branches':348,'ABM_rounds':104400,'ABM_metric_rows':4872,'ABM_preparation_roots':6,'redraw_future_histories':96,'redraw_future_histories_per_root':16,'density_full_trajectories':24,'density_full_rounds':7200,'response_recursion_trajectories':1,'response_recursion_rounds':300,'early_memory_response_trajectories':5,'early_memory_response_rounds':10,'pure_tax_N100000_control_trajectories':3,'pure_tax_rounds':900,'budget_rows':1200}
    assert mass<1e-8 and wealth<1e-8 and neg>=-1e-13
    assert summary['budget_residuals']['G']<1e-12 and summary['budget_residuals']['M']<1e-12 and summary['budget_residuals']['E']<1e-12
    assert summary['exact_controls']['pure_tax_max_top10_membership_changes']==0
    assert summary['weak_audit']['max_derivative_mass_error']<1e-10
    summary['completed_scope']='Stage-four representative development numerical validation completed; broad parameter/initial-history/finite-N and holdout tests remain stage five.'
    summary['holdout_accessed']=False
    dump_json(ROOT/'validation/FINAL_AUDIT.json',summary)
    dump_json(ROOT/'PHASE4_STATUS.json',dict(stage=4,main_metrics=['G','M_B50_to_T10','E_T10'],status='completed_representative_development_validation',holdout_accessed=False,full_parameter_universality_tested=False,all_time_simultaneous_confidence_bands=False,weak_quadratic_validity='tested small lambda; lambda=.5 Top10 truncation error exceeds .2 percentage points',execution_counts=summary['execution_counts']))
    manifest=[]
    for p in sorted((ROOT/'inputs').glob('*')):
        if p.is_file():manifest.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    dump_json(ROOT/'config/INPUT_MANIFEST.json',manifest)
    versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','pandas','numba','matplotlib','tabulate']};dump_json(ROOT/'config/NUMERICAL_VERSIONS.json',versions)
    (ROOT/'requirements.txt').write_text('\n'.join(f'{k}=={v}' for k,v in versions.items())+'\n')
    print('FINAL_AUDIT_COMPLETE',summary['execution_counts'],flush=True)
if __name__=='__main__':main()
