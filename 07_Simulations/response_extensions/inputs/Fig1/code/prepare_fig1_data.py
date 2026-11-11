#!/usr/bin/env python3
"""Rebuild journal Fig. 1 numeric inputs from archived ORIGINAL files.

Run from anywhere: python code/prepare_fig1_data.py
Only uses existing sources; no Monte Carlo simulations or invented observations.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT/'source'; OUT = ROOT/'data'; OUT.mkdir(exist_ok=True)

raw = pd.read_csv(SRC/'original_Fig2_stageA_40seeds.csv')
subset = raw.loc[raw.lambda_value.isin([0.0,4.0]), ['seed','lambda_value','chi','round','time','G','M','E']].copy()
assert subset['seed'].nunique()==40
assert subset['lambda_value'].nunique()==2
assert subset.groupby(['lambda_value','round']).size().eq(40).all()
assert subset.chi.eq(0).all()
subset = subset.rename(columns={'lambda_value':'lambda','G':'Gini','M':'bottom_to_top_fraction','E':'top_exit_fraction'})
subset['bottom_to_top_pct'] = subset['bottom_to_top_fraction']*100
subset['top_exit_pct'] = subset['top_exit_fraction']*100
subset.to_csv(OUT/'fig1_stageA_40roots_selected.csv', index=False, float_format='%.15g')

rows=[]
for (lam, rnd, time),grp in subset.groupby(['lambda','round','time'],sort=True):
    for name,col,units in [('G','Gini','fraction'), ('M','bottom_to_top_pct','percent'), ('E','top_exit_pct','percent')]:
        xx=grp[col].to_numpy(float)
        n=len(xx); sd=xx.std(ddof=1) if n>1 else 0
        ci=t.ppf(.975,n-1)*sd/np.sqrt(n)
        rows.append({'lambda':lam,'round':rnd,'time':time,'metric':name,
                     'n_roots':n,'mean':xx.mean(),'sd':sd,'ci95_low':xx.mean()-ci,
                     'ci95_high':xx.mean()+ci,'units':units})
summary=pd.DataFrame(rows)
summary.to_csv(OUT/'fig1_stageA_40roots_mean_ci.csv',index=False, float_format='%.15g')

lor=pd.read_csv(SRC/'original_initial_lorenz_6seeds.csv')
lor[['population_fraction','mean_cumulative_wealth_fraction']].to_csv(OUT/'fig1_initial_lorenz_6roots.csv', index=False, float_format='%.15g')

redraw=pd.read_csv(SRC/'original_SI_full_redraw_equivalence.csv')
assert redraw['sample'].nunique()==1 and redraw['sample'].iloc[0]=='combined96'
assert redraw['n_roots'].eq(6).all() and redraw['futures_per_root'].eq(16).all()
redraw=redraw.loc[redraw.metric.isin(['G','E'])].copy()
redraw['display_factor']=np.where(redraw.metric=='G',1e4,100.)
redraw['display_unit']=np.where(redraw.metric=='G','1e4 * (G_lam - G_0)','percentage points')
redraw['display_difference']=redraw.mean_difference*redraw.display_factor
redraw['display_ci95_half']=redraw.ci95_half*redraw.display_factor
redraw['display_ci95_low']=redraw.display_difference-redraw.display_ci95_half
redraw['display_ci95_high']=redraw.display_difference+redraw.display_ci95_half
redraw.to_csv(OUT/'fig1_full_redraw_differences_6roots16futures.csv',index=False,float_format='%.15g')

at30=summary.query('time==30 and metric in ["G","E","M"]').pivot(index='lambda',columns='metric',values='mean')
checks={'stageA_at_t30':{str(i):{k:float(v) for k,v in at30.loc[i].items()} for i in [0.,4.]},
        'stageA_N':100000, 'stageA_n_roots':40, 'stageA_n_rounds':300,
        'full_redraw_n_roots':6, 'full_redraw_future_per_root':16,
        'max_95ci_relative_equivalence_bound':float(((redraw.mean_difference.abs()+redraw.ci95_half)/redraw.equivalence_bound).max()),
        'redraw_all_95ci_inside_original_equivalence_margin':bool(((redraw.mean_difference.abs()+redraw.ci95_half) <= redraw.equivalence_bound).all()),
        'lorenz_start_Gini_derived':float(1 - 2*np.trapezoid(lor.mean_cumulative_wealth_fraction,lor.population_fraction)),
        'source_filenames':['original_approved_Fig1_art.png','original_initial_lorenz_6seeds.csv','original_Fig2_stageA_40seeds.csv','original_SI_full_redraw_equivalence.csv']}
assert abs(checks['stageA_at_t30']['0.0']['G']-0.8015885)<1e-5
assert abs(checks['stageA_at_t30']['4.0']['G']-0.8436249)<1e-5
assert abs(checks['stageA_at_t30']['0.0']['E']-47.74575)<1e-4
assert abs(checks['stageA_at_t30']['4.0']['E']-67.70825)<1e-4
assert checks['redraw_all_95ci_inside_original_equivalence_margin']
(ROOT/'validation'/'numeric_audit.json').write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')
print('prepared',*[p.name for p in sorted(OUT.glob('*.csv'))],sep='\n - ')
print(json.dumps(checks,indent=2,ensure_ascii=False))
