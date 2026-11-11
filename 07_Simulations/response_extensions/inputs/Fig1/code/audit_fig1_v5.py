#!/usr/bin/env python3
"""Independent checks on Fig. 1 data provenance, endpoints and output files."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from PIL import Image

R=Path(__file__).resolve().parents[1]; D=R/'data'; F=R/'figures'; V=R/'validation'
rr=pd.read_csv(D/'fig1_stageA_40roots_selected.csv')
assert set(rr['lambda'].unique())=={0.,4.}
assert len(rr.seed.unique())==40
assert len(rr.groupby(['seed','lambda']))==80
z=rr.loc[rr['round'].eq(0)]
p=z.pivot(index='seed',columns='lambda',values=['Gini','bottom_to_top_fraction','top_exit_fraction'])
maxinit=max(float(np.max(np.abs(p[c][0.]-p[c][4.]))) for c in ['Gini','bottom_to_top_fraction','top_exit_fraction'])
assert maxinit<1e-14, maxinit
assert np.allclose(rr.Gini.between(0,1),True)
assert rr.top_exit_fraction.between(0,1).all()
res=pd.read_csv(D/'fig1_stageA_40roots_mean_ci.csv')
for metric,val0,val4 in [('G',.8015885432517184,.8436249006727099),('E',47.74575,67.70825),('M',.37545,3.3918)]:
 a=res[(res.time==30)&(res.metric==metric)].set_index('lambda')['mean']
 assert abs(a[0]-val0)<1e-6 and abs(a[4]-val4)<1e-6
red=pd.read_csv(D/'fig1_full_redraw_differences_6roots16futures.csv')
assert red['n_roots'].eq(6).all() and red['futures_per_root'].eq(16).all()
max_ratio=float(np.max((abs(red.mean_difference)+red.ci95_half)/red.equivalence_bound))
assert max_ratio<1
lor=pd.read_csv(D/'fig1_initial_lorenz_6roots.csv')
assert (lor.mean_cumulative_wealth_fraction.diff().fillna(0)>=-1e-11).all()
assert (lor.mean_cumulative_wealth_fraction<=lor.population_fraction+1e-10).all()
assert abs(1-2*np.trapezoid(lor.mean_cumulative_wealth_fraction,lor.population_fraction)-.65052)<.0002
im=Image.open(F/'Fig1_white_v5_600dpi.png')
assert im.size[0]>7000 and im.size[1]>4000,im.size
# The plotted d-axis limits are fixed only for presentation and do not alter CSVs.
assert max(abs(red[red.metric.eq('G')].display_difference))<4.5
assert max(abs(red[red.metric.eq('E')].display_difference))<.42
script=(R/'code/plot_fig1_v5.py').read_text()
assert 'set_ylim(-4.5,4.5)' in script and 'set_ylim(-.42,.42)' in script
assert 'sp.set_visible(True)' in script
paths=[p for d in ['source','data'] for p in (R/d).rglob('*') if p.is_file()]
paths += [R/'code/prepare_fig1_data.py',R/'code/plot_fig1_v5.py',R/'code/audit_fig1_v5.py']
paths += list((R/'figures').glob('Fig1_white_v5*'))
manifest={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
(V/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
r={'status':'PASS','n_independent_ABM_roots':40,'n_full_redraw_independent_roots':6,
   'future_histories_per_redraw_root':16,'maximum_stageA_t0_branch_difference':maxinit,
   'maximum_redraw_95ci_over_equivalence_bound':max_ratio,
   'figure_png_size_pixels':im.size,'sha256_files':len(manifest)}
(V/'AUDIT_RESULT.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print(json.dumps(r,indent=2))
