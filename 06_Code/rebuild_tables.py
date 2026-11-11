#!/usr/bin/env python3
"""Rebuild all 32 main plot tables, without overwriting archived inputs.

Recomputes prepared-root statistics and paired contrasts from existing raw
records. Archived deterministic kinetic outputs are input, not regenerated here.
Fig1d reads its archived six-root summary; Fig2f preserves the archived bootstrap
CI after independently recomputing its mean. Exact enumeration is re-executed.
The report states these distinctions explicitly.
"""
from __future__ import annotations
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import argparse, importlib.util, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]

def load(path,name):
 sys.path.insert(0,str(path.parent));sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m

def ci(a):
 a=np.asarray(a,float);a=a[np.isfinite(a)];n=len(a)
 if n==0:return dict(mean=float('nan'),sd=float('nan'),lo=float('nan'),hi=float('nan'),n=0)
 if n==1:return dict(mean=float(a[0]),sd=float('nan'),lo=float('nan'),hi=float('nan'),n=1)
 mean=float(a.mean());sd=float(a.std(ddof=1));hw=float(t.ppf(.975,n-1)*sd/np.sqrt(n))
 return dict(mean=mean,sd=sd,lo=mean-hw,hi=mean+hw,n=n)

def compare_frames(a,b):
 if set(a.columns)!=set(b.columns):raise AssertionError(f'columns {set(a.columns)^set(b.columns)}')
 a=a[b.columns]
 # Known file generators can emit a different row order.  The tuple of
 # non-statistical keys provides a reproducible ordering for these tables.
 possible=['mode','process','comparison','metric','lambda_value','lambda','lam','lambda_abs','chi','round','rounds','time','tw','tau','N','seed','wealth','population_fraction','lag','ratio_before']
 keys=[k for k in possible if k in a.columns]
 if keys:a=a.sort_values(keys,kind='stable').reset_index(drop=True);b=b.sort_values(keys,kind='stable').reset_index(drop=True)
 if len(a)!=len(b):raise AssertionError(f'rows {len(a)} != {len(b)}')
 maxerr=0.
 for c in a.columns:
  if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
   x=a[c].to_numpy(float);y=b[c].to_numpy(float)
   if not np.array_equal(np.isnan(x),np.isnan(y)):raise AssertionError('NaNs '+c)
   mask=np.isfinite(x)&np.isfinite(y)
   if mask.any():maxerr=max(maxerr,float(np.max(abs(x[mask]-y[mask]))))
   if not np.allclose(x,y,atol=5e-11,rtol=2e-11,equal_nan=True):raise AssertionError(f'numeric mismatch {c}: {np.nanmax(abs(x-y))}')
  elif not np.array_equal(a[c].astype(str),b[c].astype(str)):raise AssertionError('text mismatch '+c)
 return {'rows':len(a),'max_abs_error':maxerr}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'10_Reproduced/tables');a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 work=out/'work/figures_core'
 if work.exists():raise FileExistsError(f'Use a new --out directory (workspace exists): {work}')
 shutil.copytree(ROOT/'07_Simulations/figures_core',work,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbc','*.nbi'))
 logs=out/'logs';logs.mkdir(exist_ok=True)
 def run(rel):
  with (logs/(Path(rel).stem+'.log')).open('w') as f:
   subprocess.run([sys.executable,str(work/rel)],cwd=work/Path(rel).parent,stdout=f,stderr=subprocess.STDOUT,check=True)
 run('packages/Fig1/code/prepare_fig1_data.py')
 run('packages/Fig2/code/prepare_fig2_data.py')
 # The current panel 2e uses all four time points, not the older 40-round table.
 m=load(work/'packages/Fig2/code/recompute_panel_e.py','panel_e_aggregate');d=m.D
 m.summarize(pd.read_csv(d/'direct_multitime.csv'),pd.read_csv(d/'neutral_multitime.csv'))
 # Fig3: all ensemble curves from their original per-root measurements.
 f3=work/'packages/Fig3';der=f3/'data/derived';raw=f3/'data/raw'
 long=pd.read_csv(raw/'fig3_longtime_20root_records.csv');rows=[]
 for (lam,tm),g in long.groupby(['lambda','time'],sort=True):
  for metric in ['G','M_pct','E_pct']:
   z=ci(g[metric]);rows.append(dict(lambda_value=lam,time=tm,metric=metric,mean=z['mean'],sd=z['sd'],ci95_low=z['lo'],ci95_high=z['hi'],n_roots=z['n'],N=10000))
 pd.DataFrame(rows).to_csv(der/'abc_selection_time_series.csv',index=False)
 stage=ROOT/'07_Simulations/stages0_6/02_stages/Stage4'
 full=pd.read_csv(stage/'results/stage4_main80.csv');mod=load(stage/'code/stage4_analyze.py','stage4_stats_only')
 eff=mod.effects(full);summ=mod.summarize_paired(eff)
 e=summ[(summ['mode']=='baseline')&(summ.q==.1)&summ.tw.isin([0,30,50])&(summ.metric=='Cq_lam4')]
 e[['tw','tau','mean','ci_lo','ci_hi','n_roots']].to_csv(der/'e_baseline_two_time_overlap.csv',index=False)
 f=summ[(summ.q==.1)&(summ.tau==30)&(summ.metric=='delta_Cq')]
 f[['mode','tw','mean','ci_lo','ci_hi','n_roots']].to_csv(der/'f_waiting_dependence_delta_C10.csv',index=False)
 b=full[(full['mode']=='baseline')&(full['lambda']==4)&(full.q==.1)&(full.tau==30)]
 rows=[];h=[]
 for tw,g in b.groupby('tw'):
  z=ci(g.M_B50_to_Tq*100);rows.append(dict(tw=tw,mean_pct=z['mean'],low_pct=z['lo'],high_pct=z['hi'],n_roots=z['n']))
  if tw in [0,30,50]:
   x={'tw':tw,'n_roots':len(g)}
   for k,values in [('old_top_retained_pct',100*(1-g.E)),('new_middle_pct',100*(g.E-5*g.M_B50_to_Tq)),('new_bottom_pct',500*g.M_B50_to_Tq)]:
    z=ci(values);x[k]=z['mean'];x[k+'_lo']=z['lo'];x[k+'_hi']=z['hi']
   h.append(x)
 pd.DataFrame(rows).to_csv(der/'g_retagged_bottom_entry.csv',index=False)
 pd.DataFrame(h).to_csv(der/'h_top_decile_origins_summary.csv',index=False)
 # Fig4: recompute science-panel summaries before running supplied analysis.
 raw4=work/'packages/Fig4/data/raw'
 def summary4(df,groups,metrics,rename):
  outrows=[]
  for keys,g in df.groupby(groups,sort=True):
   if not isinstance(keys,tuple):keys=(keys,)
   for metric in metrics:
    z=ci(g[metric]);outrows.append(dict(zip([rename.get(k,k) for k in groups],keys),metric=metric,mean=z['mean'],ci95_lo=z['lo'],ci95_hi=z['hi'],n=z['n']))
  return pd.DataFrame(outrows)
 late=pd.read_csv(raw4/'late40_N10000.csv');full4=pd.read_csv(raw4/'full40_N10000.csv')
 # Source late table includes five metrics, but main figures use only G and M.
 summary4(late,['tw','chi','tau'],['G','M','E','Cq','Ruw'],{}).to_csv(raw4/'late_retag_summary_40roots.csv',index=False)
 summary4(full4,['lambda_value','chi','time'],['G','M','E','Cq','Ruw','Q','Mmiddle','source_B50'],{'lambda_value':'lam'}).to_csv(raw4/'summary_42conditions_40roots_time.csv',index=False)
 run('packages/Fig4/code/prepare_science_panels.py')
 run('packages/Fig4/code/analyze_validation.py')
 run('packages/Fig4/code/analyze_refinement.py')
 run('scripts/build_plot_tables.py')
 target=out/'Main';target.mkdir(exist_ok=True);results={}
 for frozen in sorted((ROOT/'04_Plot_Data/Main').glob('*.csv')):
  fresh=work/'tables'/frozen.name
  if not fresh.exists():raise FileNotFoundError(fresh)
  results[frozen.name]=compare_frames(pd.read_csv(fresh),pd.read_csv(frozen))
  shutil.copy2(fresh,target/fresh.name)
 report={'status':'PASS','main_tables':len(results),'results':results,'scope':[
 'Prepared-root means and intervals, paired contrasts, current 2e path covariances, first-round enumeration and advantage covariance were recomputed.',
 'Fig3d/e deterministic theory checkpoints are archived inputs, not new kinetic solves.',
 'Fig1d uses archived six-root CI summary; Fig2f preserves archived bootstrap intervals and independently checks the mean.',
 'Fig4 final validation was recomputed from all archived 96-future files and theory/grid-refinement files, not resimulated.',
 'CSV output equals archived main tables within stated floating-point tolerances; no plotting data were silently replaced.']}
 (out/'table_rebuild_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
