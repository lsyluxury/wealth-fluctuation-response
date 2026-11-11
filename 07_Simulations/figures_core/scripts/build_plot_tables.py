#!/usr/bin/env python3
"""Export every numeric table used by final main figures and full Markdown lists."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];T=R/'tables';A=R/'appendices';rows=[];frames={}
for p in [T,A]:p.mkdir(exist_ok=True)
def emit(fig,panel,label,src,select=None,units='',notes=''):
 p=R/'packages'/f'Fig{fig}'/src;d=pd.read_csv(p)
 if select is not None:d=select(d).copy()
 key=f'Fig{fig}_{panel}_{label}';out=T/(key+'.csv');d.to_csv(out,index=False,float_format='%.17g');frames[key]=d
 rows.append(dict(figure=fig,panel=panel,id=key,table=str(out.relative_to(R)),source=str(p.relative_to(R)),rows=len(d),columns=len(d.columns),column_names='; '.join(d.columns),units=units,notes=notes,sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
emit(1,'b','lorenz','data/fig1_initial_lorenz_6roots.csv',units='population and cumulative wealth fractions')
emit(1,'c','selection','data/fig1_stageA_40roots_mean_ci.csv',lambda d:d[d.metric.isin(['G','E'])&(d.time<=30)],'G dimensionless; E percent','40 roots; N=100000')
emit(1,'d','full_redraw','data/fig1_full_redraw_differences_6roots16futures.csv',units='display G=10000*deltaG; display E=100*deltaE (pp)',notes='6 roots,16 futures/root')
emit(2,'a','marginal','data/a_first_round_marginal.csv',lambda d:d[d.lambda_value.isin([0,4,16])],'probability/cdf')
emit(2,'a','generated_Q_curve','data/a_first_round_Q_curve.csv',units='Q; analytic sampling, not independent realizations')
emit(2,'a','generated_Q_enumeration','data/a_first_round_Q_exact.csv',units='Q')
emit(2,'b','response_parity','data/b_kinetic_vs_ABM.csv',units='M/E coefficients use probabilities, NOT percent')
emit(2,'b','finiteN_inset','data/b_finiteN_exact.csv',units='display y=1000*cG_per_rho')
emit(2,'c','matched_full_memory','data/cd_advantage_autocovariance.csv',lambda d:d[d['mode'].isin([1,2])],'C_u raw covariance')
emit(2,'c','response','data/weak_response_all.csv',lambda d:d[d['mode'].isin([1,2])&(d.metric=='G')],'x=lambda**2;y=1000*mean')
emit(2,'d','matched_lag1_memory','data/cd_advantage_autocovariance.csv',lambda d:d[d['mode'].isin([0,6])],'C_u raw covariance')
emit(2,'d','response','data/weak_response_all.csv',lambda d:d[d['mode'].isin([0,6])&(d.metric=='G')],'x=lambda**2;y=1000*mean')
emit(2,'e','neutral_path_parity','data/panel_e_points.csv',units='display x=1000*prediction;y=1000*mean',notes='16 comparisons;12 distinct predictions;no jitter')
emit(2,'f','higher_order','data/f_higher_order_response.csv',units='relative extra response percent, not percentage points')
for panel,met,end in [('a','G',200),('b','M_pct',100),('c','E_pct',100)]:emit(3,panel,'multilambda','data/derived/abc_selection_time_series.csv',lambda d,m=met,e=end:d[(d.metric==m)&(d.time<=e)],'G dimensionless; M/E percent','20 roots;N=10000')
emit(3,'d','theory_abm','data/derived/d_theory_vs_ABM_5lambda_to100_mode0.csv',units='G dimensionless;M/E percent',notes='single representative theoretical initial state vs20-root ABM;not matched-root certification')
emit(3,'e','overlap_ABM','data/derived/e_baseline_two_time_overlap.csv',lambda d:d[d.tau.isin([0,1,2,5,10,20,30])],'normalized Cq','80 roots')
emit(3,'e','overlap_theory','data/derived/e_reference_kinetic_single_initial_state.csv',units='Cq dimensionless; E/M probability',notes='one theoretical initial state')
emit(3,'f','waiting_response','data/derived/f_waiting_dependence_delta_C10.csv',units='deltaC10=C10(lambda4)-C10(lambda0);tau=30')
emit(3,'g','retagged_entry','data/derived/g_retagged_bottom_entry.csv',units='conditional entry percent;future tau=30')
emit(3,'h','composition','data/derived/h_top_decile_origins_summary.csv',units='percent of ALL future top-decile members;sum100')
emit(4,'a','ratio_map','data/derived/a_analytic_ratio_contraction.csv',units='wealth mean1;ratio',notes='redistribution alone;Delta_t=10;x=0.1;analytic not ABM')
for panel,met in [('b','G'),('c','M')]:emit(4,panel,'late_intervention','data/derived/bc_late_common_state_timeseries.csv',lambda d,m=met:d[d.metric==m],'G dimensionless;M percent','tw30;40 roots N10000')
emit(4,'de','joint42','data/derived/de_joint_endpoint_42.csv',units='Gini;M_fraction;M_pct;rank_corr',notes='42 actual nodes;contours interpolate visualization only')
emit(4,'f','interaction','data/derived/f_nonadditivity_paired_40roots.csv',units='IG dimensionless;IM pp',notes='paired difference-in-differences')
for panel,met in [('g','G'),('h','M'),('i','E')]:emit(4,panel,'validated_endpoint','data/derived/final_acceptance_endpoints42.csv',lambda d,m=met:d[d.metric==m],'source M/E probability;display multiply100','six roots;16 futures/root')
emit(4,'SI','all_precision_checks','data/derived/final_acceptance_all_conditions_times.csv',units='G dimensionless;M/E probability;630 rows including t0')
cat=pd.DataFrame(rows);cat.to_csv(T/'PLOT_DATA_INDEX.csv',index=False);(T/'PLOT_DATA_INDEX.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
for fig in range(1,5):
 s=[f'# 图{fig}全部绘图数值列表\n','小数在Markdown中显示12位有效数字；同名CSV保存17位。概率、百分比与百分点按下列说明区分。解析采样点不是独立实验。\n']
 for rec in rows:
  if rec['figure']!=fig:continue
  s += [f"## {rec['id']}\n",f"来源：`{rec['source']}`。行数：{rec['rows']}。单位：{rec['units']}。{rec['notes']}\n",f"CSV：[下载对应表](../{rec['table']})\n",frames[rec['id']].to_markdown(index=False,floatfmt='.12g'),'\n']
 (A/f'Fig{fig}_ALL_PLOTTED_VALUES_CN.md').write_text('\n'.join(s),encoding='utf-8')
schemas=[]
for p in sorted((R/'packages').rglob('*.csv')):
 try:
  d=pd.read_csv(p);schemas.append(dict(path=str(p.relative_to(R)),rows=len(d),columns=list(d.columns),dtypes={k:str(v) for k,v in d.dtypes.items()}))
 except pd.errors.EmptyDataError:schemas.append(dict(path=str(p.relative_to(R)),rows=0,columns=[]))
for p in sorted((R/'packages').rglob('*.npz')):
 with np.load(p,allow_pickle=False) as z:schemas.append(dict(path=str(p.relative_to(R)),arrays={k:dict(shape=list(z[k].shape),dtype=str(z[k].dtype)) for k in z.files}))
for p in sorted((R/'packages').rglob('*.npy')):
 z=np.load(p,allow_pickle=False);schemas.append(dict(path=str(p.relative_to(R)),shape=list(z.shape),dtype=str(z.dtype)))
(T/'ALL_DATA_SCHEMAS.json').write_text(json.dumps(schemas,ensure_ascii=False,indent=2))
print('plot tables',len(rows),'rows',sum(r['rows'] for r in rows),'data files',len(schemas))
