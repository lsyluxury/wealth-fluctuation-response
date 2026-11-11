#!/usr/bin/env python3
"""Summarize ONLY the 20 development roots and non-reserved parameters."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
R=Path(__file__).resolve().parents[1];cfg=json.loads((R/'config/experiment_config.json').read_text())
KEYS=['stage','target_G','lambda_value','chi','elapsed_time']
METRICS=['gini','B50_T10','Etop1','Q','rho_s','variance','bottom50_share','top1_share','top10_share',
 'B50_mean_wt','B50_median_wealth','B50_q05_wealth','B50_q95_wealth','B50_share_t','B50_loss_fraction','B50_individual_change_median',
 'Middle49_mean_wt','Middle49_median_wealth','Middle49_loss_fraction','T1_mean_wt','T1_median_wealth','T1_loss_fraction','T1_remain_T10','T1_fall_B50',
 'Etop5','Etop10','fixed_partition_between_variance','fixed_partition_within_variance','threshold_T10_inclusive','threshold_T1_inclusive']

def summarize(df):
    rows=[]
    for key,g in df.groupby(KEYS,sort=True):
        assert g.seed.nunique()==len(g)==20
        for m in METRICS:
            x=g[m].to_numpy();sd=float(np.std(x,ddof=1));hw=float(t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x)))
            mean=float(np.mean(x))
            rows.append(dict(zip(KEYS,key),metric=m,n_independent_roots=len(x),mean=mean,sd=sd,ci95_low=mean-hw,ci95_high=mean+hw))
    return pd.DataFrame(rows)

def main():
    d=pd.concat([pd.read_csv(R/f'data/development/population_{s}.csv.gz') for s in cfg['developer_seeds']],ignore_index=True)
    assert d.seed.nunique()==20 and not d.parameter_heldout.any()
    out=R/'summaries/development';out.mkdir(parents=True,exist_ok=True)
    summary=summarize(d);summary.to_csv(out/'summary.csv',index=False)
    differences=[]
    merge=['seed','stage','target_G','elapsed_time']
    for stage,pcol in [('selection','lambda_value'),('tax','chi')]:
        x=d[d.stage==stage]
        control=x[x[pcol]==0][merge+METRICS]
        q=x.merge(control,on=merge,validate='many_to_one',suffixes=('','_control'))
        for m in METRICS:q[m]=q[m]-q[m+'_control']
        differences.append(q[['seed']+KEYS+METRICS])
    paired=summarize(pd.concat(differences,ignore_index=True))
    paired.to_csv(out/'paired_contrasts.csv',index=False)
    endpoints=summary[summary.elapsed_time==30]
    endpoints.to_csv(out/'endpoint_summary_long.csv',index=False)
    endpointwide=d[d.elapsed_time==30].groupby(KEYS,as_index=False)[METRICS].mean()
    endpointwide['n_independent_roots']=20
    endpointwide.to_csv(out/'endpoint_means_wide.csv',index=False)
    d[d.elapsed_time==30].to_csv(out/'endpoint_runs.csv',index=False)
    plan=pd.read_csv(R/'validation/PRE_RUN_PRECISION_PLAN.csv')
    lines=['# 开发集结果预览','',
      '**只包含原20个开发根种子和非保留参数。新增40个根种子及保留参数结果没有进入这些汇总。**',
      '这不是60种子总体结果，也不是动力学预测验收。A/B群体分别在各阶段起点标记；下面M和E以百分比表示。','',
      '## G*=0.65，阶段终点t=30','',
      '|阶段|λ|χ|Gini|B50→T10（%）|T1退出（%）|起点B50财富下降者（%）|',
      '|---|---:|---:|---:|---:|---:|---:|']
    for _,z in endpointwide[endpointwide.target_G==.65].iterrows():
        lines.append(f"|{z.stage}|{z.lambda_value:g}|{z.chi:g}|{z.gini:.6f}|{100*z.B50_T10:.4f}|{100*z.Etop1:.4f}|{100*z.B50_loss_fraction:.3f}|")
    lines += ['','## 解释限制','',
       '“财富下降者”是相对自身起点，不是相对同期控制的处理伤害比例。不得把“固定群体平均财富提高”写成“多数人变富”。',
       'summary.csv给出逐点Student-t区间，paired_contrasts.csv给出同一根种子与同期控制的配对差区间。没有多重比较修正；不能仅据若干点的p值选择阈值。',
       '60种子精度投影只基于先前试算方差，最宽的T1退出配对差95%半宽约0.428个百分点。主矩阵完成不等于所有0.2个百分点的理论工作目标已经可判定。','']
    (out/'DEVELOPMENT_PREVIEW_CN.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='complete',source='development_only',root_count=20,metric_rows=len(d),
          summarized_conditions=summary[KEYS].drop_duplicates().shape[0],summary_rows=len(summary),paired_rows=len(paired)),indent=2))
    print(endpointwide[(endpointwide.target_G==.65)&(((endpointwide.stage=='selection')&endpointwide.lambda_value.isin([0,1,4,8,16]))|
                ((endpointwide.stage=='tax')&endpointwide.chi.isin([0,.01,.03,.1])))][['stage','lambda_value','chi','gini','B50_T10','Etop1']].to_string(index=False))
if __name__=='__main__':main()
