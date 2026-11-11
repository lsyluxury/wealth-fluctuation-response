#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];D=R/'data/derived';V=R/'validation'
base=json.loads((V/'ACCEPTANCE_SUMMARY.json').read_text())
final=json.loads((V/'FINAL_ACCEPTANCE_SUMMARY.json').read_text())
a=pd.read_csv(D/'final_acceptance_all_conditions_times.csv');p=a[a['round']>0];end=a[a['round']==300]
grid=pd.read_csv(D/'grid_convergence_summary.csv');more=pd.read_csv(D/'additional_mesh_halving.csv')
audits=[json.loads(f.read_text()) for folder in ['theory','refinement'] for f in (R/'data/raw'/folder).glob('*.json')]
integrity={key:max(x.get(key,0) for x in audits) for key in ['max_probability_error','max_mean_wealth_error','max_tag_mass_error','skipped_mass','overflow_mass','overflow_wealth']}
integrity['minimum_density']=min(x.get('min_density',0) for x in audits)
integrity['theory_runs']=len(audits)
(V/'GLOBAL_NUMERICAL_INTEGRITY.json').write_text(json.dumps(integrity,indent=2))
def table_rows(counts):
 rows=[]
 for m in ['G','M','E']:
  z=counts.get(m,{})
  rows.append(f"| {m} | {z.get('supported',0)} | {z.get('inconclusive',0)} | {z.get('outside',0)} |")
 return '\n'.join(rows)
metrics=[]
for m,factor,u in [('G',1,'Gini单位'),('M',100,'个百分点'),('E',100,'个百分点')]:
 metrics.append(f"| {m} | {final['max_mean_abs_error'][m]*factor:.6f} | {final['max_CI_extent'][m]*factor:.6f} | {u} |")
fail=p[p.status!='supported'].copy();fail['factor']=np.where(fail.metric=='G',1,100)
fail['error_display']=fail['mean']*fail.factor;fail['lo_display']=fail.ci_lo*fail.factor;fail['hi_display']=fail.ci_hi*fail.factor
fail.to_csv(D/'not_certified_points.csv',index=False)
lines=[]
for z in fail.itertuples(index=False):lines.append(f"| {z.lambda_value:g} | {z.chi:g} | {z.time:g} | {z.metric} | {z.error_display:+.4f} | [{z.lo_display:+.4f}, {z.hi_display:+.4f}] | {z.status} |")
no_pass='全部504个非零时点检验均满足原定判据。' if len(fail)==0 else f'仍有{len(fail)}个“条件×时点×指标”检验未获完整容差认证，逐项列于第5节；不能称全部时点均已通过。'
report=f'''# Fig.4：42条件动理学高精度验收报告

## 1. 完成范围与判定

已完成**42条件×6个归档初态×300轮**的统一精细网格计算，且全42条件均有中等网格对照。每个初态N=100,000；每个条件的ABM按同一初态重复16条独立未来，先在初态内平均，再以6个独立初态作配对统计。{no_pass}

- 条件：λ∈{{0,0.5,1,2,4,8,12}}，χ∈{{0,0.005,0.01,0.02,0.03,0.06}}。
- 统一比较时间t=5、10、20、30，分别为50、100、200、300轮；t=0只作初态与数值审计。
- G、M、E各168次检验，合计504次。初态不是96个：96是6×16条条件未来，统计独立准备态仍为6个。
- 另保留原40个N=10,000初态的联合干预与晚期实验，用于主图的经济机制结果；不把两套样本混合计算CI。

### 统一精细网格（Δlogw=0.01，96优势格）的原始验收

| 指标 | 通过 | 证据不足 | 明确超出容差 |
|---|---:|---:|---:|
{table_rows(base['test_counts'])}

### 进一步网格复核后的最终验收

| 指标 | 通过 | 证据不足 | 明确超出容差 |
|---|---:|---:|---:|
{table_rows(final['all_time_counts'])}

### 终点t=30的42条件验收

| 指标 | 通过 | 证据不足 | 明确超出容差 |
|---|---:|---:|---:|
{table_rows(final['endpoint_counts'])}

| 指标 | 最大平均绝对偏差 | 最大95%区间端点绝对值 | 单位 |
|---|---:|---:|---|
{chr(10).join(metrics)}

## 2. 完全匹配的模型与初态

每轮不重叠随机配对，交换额εmin(w_i,w_j)，ε=0.14；胜率为logistic[λ(u_i−u_j)]。优势采用原高斯–泊松更新，h=0.3、μ=1、r=10，不替换为均匀重抽。税收返还映射为w⁺=ρw⁻+(1−ρ)，ρ=exp(−χ/10)。

理论直接投影6个归档微观初态的财富、优势及B50/M40/T10标签；保留经验联合分布，不作财富—优势独立分解。理论程序不读取ABM干预后的状态。ABM的第0条未来复用原种子，复现原42条件数据；其余未来使用固定独立种子规则。

## 3. 网格与数值复核

全网格中等分辨率为Δlogw=0.025、48优势格；精细分辨率为0.01、96格。全42条件、六初态、全部观测时间均完成两者对照。

针对弱选择下的顶部退出率网格敏感性，另做：

- 所有λ=0条件：在0.01、0.005、0.0025财富网格上计算。当胜率恒为1/2时，对优势积分与财富/标签演化严格可交换，因此可使用精确的优势积分后方程。0.01网格上与完整96格程序的G/M/E最大差为{max(final['scalar_reduction_max_error'].values()):.3e}。该降维仅用于λ=0，不用于非零选择，也不用于检验R_uw。
- 所有λ=0.5且χ>0的五个条件，以及(λ,χ)=(4,0.03)：全部六初态、完整300轮进一步加密至Δlogw=0.005、96优势格。
- 最终理论结果按每个完整参数条件的最细已计算空间分辨率选取，不按照与ABM的吻合程度选择，也不按时间或指标挑选。统一0.01/96结果完整保留，可单独审查。

两个计算优化均独立核验：平滑优势更新矩阵的低秩表示误差约机器精度；税收后严格不可能被占据的空网格列被删除，未丢弃非零质量。总质量小于1e−18的节点本轮保留原位，累计影响单独记录；这不是把密度重新归一化。

| 全部理论计算的数值审计量 | 最大值/最小值 |
|---|---:|
| 总概率误差 | {integrity['max_probability_error']:.3e} |
| 平均财富误差 | {integrity['max_mean_wealth_error']:.3e} |
| 群体质量误差 | {integrity['max_tag_mass_error']:.3e} |
| 累计微小节点保留质量 | {integrity['skipped_mass']:.3e} |
| 上边界溢出质量 | {integrity['overflow_mass']:.3e} |
| 最小密度 | {integrity['minimum_density']:.3e} |

最小负值约为双精度最小次正规数，属于舍入下溢；未作事后裁剪或重新归一化。守恒误差小不等于G/M/E预测误差小，两类结果分开核查。

## 4. 原定统计验收标准

令d_s=T_s−(1/16)Σ_fA_sf，其中s为独立初态，f为同初态未来。以6个d_s计算平均和自由度5的双侧95% Student-t区间。

G的容差为±0.005；M、E为±0.002（概率），即±0.2个百分点。区间整体落入容差窗才记“通过”。区间整体在窗外记“明确超出”，其余记“证据不足”。置信区间包含0不等于通过等效性检验。

这些是**点态95%区间**，不是全部504项同时95%覆盖。格点差异属于确定性数值诊断，不用增加ABM重复来掩盖，也不把它当作额外独立初态。R_uw只做附加诊断，没有沿用G/M/E的精度认证。

## 5. 未认证项目

| λ | χ | t | 指标 | 平均偏差 | 95%区间 | 判定 |
|---|---|---|---|---:|---|---|
{chr(10).join(lines) if lines else '| — | — | — | — | — | — | 无 |'}

M和E在此表以百分点显示，G为Gini单位。完整630行（含t=0）在`data/derived/final_acceptance_all_conditions_times.csv`；不通过项目单独保存在`not_certified_points.csv`。

## 6. 图4的最终安排

a：保序收缩和一次超越阈值；b/c：共同晚期状态出发的Gini与底部进入；d：42条件的M颜色图与G等值线；e：G–M–R_uw三指标；f：非加性相互作用；g/h/i：全部42条件在终点的独立Theory–ABM对照，分别展示G、M、E。

a为解析示意，b–f仍使用原40根结果。后三个验证面板使用匹配的六个十万人初态和16条条件未来。晚期干预与初始联合干预是不同初态设计；没有把联合扫描理论曲线错误叠加到晚期实验上。新附图展示全部时点的验收比值、网格差异及残差时间曲线。

本参数图是有限时间结果，不是稳态相图；M/E是固定群体的终点占据比例，不是首次通过概率。χ不是现实税率推荐，R_uw不是福利或生产率。
'''
(R/'VALIDATION_REPORT_CN.md').write_text(report)
readme='''# Fig.4 final package — joint-intervention validation

## 先看哪些文件

- `figures/Fig4_final_42validated_preview.png`：完整最终图。
- `figures/Fig4_final_42validated_600dpi.png`、同名PDF/SVG：正式图片。
- `VALIDATION_REPORT_CN.md`：计算覆盖、验收结果、数值精度与适用边界。
- `FIG4_CAPTION_EN_CN.md`：中英图注。
- `data/derived/final_acceptance_all_conditions_times.csv`：逐条件、逐时间、逐指标验收。
- `figures/SI_full_grid_precision.png`：全部时点的精度地图。

## 复现

安装`requirements.txt`中的依赖；运行`replot.sh`（Windows运行`replot.bat`）可从附带的原始数值记录重新计算统计并绘制图片，不再运行模拟。

运行`recompute.sh`会从`inputs/`中的六个原始准备态重新执行两网格全扫描、16条条件未来及额外网格复核，然后重建图表。程序会跳过已经完成且CSV和审计JSON均存在的计算。要全新重算，请先将`data/raw/theory`、`data/raw/futures`、`data/raw/refinement`移到另一个目录。

统计单位始终是6个初态，而非6×16条未来。模型和容差的机器可读定义见`validation/PROTOCOL.json`。全部绘图数据均为CSV；初始微观态保存在NPZ。

## 数据目录

`data/raw/theory`：全42条件、6初态、两套网格的理论观测量与审计。
`data/raw/futures`：每个初态16条未来，42条件共享外生随机数。
`data/raw/refinement`：额外网格减半与λ=0精确降维核对。
`data/derived`：统计量、原始/最终两套验收、格点差与主图数据。
`inputs`：六个归档财富—优势初态。

`code/density_solver.py`与`common.py`保留原算子的定义；`fast_solver.py`是经过独立对照的计算实现。所有脚本使用相对于包根目录的路径。
'''
(R/'README_CN.md').write_text(readme)
print('Reports written')
