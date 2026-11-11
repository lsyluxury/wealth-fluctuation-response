#!/usr/bin/env python3
"""Generate a factual completion handoff only after the full audit has passed."""
from __future__ import annotations
import hashlib,json,datetime
from pathlib import Path
R=Path(__file__).resolve().parents[1]

def main():
    a=json.loads((R/'validation/FINAL_AUDIT.json').read_text());assert a['status']=='PASSED'
    c=json.loads((R/'config/experiment_config.json').read_text())
    comp=[json.loads(p.read_text()) for p in (R/'logs').glob('completed_*.json')]
    assert len(comp)==60
    cfgsha=hashlib.sha256((R/'config/experiment_config.json').read_bytes()).hexdigest()
    h=hashlib.sha256()
    for name in ['engine.py','records.py','run_experiments.py']:
        h.update(name.encode());h.update((R/'src'/name).read_bytes())
    codesha=h.hexdigest()
    assert {x['config_sha256'] for x in comp}=={cfgsha} and {x['source_sha256'] for x in comp}=={codesha}
    status=dict(date='2026-10-14',phase_1='completed_and_verified',phase_2_main_matrix='completed_and_verified',
      phase_2_long_windows='not_started_by_predefined_stage3_gate',
      independent_root_count=60,phase_trajectories=4860,full_metric_records=68040,round_budget_records=1458000,
      common_anchors=360,joint_distribution_files=1512,microstate_files=40,branch_microstate_snapshots=120,
      developers=list(c['developer_seeds']),heldout_roots=list(c['holdout_seeds']),
      holdout_parameters=c['heldout_parameters'],holdout_outcomes_unblinded=False,
      holdout_QA='automatic numerical/structural checks only',
      config_sha256=cfgsha,simulation_source_sha256=codesha,
      not_performed=['independent_density_solver','new_analytic_predictions','holdout_theory_validation','t50_t100_extensions','final_main_figures'],
      follow_on='Stage 3: finite-stake total and fixed-cohort density dynamics; start with G*=.65 lambda=0/4 chi=0/.03.',
      completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (R/'PHASE12_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    reg=a['original_pilot_regression'];snap=a['independent_snapshots'];q=a['max_errors']
    text=f'''# 新版财富交换研究：阶段一／二执行报告

日期：2026-10-14。

## 完成结论

**阶段一已完成并核验；阶段二的60种子主矩阵已全部实际运行并通过数据验收。不是只交付待运行程序。** 主矩阵没有缩小N、减少种子或用已有结果填充。旧研究协议下的图点和数据没有进入本轮结果。

50/100时间单位的延伸按路线中“阶段三代表条件验证后再做次要扩展”的限制暂未启动。完整动力学、解析预测及四张正式主图不在本轮声称的完成范围。

## 1. 实际规模

|项目|实际完成|
|---|---:|
|人口规模N|100000|
|独立根种子|60，原20+新增40|
|中性准备背景|G*=0.50、0.65、0.80|
|A能力选择强度|14个λ点|
|B税收强度|13个χ点，固定λ=4|
|每阶段主窗口|30单位=300完整配对轮次|
|阶段轨迹|4860|
|完整指标观察时点|14个|
|种子—条件—时点记录|68040|
|分支—轮次收支记录|1458000|
|完整A/B共同起点|360|
|代表联合分布文件|1512|
|完整微观快照文件|40，每文件3个代表分支|
|可独立复算的分支快照|120|

独立样本数是60，不是4860、68040或1458000。三种背景出自同一根种子的同一中性轨迹；同阶段多个分支和同轨迹多个时间点也不是独立重复。原360条试算分支没有额外加入本轮样本数。

模型参数保持ε=0.14、r=10、μ=1、h=0.3。A的λ=[0,.1,.25,.5,.75,1,1.5,2,3,4,6,8,12,16]；B的χ=[0,.001,.002,.003,.005,.0075,.01,.015,.02,.03,.04,.06,.10]。不是λ×χ全交叉扫描。

完整指标时点为0、0.1、0.2、0.3、0.5、1、2、3、5、10、15、20、25、30。每轮另记录实际交易收支、条件期望收支和能力更新，而不是只保留终点。

## 2. 阶段一已完成的工作

协议、参数矩阵、根种子和未来流、群体定义、控制分支、时间网格、保存层级、开发/保留区及数值阈值均由正式运行前的机器配置固定。配置SHA256：`{cfgsha}`。

A起点是对应中性准备状态；B起点是对应A的λ=4、t=30完整终点，保留财富—能力联合状态。A/B分别固定自己的初始群体，B效果只与B的χ=0比较。

新增记录没有更改微观规则或随机抽样顺序。共同起点保存w、a、u、ID、排列、初始排名、群体标签和完整随机数状态。正态随机数缓存也被保存，并通过真实N=100000的断点重放检查。

增强程序还保留固定群体财富均值/分位数、财富减少者比例、当前群体份额、Top1%—10%退出与去向、Q和Spearman，防止把排名流动直接等同于财富受益。

## 3. 数据和数值核验

|核验|结果|
|---|---|
|主矩阵缺项/重复条件|0|
|跨分支共同起点、B继承A完整终点|全部通过|
|与此前新版试算2160条记录对照|{reg['field_comparisons']}个数值字段，最大绝对差{reg['max_absolute_error']:.4g}|
|此前新版保存的30个微观数组|逐元素完全一致|
|独立统计实现复算|{snap['branch_snapshots']}个分支快照，{snap['metric_checks']}项指标；最大绝对差{snap['max_absolute_error']:.4g}|
|独立复算最大归一化差|{snap['max_normalized_error']:.4g}|
|财富守恒最大相对误差|{q['mass']:.4g}|
|负财富观测、零财富观测、分位切点并列|全部为0|
|逐轮Q收支最大重建残差|{q['Q_budget']:.4g}|
|逐轮方差收支最大重建残差|{q['variance_budget']:.4g}|
|固定三群体方差重组最大残差|{q['partition_variance']:.4g}|
|联合直方图人口、财富和CDF一致性|全部通过|
|完整随机状态恢复与单分支重放|通过|

此前试算对照只针对新版协议同条件。该检查证明扩展参数和记录没有改变已有轨迹，不是用旧协议模拟替代新版计算。独立快照统计采用另写的统计函数；整套ABM没有另写第二套完全独立随机模拟器重复60种子，因此不作这种声称。

每轮收支使用实际σ和条件期望m分别记录。实际Q/方差重建通过，不等于总体密度动力学或Gini/群体转移的独立理论预测已经通过。

## 4. 开发区和保留区

20个开发根种子：2026101201—2026101220。新增40个保留根种子：2026101401—2026101440。

预留参数λ=1.5、6、12，χ=.0075、.015、.06没有进入开发汇总，即使其根种子属于开发组。新增40根种子的科学结果没有做跨种子汇总或解释；只运行自动完整性、守恒、有效范围和事件数等质量核查。

因此，`summaries/development/`中的均值、置信区间和配对差**只代表20个开发根种子与非保留参数**，不能标成60种子结果。保留区是流程隔离，并未加密或第三方托管；后续必须先冻结求解器和预测，再解盲。

主数据的机械合并表保留为两份：development共17640条，holdout_locked共50400条。合并没有计算保留区的干预均值或效应。

## 5. 开发集的代表结果：不是60种子正式统计

下表G*=.65，终点t=30；仅用20个开发根种子。M和E以百分比表示。

|阶段|λ|χ|Gini|B50→T10（%）|T1退出（%）|
|---|---:|---:|---:|---:|---:|
|A|0|0|0.801589|0.3736|76.670|
|A|1|0|0.815881|1.2904|81.880|
|A|4|0|0.843743|3.3889|88.935|
|A|8|0|0.848893|3.7599|89.990|
|A|16|0|0.850592|3.8905|90.230|
|B|4|0|0.890680|0.1302|61.935|
|B|4|.01|0.772310|3.9331|65.800|
|B|4|.03|0.670911|7.3034|79.735|
|B|4|.10|0.538112|9.8237|98.250|

新增参数填补了弱选择区间和更宽税收范围；这些是开发数据的描述，不提供任意初态或无限时间的普遍结论。A/B行不能直接相减当税收效应，因为两阶段在各自起点重新定义群体。

逐点Student-t区间和同根种子配对差见开发汇总CSV。无多重比较修正；没有为使曲线符合预期而删除任何参数点。

## 6. 当前优势与风险

优势是三项主指标之外，已具有早期逐轮收支、完整共同初态、固定群体联合分布和可精确重放的随机流。后续可以从同一输入检验理论，而不是继续追加只有终点的曲线。

风险是尾部排名事件仍有抽样不确定度。正式运行前，仅用此前20种子方差投影，60种子最宽的T1退出配对差95%区间半宽约0.428个百分点。这意味着60种子**不能保证所有条件都能分辨0.2个百分点的预测误差**。该限制已保存，不用增设时间点或分支来虚增独立样本，也不把数据完整性验收混同预测精度验收。

对本轮完成范围、数值复现和数据一致性的置信度：高。完整理论预测、其他准备历史和长时间的适用性：本轮未检验。

## 7. 下一阶段入口

下一步应执行阶段三，从G*=.65、A的λ=0/4、B的χ=0/.03开始建立有限金额的总体与固定群体动力学。先使用开发起点、开发数据和已冻结协议，不读取保留结果调参。

阶段三轻量包包括前6个开发种子在G*=.65的完整A/B起点、对应初态联合分布、开发统计及全部接口说明。完整归档分为CORE、开发起点、代表分布/微观态和两个保留起点包。大数据包与CORE解压到同一上级目录即可合并。

具体文件：`docs/PROTOCOL_V1.md`、`config/experiment_config.json`、`config/seed_registry.csv`、`config/run_registry.csv`、`docs/metric_dictionary.md`、`docs/prediction_validation_plan.md`、`validation/FINAL_AUDIT.json`、`PHASE12_STATUS.json`。
'''
    (R/'docs/PHASE12_COMPLETION_REPORT_CN.md').write_text(text)
    (R.parent/'WEALTH_PHASE12_COMPLETION_REPORT_CN_20261014.md').write_text(text)
    print(json.dumps(status,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
