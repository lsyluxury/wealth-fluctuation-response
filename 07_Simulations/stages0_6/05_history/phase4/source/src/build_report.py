"""Create the Chinese report from completed numerical outputs."""
from pathlib import Path
import json
import pandas as pd
from common import ROOT

def table(df):return df.to_markdown(index=False)
def main():
    a=json.loads((ROOT/'validation/FINAL_AUDIT.json').read_text());S=ROOT/'results/summary'
    w=pd.read_csv(S/'weak_selection_error.csv').pivot(index='lam',columns='metric',values='max_abs_error').reset_index()
    wt=pd.DataFrame({'λ':w.lam,'Gini最大绝对误差':w.G.map(lambda x:f'{x:.7f}'),'底层进入误差（百分点）':(100*w.M).map(lambda x:f'{x:.5f}'),'前10%退出误差（百分点）':(100*w.E).map(lambda x:f'{x:.5f}')})
    pe=pd.read_csv(S/'persistence_effects_endpoint.csv');pt=pe.pivot(index=['mode','lag_score_corr'],columns='metric',values='mean_effect').reset_index().sort_values('lag_score_corr')
    names={'fast_mu5':'μ=5，h=0.3','sparse_h0':'μ=1，h=0','base':'原模型：μ=1，h=0.3','persistent_h09':'μ=1，h=0.9','frozen_mu0':'能力固定：μ=0'}
    ptable=pd.DataFrame({'能力更新条件':pt['mode'].map(names),'一轮能力分数相关':pt.lag_score_corr.map(lambda x:f'{x:.6f}'),'ΔGini':pt.G.map(lambda x:f'{x:+.6f}'),'Δ底层进入（百分点）':(100*pt.M).map(lambda x:f'{x:+.3f}'),'Δ前10%退出（百分点）':(100*pt.E).map(lambda x:f'{x:+.3f}')})
    eq=pd.read_csv(S/'full_redraw_equivalence_summary.csv');equ=[]
    for sample in ['original24','new72','pooled96']:
        row={'样本':{'original24':'最初24条未来历史','new72':'新增72条未来历史','pooled96':'合计96条未来历史'}[sample]}
        for m in ['G','M','E']:
            z=eq[(eq['sample']==sample)&(eq.metric==m)].iloc[0];row[m]=f'{int(z.passed)}/{int(z.tested)}'
        equ.append(row)
    cv=pd.read_csv(S/'grid_convergence.csv');cvt=[]
    for label in cv.refinement.unique():
        g=cv[cv.refinement==label].set_index('metric');cvt.append({'细化对照':{'dx0.005_na96':'财富网格 Δlog(w)：0.01→0.005','dx0.01_na192':'能力分箱：96→192'}[label],'Gini最大变化':f'{g.loc["G","max_abs_difference"]:.7f}','底层进入最大变化（百分点）':f'{100*g.loc["M","max_abs_difference"]:.5f}','前10%退出最大变化（百分点）':f'{100*g.loc["E","max_abs_difference"]:.5f}'})
    text=f'''# 财富交换新版研究：第四阶段数值验证报告

日期：2026-10-14。

## 结论与完成边界

**计算环境已恢复。本轮实际完成了第四阶段代表条件下的解析响应求解、机制对照、收支核对与前10%群体的数值收敛检查。** 不是仅提供待运行程序，也没有把第三阶段旧结果当成本轮新增计算。

主要发现：选择作用取决于能力优势能否跨轮持续；每轮全员独立重抽能力后，选择不再改变财富统计；小选择强度的响应系数现在可以由方程直接算出，而非从模拟拟合；再分配当场不改变顶层身份，但会压缩财富差距并改变后续交易条件。正的财富—能力关联不是Gini上升的普遍充分条件。

**这是开发条件验证，不是40个保留根种子的盲测。** 正文指标为G、M_B50→T10、E_T10；前1%仍作为尾部稳健性指标。Top10的更换发生于开发分析之后、保留集解盲之前，已写入协议修订。

## 1. 实际运行规模与对象

| 项目 | 实际完成 |
|---|---:|
| 人口规模 | N=100000，不缩小 |
| 原参数 | ε=.14，r=10；基准μ=1，h=.3 |
| A/B主窗口 | 每阶段300轮，即30模型时间单位 |
| 新ABM参数分支轨迹 | 348条、104400轮 |
| 新ABM观察记录 | 4872条，14个观察时点 |
| 全员重抽未来历史 | 6个准备态×16个未来随机流=96条；每条λ=0/4/16三个分支 |
| 持续性切片 | 6个准备态×5种更新条件×λ=0/4，共60分支 |
| 独立完整密度轨迹 | 24条、7200轮 |
| 二阶响应递推 | 1条300轮；另5种持续性条件各2轮检查 |
| 纯税收人口控制 | 3条N=100000、300轮控制 |
| Gini/群体逐轮收支 | 1200条 |

**独立准备态只有6个，不是96个或348个。** 新增未来随机历史用于减小条件动态波动，统计区间先在同一准备根内对未来重复取均值，再按6个根做配对t推断。没有把不同分支或时间点当作独立人口样本。

主密度和网格诊断采用第一开发初态2026101201；ABM机制切片采用2026101201—2026101206。A/B仍连续求解：B从理论A λ=4终点重标记B50/M40/T10，不读取模拟B起点或期末门槛。新输入文件哈希见`config/INPUT_MANIFEST.json`。

弱选择递推另有明确初态：保留同一准备态的财富及群体财富边缘，重新组成P_0(w)φ(a)。这是检验中性独立初态命题所需的概率分布；**不是声称单个经验能力样本与财富完全独立**。该区别在所有响应数据中保留。

## 2. 每轮全员重抽能力：解析、密度与ABM三层一致

### 2.1 精确枚举与密度求解

四人枚举显示，独立能力初态下首轮财富律对λ不变，最大概率差{a['exact_controls']['first_round_max_law_error']:.3e}；首轮Q注入恒等式误差{a['exact_controls']['first_round_max_Q_error']:.3e}。每轮重新抽能力时，两轮财富律与公平交易的总变差距离不超过{a['exact_controls']['two_round_full_redraw_max_TV']:.3e}。保留能力时，两轮财富律出现非零差异，详见`finite_population_two_round.csv`。

同一财富初态，独立密度程序分别推进λ=0、4、16各300轮并在每轮重新因子化能力。三项指标相对λ=0的最大差为：

- G：{a['redraw_density_max_difference']['G']:.3e}；
- M：{a['redraw_density_max_difference']['M']:.3e}；
- E：{a['redraw_density_max_difference']['E']:.3e}。

这是数值程序对解析恒等关系的核验，不是将λ直接置零，也没有以公平交易ABM代替需要检验的选择ABM。

### 2.2 N=100000的额外ABM检验

全员重抽包括首轮前独立抽取能力，之后每轮交易后独立重抽用于下一轮。各λ分支共用交易流和能力流，但保留各自按选择胜率产生的真实财富历史。

事先采用等价界限：G为±0.001，M为±0.001即±0.1个百分点，E为±0.002即±0.2个百分点。用配对90%区间完全进入等价区间作为TOST判据，**不以p>0.05冒充相同**。

最初24条未来历史的退出率精度不足。随后另行冻结一次性增加72条历史的计划，界限没有放宽，也没有边运行边检查直到通过。原24、新72与合计96分别保留：

{table(pd.DataFrame(equ))}

分母26=两个λ干预×13个非零观察时点。新增72条历史和合计96条历史的三指标均通过全部逐点等价检验。新增72条历史的E平均差最大为{100*eq[(eq['sample']=='new72')&(eq.metric=='E')].iloc[0].max_abs_mean_difference:.4f}个百分点；合计96条对应{100*eq[(eq['sample']=='pooled96')&(eq.metric=='E')].iloc[0].max_abs_mean_difference:.4f}个百分点。

这不是连续时间同时置信带，也不是新增独立准备态的外推检验。所有未通过的原24条结果仍保留，见`full_redraw_equivalence.csv`。

## 3. 能力越能持续，当前窗口中的选择效应越强

下表为六个准备根、每个条件一个新未来流的配对效应，比较同条件λ=4与λ=0，终点t=30。正数分别表示Gini、底层进入和顶层退出增加。

{table(ptable)}

在这五个预定切片上，三项效应均随实际能力持续性增强而增大；方向与同一代表初态的独立密度演化相符。逐根值及95%配对区间完整保存，不将一个代表密度解与六根均值的比较称为新的全面精度验收。

**h=0仍不等于每轮全员重抽。** μ=1、r=10时多数家族本轮没有能力更新事件，故h=0时一轮能力分数相关仍为e^(-0.1)=0.904837。只有每轮全员重抽的控制才令这一相关为0。

上述单调次序是当前背景、参数切片与30单位窗口内的结果，不是任意初态或无限时间的单调定理。

## 4. 非拟合的弱选择预测已经实际算出

令J_0、J_1、J_2分别为总体和固定群体密度的λ^0、λ^1、λ^2系数。直接对完整有限交易额算子展开，逐轮推进这三个系数。能力算子和税收映射仍保留原规则；有正有负的导数密度不做正性裁剪。随机伙伴分布的归一化也按阶求导，避免把数值质量误差放大成虚假响应。

在独立对称初态上：

\\[
Q(λ,t)=k_1(t)λ+O(λ^3),\\qquad
X(λ,t)=X(0,t)+c_X(t)λ^2+O(λ^4),\\quad X=G,M,E.
\\]

不从ABM曲线拟合任何系数。t=30的实际计算值为：

| 系数 | 数值 |
|---|---:|
| c_G | {a['weak_final_coefficients']['G2']:.10f} |
| c_M | {a['weak_final_coefficients']['M2']:.10f} |
| c_E | {a['weak_final_coefficients']['E2']:.10f} |
| k_1 | {a['weak_final_coefficients']['Q1']:.10f} |

M、E在此以0—1概率计，不是百分比。全部时间曲线见`response_coefficients.csv`。

### 4.1 首轮和第二轮

首轮财富响应的最大数值残差为{a['weak_audit']['first_round_wealth_response_max']:.3e}；Q已产生非零一阶项。首轮“同分布”是指不同λ之间，不是财富没有发生交易变化。

在同一非均等财富背景下，分别改变μ/h，两轮后的G与E二阶系数除以同一能力网格计算的一轮相关后相同，比例差处于约10^-18—10^-20；M在这两轮内为零。能力优势跨轮延续是这一早期财富效应所需的因素。

从均等财富出发，还得到可手算的平均场控制式：

\\[
G_2(λ)-G_2(0)=\\frac{{ε(1-ε)}}{{24}}\\rho_u\\,λ^2+O(λ^4).
\\]

本参数系数为{a['exact_controls']['equal_wealth_two_round_coefficient']:.10f}。使用精确的离散财富落点、48/96/192能力积分和小λ直接求解核验，而非用财富网格拟合此系数。有限人口四人枚举和平均场二阶式属于不同层次，未混称为同一有限N定理。

### 4.2 对独立完整密度解的预测精度

保持相同初态和同一网格，另行完整求解λ=.1、.25、.5，并与零阶+二阶预测比较。全14观察时点的最大误差为：

{table(wt)}

**λ=.1、.25时展开较准确；λ=.5时E的二阶截断误差达到约0.479个百分点，已经超过0.2个百分点目标。** 这明确划定了目前二阶近似的使用边界，不能将它直接用于主实验λ=4。

另外实际求解λ=.01和.02，以差商独立检查系数。步长减半后四项系数误差约缩小到四分之一，符合被省略的下一阶项。单步导数算子还用非零质量导数的独立有限差分核对，二阶相对L1误差降至约2.7×10^-8。**这些差商只用于验算，不是系数的生成方法。**

当前完整响应系数以Δlog(w)=.01、96能力箱给出；三主指标另有网格细化检验，但未对每一个响应系数进行独立连续极限外推。

## 5. Gini与跨门槛收支：已按1200轮逐项核对

对每轮当前相同联合分布，先计算公平交易和实际选择交易。定义公平交易的Gini增加U、选择相对公平交易的额外变化S。税收系数ρ=e^(-χ/r)，则物理收支为：

\\[
G_{{n+1}}-G_n=U_n+S_n-(1-ρ)G_{{\\mathrm{{trade}},n}}.
\\]

有限网格另有明确记录的税收重映射误差。将该数值误差单独列出后，逐轮恒等式残差{a['budget_residuals']['G']:.3e}。税收重映射单轮最大误差为{a['budget_residuals']['max_single_round_tax_remap_G_error']:.3e}，没有隐藏在选择贡献里。

同样把群体排名变化分成“固定旧门槛时群体财富变化”和“总体门槛移动”两项，M与E收支残差分别为{a['budget_residuals']['M']:.3e}和{a['budget_residuals']['E']:.3e}。

公平交易项在全部主轨迹上为正，最小值{a['budget_residuals']['min_fair_G_contribution']:.3e}。选择额外项在原经验初态的第一轮可以微负，这与初态有限人口经验关联有关，并不违反概率律下独立准备的首轮等价。

**严格反例亦已数值枚举：** w=(.01,.02,1.90,2.07)、u=(-.8,-.8,.8,0)、λ=ln(3)/.8，Q=0.574>0，但选择额外Gini项S=-0.00138。三种匹配、四种胜负组合全部计算。因此不能把Q>0当作Gini必增的普遍充分条件。

沿实际路径累计的U和S是当前状态的收支，不是整段干预效应的唯一因果分解；选择也改变了后续参照分布。累计值见`cumulative_budgets.csv`。

## 6. 税收的直接核验与能力关联的模态核验

### 6.1 纯税收与最低财富界

N=100000的无交易税收控制核实G^+=ρG^-，方差和Q分别按ρ²、ρ缩放。最大Gini误差{a['exact_controls']['pure_tax_max_G_error']:.3e}；前10%成员变化数为0。

浮点计算使极少数非常接近的底部财富值合并成相同数值：最多25个重复值、全人口排序数组最多23个位置发生并列排序差异。它们不涉及前10%门槛，**不能声称整个人口排序数组逐元素完全不变**；数学上的严格单调映射与浮点并列需要分开。

一笔交易的反超条件为y/x<1+2ε=1.28。税收虽不立即改变排名，却压缩财富比，使原本赢一次仍追不上的某些配对进入可反超范围。这里只是局部条件，不以此替代30单位的整体流动预测。

最不利交易路径满足w_{{n+1}}≥ρ(1-ε)w_n+1-ρ。几何递推下界已独立数值核实，长期下界为(1-ρ)/[1-ρ(1-ε)]。χ=.01和.03分别约.00710和.02101。这是理论保底界，不是实测最低财富。

### 6.2 Poisson更新与关联截断

对归一化Hermite模态m_k，γ_k=exp[-μ(1-h^k)/r]，递推m_{{k,n+1}}=ργ_k(m_{{k,n}}+J_{{k,n}})。连续高斯条件积分检验的最大误差为{a['exact_controls']['mode_quadrature_max_error']:.3e}。

对实际有限能力箱，用离散核的转置作用于观测函数检查，矩递推误差{a['mode_kernel_recurrence_max_error']:.3e}。直接把连续γ_k套到96箱的高阶模态，最大单轮残差约{a['mode_scalar_grid_max_error']:.3e}，这是能力分箱误差，不能隐藏。

用第1阶、第1/3阶、第1/3/5阶重构Q，全部主条件的最大绝对误差依次为{a['Q_reconstruction']['1']:.6f}、{a['Q_reconstruction']['3']:.6f}、{a['Q_reconstruction']['5']:.6f}。这表明低阶模态近似有实际用处，也给出了截断误差量级。J_k仍由当前联合分布给出，因此**该重构不是新的、不依赖分布演化的独立预测**。

## 7. 前10%的数值稳定性与身份计数

{table(pd.DataFrame(cvt))}

这些是同一开发初态、四个A/B主条件、完整观察窗的两档差异，不是严格的连续极限误差上界，也不自动解决有限N和抽样误差。

新B50/M40/T10分群与此前五群体Top10程序的最大差：G为{a['cohort_refactor_regression']['G']:.3e}，E为{100*a['cohort_refactor_regression']['E']:.5f}个百分点。新分群没有实质改变先前Top10预测。

新三群体给出E=5M_B+4M_I。所有ABM观察和密度群体检查均核对人数守恒；进入顶层者中原底层占比为5M_B/E。该比例的分母是新进入者，不是全体底层人口。

主密度最大概率与财富误差分别为{a['density_max_mass_error']:.3e}、{a['density_max_wealth_error']:.3e}。保存的主收敛运行没有负密度；导数密度允许正负，不能用同一“非负”标准判断。

## 8. 可以写入论文的结论与不可越过的边界

可以写入：在中性共同初态下，选择先生成财富—能力关联；这种优势的跨轮延续使财富统计逐步受影响。每轮全员重抽时选择效应消失。当前持续性切片重现了选择效应随持续性增强而增强。再分配压缩差距但不当场重排排名，其排名作用来自后续交易。弱选择响应与逐轮收支已经由方程计算并核验。

不可写成：任意初态下Q>0必导致Gini上升；二阶公式在λ=4仍精确；所有μ/h、所有历史准备、无限时间的单调普遍定理；40种子盲测已通过；全部误差已经由有限人数解释。

**下一阶段入口：** 冻结当前求解器及预测验收规范，再执行第五阶段的独立检验与必要边界切片。保留集未解盲；本轮未运行全部准备背景、N缩放或50/100时间延伸。

## 9. 复现与文件

- `src/run_all.py`：完整顺序复现；每一步失败即停止，不会静默跳过。
- `src/weak_response.py`：真正的二阶系数递推；`check_weak_small_parameters.py`与`validate_response_operator.py`用于独立验算。
- `src/run_abm_mechanisms.py`、`extend_redraw_precision.py`：新增机制ABM。
- `src/run_budget_convergence.py`、`run_density_mechanisms.py`：独立密度、收支与网格控制。
- `src/exact_controls.py`：有限人口枚举、原子财富两轮检验、纯税收与能力更新积分。
- `results/`保留逐运行数据、原24/新72/合计96区间、全部未通过条件。
- `validation/FINAL_AUDIT.json`：机器可读规模、误差和状态。
- `config/`保存冻结计划、精度扩展计划、输入哈希及版本。

对本轮模型内恒等关系与代表条件数值核验的置信度：高。跨准备历史、全参数和保留样本的推广精度：本轮不作结论。
'''
    (ROOT/'PHASE4_COMPLETION_REPORT_CN.md').write_text(text,encoding='utf8')
    (ROOT.parent/'WEALTH_PHASE4_COMPLETION_REPORT_CN_20261014.md').write_text(text,encoding='utf8')
    print('REPORT_WRITTEN',len(text),flush=True)
if __name__=='__main__':main()
