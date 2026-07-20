# Gate A：目标局部概念覆盖问题存在性检验

日期：2026-07-20
分支：`codex/target-coverage-gate-a`
基线提交：`d1a980a1e0d717b0e0c555aacdffc73128503858`

## 目的与边界

本实验只回答一个问题：在已知学生、随机交互划分的认知诊断设置中，当测试题所需概念未被该学生的训练历史充分覆盖时，强外部模型是否出现可重复的预测损失上升。

本实验不是模型选择或调参实验，不访问任何 test 文件。所有变量均由 standard train history、standard validation 行、冻结的 Full validation 预测和逐行对齐的外部 validation 预测构造。

## 数据与定义

使用 ASSIST17、MOOCRadar、XES3G5M 和 Junyi。对学生 (u) 和 validation 题目 (e)，定义：

[
c(u,e)=\frac{|O_u \cap R_e|}{|R_e|},
]

其中 (O_u) 是学生在 train history 中接触过的概念集合，(R_e) 是题目 Q 向量对应的需求概念集合。主回归量为未覆盖比例 (1-c(u,e))。

Junyi 的当前协议是一题一概念，且学生—题目划分使 validation 行全部为 exact-zero。它可以用于报告该现象的发生率，但若 (c) 没有变异，双向固定效应会将其明确标记为 `not_identified`，不参与 Gate A 的支持数据集计数。

## 主估计量

主结果使用外部模型逐行 log-loss：

[
L_{ue}=\beta(1-c(u,e))+\alpha_u+\gamma_e+\epsilon_{ue}.
]

- 学生固定效应 (alpha_u) 控制学生能力、历史长度、历史正确率和全局概念覆盖等所有学生层面的不变因素。
- 题目固定效应 (gamma_e) 控制题目难度、训练频率、Q 概念数和题目身份等所有题目层面的不变因素。
- 因而 (eta) 由“同一学生面对不同覆盖题目”与“同一题目被不同覆盖学生作答”的交叉变化识别。
- (eta>0) 表示目标局部覆盖越缺失，外部模型预测损失越高。

同时报告 external Brier、Full log-loss 和 Full 相对 external 的 log-loss advantage，但它们不是 Gate A 的主判据。

不把不同全局覆盖程度的学生直接混在一起解释：除学生固定效应外，还按学生全局概念覆盖五分位分层重复 external log-loss 检验。

## 不确定性与门槛

- 固定效应点估计使用交替去均值求解。
- 95% CI 使用学生为聚类单位的 pairs bootstrap；固定 2,000 次，`seed=2024`。
- Gate A 预先规定：至少 3 个数据集的 external log-loss 效应 95% CI 下界严格大于 0 才通过。
- 无可识别覆盖变异的数据集记为 N/A，不视为支持，也不人为补造对照。
- bootstrap 不是多 seed；模型预测、划分和所有训练产物均保持冻结。

## 资产

| 数据集 | train/valid | Full validation | 外部 validation |
|---|---|---|---|
| ASSIST17 | `knofield_data/assist_17` | `final_standard_v9/a17_predictions.csv` | ORCDF standard validation |
| MOOCRadar | `knofield_data/moocradar` | `final_standard_v9/moo_predictions.csv` | SVGCD standard validation |
| XES3G5M | `knofield_data/xes3g5m` | `final_standard_v9/xes_predictions.csv` | ORCDF standard validation |
| Junyi | `knofield_data/junyi` | `final_standard_v9/junyi_predictions.csv` | ORCDF standard validation |

脚本保存输入 SHA-256、逐行丰富表、覆盖发生率、全局覆盖分层指标、固定效应结果和机器可读总结。任何预测行的学生、题目或标签顺序不一致都会直接报错。

## 复现

```bash
/home/xph/anaconda3/envs/decoupled_cd/bin/python \
  scripts/analyze_target_coverage_gate.py \
  --dataset-spec ASSIST17 \
    /home/xph/jwc/research/knofield_data/assist_17 \
    /home/xph/jwc/research/decoupled_cd_codex/results/goal_two_module/final_standard_v9/a17_predictions.csv \
    /home/xph/jwc/research/decoupled_cd_codex/results/external_benchmarks/legacy/orcdf_a17_standard/valid_predictions.csv \
  --dataset-spec MOOCRadar \
    /home/xph/jwc/research/knofield_data/moocradar \
    /home/xph/jwc/research/decoupled_cd_codex/results/goal_two_module/final_standard_v9/moo_predictions.csv \
    /home/xph/jwc/research/decoupled_cd_codex/results/external_benchmarks/legacy/svgcd/moocradar/standard/valid_predictions.csv \
  --dataset-spec XES3G5M \
    /home/xph/jwc/research/knofield_data/xes3g5m \
    /home/xph/jwc/research/decoupled_cd_codex/results/goal_two_module/final_standard_v9/xes_predictions.csv \
    /home/xph/jwc/research/decoupled_cd_codex/results/external_benchmarks/legacy/orcdf_xes_standard/valid_predictions.csv \
  --dataset-spec Junyi \
    /home/xph/jwc/research/knofield_data/junyi \
    /home/xph/jwc/research/decoupled_cd_codex/results/goal_two_module/final_standard_v9/junyi_predictions.csv \
    /home/xph/jwc/research/decoupled_cd_codex/results/external_benchmarks/orcdf/junyi/standard/valid_predictions.csv \
  --bootstrap 2000 \
  --seed 2024 \
  --output-dir results/goal_two_module/gate_a_target_coverage
```

## 结果

运行完成于 2026-07-20。下表中的“覆盖不完整”包括 zero、low 和 partial；`β` 表示目标覆盖从 1 降至 0 时，外部模型逐行 log-loss 的双向固定效应变化。

| 数据集 | 覆盖不完整行/占比 | β | student-cluster 95% CI | 判定 |
|---|---:|---:|---:|---|
| ASSIST17 | 2,302 / 7.54% | +0.08024 | [+0.04249, +0.11946] | 支持 |
| MOOCRadar | 23,944 / 62.02% | −0.00941 | [−0.04163, +0.02341] | 不支持 |
| XES3G5M | 9,076 / 43.82% | +0.00249 | [−0.02784, +0.03171] | 不支持 |
| Junyi | 24,299 / 100.00% | N/A | N/A | 不可识别 |

ASSIST17 的五个全局覆盖分层点估计均为正，其中 Q4、Q5 的区间下界大于 0。MOOCRadar 的分层方向不一致，Q3 反而呈显著负效应；XES3G5M 没有分层区间排除 0。该结果不是由简单的全局覆盖差异或题目难度分布直接造成，因为主估计已经吸收学生和题目固定效应。

## Gate 判决

**Gate A 未通过：只有 1 个数据集支持，预注册门槛要求至少 3 个。**

因此不能把“目标局部概念未覆盖普遍导致现有模型失效”写成跨数据集研究事实。覆盖不完整本身在四个数据集中确实出现，但“存在”不等于“造成额外误差”；MOOCRadar 中 zero 行未经控制时甚至更容易，控制题目和学生后仍无正效应。

当前 Full 也不能据此声称解决了该问题。Full 相对 external 的 log-loss advantage 随未覆盖比例变化为：

- ASSIST17：−0.05510，95% CI [−0.07777, −0.03505]；
- MOOCRadar：+0.01230，95% CI [−0.00125, +0.02603]；
- XES3G5M：−0.00292，95% CI [−0.01104, +0.00545]。

也就是说，即便 ASSIST17 确实存在覆盖缺口，当前 Full 的相对优势并未随缺口扩大，反而显著下降。该 Gate 应作为否定性证据保留，不应通过更换分桶、结果变量或事后缩窄门槛来挽救原叙事。

完整机器结果位于被 git 忽略的 `results/goal_two_module/gate_a_target_coverage/`；核心文件为 `gate_a_summary.json`、`two_way_fixed_effects.csv` 和 `global_coverage_strata_fixed_effects.csv`。
