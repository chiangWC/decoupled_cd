# Two-Module v10：剩余数据集与匹配分析

## 范围与协议

- 日期：2026-07-14。
- 模型 seed 固定为 42，不做多 seed。
- 正式扩池结果只接受冻结 architecture fingerprint
  `099906acdba8c3b4`；standard 与 holdout 必须共享同一数据集配方。
- 所有训练只看 validation。两个补充数据集均未形成普通胜局，因此未运行
  test confirmation。
- target coverage 使用完整 Q-matrix 和 train-only 学生历史：ASSIST09 为
  exact-zero，NIPS34 为 low-coverage。

## 有限调参

ASSIST09 从 `dim64/lr1e-3/student-batch64/context-target0.2` 出发，依次检查
学习率、student batch、context-target hiding 和容量。`dim128` 产生不同参数形状
及 fingerprint，只保留为容量诊断，不进入同架构结果。最终同 fingerprint 配方为：

```text
dim64, lr1e-3, student-batch64, 100 epochs, context-target=0.8
```

NIPS34 检查 `lr2e-3`、student-batch128、dim128 和更强 context-target hiding。
最终配方为：

```text
dim64, lr1e-3, student-batch64, 80 epochs, context-target=0.4
```

## 扩池 validation 结果

| 数据集 | S | H | T | S margin | H margin | T margin | ordinary |
|---|---:|---:|---:|---:|---:|---:|:---:|
| ASSIST09 | 0.765720 | 0.756552 | 0.744129 | -0.010712 | -0.012396 | -0.011311 | 否 |
| NIPS34 | 0.780253 | 0.778957 | **0.763795** | -0.008224 | -0.005291 | **+0.001675** | 否 |

外部 validation 线分别为：ASSIST09/SVGCD
`0.776432/0.768947/0.755440`；NIPS34/ORCDF
`0.788478/0.784249/0.762120`。

调参相对冻结基础配方的变化：

| 数据集 | ΔS | ΔH | ΔT |
|---|---:|---:|---:|
| ASSIST09 | +0.003675 | +0.006497 | +0.004113 |
| NIPS34 | +0.000954 | +0.000007 | +0.001065 |

ASSIST09 的 dim128 容量诊断达到 S/H/T
`0.764285/0.757905/0.748157`，仍不胜出，且 fingerprint 为
`d4609d8505353d65`，不计入正式同架构表。NIPS34 的调参扩大了 target
领先，但没有解决 overall 差距。

## matched-item / matched-student 小实验

对 holdout validation 的 exact-zero 与 full-coverage 行进行匹配，估计量为：

```text
(Ours - External advantage on target)
- (Ours - External advantage on matched full-coverage rows)
```

逐行 Brier/log-loss advantage 定义为 `external loss - ours loss`，因此差分为正表示
我方相对优势更集中在 target。matched-item 对题目聚类 bootstrap，matched-student
对学生聚类 bootstrap；均为 2000 次。AUC 列是相同匹配权重下 target margin 与
matched-seen margin 的差，未给 bootstrap CI。

| 数据集 | 匹配 | 单元数 | AUC margin DID | Brier advantage DID [95% CI] | Log-loss advantage DID [95% CI] |
|---|---|---:|---:|---:|---:|
| ASSIST17 | item | 1916 | +0.006099 | +0.003940 [0.000151, 0.007913] | +0.012017 [-0.000202, 0.023938] |
| ASSIST17 | student | 562 | +0.001384 | +0.015542 [0.006812, 0.023806] | +0.044385 [0.021161, 0.065808] |
| MOOCRadar | item | 355 | +0.003185 | -0.002428 [-0.011651, 0.006825] | -0.008391 [-0.035044, 0.016601] |
| MOOCRadar | student | 928 | -0.000301 | -0.000700 [-0.003031, 0.001673] | -0.001928 [-0.010315, 0.006640] |
| XES3G5M | item | 670 | +0.003844 | +0.001485 [-0.004391, 0.007228] | +0.002769 [-0.013819, 0.018256] |
| XES3G5M | student | 995 | +0.008970 | -0.000240 [-0.002727, 0.002283] | +0.002645 [-0.004634, 0.010359] |

结论：ASSIST17 支持 target-specific advantage，尤其 matched-student 的两个逐行
损失指标 CI 均完全高于 0；matched-item 的 Brier CI 也高于 0。MOOCRadar 不支持该
叙事，XES3G5M 只有 AUC 点估计为正而逐行损失 CI 跨 0。因此这项实验可作为
ASSIST17 case study，不能写成跨数据集普遍规律。

## 资产与审计

- 扩池训练、预测、coverage 与冻结汇总：
  `results/goal_two_module/pool_extension_v10/`。
- 配方、margin、fingerprint 与数据哈希审计：
  `results/goal_two_module/pool_extension_v10/final_summary.json`。
- 匹配汇总与 forest plot：
  `results/goal_two_module/matched_target_v10/`。
- 分析入口：`scripts/analyze_matched_target_advantage.py`。
- standard/holdout 的 train/valid/test/Q 文件哈希均匹配 live registry；预测入口逐行
  校验 `stu_id/exer_id/label`。
