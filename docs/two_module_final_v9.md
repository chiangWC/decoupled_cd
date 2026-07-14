# Two-Module Final v9

## 冻结状态

- 代码提交：`9531852`（模型冻结）；其后的改动只修复 DOA 评估内存。
- architecture fingerprint：`099906acdba8c3b4`。
- 固定 `seed=42`，不做多 seed；数据 split seed 为 2024。
- 主动数据流只有两个论文模块：Calibrated Evidence Representation 与 Exercise-Specific Requirement Query。
- Q semantic nodes、concept prior、state completion 和 response diagnosis 是固定底座函数，不列为贡献模块；所有 rejected candidate 分支默认不激活。

## 两个论文模块

### Calibrated Evidence Representation

输入 train-only 学生作答历史、尝试题目语义和训练集题目难度统计，输出唯一供 state completion 消费的 `student_evidence`。整块 control 保留同容量学生边际统计编码器，但移除题目集合语义和难度校准数据流。

### Exercise-Specific Requirement Query

输入目标题目的 Q 概念需求与 population-trained exercise-specific 语义，输出 Diagnosis 唯一消费的 `q_repr`。`w/o Module` 为 Q-only；capacity control 为按 Q 聚合的 concept-conditioned exercise prototype。三路参数、初始化和输入宽度相同，control 不读取目标题 ID。

## Validation 胜局

| 数据集 | S | H | T | S/H/T margin | ordinary | strict |
|---|---:|---:|---:|---|:---:|:---:|
| ASSIST17 | 0.801455 | 0.799584 | 0.796573 | +0.017082 / +0.015731 / +0.015236 | 是 | 是 |
| MOOCRadar | 0.929467 | 0.926027 | 0.935420 | -0.000086 / +0.001848 / +0.002312 | 是 | 否 |
| XES3G5M | 0.789297 | 0.786737 | 0.785070 | +0.002901 / +0.005117 / +0.006543 | 是 | 是 |
| Junyi | 0.829147 | 0.828647 | 0.828647 | +0.008713 / +0.008658 / +0.008658 | 是 | 是 |

结果为 4 个 ordinary validation wins、3 个 strict validation wins。

## Validation-driven test confirmation

既往 test 已在项目历史中查看，因此以下结果只称 confirmation，不声称 blind test。

| 数据集 | S | H | T | S/H/T margin | ordinary | strict |
|---|---:|---:|---:|---|:---:|:---:|
| ASSIST17 | 0.801965 | 0.800545 | 0.796217 | +0.015065 / +0.016885 / +0.015411 | 是 | 是 |
| MOOCRadar | 0.929030 | 0.930463 | 0.946765 | -0.000970 / +0.001164 / +0.001370 | 是 | 否 |
| XES3G5M | 0.793260 | 0.788934 | 0.786587 | +0.000760 / +0.004058 / +0.004696 | 是 | 是 |
| Junyi | 0.834028 | 0.828365 | 0.828365 | +0.009441 / +0.009062 / +0.009062 | 是 | 是 |

结果为 4 个 ordinary test confirmations、3 个 strict confirmations。

## 干净消融

| 模块 | 数据集 | Full T | Control T | ΔT | clustered-bootstrap 95% CI |
|---|---|---:|---:|---:|---:|
| Evidence | ASSIST17 | 0.796573 | 0.777750 | +0.018823 | [+0.015791, +0.022016] |
| Evidence | Junyi | 0.828647 | 0.817022 | +0.011626 | [+0.010140, +0.013342] |
| Requirement Query | ASSIST17 | 0.796573 | 0.784683 | +0.011890 | [+0.009019, +0.014802] |
| Requirement Query | XES3G5M | 0.785070 | 0.769736 | +0.015335 | [+0.010817, +0.019966] |

所有 bootstrap 使用 2000 次学生聚类重采样，`P(ΔT>0)=1.0`；这不是多 seed。

## DOA 诊断

`min_responses=3`，不参与模型晋级。

| 数据集 | DOA | weighted DOA | 95% CI | Spearman | holdout DOA |
|---|---:|---:|---:|---:|---:|
| ASSIST17 | 0.638646 | 0.625032 | [0.614018, 0.664806] | 0.338310 | 0.628527 |
| MOOCRadar | 0.721926 | 0.796003 | [0.690784, 0.751348] | 0.353480 | 0.743646 |
| XES3G5M | 0.383968 | 0.377275 | [0.332370, 0.435230] | -0.223870 | 0.413775 |
| Junyi | N/A | N/A | N/A | N/A | N/A |

Junyi 的 exercise–concept 一一对应使 test 中每个 student–concept 不满足三次作答门槛，因此没有可计算概念，不能填 0。

## 审计与结果位置

- 数据与逐行对齐：`results/goal_two_module/final_audit_v9/`。
- validation：`final_standard_v9/`、`final_holdout_v9/`、`target_screen_v9/`、`evidence_final_v9/`。
- test confirmation：`results/goal_two_module/final_test_v9/`。
- bootstrap：`bootstrap_v9/`、`bootstrap_v9_final/`。
- DOA：`results/goal_two_module/final_doa_v9/`。

四个胜出数据集的 standard/holdout 文件哈希均匹配 live registry；我方与外部 validation predictions 的 `stu_id/exer_id/label` 逐行完全一致。
