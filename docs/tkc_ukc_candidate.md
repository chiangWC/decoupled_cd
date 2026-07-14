# Goal 两模块候选与拒绝记录

## 已拒绝：Personalized TKC/UKC State Completion

输入普通 student evidence、concept nodes、train-only population concept prior 和 observed concept evidence，输出下游唯一消费的完整 framework state：

- Full 通过 student–concept 乘性交互一次生成全部 TKC/UKC 状态。
- Control 对 observed concept 直接投影原始统计，对 UKC 使用学生无关的静态 concept prior。
- Strong Control 同样消费 student evidence、concept 和 population prior，仅移除显式乘性交互。
- 两条 decoder 容量完全相同，Diagnosis 不存在其他学生特异旁路。

Holdout validation 首屏：

| 数据集 | Full H | Full T | Control H | Control T | ΔT |
|---|---:|---:|---:|---:|---:|
| MOOCRadar | 0.925962 | 0.935233 | 0.854393 | 0.760626 | +0.174607 |
| XES3G5M | 0.786172 | 0.783953 | 0.720483 | 0.708437 | +0.075516 |

Full 在两者均保持 validation strict win，但上表只相对弱 direct-prior control。加入同样消费 student evidence、concept 和 population prior 的 additive strong control 后，MOO/XES 的 target 增益仅 +0.000113/-0.000148；ASSIST17/Junyi 也只有 +0.001415/+0.000067。旧大幅度来自删除全部学生信息的弱对照，该模块拒绝。

## 已通过：Calibrated Evidence Representation

在修正逐行预测导出后，Full 相对 raw-summary control 的 holdout-validation target 结果为：

| 数据集 | Full T | Control T | ΔT | student-clustered 95% CI |
|---|---:|---:|---:|---:|
| ASSIST17 | 0.795974 | 0.776803 | +0.019171 | [+0.016157, +0.022370] |
| Junyi | 0.829230 | 0.817245 | +0.011985 | [+0.010426, +0.013657] |

两者 Full 均胜 validation 外部线，因此 Evidence 是当前第一个正式通过的框架模块。

## 已拒绝：Population-Calibrated Concept Prior

相对较强 semantic-Q control，ASSIST17/Junyi target 均只提升约 +0.000306，拒绝。

## 已拒绝：Q-Specific Semantic Node Alignment

Full 使用 Q 特异双向邻域，对照为 raw identity 与 Q 无关的 global context。相对逐数据集较强对照，ASSIST17 target 提升 +0.002549，Junyi 下降 0.000181，拒绝。

## 未激活：Peer Transfer 与 Response-Noise Calibration

train-only peer 审计在四个胜出数据集与 Full 融合的最佳 target 增益均低于 0.003；现有 instance-dependent guess/slip 相对 cognitive probability 的 target 增益也均低于 0.004。二者不进入代码级模块探索。

## 当前候选：Observed-Anchor Attentive State Field

该候选借鉴 Attentive Neural Processes 的 query-specific context reading，将学生已观察概念及其作答统计作为 anchors、每个待诊断概念作为 query，一次输出完整 framework state。Capacity control 使用完全相同的 context encoder、Q/K/V、decoder 和参数量，但每个学生只生成一个 global query。Direct control 是已有 additive MLP。候选只在 validation 通过既定幅度门后扩展。

## 已拒绝：Target-Conditioned Diagnosis

输入完整 framework state、Q 和题目表示，输出作答概率：

- Full 使用目标状态与题目需求的乘积/差异交互，以及由目标状态条件化的有界 guess/slip。
- Control 是标准单调诊断：将 Q 加权 mastery 作为唯一学生变量，以正 discrimination、题目偏置/难度和仅题目条件化的 bounded guess/slip 生成概率。
- Full/control 的 cognitive、guess、slip 网络容量相差不超过 10%；两者共享完整状态和其他所有训练条件。

MOO target 只提升 +0.000783；XES target 反而下降 0.001070。该机制明确拒绝，不作为论文模块。

最终门槛不变：同一架构至少三胜；两个模块各自在至少两个 Full 胜出数据集 target 提升不低于 0.005，其中一个不低于 0.01，并至少一个 student-clustered paired-bootstrap CI 下界大于 0。
