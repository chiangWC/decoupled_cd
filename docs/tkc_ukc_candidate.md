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

## 已拒绝：Calibrated History Representation

旧 `raw_summary_control` 同时删除题目语义和难度校准信息，因而不能归因
attempted-item History。最终 clean gate 固定同一个信息匹配的
`factorized_item_control` Requirement：Full 使用 `calibrated_history`，强
control 使用保留正确率、难度校准、置信度和 coverage 的
`calibrated_summary_control`，只移除 attempted-item semantic pool。

在 Q-consistent holdout validation 上，四个数据集的初始化哈希逐一相同：

| 数据集 | Full H | Control H | ΔH | Full T | Control T | ΔT | student-clustered 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASSIST17 | .799879 | .784089 | +.015790 | .781492 | .775977 | +.005515 | `[+.001274, +.009824]` |
| MOOCRadar | .926716 | .926557 | +.000160 | .935352 | .935386 | -.000034 | `[-.001172, +.001147]` |
| XES3G5M | .787696 | .786159 | +.001537 | .785818 | .784460 | +.001359 | `[-.000271, +.002989]` |
| Junyi | .824025 | .822692 | +.001332 | .824025 | .822692 | +.001332 | `[+.000329, +.002343]` |

门槛要求至少两个胜出数据集 `ΔT >= 0.005`、其中一个 `>= 0.01`。实际只有
ASSIST17 达到 0.005，且没有数据集达到 0.01；虽然 ASSIST17、Junyi 的 CI
下界大于零，联合门槛仍失败。必要 T 门已经失败，因此不再补 standard
control：S 结果无法逆转拒绝结论，只会补全一个失败模块的表格。完整记录见
`docs/experiments/2026-07-21-history-clean-gate.md`。

## 已拒绝：Population-Calibrated Concept Prior

相对较强 semantic-Q control，ASSIST17/Junyi target 均只提升约 +0.000306，拒绝。

## 已拒绝：Q-Specific Semantic Node Alignment

Full 使用 Q 特异双向邻域，对照为 raw identity 与 Q 无关的 global context。相对逐数据集较强对照，ASSIST17 target 提升 +0.002549，Junyi 下降 0.000181，拒绝。

## 未激活：Peer Transfer 与 Response-Noise Calibration

train-only peer 审计在四个胜出数据集与 Full 融合的最佳 target 增益均低于 0.003；现有 instance-dependent guess/slip 相对 cognitive probability 的 target 增益也均低于 0.004。二者不进入代码级模块探索。

## 已拒绝：Observed-Anchor Attentive State Field

该候选借鉴 Attentive Neural Processes 的 query-specific context reading，将学生已观察概念及其作答统计作为 anchors、每个待诊断概念作为 query，一次输出完整 framework state。Capacity control 使用完全相同的 context encoder、Q/K/V、decoder 和参数量，但每个学生只生成一个 global query。Direct control 是已有 additive MLP。

ASSIST17 Full target 为 0.765315，capacity control 为 0.765686，direct control 为 0.795655；Full 相对较强对照下降 0.030340，且 Evidence 增益在该 State 下缩至 +0.000029。Junyi Full 相对 capacity control 也下降 0.000069。该完整 State 替换明确拒绝。

## 已拒绝：Target-Conditioned Diagnosis

输入完整 framework state、Q 和题目表示，输出作答概率：

- Full 使用目标状态与题目需求的乘积/差异交互，以及由目标状态条件化的有界 guess/slip。
- Control 是标准单调诊断：将 Q 加权 mastery 作为唯一学生变量，以正 discrimination、题目偏置/难度和仅题目条件化的 bounded guess/slip 生成概率。
- Full/control 的 cognitive、guess、slip 网络容量相差不超过 10%；两者共享完整状态和其他所有训练条件。

MOO target 只提升 +0.000783；XES target 反而下降 0.001070。动态扩池后，ASSIST17 提升 +0.014351，但 Junyi、ASSIST09、NIPS34、EdNet 分别为 -0.000158、-0.003344、-0.000642、-0.003570；其中 ASSIST09、NIPS34、EdNet Full 也不是普通胜局。六个数据集只有一个幅度过门点，该机制最终拒绝。

## 已拒绝：Item-Conditioned Low-Rank Hypernetwork Diagnosis

借鉴 Sarafian、Keynan、Kraus 在 ICML 2021 提出的条件 hypernetwork 对笛卡尔积输入的处理，由目标题表示生成低秩诊断权重，再作用于学生状态；模块独立输出 cognitive、guess、slip 和最终作答概率。论文机制映射为 `target item -> conditional weights`、`student state -> conditional network input`，实现只依据论文描述和公式独立编写，不移植作者代码。来源：[Recomposing the Reinforcement Learning Building Blocks with Hypernetworks](https://proceedings.mlr.press/v139/sarafian21a.html)。

Capacity control 是当前 target-conditioned MLP，Direct control 是 monotonic NCD。三路共用完整 Evidence/State、初始化、数据顺序和训练配方，活跃参数量差异不得超过 10%。ASSIST17 Full 相对较强 capacity control 的 H/T 分别下降 0.014242/0.015678；MOO 的 H 下降 0.000098、T 只提升 0.000358。候选不调参，原样拒绝。

## 已拒绝：Outcome-Partitioned Multi-Set Evidence Refinement

该候选借鉴 Selby 等人对“多个置换不变集合上的函数”的建模问题，将 train-only 正确作答与错误作答视为两个有关联但不可混同的 evidence sets；分别汇总题目语义，并显式提供 correct–incorrect contrast，再整块输出下游唯一消费的 student evidence。来源：[Learning Functions on Multiple Sets using Multi-Set Transformers](https://arxiv.org/abs/2206.15444)。这里只采用 multi-set 问题定义并独立实现紧凑关系池化，不复制其 Transformer 或作者代码。

Capacity control 保留相同输入、全局作答统计、四倍语义输入宽度和完全相同参数量，但把两个集合都替换为不区分结果的 attempted-item set；Direct control 只消费原 Evidence 输出与全局统计。ASSIST17/MOOCRadar 相对逐数据集较强对照的 target 增益只有 +0.000391/+0.000741，拒绝。

## 已拒绝：Bottleneck-Aware Monotone Requirement Surface

该 Diagnosis 候选借鉴 Hierarchical Lattice Layer 的硬单调曲面，以目标
概念 readiness 的最小值与均值作为二维输入；同容量强对照是在 logit
尺度可加的两个一维 lattice，另有只读均值的 Pooled 和正权 NCD 对照。

正式 train-only Stage 1 中，ASSIST17/MOOCRadar 相对逐数据集最强对照的
Q2 AUC 增益分别为 -0.000188/+0.001095，student-clustered 95% CI 为
[-0.002282,+0.001621]/[-0.006344,+0.004211]。两处曲面虽均 noncollapsed，
但没有形成足够、稳定的预测收益；MOO 的小增益在去除 top-5 Q-pair 后
转负。候选拒绝且不进入 Claude v2。完整记录见
`docs/experiments/2026-07-21-bottleneck-requirement-surface-result.md`。

## 历史“已通过”结论已推翻：Exercise-Specific Requirement Query

该候选从已训练模型反向归因得到：Q view 表示题目要求哪些概念，exercise-specific view 表示同一 Q 组合在具体题目中的实现方式；二者共同生成下游 Diagnosis 唯一消费的 target requirement query。这个“协同身份 + 内容侧信息”映射借鉴 hybrid recommendation 的问题分解，例如 [Collaborative Deep Learning for Recommender Systems](https://dl.acm.org/doi/10.1145/2783258.2783273)，但 CD 模块与代码独立实现，不移植原模型。

`w/o Module` 将 exercise-specific view 替换为 Q-only view；Capacity control 使用 train-population 内按 Q 聚合的 concept-conditioned exercise prototype，保留完全相同的投影参数和输入宽度，但不能读取目标题 ID。正式重训后，逐数据集取两个对照中 target 更强者：

| 数据集 | Full T | Stronger control T | ΔT | student-clustered 95% CI |
|---|---:|---:|---:|---:|
| ASSIST17 | 0.796573 | 0.784683 | +0.011890 | [+0.009019, +0.014802] |
| XES3G5M | 0.785070 | 0.769736 | +0.015335 | [+0.010817, +0.019966] |

旧 `q_only_control` 与 `concept_prototype_control` 都删除了目标题身份，因此
上表把“保留 item 信息”的收益误归因为 Q-item 联合机制。新的
`factorized_item_control` 保留完全相同的 Q view、exact target item ID、
item representation、item difficulty 和公共 Diagnosis，参数量也完全相同；
它只把 joint hidden-unit nonlinear composition 替换为 factorized additive
composition。

ASSIST17 的旧 Full、旧外部预测和当前协议还分别使用了 row/row、row/Q 与
Q/Q 三种 target mask。以当前 Q/Q mask 对 Full 和强 control 成对重跑后：

| 数据集 | Full T | factorized control T | ΔT | student-clustered 95% CI |
|---|---:|---:|---:|---:|
| ASSIST17 | 0.780853 | 0.781492 | -0.000639 | [-0.003034, +0.001680] |
| MOOCRadar | 0.936415 | 0.935352 | +0.001063 | [+0.000101, +0.002099] |
| XES3G5M | 0.785588 | 0.785818 | -0.000230 | [-0.001974, +0.001441] |
| Junyi | 0.825002 | 0.824025 | +0.000978 | [+0.000051, +0.001952] |

Full 在四个数据集仍均为 validation strict external win，但没有任何数据集
达到预注册的 `ΔT >= 0.002`，更没有达到 `0.003`。因此 Requirement gate
明确失败，剩余 14 个 2x2 任务不运行；性能强不能替代模块归因。

History clean gate 也已失败。当前状态精确为：现有架构有 4 个 validation
strict external wins，但 qualified paper modules 为 0。后续模块必须遵守
`docs/research_goal.md` 的强对照与幅度门槛，不能再使用上述两组旧结论组成
双模块论文叙事。
