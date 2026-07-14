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

## 当前候选：Outcome-Partitioned Multi-Set Evidence Refinement

该候选借鉴 Selby 等人对“多个置换不变集合上的函数”的建模问题，将 train-only 正确作答与错误作答视为两个有关联但不可混同的 evidence sets；分别汇总题目语义，并显式提供 correct–incorrect contrast，再整块输出下游唯一消费的 student evidence。来源：[Learning Functions on Multiple Sets using Multi-Set Transformers](https://arxiv.org/abs/2206.15444)。这里只采用 multi-set 问题定义并独立实现紧凑关系池化，不复制其 Transformer 或作者代码。

Capacity control 保留相同输入、全局作答统计、四倍语义输入宽度和完全相同参数量，但把两个集合都替换为不区分结果的 attempted-item set；Direct control 只消费原 Evidence 输出与全局统计。候选首先固定 ASSIST17/MOOCRadar 原配方运行，不做候选专属调参；若候选过门，还必须重新验证已通过 Evidence 模块在新数据流下仍满足幅度门。

最终门槛不变：同一架构至少三胜；两个模块各自在至少两个 Full 胜出数据集 target 提升不低于 0.005，其中一个不低于 0.01，并至少一个 student-clustered paired-bootstrap CI 下界大于 0。
