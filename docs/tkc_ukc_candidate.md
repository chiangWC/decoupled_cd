# TKC/UKC 两模块结果反推候选

前两轮候选已经按验证结果拒绝：

- Neural-Process Evidence Posterior + Attentive Concept Query：Full 在 MOOCRadar/XES3G5M 均形成普通胜局，但模块 target 增益分别仅为 MOO +0.002264/+0.000337、XES +0.000652/+0.000178。
- Relational TKC Evidence + Personalized UKC Attention：MOO 的两个模块 target 增益为 +0.001931/+0.004205；XES 为 +0.001041/−0.000546，且 Full 的 holdout overall 不达外部普通胜局线。
- Partial-flow 草案 Full 的 holdout AUC 仅为 MOO 0.920065、XES 0.773759，未进入消融；与历史 Partial-VAE 的逐行误差相关性审计也表明专家融合上限不足，故活动代码已删除。

当前候选不再从新机制名称出发，而是把历史已取得三胜的 marginal 路径重新拆成两个唯一数据流方框。机制来自本项目 README_spec 的 TKC/UKC 职责和历史验证结果，不复制外部模型代码。

## Module 1: Calibrated Evidence Representation

输入 train-only 学生作答历史、题目/Q 表示和 train-only 题目难度统计，输出唯一 student evidence：

- Full 使用已作答题目表示的集合均值、整体正确率、相对已作答题目难度的残差、历史置信度和覆盖率。
- Control 只使用原始整体正确率、置信度和覆盖率，不读取题目身份或难度校准。
- 两条路径具有完全相同的编码器容量。

## Module 2: Personalized TKC/UKC State Completion

输入 Module 1 的 student evidence、concept nodes、train-only population concept prior 和 observed concept evidence，输出完整 framework state：

- Full 使用 student–concept 乘性交互，一次生成全部 TKC/UKC 状态。
- Control 对 observed concept 直接投影原始统计，对 UKC 使用学生无关的静态 concept prior。
- 两条 decoder 容量完全相同，且 Diagnosis 不存在其他学生特异旁路。

固定的 Q-conditioned pooled NCF Diagnosis 只消费 framework state，不作为论文贡献。四种 Full/单模块消融组合共享同一个 state dict、初始化哈希和 architecture fingerprint；训练统一采用 20% context-target hiding，目标 response 不得进入自身历史。

晋级门槛保持 Goal 约束：Full 至少三胜；每个模块相对其较强合理对照在至少两个胜出数据集 target 提升不低于 0.005，其中一个不低于 0.01，并至少一个 student-clustered paired-bootstrap CI 下界大于 0。
