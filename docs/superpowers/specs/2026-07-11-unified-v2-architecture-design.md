# Unified V2 同架构优化设计

## 目标与论文口径

构建一套在所有最终入选数据集上计算图与模块开关完全一致的 DecoupledCD V2。数据集之间只允许调整数值型训练超参数，不允许按数据集启停模块。最终主实验不固定为 ASSIST17、MOOCRadar、XES3G5M，但必须包含至少三个数据集，并全部来自同一 architecture fingerprint。不能继续把当前 `v2 base` 与 MOOCRadar 的 `hybrid+monotonic+UKC` 结果合并为“同一模型 3/3”。

统一模型以零知识覆盖预测为主轴，同时始终输出逐学生–知识点 mastery，使所有入选数据集都能计算 DOA。正式目标为：至少三个数据集超过各自预注册的 zero-AUC 对手门槛，standard/holdout overall AUC 不回退，且 DOA 不再缺失。

## 数据集池与冻结规则

当前远端数据集分为两级：

- 可立即进入统一实验的五个数据集：ASSIST09、ASSIST17、NIPS34、MOOCRadar、XES3G5M；它们已有 standard/holdout、Q-matrix 和 seed-2024 划分资产。
- 候补数据集：Junyi、EdNet-ICDM；当前只有交互划分和 `cpt_seq`，进入候选池前必须以 train-only 数据构建 Q-matrix、生成相同协议的 student-concept holdout，并通过泄漏与覆盖切片审计。

先用 Unified-V2-B0 在所有合格数据集上运行 validation，据此冻结一个不少于三个数据集的 primary cohort。冻结只依据预先声明的可训练性、zero-coverage 样本量、匹配基线安全门和 validation 指标，不读取新 test。模块搜索开始后 primary cohort 不得随候选模块更换，防止每一轮挑选不同的有利数据集。未入选数据集保留完整内部记录；同一最终架构若随后在候补数据集通过全部硬门，可增加到主表，但不能替换失败的已冻结成员来美化原 cohort。

## 架构不变量与允许差异

以下内容组成 architecture fingerprint，在所有 primary cohort 和新增主表数据集上必须逐项相同：

- 模型类与所有布尔模块开关；
- TKC/UKC 传播路径、读出路径和门控拓扑；
- mastery head 的类型、输入与输出定义；
- 训练损失的组成项及其作用位置；
- 各框架模块的参数初始化规则；
- forward 输出协议。

允许按数据集改变：concept dimension、batch size、learning rate、weight decay、epochs、patience，以及已预先声明的数值型 loss weight。学生数、题目数、知识点数等数据决定的张量尺寸不计作架构差异。模块不能通过数据集专属默认值、固定为零的 gate 或零 loss weight 被变相关闭。

## 统一起点 Unified-V2-B0

`Unified-V2-B0` 由必要的两端组成：已测知识证据编码器 M1 和单调诊断解码器 M4。M1 从作答历史产生逐学生、逐知识点的已测状态；对没有状态的知识点，B0 使用同一套可学习 concept prior 填充。M4 始终输出统一语义的 `m[s,k]`，再结合题目难度、区分度和 Q-matrix 计算答对概率。训练阶段使用同一类 mastery 监督/排序目标；若不同数据集采用不同 loss weight，该权重只能通过 validation 选择且不得等于零。

统一模型不保留旧 base 与新路径的并行 residual，也不使用 hybrid readout 兜底。每个候选直接替换其负责的完整功能，并用相同训练预算和 validation 协议与 B0 比较。稳定初始化可以作为优化细节，但不得改变模块定义或成为论文贡献。

## 框架级模块

最终框架只允许出现能够独立画成方框、具有明确输入输出和研究问题的模块：

1. **M1 已测知识证据编码器（Tested-Knowledge Evidence Encoder）**：输入学生作答历史和 Q-matrix，输出学生条件化的 TKC states。它是所有组合的必需模块。
2. **M2 未测知识推断网络（Untested-Knowledge Inference Network）**：输入 TKC states 与知识关系图，直接生成学生条件化的 UKC states。它替换 B0 的 concept-prior 填充和旧的静态 UKC 路径，不以 residual 形式附着在旧路径上，也不能把同一 UKC 表征广播给所有学生。
3. **M3 覆盖感知状态融合器（Coverage-Aware State Composer）**：输入 TKC states、UKC states、观测覆盖和证据可靠性，输出唯一的学生–知识点 mastery map。覆盖率、图可达性、propensity 等只是其内部信号，不能单独列为模块。
4. **M4 单调诊断解码器（Monotonic Diagnosis Decoder）**：输入 mastery map、目标题目的 Q 向量、难度与区分度，输出答对概率；mastery 提升不得导致答对概率下降。它是所有组合的必需模块，并保证 AUC 与 DOA 消费同一认知状态。

`corrected support`、mastery auxiliary/separation、masked reconstruction、zero initialization、warm-up、dropout 和 loss weight 均归入内部机制或训练策略，不进入主框架图，也不作为独立模块计数。

## 组合顺序

模块识别只比较以下可解释组合：

1. `M1 + M4`：无 UKC 推断的统一诊断基线 B0；
2. `M1 + M2 + M4`：用观测 mask 对 TKC states 与 M2 的 UKC states 作确定性拼接，识别未测知识推断是否有效；该 mask assembly 不是可学习模块；
3. `M1 + M2 + M3 + M4`：识别覆盖感知融合能否同时改善 zero AUC、保住 standard/holdout overall AUC；
4. 在最终推理架构不变的前提下比较训练目标，训练目标结果单列，不记为新增模块。

每个组合都是全局架构：同一次候选评估必须在冻结的 primary cohort 及其 standard/holdout validation 上全部运行。M2 或 M3 只有通过全局硬门才可保留；失败则全局删除，不允许变成数据集专用开关。

## 文献驱动的模块迭代

M1–M4 固定的是功能职责与输入输出接口，不固定内部算法、论文来源或当前实现。任一模块经 validation 证明存在明确失效模式后，都可以继续搜索认知诊断及相邻领域论文，整体替换该模块的实现。每次迭代必须先写出“失效指标 → 检索问题 → 借鉴假设 → 新模块输入输出 → 可证伪硬门”，不能仅因某一数据集 test 数值不佳而新增结构。

- **Incomplete Multi-view Learning**：可把有作答的 TKC view 与缺失的 UKC view 视为不完整多视图，检索 missing-view inference、consensus representation、quality-aware fusion 等方向。
- **Semi-supervised Node Learning / Missing Node Features**：可把 TKC 视为有观测节点、UKC 视为缺失特征节点，检索 propagation、diffusion、graph imputation、uncertainty-aware message passing 等方向。
- **Recommendation Exposure Bias / MNAR**：可把“学生是否在某知识点作答”建模为观测过程，检索 exposure、propensity、causal recommendation、doubly robust learning 等方向。
- **Positive-Unlabeled / Weakly-supervised Learning**：可用于重新审视未观测知识状态的风险估计、伪标签和置信度学习；仅有 loss 变化时仍归为训练策略。
- **Noisy-label / Robust Learning**：只有确认作答标签噪声是主要失效原因时才引入；双网络、样本选择等训练范式不能自动包装成本文核心模块。
- **开放检索**：上述领域不是白名单。若失效模式更接近冷启动推荐、矩阵补全、图信号恢复、因果表示、领域泛化或其他问题，可继续扩展关键词和论文池。

文献借鉴不是复制论文名称：必须说明认知诊断中的变量对应关系，并保留原论文引用。一次只整体替换一个框架模块；新旧实现不能以 residual、adapter 或 mixture-of-experts 方式同时堆叠。文献和实现尝试不设固定次数上限，但每次继续搜索必须由新的 validation 证据或新的失效诊断驱动，并完整保留负面结果。成功条件满足后立即停止新增结构。

### 初始文献入口

以下仅是第一批检索种子，不构成必选列表、白名单或实现承诺。后续可根据失效诊断增加、替换或放弃：

- Wen et al., *Unified Embedding Alignment with Missing Views Inferring for Incomplete Multi-View Clustering*, AAAI 2019: <https://doi.org/10.1609/aaai.v33i01.33015393>
- Liu et al., *Quality-aware and Soft Consistency Driven Representation Fusion for Incomplete Multi-view Multi-label Classification*, AAAI 2026: <https://doi.org/10.1609/aaai.v40i28.39564>
- Gasteiger et al., *Predict then Propagate: Graph Neural Networks meet Personalized PageRank*, ICLR 2019: <https://iclr.cc/virtual/2019/poster/1117>
- Rossi et al., *On the Unreasonable Effectiveness of Feature Propagation in Learning on Graphs with Missing Node Features*, LoG 2022: <https://openreview.net/forum?id=qe_qOarxjg>
- Liang et al., *Modeling User Exposure in Recommendation*, WWW 2016: <https://arxiv.org/abs/1510.07025>
- Wang et al., *Doubly Robust Joint Learning for Recommendation on Data Missing Not at Random*, ICML 2019: <https://proceedings.mlr.press/v97/wang19n.html>
- Kiryo et al., *Positive-Unlabeled Learning with Non-Negative Risk Estimator*, NeurIPS 2017: <https://proceedings.neurips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html>
- Han et al., *Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels*, NeurIPS 2018: <https://proceedings.neurips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html>

## Validation 硬门与排序

每个候选固定 seed 42，并与同训练预算的前一统一架构比较。候选只有同时满足以下条件才可保留：

- primary cohort 每个数据集的 standard overall AUC 均不下降；
- primary cohort 每个数据集的 holdout overall AUC 均不下降；
- primary cohort 每个数据集的 weighted DOA 均不下降；
- 至少 `ceil(2N/3)` 个数据集的 zero AUC 严格提升，其中 `N` 为 cohort 大小；
- 至少一个数据集的 zero AUC 提升不小于 `0.001`；
- 至少 `ceil(2N/3)` 个数据集的 ordinary DOA 严格提升；
- 无 NaN、OOM 后静默改 batch、空 mastery 或预测顺序错误。

比较使用保存的原始双精度指标，不通过四舍五入制造“持平”。多个候选通过时，依次最大化：cohort mean zero-AUC delta、最差数据集 zero-AUC delta、cohort mean ordinary DOA delta；仍相同时选择参数量更少、实现更简单者。

## 实验阶段与停止规则

阶段零审计数据集池并冻结 primary cohort。阶段一使用现有每数据集训练配方作为数值超参数起点，只运行 validation。阶段二只对通过模块硬门的架构做数值型调参；模块集合不得变化。阶段三冻结唯一统一架构和每数据集数值配置，再进行 test 确认性复现。

正式成功要求：

- 最终主表至少包含三个数据集；
- ASSIST17 若入选，zero AUC `> 0.7808065`；
- MOOCRadar 若入选，zero AUC `> 0.9454`，overall 相对 `0.9300` 回退不超过 `0.002`；
- XES3G5M 若入选，zero AUC `> 0.7845`；
- ASSIST09、NIPS34、Junyi、EdNet-ICDM 若入选，须先用 validation 选择并冻结 fresh external baseline，再把该 baseline 的固定 test 指标登记为统一模型的对手门槛；统一模型不得根据自己的 test 结果继续改架构；
- 每个入选数据集的 standard/holdout overall AUC 均不低于各自匹配基线；
- 每个入选数据集均生成非空、定义一致的 mastery 与 DOA；
- 所有冻结配置的 architecture fingerprint 完全相同。

任一阶段若“当前成功数 + 尚可尝试数据集数 < 3”，停止该候选。连续两轮新增模块均不能通过全局硬门时，停止扩展架构，回到最近通过的统一版本。旧 test 已被历史实验查看，因此新结果只能称为 validation 驱动的确认性复现，不能称全局盲测或历史首次 test-once。

## 工程、审计与测试

统一 architecture manifest 由训练 CLI 的架构字段规范化后计算 SHA-256；冻结、checkpoint、summary 和评估产物都记录该 fingerprint。campaign runner 在启动 cohort 前校验所有数据集 fingerprint 一致，不一致即拒绝运行。

新增标准库 `unittest` 覆盖：M1–M4 的输入输出契约；mastery 始终非空且形状为 `[students, concepts]`；单调解码约束；M2 输出随学生 TKC states 改变；M3 在 TKC/UKC 缺失边界上行为明确；不存在 legacy/hybrid readout 绕过；模块不能被数据集配置关闭；任意数量 manifest 的 fingerprint 一致；primary cohort 冻结后不可替换成员；standard/holdout 分发；DOA 输入语义；validation-only 模块选择；失败 attempt 不静默改变 batch。每个候选先进行 synthetic CPU/GPU smoke，再进入真实数据训练。

GPU 继续优先选择空闲卡；显存未过半的卡允许并发，但同一卡使用 `flock` 串行保护。所有数据、日志、checkpoint、预测和 mastery 留在远端资产目录，不进入 Git。实现里程碑以 `chiangWC <215551297+chiangWC@users.noreply.github.com>` 提交并生成 bundle，不 push，也不修改 `decoupled_cd_v2`。

## 论文结果边界

当前异构配置的 coverage 3/3 只保留为探索上限，不进入统一模型主表。主表只报告最终 fingerprint 相同的模型；未通过全局硬门的模块进入消融或负面分析。可从全部合格数据集中冻结不少于三个主数据集，也允许增加更多通过者；若最终只有两个数据集通过，则论文必须报告统一模型 2/3，不得用异构配置或搜索中途更换 cohort 补足第三个胜出。
