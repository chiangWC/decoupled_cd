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
- residual/gate 的初始化规则；
- forward 输出协议。

允许按数据集改变：concept dimension、batch size、learning rate、weight decay、epochs、patience，以及已预先声明的数值型 loss weight。学生数、题目数、知识点数等数据决定的张量尺寸不计作架构差异。模块不能通过数据集专属默认值、固定为零的 gate 或零 loss weight 被变相关闭。

## 统一起点 Unified-V2-B0

`Unified-V2-B0` 复用现有 V2 base 的预测主干，并增加始终存在的 mastery 接口。mastery head 从统一的逐知识点学生状态产生 `m[s,k]`，所有数据集返回相同形状语义的 mastery。训练阶段使用同一类 mastery 监督/排序目标；若不同数据集采用不同 loss weight，该权重只能通过 validation 选择且不得等于零。

新增路径必须使用 base-equivalent 初始化：残差分支的初始贡献为零，融合 gate 的初始输出保持原 base 预测。单元测试需证明新增模块系数为零或 residual 初始化时，预测与冻结的 V2 base 在数值容差内一致。这样可区分“模块带来增益”与“随机初始化改变基线”。

## 候选模块与组合顺序

第一轮分别在 B0 上加入一个模块：

1. 学生条件化 TKC→UKC propagation；
2. monotonic readout residual；
3. base/structured hybrid readout 与可学习融合 gate；
4. corrected support-aware gate；
5. mastery auxiliary/separation objective；
6. 覆盖感知证据置信度模块，仅在前五项均无法满足硬门时实现。

每个候选都是全局架构：同一次候选评估必须在冻结的 primary cohort 及其 standard/holdout validation 上全部运行。通过硬门的最优单模块形成 B1；下一轮仅测试 `B1 + 一个剩余模块`。没有单模块通过时，只测试具有明确依赖关系或互补机制的二模块组合，例如 `propagation+hybrid`、`mastery+monotonic`，不进行无边界排列组合。

新模块必须先说明它弥补的可测失效模式，并有对应消融。不能仅因某一数据集 test 数值不佳而新增模块。

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

新增标准库 `unittest` 覆盖：mastery 始终非空且形状为 `[students, concepts]`；base-equivalent 初始化；模块不能被数据集配置关闭；任意数量 manifest 的 fingerprint 一致；primary cohort 冻结后不可替换成员；standard/holdout 分发；DOA 输入语义；validation-only 模块选择；失败 attempt 不静默改变 batch。每个候选先进行 synthetic CPU/GPU smoke，再进入真实数据训练。

GPU 继续优先选择空闲卡；显存未过半的卡允许并发，但同一卡使用 `flock` 串行保护。所有数据、日志、checkpoint、预测和 mastery 留在远端资产目录，不进入 Git。实现里程碑以 `chiangWC <215551297+chiangWC@users.noreply.github.com>` 提交并生成 bundle，不 push，也不修改 `decoupled_cd_v2`。

## 论文结果边界

当前异构配置的 coverage 3/3 只保留为探索上限，不进入统一模型主表。主表只报告最终 fingerprint 相同的模型；未通过全局硬门的模块进入消融或负面分析。可从全部合格数据集中冻结不少于三个主数据集，也允许增加更多通过者；若最终只有两个数据集通过，则论文必须报告统一模型 2/3，不得用异构配置或搜索中途更换 cohort 补足第三个胜出。
