# 统一 Mastery 补全模型设计

日期：2026-07-12

远端实验仓库：`/home/xph/jwc/research/decoupled_cd_codex_worktrees/complete_model`

目标分支：`codex/remote-complete-model-20260710`

## 1. 背景与问题定义

Task 9 的 NeuralCDM M4 在 ASSIST09、ASSIST17 和 MOOCRadar 上相对匹配的旧线性 M4 显著提升 AUC，但没有达到论文所需的绝对性能，且 ASSIST17/MOOCRadar 的 weighted DOA 回退。该结果只能证明新 M4 优于内部弱解码器，不能视为胜过外部模型。

历史强结果也不能直接作为统一模型结果：ASSIST17 使用 `v2_base`，MOOCRadar 使用 `hybrid+mono+UKC` minibatch，XES3G5M 使用 `base_long`，计算图不同。可比的历史 validation overall AUC 分别为 ASSIST17 `0.7855566`、MOOCRadar `0.9245100`、XES3G5M `0.7821793`，明显高于当前统一模型。

对历史 checkpoint 的只读推理消融进一步发现：

| 数据集 | 完整 validation AUC | cognitive-only AUC | mean guess | mean slip |
|---|---:|---:|---:|---:|
| ASSIST17 | 0.7855566 | 0.2154143 | 0.9734055 | 0.9736458 |
| MOOCRadar | 0.9245100 | 0.9241838 | 0.0904973 | 0.0048430 |
| XES3G5M | 0.7821793 | 0.7807968 | 0.0063895 | 0.0154511 |

历史 ASSIST17 通过 `guess+slip>1` 将反向 cognitive 排序再次翻转，不能作为可信 mastery 路径继承。因此，新模型必须恢复强转导建模能力，同时排除这种退化。

## 2. 论文目标与不变量

最终模型至少在三个冻结数据集上满足：

- zero/目标切片 AUC 超过最强的同协议、可审计外部模型；
- standard overall AUC 和 holdout overall AUC 均不低于对应最强外部结果；
- 所有入选数据集使用完全相同的 architecture fingerprint；
- 始终输出唯一、非空的逐学生–概念 mastery；
- cognitive probability 与最终答对概率均随所需概念 mastery 单调不减。

DOA 完整保留为内部诊断与软优化目标，不再作为候选淘汰硬门。若最终缺乏竞争力，可不列为论文主指标，但不得删除内部负面记录。

数据集之间只允许改变数值型超参数：维度、batch size、学习率、weight decay、epoch、patience 和固定损失项的非零权重。模型类、模块、损失组成、初始化语义、mastery 定义和预测公式必须一致。seed 固定为 `42`。

## 3. 稳定职责与可替换模块

框架固定四个能够独立画入框架图的职责槽位，内部算法可在 validation 诊断后整体替换。

### 3.1 已测知识掌握估计器

输入训练作答与 Q-matrix，输出已测学生–概念 mastery 和可靠性。首轮使用可训练 `student×concept` logits，并以训练集平滑正确率初始化：

```text
theta_init[s,k] = logit((correct[s,k] + alpha) /
                        (attempt[s,k] + 2*alpha))
m_obs[s,k] = sigmoid(theta[s,k])
```

未测单元不由该模块预测。后续可整体替换为响应图、贝叶斯后验或集合编码器。

### 3.2 未测知识掌握补全器

输入已测 mastery、观测 mask 与可靠性，只输出未测单元。首个候选使用低秩学生–概念补全：

```text
m_miss[s,k] = sigmoid(u[s]^T v[k] + b_student[s] + b_concept[k])
```

学生因子从该学生其他已测概念获得监督，概念因子从其他学生对该概念的记录获得监督。后续可根据失效机制整体替换为图补全、曝光去偏矩阵补全、VAE、不完整多视图或其他匹配方法。

### 3.3 单调认知响应解码器

输入唯一 mastery、Q、题目难度和区分度，输出 cognitive probability。首轮使用 NeuralCDM：item-by-concept difficulty、正区分度、Q 精确屏蔽和非负有效权重的单调交互网络。该实现不是永久锁定，但任何替代实现必须保留单调性和唯一 mastery 输入。

### 3.4 条件行为响应模型

输入学生行为 embedding、题目 embedding 和题目属性，输出 guess/slip 参数，不生成第二份 mastery 或独立答题概率：

```text
[guess, slip, cognitive_weight] = softmax(z_guess, z_slip, z_weight)
p = (1 - slip) * p_cognitive + guess * (1 - p_cognitive)
```

不设置 `0.3` 等固定上限，但由 simplex 保证 `guess+slip<1`，因此：

```text
d p / d p_cognitive = 1 - guess - slip = cognitive_weight > 0
```

### 3.5 唯一 Mastery 组装

已测与未测单元按 train-only 观测 mask 硬组装，不使用可学习 gate、residual 或 mixture-of-experts：

```text
mastery = observed_mask * m_obs + (1 - observed_mask) * m_miss
```

## 4. A0 与首个可证伪候选 A1

统一 A0 用于数值对齐和 cohort 筛选：已测 mastery 估计器、全局 concept prior 未测填充、单调认知解码器、条件行为模型。A0 不声称解决未测 mastery。

A1 只做一次整模块变化：用低秩未测 mastery 补全器整体替换 A0 的 concept-prior 填充。其他模块、训练预算和 validation 协议保持一致。

候选内部的损失组成跨数据集固定。A0 使用前两项，A1 在保持前两项不变的基础上加入第三项：

1. 最终答题 BCE；
2. 已测 mastery 的训练作答证据约束；
3. A1 补全器在已测单元上的 masked completion 约束。

首轮不加入 DOA 排序损失。若 AUC 达标而 DOA 较差，可将训练目标作为后续独立迭代，但不能伪装成新框架模块。

## 5. 文献驱动的模块探索约束

路线 A 是首个注册假设，不锁死低秩补全或 NeuralCDM。每次迭代必须先登记：

```text
failure_id -> 失败数据集/划分 -> 原始双精度差值 -> 责任模块
-> 机制假设 -> 检索问题 -> 变量映射 -> 新输入/输出 -> acceptance gate
```

检索从认知诊断出发，可扩展到矩阵补全、协同过滤、Incomplete Multi-view Learning、MNAR/曝光偏差、因果推荐、图缺失特征、半监督节点学习、PU/弱监督、噪声标签、冷启动、贝叶斯推断或其他符合诊断的问题。既有论文清单不是白名单。每轮最多选择三个机制做代码级审阅，并保留论文、venue、URL 和认知诊断变量映射。

一次只能整体替换一个职责模块。禁止 scalar loss 冒充模块、legacy/hybrid 旁路、residual、adapter、旧新模块 mixture-of-experts、数据集专属开关和第二个独立答题头。行为模块只产生唯一概率公式的参数，因此不构成第二预测头。

每个候选必须覆盖整个 frozen cohort。只有新的 validation 证据产生不同失效诊断时才允许继续搜索；负面结果完整保留。连续两次整模块替换未通过全局 AUC 门时，回到最近通过版本重新诊断。达到至少三个数据集外部胜出后立即停止增加结构。

## 6. 基线审计与 Primary Cohort

探索池固定为 ASSIST09、ASSIST17、NIPS34、MOOCRadar 和 XES3G5M。先审计 ORCDF、SVGCD、KaNCD 及其他可公平复现强基线。

历史基线只有在数据/Q 哈希、standard/holdout 划分、指标定义、seed、checkpoint、配置和预测顺序全部匹配时才直接复用；否则仅重跑缺失或不一致项。同协议的多个可审计结果一律取较强值。协议不同的历史结果只作补充说明，不进入硬门。

A0 模块固定，先使用历史强数值配方及 r20–r23 候选完成 validation-only 数值对齐。数据集需满足：standard/holdout/zero 指标可计算、zero 至少 1000 条且正负标签各至少 100 条、无 NaN/OOM/空 mastery/顺序错误、外部基线完整可用。

合格数据集依次按 A0 相对最强外部 zero-AUC 差距、standard/holdout overall 中较差的差距、zero 样本量和训练稳定性排序，冻结前三名为 primary cohort。冻结发生在 A1 模块搜索前，之后不得用探索池其他成员替换失败数据集。追加集只有在最终统一架构独立通过全部门槛时才可加入主表。

## 7. Validation 门与最终成功条件

模块候选相对前一统一架构必须满足：

- primary cohort 每个数据集 standard overall AUC 不下降；
- 每个数据集 holdout overall AUC 不下降；
- 至少 `2/3` 数据集 zero AUC 严格提升；
- 至少一个 zero AUC 提升不低于 `0.001`；
- 唯一 mastery、单调性、数据顺序和同一 fingerprint 全部成立。

DOA 不参与淘汰；多个 AUC 候选近似时，依次选择 mean zero-AUC 更高、最差数据集 zero-AUC 更高、weighted DOA 更高、参数更少者。

最终三个 primary 数据集必须同时超过各自最强同协议外部 zero AUC，并守住 standard/holdout overall AUC。配置冻结后才能运行 test；不得根据新模型 test 结果继续修改架构。历史 test 已被查看，最终结果只能称为 validation 驱动的确认性复现。

## 8. 精简测试与真实 GPU 验证

每次整模块替换只运行六项聚焦测试：train-only 数据边界、唯一 mastery/mask 组装、M2 只负责未测单元、认知单调性、`guess+slip<1` 与最终正导数、跨数据集 fingerprint 一致。

每次架构变化只运行一次 1-epoch GPU smoke；纯数值配置变化不重复 smoke，也不做 CPU/GPU 全排列一致性。完整测试发现只在最终架构冻结和打开 test 前运行。文档、账本和纯数值配置变化仅需 `git diff --check`、配置与 fingerprint 校验。

真实 GPU runner 必须持续检查数据哈希、dirty tree、NaN/OOM、mastery 非空、预测标签顺序、产物完整性和 test-once。失败 attempt 不得静默改变 batch、维度或训练模式。

## 9. GPU 并行与远端约束

不限定单卡。每次启动前动态检查 GPU，优先空闲卡，也允许使用显存低于一半的卡。不同物理 GPU 可并行运行不同数据集或架构的独立 attempt；同一物理 GPU 使用 `flock` 串行。相同 controller state、artifact root 或 attempt 目录禁止并发写入。全局候选判定必须等待 frozen cohort 的所有 required proof 完整生成，不能因先完成的有利结果提前通过。

CPU 的 coverage/DOA/哈希计算可与其他 GPU 训练并行，但同一 attempt 的最终 proof 必须在所有派生指标完成后原子发布。

所有工作只在 `decoupled_cd_codex` 远端实验分支进行，Git 身份保持 `chiangWC <215551297+chiangWC@users.noreply.github.com>`。里程碑提交后生成 Git bundle，不 push。`decoupled_cd_v2` 始终只读，数据、checkpoint、预测、mastery 和日志不提交 Git。
