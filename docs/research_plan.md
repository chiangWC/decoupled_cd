# 解耦式认知诊断（Decoupled CD）研究计划

> **生成日期**：2026-07-06
> **来源**：多智能体分析（文献与评估协议调研、实验矩阵设计、模块架构提案、对抗性审稿、代码存活率评估）＋ 逐条代码可行性核查（文中 file:line 引用均已在本仓库逐一核实）。

## 目录

1. [摘要](#1-摘要)
2. [已确认的问题清单（P1–P6）](#2-已确认的问题清单p1p6)
3. [迁移计划（hybrid，七步）](#3-迁移计划hybrid七步)
4. [相关工作与定位](#4-相关工作与定位)
5. [实验矩阵](#5-实验矩阵)
6. [模块设计](#6-模块设计)
7. [统计严谨性与缺失项](#7-统计严谨性与缺失项missing-清单)
8. [执行环境](#8-执行环境)

---

## 1. 摘要

### 1.1 项目现状

本项目提出"解耦式认知诊断"：把学生对已测知识点（TKC, tested knowledge concepts）与未测知识点（UKC, untested knowledge concepts）的掌握状态解耦——TKC 分支接受作答行为监督，UKC 分支不接受任何直接行为监督、只接受知识结构先验与从 TKC 侧传播来的认知证据，再经覆盖自适应门控融合输出预测与逐知识点掌握度。

代码现状（全仓约 1 万行）：

- 核心传播模块 `models/hetero_propagation.py` 仅 277 行，UKC 分支学生无关（P1）——"薄"不只是代码量问题，而是现有模块没有实现论文声称的机制（个性化诊断未测知识点）。
- `models/decoupled_cdm.py` 1204 行，但约 850 行为适配器栈（虚胖，多为平滑计数统计打补丁，归因不清，P4）。
- 概念转移图默认从全量数据（含验证/测试标签）构建（P3，数据泄漏）。
- 读出层全局池化，逐知识点状态算完即弃，无掌握度输出、无法算 DOA（P2）。
- readout 无单调约束，guess/slip 无界且以认知状态为条件（P5，可辨识性缺失）。
- 训练为转导式全批、历史非时序（P6）。
- 训练引擎（1415 行）、数据管线（约 730 行）、评测脚本（约 5900 行）是模型无关的成熟管道，带泄漏守卫、三种训练模式、checkpoint/早停/SWA、coverage-slice 与 history-hiding 压力测试——这些恰是验证后续修复所必需的仪器。

### 1.2 核心结论：hybrid v2 路线

存活率评估判决为 **hybrid（混合方案）**：同仓库、同训练框架内新建 v2 模型，旧模型冻结为对照，**坚决不换仓库重写**。理由：

- 全仓约 1 万行代码中，真正被五项修复判死刑的只有 `decoupled_cdm.py` 约 850 行适配器栈（修复 P4 本就要求移出主线，留在冻结 v1 中一行不删）和 `hetero_propagation.py` 的 UKC 路径（约 100 行）。加权估算约 **75% 的代码在混合方案下原样存活**。
- 重写的两个致命代价：`results/experiment_results.csv` 在 `.gitignore` 中且本 checkout 不存在——它是只存在于实验机上的本地追加台账（`train.py:893` 追加），换仓库即孤立全部历史实验记录；仓库无任何 tests/ 目录，可运行的旧基线是唯一回归护栏，重写期间将同时失去"能跑的模型"和"能比的数字"。

对抗性审稿在肯定 hybrid 结论方向正确、证据扎实（引擎关键字接口、gitignore 台账、无测试护栏三点均经核实）的同时，指出了三处足以翻车的结构性弱点：

1. **论证逻辑**："行为污染/净化/记忆 vs 推断"这组核心修辞没有一个可操作定义；M2 的漂移指标与修复 P1 的目标直接冲突（个性化必然带来漂移）；M3 的"退化平缓=更好"存在方向性混淆（v1 学生无关的 UKC 分支恰好会伪装成鲁棒）；"两难困境"把待验证的经验命题（KaNCD 会落入失效角）当成了公理——这条故事线必须先跑 KaNCD 的 none_seen 数字再定稿，并把修辞全部替换为可测量表述。
2. **自洽性**：模块四的强版本（掩码概念 BCE）与"UKC 无直接行为监督"的核心主张正面矛盾，此前四份报告无一发现；实验矩阵的主表基线漏掉了文献报告钦定的三个必选对手（KaNCD/KSCD/ORCDF）；DisKCD 的"必须对比"没有任何落地方案。
3. **工程判断**：已核实 full_batch 模式对全体学生一次性前向（`engine.py:773`），逐学生传播的稠密实现在 Junyi 量级不可行，v2 事实上会被锁死在 minibatch 模式——"机械改动"的定性需修正为"接口协议＋稀疏实现＋训练模式对齐"三项实质工作。

**可辩护的论文版本**：贡献收缩为"**监督解耦的归纳偏置（以情景式共享传播函数的形式，诚实声明监督路径）＋ 未测概念诊断的分层留出评测协议（含留出 DOA、MND、防泄漏全链路）**"，机制新颖性降为次级卖点；承诺从"不牺牲整体"改为"**整体持平前提下失效切片显著领先**"。按此收缩，配齐统计严谨性（5 种子＋显著性＋划分种子）与缺失基线，这份计划有成为一篇扎实 SIGIR/KDD 投稿的底子；按原样执行，则大概率死于"两难困境不成立"或"与 DisKCD 差异不足"其中之一。

---

## 2. 已确认的问题清单（P1–P6）

以下问题全部经逐条代码核查确认（status: confirmed），行号以当前仓库为准。

| 编号 | 问题 | 关键位置 |
|---|---|---|
| P1 | UKC 分支学生无关，未实现"个性化诊断未测知识点" | `models/hetero_propagation.py:156, 126-127, 149-150` |
| P2 | 读出全局池化，逐知识点状态算完即弃，无法算 DOA | `models/decoupled_cdm.py:374-395, 387-390` |
| P3 | 概念图从全量数据（含 valid/test 标签）构建，泄漏 | `data/concept_graph.py:37-54`；`configs/defaults.py:10` |
| P4 | 适配器堆叠（7 个生效模块，1 个无 flag 常开），归因不清 | `models/decoupled_cdm.py:274-278, 393-395`；`run_assist09_baseline.sh:34-46` |
| P5 | readout 无单调约束；guess/slip 无界且以认知状态为条件 | `models/decoupled_cdm.py:248-249, 480-492` |
| P6 | 转导式全批训练、dense 矩阵、历史非时序 | `trainers/engine.py:773, 901, 1029` |

### P1 — UKC 分支学生无关（致命，核心机制未实现论文主张）

- `models/hetero_propagation.py:156`：`ukc_states = student_ukc_mask.unsqueeze(-1) * ukc_neighbor_component`，其中 `ukc_neighbor_component` 由 126–127 行的 `concept_graph @ ukc_concept_to_concept(concept_embeddings)` 计算，经 149–150 行 expand 广播到所有学生——学生间仅差一个 0/1 掩码。对同一 TKC/UKC 划分的所有学生，UKC 输出完全相同。
- TKC 分支的图邻居消息（126–127 行）同样学生无关。
- 每学生仅有一个全局标量融合门 w_u：52–56 行定义，输入为 `[coverage, tkc_mean, ukc_mean]`（187–188 行），161–166 行融合——把学生对所有题的 TKC/UKC 混合比压成一个数，粒度过粗。
- 旁证（承认 P1 的膏药）：`models/decoupled_cdm.py:782-896` 的 `student_conditioned_ukc_readout_residual`，849–850 行对 `tkc_states`/`ukc_states` 做了 `.detach()`——梯度无法回流塑造传播；893–896 行 `target_mask = (seen_concept_count<=0) & (mean_neighbor_count>0)` 证实其只在目标知识点全未测时激活。
- 后果：同覆盖不同水平的学生 UKC 预测相同，学生-概念留出实验下留出 AUC 接近先验；UKC 表征在学生维度完全塌缩，ORCDF 的 MND 指标一测便知（MND≈0）。

### P2 — 读出层全局池化，逐知识点状态算完即弃

- `models/decoupled_cdm.py:374-395`：`student_state = propagated.student_state[state_target_student_ids]`，主读出只用全局池化的 student_state 与 q_repr 做 NCF 匹配，`match_inputs = [h, q, h*q, |h-q|]`（387–390 行）。
- 逐知识点的 tkc/ukc 状态只进可选残差适配器、不进主读出。
- 后果：模型退化为矩阵分解，无法输出逐知识点掌握度、无法计算 DOA，整个"认知诊断"定位不成立。
- 修复所需的 topk-gather 取目标题知识点状态的现成代码模式在 837–861 行（残差内部）。

### P3 — 概念图从全量数据构建（数据泄漏）

- `data/concept_graph.py:37-54`：correct-correct 相邻转移逐生计数（逐生相邻两次全对才计数），包含 valid/test 标签——泄漏确认。
- `scripts/build_assist09_transition_graph.py:19-23`：默认吃 `data.csv`（全量），`--interactions` 可指向 `train.csv`。
- `configs/defaults.py:10`：assist_09 训练默认使用这张泄漏图。
- `scripts/split_student_concept_holdout.py:382-384`：默认行为把全量图拷贝为"外部先验"（`--no-copy-transition-graph` 开关存在于 40–47 行，但默认不启用）——holdout 划分下默认仍泄漏。
- 论文主张恰是"结构先验推断未测概念"，先验被测试标签污染等于循环论证；ORCDF（KDD 2024）已确立"仅用训练数据构图"的规范，一旦被审稿人发现是直接拒稿理由。

### P4 — 适配器堆叠，贡献归因不清

- `models/decoupled_cdm.py` 实测 1204 行，其中约 850 行为适配器栈（init 校验/属性约 250 行＋forward 分支约 110 行＋残差构建器约 490 行）；115–228 行为适配器超参校验＋赋值样板（实测约 114 行，提案称 250 行略夸大但量级对）；全部适配器 zero-init（318–325 行）。
- 隐藏适配器：`cognitive_difficulty_adapter` 于 274–278 行定义、318 行 zero-init、393–395 行**无条件**加到 `cognitive_logits` 上，`train.py` 中无对应开关——是当前所有实验的隐形混淆变量。其输入是 detach 过的 `match_inputs`（391 行）。
- 官方基线配置核查（修正原表述）：`scripts/run_assist09_baseline.sh:34-46` 实开 **6 个可选适配器**（high-concept-logit、pairwise-history、gs-difficulty、interpretable-readout-expert(3)、student-conditioned-ukc-readout、concept-evidence-readout）；`concept_evidence_prior_residual` 与 `history_evidence_logit_prior_residual` 并未启用；加上常开的 `cognitive_difficulty_adapter` 才凑成 7 个生效模块。
- 后果：审稿人会要求逐一消融＋纯计数统计 logistic 基线，证明增益来自 TKC/UKC 解耦结构而非历史频次先验；裸模型（零适配器）的数字必须出现在正文。

### P5 — 无单调约束的读出 + 无界且认知耦合的 guess/slip（可辨识性缺失）

- `cognitive_match_mlp` 无单调性约束：掌握度上升，预测概率可以下降，诊断解释失效（违反 NCDM 确立的单调性假设，ID-CDF 批评的对象）。
- `models/decoupled_cdm.py:248-249`：guess/slip 为每学生 `nn.Embedding(num_students, 1)`——既破坏可辨识性又破坏归纳性。
- 482–485 行：conditional 模式下 `non_cognitive_inputs = cat([student_state, q_repr])`——认知信号可从非认知通道泄漏，g/s 与掌握度不可分。
- 491–492 行：裸 sigmoid，g/s 可逼近 1，无 g_max/s_max 上界。
- `gs_mode` 仅有 constant/conditional 两档（`decoupled_cdm.py:115`，`train.py:172`），可按提案扩展。

### P6 — 转导式全批训练、dense 矩阵、历史非时序

- `trainers/engine.py:773`：full_batch 模式每 epoch 对全体交互一次性调用 `model(...)`，不做学生子集化（`use_student_subset=True` 仅出现在两个 minibatch 路径，行 901/1029）。
- ID 映射跨 train+valid+test（转导式，见存活率报告对数据管线的备注）；历史张量非时序构建。
- 在 ICDM（WWW 2024）系统提出的转导 vs 归纳评测对照下显得过时；"历史非时序"是审稿常规追问，协议建议要求至少在一个数据集上做时间序切分敏感性分析。

---

## 3. 迁移计划（hybrid，七步）

### 3.1 组件存活率分析

| 组件 | 规模 | keep_ratio |
|---|---|---|
| 数据管线 `data/` | 约 730 行 | 0.90 |
| 传播模块 `models/hetero_propagation.py` | 277 行 | 0.55 |
| 读出层 / DecoupledCDM 核心（非 adapter） | 约 250 行 | 0.45 |
| Adapter 栈（decoupled_cdm.py 内） | 约 850 行 | 0.15 |
| 训练引擎 `trainers/engine.py` | 1415 行 | 0.85 |
| 脚本与评测工具 `scripts/` | 约 5900 行 | 0.85 |
| 实验日志与工具 `utils/` + results 约定 | 约 240 行 | 0.95 |

各组件要点（来自存活率报告，均经核实）：

- **数据管线（0.90）**：五项修复几乎不动它。历史张量（掩码、响应矩阵、证据计数）本来就只用 train split 构建，修复 P1/P2 所需的输入张量（student_tkc_mask、per-concept evidence）已经存在。P3 的泄漏点不在 `build_transition_matrices` 本身（它只接收一个 DataFrame），而在两个调用点：`build_assist09_transition_graph.py` 喂了全量数据、`split_student_concept_holdout.py` 默认把旧图拷贝为"外部先验"（第 41–44、301、383–384 行）。改这两处调用点即可。遗留问题（不在五项修复范围内）：itertuples 逐行循环较慢、ID 映射跨 train+valid+test（转导式）。
- **传播模块（0.55）**：修复 P1 主要是替换 UKC 路径（156 行及其上游静态邻居消息）；但 TKC 支路（对/错习题变换、行为门控、图消息融合门，约 102–155 行）、dual-graph 融合、masked_average、gate 先验初始化等约一半代码原样可用。建议不在原文件上改，而是新建 `personalized_propagation.py` 复用这些构件——`PropagationOutput` dataclass 加字段即可。原模块保留供冻结基线使用。
- **读出层 / CDM 核心（0.45）**：修复 P2 重构 forward 匹配逻辑（374–395 行的池化匹配改为 gather 目标习题概念的 per-concept 状态）；修复 P5 替换 `cognitive_match_mlp`（改为正权/单调 MLP 或 NCDM 式交互）并给 guess/slip 头加界。能存活的：embedding 表、`exercise_difficulty`、`_summarize_exercise_concepts`、`_build_exercise_q_representation`（注意力池化）、g/s 双头结构骨架、`DecoupledForwardOutput` 接口。建议新建 `DecoupledCDMv2` 类而非原地改——原类冻结做对照。
- **Adapter 栈（0.15）**："不存活"恰是目的而非损失。其中 `_build_history_evidence_logit_prior_residual` 和 concept-evidence 系列的平滑计数逻辑值得抽出，单独立为"纯历史统计先验"基线模型（回答 P4 归因问题）。混合方案下这 850 行留在冻结 v1 类里一行不动，v2 完全不带；不需要删代码，只需要不迁移。
- **训练引擎（0.85）**：引擎与模型的耦合极浅——四个 `model(...)` 调用点（evaluate_model＋三个 epoch 函数）全用固定关键字接口；仅三处摸模型内部：`build_exercise_difficulty_prior_target`（395 行）、`_build_history_evidence_logit_prior_residual`（1117 行，只在默认权重为 0 的 alignment loss 里）、`exercise_difficulty.weight`（1413 行）。checkpoint 选择、早停、SWA/EMA、泄漏守卫 `_validate_history_visibility`（73–83 行）、三种训练模式全部通用。v2 只要把 forward 设计成现有 kwargs 的超集，引擎改动是机械的；1097–1415 行的 alignment loss 家族随修复 P4 沦为休眠代码，加 hasattr 守卫即可、不必删。
- **脚本与评测工具（0.85）**：外部基线 runner（pyedmine/scd/svgcd，约 1670 行）零改动；coverage-slice、history-hiding stress、slice-analysis 等评测脚本消费标准 bundle＋checkpoint 接口，加 `--model` 分发即可复用——这些正是验证 v2 是否真的改善 UKC 诊断所必需的工具，重写会连它们一起丢掉。需要实质修改的只有三个：`build_assist09_transition_graph.py`（只喂 train）、`split_student_concept_holdout.py`（默认改为不拷贝旧图/从 holdout 后的 train 重建）、`train.py`（加 `--model v2` 与新 flag）。train.py 的 84 个旧 flag 保留：冻结基线复现需要它们，它们同时是这项研究的实验台账。新增 DOA 评测脚本是纯增量。
- **实验日志与工具（0.95）**：utils 是小而通用的辅助（metrics/logging/io/seed/device），唯一改动是 `metrics.py` 加 DOA。关键发现：`results/` 在 `.gitignore` 里且本 checkout 中不存在——`experiment_results.csv` 是只存在于实验机上的本地追加日志（train.py:893 追加），换仓库重写会直接孤立全部历史实验记录；且仓库里没有 tests/ 目录，回归安全完全依赖重跑基线对数。这两点都强烈反对推倒重写。

### 3.2 七个迁移步骤

**第一步（锚点）**：给当前 commit 打 tag（如 `v1-freeze`）；声明 `models/decoupled_cdm.py`、`models/hetero_propagation.py` 只读。

**第二步（先修数据层泄漏，P3）**：改 `scripts/build_assist09_transition_graph.py` 只喂 train split；`scripts/split_student_concept_holdout.py` 默认不拷贝旧图（翻转 `--no-copy-transition-graph` 默认值），改为从 holdout 后的 train 重建；图产物带版本目录（如 `transition_graph_trainonly_v2/`）。然后用 train-only 图重跑一次 v1 官方基线（`run_assist09_baseline.sh`），把结果作为后续一切比较的新锚点写入日志并注明 graph_version。

> **审稿修正（严重度：低）**："一天完成"只算了代码改动。correct-correct 转移图在只用 70% 训练作答后会显著变稀（低频概念对的边直接消失），图连通性变化会级联影响：UKC 节点可达性（部分 UKC 节点在 train-only 图上失去全部 TKC 邻居，静态先验回退路径触发比例上升）、门控先验的合理初值、holdout 划分下"从 holdout 后的 train 重建"的图更稀。交付物须加一份**图诊断报告**：全量图 vs train-only 图 vs holdout-train 图的边数、密度、UKC 节点的 TKC 邻居覆盖率、连通分量对比；若 UKC 孤立节点比例超阈值（如 10%），提前设计回退策略（专家先修图补边、相似度图兜底）；A5 泄漏量化消融的解读须与图稀疏化效应分离（泄漏效应 = 全量图增益 − 同等密度随机降采样图增益）。

**第三步（新传播模块，P1）**：新建 `models/personalized_propagation.py`，复用现有 TKC 支路、门控与 masked_average 构件（约一半代码可直接搬）；UKC 路径改为"先算出学生的 TKC 概念状态，再沿概念图传播到 UKC 节点"，保持"UKC 不受作答直接监督"的原则（监督信号只经由 TKC 间接流入）。`PropagationOutput` 加字段而不改旧字段。

> **审稿修正（严重度：中）**：已核实 `engine.py:773` 的 full_batch 每 epoch 对全体交互调用一次 `model(...)`、不做学生子集化。修复 P1 后传播从共享 (K,D) 变为逐学生 (S,K,D)，稠密 (S,K,K) 边张量在 assist09 约 250MB——未计 L 层、反向传播保存的激活与梯度（实际 3–5 倍）；Junyi 量级（S≈万级, K≈700）下是 TB 级，"按学生分块或稀疏化"只是一句话。**v2 在 full_batch 模式下大概率直接不可用**，被迫全线切到 `student_recompute_minibatch`——这不是"机械改动"，而是改变了优化动态（与 v1 全批结果的可比性）。应对：(1) 接受"v2 只支持 student minibatch 模式"为设计决定并显式写入迁移计划；(2) 实现上放弃稠密 (S,K,K)——概念图本就稀疏，用共享 nnz 边索引＋逐学生边值 (S, nnz) 做 scatter/segment-sum 传播，内存 O(S·nnz·D 分块)；(3) 把"**v1 也在 minibatch 模式下重跑一遍**"加入锚点实验，消除训练模式混淆变量后再比 v1 vs v2。

**第四步（新模型类，P2＋P5）**：新建 `models/decoupled_cdm_v2.py`。读出层 gather 目标习题 Q 向量对应概念的 per-concept tkc/ukc 状态做目标概念感知匹配，天然产出逐知识点掌握度；认知匹配用正权（单调）MLP 或 NCDM 式交互；guess/slip 加上界（如 sigmoid 后乘 0.3/0.2）且不以认知状态为条件。forward 签名设计为现有关键字接口的超集，engine 四个调用点即可通用。不带任何 adapter。

> **审稿修正（严重度：中）**："评测脚本加个 --model 分发即可复用"（keep_ratio 0.85）低估了诊断类脚本与模型内部表征的形状耦合：`evaluate_gate_diagnostic`/`visualize_ukc`/M4 新脚本都直接前向提取 propagation 的 tkc_states/ukc_states，而 P1 修复后 `ukc_states` 从共享 (K,D) 变为逐学生 (S,K,D)——全量学生提取的内存与这些脚本假设的张量形状同时改变；E4/E5 还依赖 v2 才有的逐概念掌握度头。应在 v2 上定义**稳定的诊断接口协议**（如 `model.export_mastery(student_ids) -> (B,K)` 与 `model.export_branch_states(student_ids)`），按学生分块流式导出；所有诊断脚本改为消费该接口而非直接摸 propagation 输出。此接口工作显式纳入第三/四步，而不是归入"机械改动"。

**第五步（基线归因，P4）**：把 `history_evidence_logit_prior` 的平滑计数逻辑抽成一个独立的"纯历史统计先验"基线模型 B0（不含图、不含传播），回答"增益来自解耦还是计数"的归因问题。`engine.py:1117` 对模型私有方法的调用加 hasattr 守卫。

**第六步（接线与评测）**：`scripts/train.py` 加 `--model {v1,v2}` 分发，84 个旧 flag 原样保留（v1 复现需要，也是实验台账）；`utils/metrics.py` 加 DOA；`evaluate_coverage_slice` / `history_hiding_stress` 等脚本加模型分发后直接复用。v2 结果写入独立的 `results/experiment_results_v2.csv`，避免列漂移。

**第七步（消融矩阵定案）**：同一 train-only 图、同一 split、同一组 seed 上跑齐四列——v1 全 adapter、v1 裸模型、纯计数先验基线 B0、v2——特别在 concept-holdout 与低覆盖率切片上比较，这是论文叙事（UKC 诊断能力）的直接证据。

**时间预期**：第二步一天内可完成（代码部分；图诊断报告另计）；第三、四步是核心工作量（新代码约 500–700 行，大部分构件可搬）；全程任何时刻仓库里都有一个能跑、能复现历史数字的基线。

### 3.3 迁移风险清单

1. **图重建改变数据本身**：train-only 图上的新结果与旧 `experiment_results.csv` 里的全量图结果不可直接比较。必须先用 train-only 图重跑一次冻结基线、建立新对照锚点，并给图工件加版本号，否则整个消融矩阵失去参照系。
2. **engine 关键字接口**：四个 `model(...)` 调用点用硬编码关键字；若 v2 的 forward 签名不是现有 kwargs 的超集（比如个性化 UKC 需要新张量），会在四处同时炸。设计 v2 接口时先定协议再写实现，或在 engine 加一个薄的模型分发层。
3. **engine:1117 私有方法调用**：engine 直接调用模型私有方法 `_build_history_evidence_logit_prior_residual`；若有人对 v2 开启 `history_evidence_cognitive_alignment_weight > 0` 会 AttributeError。需加 hasattr 守卫或在 v2 上显式报错。
4. **结果 CSV 列漂移**：`results/experiment_results.csv` 是 gitignore 的本地文件且列结构由 summary_row 决定；v2 新增字段时 `append_summary_csv` 的列漂移可能损坏历史行的可读性——建议 v2 写入独立的 `results/experiment_results_v2.csv` 或先做列并集迁移。
5. **无单元测试**：冻结基线是唯一的回归护栏。动 `hetero_propagation.py` 或 `decoupled_cdm.py` 任何共享代码前，先固化一条小数据 smoke run（固定 seed，记录 AUC 到小数点后 4 位）作为快速回归检查。
6. **显存与训练模式**：个性化 UKC（学生态沿图传播）会把 UKC 支路从"一份共享张量"变成 S×K×D 的逐学生计算；full_batch 转导模式下显存上升，可能被迫默认切到 `student_recompute_minibatch`——这本身又是和历史结果可比性的变量（优化动态不同）。
7. **混合方案的社会性风险**：旧模型＋84 个旧 flag 留在仓库里，容易诱惑"再给 v1 打一个 adapter 补丁"。需要纪律：v1 目录只读，新想法一律进 v2；否则混合方案退化为现状延续。

---

## 4. 相关工作与定位

### 4.1 论文清单（17 项）

**1. NCDM / NeuralCD（Neural Cognitive Diagnosis for Intelligent Education Systems）** — AAAI 2020

- 相关性：领域奠基工作，确立神经认知诊断标准范式（学生/习题嵌入＋Q 矩阵掩码＋单调性假设的正权重 MLP）。本项目的 readout 违反其单调性假设（P5），审稿人会以它为参照。
- 评估协议：随机作答划分（按学生内作答随机切分）；AUC/ACC/RMSE；首次提出 DOA（Degree of Agreement）作为解释性指标——DOA 只在有作答记录的概念上定义，未涉及未测概念。数据集 ASSIST09、Math（私有）。

**2. KaNCD（NeuralCD 扩展版）** — IEEE TKDE 2022

- 相关性：最早正式提出"低知识覆盖问题（low knowledge coverage）"——学生掌握度向量中大量概念无作答支撑——并用学生×概念低秩隐因子分解外推未测概念掌握度。这是本项目核心主张最直接的"隐式"竞争者：**如果解耦式 UKC 分支打不过 KaNCD 的低秩外推，论文不成立。必选基线**。
- 评估协议：随机作答划分；AUC/ACC/RMSE＋DOA；ASSISTments、SLP、Junyi。未对未测概念做专门留出评测，外推质量只通过整体预测指标和 DOA 间接体现——这正是本项目可以补的协议空白。

**3. KSCD（Knowledge-Sensed Cognitive Diagnosis）** — CIKM 2022

- 相关性：引入概念间隐式关联的可学习概念嵌入来缓解知识覆盖稀疏，与 KaNCD 同属"隐式概念关联"路线，是常规基线（ORCDF/ICDM 均对比）。
- 评估协议：随机作答划分；AUC/ACC/RMSE＋DOA；ASSISTments、Junyi 等。无未测概念专门评测。

**4. RCD（Relation Map Driven Cognitive Diagnosis）** — SIGIR 2021

- 相关性：图式 CDM 的代表（学生-习题-概念多层关系图＋概念依赖图注意力传播）。本项目的传播骨架与之最接近，是必须重点区分的对象。其概念依赖图部分由数据统计导出且代码库未说明只用训练集构图，与本项目 P3 是同类问题；ORCDF（KDD 2024）已明确改为"仅用训练数据构图"，形成事实上的新规范。
- 评估协议：随机作答划分；AUC/ACC/RMSE；ASSIST09、Junyi。图构建脚本基于响应日志，README 未声明 train-only；无未测概念评测；大数据集上有 OOM 扩展性问题（ORCDF 指出）。

**5. HierCDF（Bayesian Network-based Hierarchical Cognitive Diagnosis Framework）** — KDD 2022

- 相关性：用贝叶斯网络把概念层级（先修结构）注入任意 CDM，代表"结构先验驱动掌握度"路线——本项目 UKC 分支"仅由图结构推断"的思想与其同源，应引用并对比：HierCDF 的结构影响是**学生条件化的**（基于父概念掌握度），而本项目当前 UKC 分支是学生无关的（P1），这一对比恰好暴露短板。
- 评估协议：随机作答划分；AUC/ACC/RMSE＋DOA；ASSIST09、Junyi 等；层级来自专家标注先修关系（非数据统计，天然无泄漏）。

**6. SCD（Self-supervised Graph Learning for Long-tailed Cognitive Diagnosis）** — AAAI 2023

- 相关性：针对长尾（作答稀疏）学生的自监督图对比学习。处理"学生级稀疏"，与本项目"学生内概念级稀疏"互补，可作稀疏场景基线并用于切片对比。仓库已有 `scd_baselines.py`。
- 评估协议：随机作答划分＋按学生作答量分桶的长尾切片评测（重点报告稀疏学生上的 AUC/ACC 提升）——文献中最接近"覆盖率切片"的先例，可直接类比到概念覆盖率分桶。

**7. TechCD（Leveraging Transferable Knowledge Concept Graph Embedding for Cold-Start Cognitive Diagnosis）** — SIGIR 2023（注意：不是 KDD 2023）

- 相关性：用教学知识概念图（KCG）作跨域中介，把老域学生认知信号传到零样本冷启动域；GCN 丢弃底层的技巧。与本项目同样用"概念图传播补无监督区域"，但设定是跨域学生级冷启动、不是学生内未测概念——定位区分的关键引文。
- 评估协议：跨域冷启动协议（源域训练、目标域测试）；AUC/ACC；概念图为外部教学先验（非数据统计）。

**8. DCD（Disentangling Cognitive Diagnosis with Limited Exercise Labels）** — NeurIPS 2023

- 相关性：**名称撞车风险**——同叫"解耦/Disentangle CD"，但语义完全不同：它解决 Q 矩阵标签不全（习题-概念标注缺失），用组式解缠＋有限标签对齐。若本项目标题用 decoupled/disentangled，引言必须显式区分，否则审稿人会混淆。
- 评估协议：部分习题有概念标签的半监督设定；学生表现预测（AUC/ACC）＋概念层面有限真实标签对齐验证；随标签稀疏度报告性能退化曲线。

**9. Zero-1-to-3（Domain-level Zero-shot Cognitive Diagnosis via One Batch of Early-bird Students）** — AAAI 2024

- 相关性：域级零样本诊断（早鸟学生认知信号迁移＋模拟作答生成）。属于域级冷启动谱系，引用以划清"我们不是冷启动，而是同一学生内的概念级盲区诊断"。
- 评估协议：六个真实数据集；域级零样本划分；AUC/ACC＋下游习题推荐任务验证诊断结果可用性——"用下游任务间接验证不可直接评测的诊断量"是值得借鉴的手法。

**10. ICDM（Inductive Cognitive Diagnosis for Fast Student Learning in Web-Based Online Intelligent Education Systems）** — WWW 2024

- 相关性：系统性提出转导 vs 归纳评测之分（新学生不重训练即诊断）。本项目 engine 的全批转导训练＋dense 矩阵（P6）在此对照下显得过时；其新学生留出协议可直接借用作附加实验。
- 评估协议：FrcSub/EdNet-1/Assist17/NeurIPS20(Eedi) 四数据集；80/20 作答划分＋20% 学生整体留出为 unseen（归纳场景）；AUC/ACC/RMSE/DOA＋新学生专用 ACC；基线含 DINA/MIRT/NCDM/RCD/KSCD/KaNCD 及重训练变体；报告推断延迟（26ms vs 重训分钟级）。

**11. ID-CDF（Towards the Identifiability and Explainability for Personalized Learner Modeling: An Inductive Paradigm）** — arXiv 2309.00300（bigdata-ustc 系）

- 相关性：把可辨识性（identifiability）和单调性条件作为诊断函数的显式约束，response-proficiency-response 范式。本项目 guess/slip 无界＋非单调 readout（P5）正是它批评的对象；引用它来论证加单调约束与有界 g/s 的必要性。
- 评估协议：AUC/ACC/RMSE＋DOA；强调只有同时用 Q 矩阵信息和单调性约束 DOA 才显著提高；做了可辨识性诊断实验（同一响应模式应映射到同一诊断结果）。

**12. ORCDF（An Oversmoothing-Resistant Cognitive Diagnosis Framework）** — KDD 2024

- 相关性：当前图式 CDM 的 SOTA 框架，且明确写出"ResG 仅用训练数据构建"——防泄漏构图的正面规范引文，直接支撑修 P3 的动机。其 MND 过平滑指标对本项目尤其重要：UKC 分支学生无关（P1）意味着 UKC 表征在学生维度完全塌缩，MND≈0，审稿人用此指标一测便知。
- 评估协议：Assist17/EdNet-1/Junyi/XES3G5M；作答 7:1:2 划分；构图 train-only；AUC/ACC＋DOA＋MND（学生掌握度两两 l2 距离均值/概念数）；基线 IRT/MIRT/NCDM/CDMFKC/KSCD/KaNCD/RCD/LightGCN/HierCDF；下游 CAT（Random/MAAT/BECAT 策略，第 5/10/15 步的 AUC/ACC）验证诊断可用性。

**13. ISG-CD（Exploring Heterogeneity and Uncertainty for Graph-based Cognitive Diagnosis Models）** — KDD 2025

- 相关性：图式 CDM 最新工作，处理响应边的语义不确定性（答对可能是猜的、答错可能是失误）——与本项目 guess/slip 层动机相同但实现于图层面。展示 KDD 审稿人当前对图式 CD 的关注点：边不确定性、DOA 提升幅度（>1% 即可发表级）。
- 评估协议：三个真实数据集；AUC＋DOA（强调 DOA 稳定提升 1%+）；以 KaNCD 为骨干，对比 RCD/SCD/ORCDF。

**14. DisKCD（Disentangling Heterogeneous Knowledge Concept Embedding for Cognitive Diagnosis on Untested Knowledge）** — arXiv 2405.16003（2024-05 首发，2024-10 v2，截至检索仍未见正式发表）

- 相关性：**最高优先级：与本项目正面撞车**（详见 4.2）。
- 评估协议：五数据集——JAD/SDP/Math2（线下课，带课程成绩、课件资源）＋Junyi（589 TKC/246 UKC）/ASSIST09（100 TKC/23 UKC）；核心协议 RQ1 = 用 TKC 相关作答训练、在 UKC 相关习题的作答上做预测（间接评测未测概念诊断，因 UKC 无直接标签）；ACC/RMSE/AUC＋DOA（RQ4 解释性）；嵌入插拔到 DINA/IRT/MIRT/NeuralCD/RCD 五个骨干上验证，ACC 提升 2–6%。这个"UKC 习题留出"协议正是本项目 `split_student_concept_holdout.py` 所做事情的数据集级版本。

**15. Breaking student-concept sparsity barrier for cognitive diagnosis** — Frontiers of Computer Science 2025

- 相关性：2025 年针对"学生-概念稀疏壁垒"的工作：指出学生作答只覆盖概念小子集而现有 CDM 输出全概念掌握度向量；用轻量矩阵分解层借相似学习者的模式推断未练习概念掌握，并据真实作答修正习题-概念映射，数据稀缺场景准确率提升至 6%。与 KaNCD 同属隐式外推路线的最新版，建议作为对比或至少引用。
- 评估协议：标准作答划分＋数据稀缺（稀疏）场景对比；AUC/ACC 类指标（经由 Springer 页面受限，细节未能全文核实，引用前需读原文确认）。

**16. Towards Accurate and Fair Cognitive Diagnosis via Monotonic Data Augmentation** — NeurIPS 2024

- 相关性：用单调性数据增强同时提升准确性与公平性，说明单调性已从建模假设升级为 NeurIPS 级别的独立研究对象；支撑本项目加入单调 readout 的修改方向（P5）。
- 评估协议：标准 CD 评测（AUC/ACC）＋公平性指标；通过构造保持单调性的增强样本训练。

**17. A Survey of Models for Cognitive Diagnosis: New Developments and Future Directions** — arXiv 2407.05458（bigdata-ustc 系综述）

- 相关性：领域权威综述，确认 AUC/ACC/RMSE/DOA 为标准指标组合，把 KaNCD 定位为"低知识覆盖问题"的解法，并把可解释性评测单列一节。写相关工作与指标合法性论证时的引用锚点。
- 评估协议：综述性质；总结心理测量系与机器学习系 CDM 的评测惯例；**未系统讨论图构建泄漏问题——这本身说明该问题尚无标准处理，是本项目可主张的贡献点**。

### 4.2 DisKCD 撞车警告

DisKCD（arXiv 2405.16003）与本项目**正面撞车**：它使用一模一样的 TKC/UKC 术语（tested/untested knowledge concepts），同样把概念集拆为 K_tested ∪ K_untested，用异构关系图（学生-习题、习题-TKC、TKC-TKC、TKC-UKC、UKC-UKC）分层消息传递把信息从 TKC 传到 UKC，并自称"首次用关系图网络评估学生对 UKC 的掌握"。**必须引用、对比、并在术语和贡献点上做出明确差异化，否则新颖性直接被否**。好消息：它仍是预印本，窗口仍在；且据文献报告，其 UKC 表征同样主要靠图结构传播、未解决学生个性化问题，也没有防泄漏协议或单调性约束。

文献报告给出的差异化路径：

1. 不要把 TKC/UKC 缩写当成首创卖点，可改用 observed/unobserved concept mastery 或 within-student concept-level cold start 之类表述；
2. 把差异点放在 DisKCD 没有的东西上——按学生覆盖率自适应的门控融合（w_u）、"UKC 不受任何直接行为监督"的显式归纳偏置及其对应的防泄漏评测协议、单调可辨识的 readout、以及（修复 P1 后的）学生条件化 UKC 传播；
3. 直接采纳并扩展它的"UKC 相关习题留出"评测为分层的学生-概念留出协议，把协议本身写成贡献（社区目前没有未测概念诊断的标准评测）。

> **审稿修正（严重度：高）**：文献报告断言 DisKCD"UKC 表征同样主要靠图结构传播、未解决学生个性化问题"，并把这作为新颖性窗口的主要依据——**该断言未经核实**。DisKCD 的异构图包含学生-习题、习题-TKC 边，消息传递从学生作答出发流向 UKC，其 UKC 表征很可能（至少形式上）已是学生条件化的。如果核实后 DisKCD 确实个性化，则修复 P1 后的模块一与 DisKCD 的机制差异收窄到"证据置信度加权＋单向传播＋无直接监督约束"这类二阶细节，新颖性主张需要整体重写。**把论文成败押在对一篇未精读预印本的单句概括上是高危行为**。修正：立即精读 DisKCD 全文并逐条核对——其 UKC 嵌入是否随学生变化、训练目标是否用 UKC 相关作答（若用了，它反而没有"无直接监督"约束，这才是本项目的真差异点）、是否有防泄漏协议与单调性。据此重写差异化段落，把差异点锚定在可验证的三件事上：**监督路径约束、覆盖自适应门控＋防泄漏评测协议、单调可辨识读出**；并按文献报告建议复现其 UKC 留出协议做直接对比，而不是只在相关工作里口头区分。

### 4.3 其他定位边界与措辞风险

- **边界一，对 RCD 式图扩散**：RCD/ORCDF/ISG-CD 把图当作增强所有概念表征的手段、不区分监督来源；本工作的卖点应是"监督解耦"——测过的概念由行为信号驱动、未测概念只允许结构信号驱动，并用留出实验证明这种解耦比无差别扩散在未测概念上更准、且（通过 MND/DOA）不过平滑。**注意：如果 P1 不修，UKC 分支就退化成学生无关的图平滑先验，恰好落回被你批评的 RCD 式扩散，这条定位立不住**。
- **边界二，对冷启动系**（TechCD SIGIR 2023、Zero-1-to-3 AAAI 2024、ICDM WWW 2024）：它们解决的是新学生/新域整体无数据的冷启动，本工作是同一学生内部的概念级盲区，两者正交；引言里应明确"我们假设学生有充足作答但概念覆盖不全"，并引用 KaNCD（TKDE 2022）首次形式化的 low knowledge coverage problem 作为问题源头，把 KaNCD 立为最重要的隐式外推对手。
- **措辞风险**：标题若含 decoupled/disentangled，会触发与 NeurIPS'23 DCD（Q 矩阵标签缺失的解缠）和 NeurIPS'24 DisenGCD（图表征解缠）的联想，引言需一句话区分"我们解耦的是监督信号来源，不是表征因子"。
- **把"无泄漏"做成叙事优势**：指出图式 CDM 的先验构建缺乏 train-only 规范（ORCDF 是少数明确声明者），而本工作的概念图、历史张量、留出协议全链路防泄漏，恰与"诊断未测概念"主张的可信性绑定——这在 KDD/SIGIR 审稿人眼中是显著的严谨性信号。

### 4.4 审稿人期望（8 条）

1. **未测概念主张要直接证据而非整体 AUC**：必须有学生-概念级留出实验（把某学生在某些概念上的全部作答从训练与传播历史中剔除，再在这些作答上评测），并证明胜过 KaNCD/KSCD 的隐式低秩外推——否则被判"用复杂图机制重新发明了矩阵分解"。P1 在该实验下立刻现形：同覆盖不同水平的学生 UKC 预测相同，留出 AUC 接近先验。
2. **DOA 是硬性解释性指标**（NCDM 首创，RCD/KaNCD/ORCDF/ICDM/ISG-CD 全部报告），且审稿人会追问：DOA 只在有作答的概念上有定义，未测概念掌握度如何验证？需给出留出式 DOA 变体或 DisKCD 式"UKC 相关习题间接评测"，不能只给整体 DOA。
3. **图先验必须无泄漏构建**：ORCDF 已明确写出 ResG"仅用训练数据构建"，这是 KDD/SIGIR 当前规范；本项目从全量数据统计 correct-correct 转移建概念图（P3），一旦被发现是直接拒稿理由——论文主张恰是"结构先验推断未测概念"，先验被测试标签污染等于循环论证。
4. **消融归因要干净**：7 个适配器（多为学生/习题/概念正确率计数统计）叠在主模型上，审稿人会要求逐一消融＋一个纯计数统计的 logistic 基线，证明增益来自 TKC/UKC 解耦结构而非历史频次先验（P4）。裸模型（零适配器）的数字必须出现在正文。
5. **单调性与可辨识性**：readout MLP 无单调约束、guess/slip 依赖认知状态且无界（P5），会被 ID-CDF、NeurIPS 2024 单调增强这条线的审稿人质疑"掌握度上升预测概率可能下降，诊断量不可辨识"；需要非负权重/单调网络＋g/s 上界。
6. **新颖性核查**：审稿人检索 "untested knowledge concept cognitive diagnosis" 第一条就是 DisKCD（arXiv 2405.16003），术语 TKC/UKC 完全相同；不引用、不对比、不差异化 = 新颖性直接被否。同时 "decoupled/disentangled" 措辞会触发与 NeurIPS'23 DCD、NeurIPS'24 DisenGCD 的比对。
7. **实验广度**：至少 3–4 个公开数据集（仅 ASSIST09＋Junyi 会被嫌少）、标准 7:1:2 划分、5 个以上随机种子＋显著性检验、IRT/MIRT/DINA/NCDM/KaNCD/KSCD/RCD/ORCDF 基线全家桶；图式方法还会被问大数据集扩展性（RCD 在 Junyi 上 OOM 是公开梗）与新学生归纳能力（ICDM 之后成为常规问题）。
8. **诊断结果的下游可用性**：ORCDF 用 CAT、Zero-1-to-3 用习题推荐验证诊断量的实用价值；对"未测概念诊断"这种无直接标签的主张，下游任务（如据 UKC 掌握度推荐补练习题）是审稿人认可的间接验证手段。

### 4.5 评估协议建议（8 条）

1. **主表**：随机作答 7:1:2 划分（对齐 ORCDF）；数据集在 ASSIST09、Junyi 之外至少加 NeurIPS2020-Eedi（自带专家概念树）和 XES3G5M 或 EdNet-1 之一；指标 AUC/ACC/RMSE＋DOA；≥5 种子报均值±方差与显著性。ECE/Brier 可保留为附加特色（文献很少报校准，能成加分项），但不可替代 DOA。
2. **核心主张实验（论文成败所系）**：学生-概念留出协议——扩展现有 `scripts/split_student_concept_holdout.py`，对每个测试学生随机留出 20–30% 概念的全部作答，从历史张量、TKC/UKC 掩码和概念图构建中彻底剔除，在留出作答上报 AUC/ACC/RMSE；对照组必须含 KaNCD（隐式外推）、DisKCD（同题方法）、tkc_only/mean 消融；并按学生概念覆盖率分桶（如 <30%/30–60%/>60%）做切片曲线（类比 SCD 的长尾分桶）。
3. **未测概念 DOA 变体**：在留出的学生-概念对上，用留出作答正确率构造学生对的偏序，计算 held-out DOA（"两个学生谁在被隐藏概念上表现更好，模型的 UKC 掌握度是否排对"）；同时报 ORCDF 的 MND 指标验证 UKC 表征未在学生维度塌缩——修复 P1 前后 MND 对比本身就是一张好图。
4. **图先验去泄漏（修 P3）**：概念转移图只用训练划分构建并做敏感性分析；更优方案是把 Junyi/Eedi 数据集自带的专家先修关系作为外部图先验，做"训练数据图 vs 专家图 vs 无图"三方消融——既回应 RCD 式构图疑虑，又证明方法不依赖特定图来源。
5. **基线集**：必选 IRT/MIRT/DINA/NCDM/KaNCD/KSCD/RCD/ORCDF（项目已有 `pyedmine_cd_baselines.py` 可覆盖大部分）；稀疏切片上加 SCD；未测概念协议上必须复现或至少对比 DisKCD；可选 ICDM（归纳）、HierCDF（结构先验）。
6. **适配器归因（修 P4）**：把适配器拆成"计数统计先验组"和"结构组"，单独报一个纯计数 logistic 回归基线；主结果用零适配器裸模型＋至多 1–2 个必要组件，其余移到附录消融矩阵；否则贡献归因会被质疑。
7. **可辨识性（修 P5）**：readout 对学生掌握度施加单调约束（非负权重 MLP 或单调网络），guess/slip 加 sigmoid 上界（如 g≤0.4、s≤0.3）并报超参敏感性；附一个"同响应模式→同诊断"的可辨识性 sanity 实验（参照 ID-CDF）。
8. **加分实验**：(a) 新学生归纳评测——留出 20% 学生整体（ICDM 协议），用 `student_recompute_minibatch` 模式测不重训诊断；(b) 下游任务——用 UKC 掌握度做补练推荐或模拟 CAT（参照 ORCDF 的 BECAT/MAAT 设置），间接验证未测概念诊断的实用价值；(c) 时间序切分敏感性分析，回应"历史非时序"的质疑（P6）。

---

## 5. 实验矩阵

### 5.1 研究说法（story）

一段话版本：现有认知诊断模型（NCDM/KaNCD/RCD 等）把学生对 TKC 和 UKC 的状态耦合在同一组由作答行为直接监督的参数里；同时其整体指标主要由高覆盖交互支撑，性能依赖对个体历史作答模式的记忆而非对知识结构的推断。我们提出解耦式认知诊断：TKC 分支接受作答监督形成可靠认知状态，UKC 分支不接受任何直接行为监督，只接受知识结构先验与从 TKC 侧经知识图谱个性化传播来的"净化后"认知证据，再以覆盖自适应门控融合并经目标知识点感知的单调读出输出预测与逐知识点掌握度。实验设计为两条证据链：**动机组**（M1–M4）用覆盖分层退化、行为污染探针、历史遮蔽应力和 UKC 表征坍缩四个实验证明"未测知识点诊断失效/行为污染"是耦合模型的普遍且可测量的问题；**主实验组**（E1–E6）用严格学生-知识点留出协议、覆盖切片、留出知识点 DOA、低覆盖校准与门控机制分析证明解耦恰好在该失效切片上修复问题且不牺牲整体性能；**消融组**（A1–A8）定位增益来源（个性化 UKC 传播、目标感知读出 vs 计数先验适配器）并量化全量数据建图的泄漏，使归因干净。

证据链对应关系（问题实验与修复实验一一对应，是审稿人最认可的结构）：

| 动机实验 | 对应修复验证 | 口径 |
|---|---|---|
| M1 覆盖分层退化（zero/low 桶） | E3 覆盖切片胜出；E2 none_seen 子集 | 同四桶切片 / none_seen≈zero 桶 |
| M2 行为污染探针 | E6 污染抑制 | 同噪声注入协议 |
| M3 历史遮蔽应力 | E6 遮蔽鲁棒性 | 同遮蔽-重建-评测协议 |
| M4 表征两难诊断 | E4 留出 DOA（反驳两难的两个角） | 同留出划分与学生对 |

> **审稿修正（严重度：高，针对 story 的"两难困境"）**："未测概念估计要么无个性化、要么被污染"被当成公理写进说法，但它是尚未验证的经验命题，且最强反例就在必选基线里：**KaNCD 的学生×概念低秩分解既是学生个性化的，也不经过图扩散——它不在两难的任何一角**。若 KaNCD（或 RCD）在 M4 二维散点图上落在"个性化且留出预测力不差"的位置，整个论证枢纽崩塌，而文献报告自己都承认"如果解耦打不过 KaNCD 的低秩外推，论文不成立"。修正：把说法从"两难困境"降级为"**谱系与空白**"——现有方法在留出未测概念上的个性化质量未被系统评测（协议贡献），我们的方法在该协议下取得最优；并预注册两种结果的解释（见 M4 条目）。**任何情况下先跑 KaNCD 的 none_seen 数字再定叙事，这应是第 0 优先级实验**。
>
> **审稿修正（严重度：中，针对"不牺牲整体性能"承诺）**：v2 同时做了四件降拟合力的事——删 7 个适配器（其中计数先验对整体 AUC 贡献可能很大）、单调约束、有界 g/s、换掉有记忆能力的池化 NCF 读出——模块提案自己两处承认"预期指标会掉"。若 v2 整体 AUC 低于 v1 全适配器版乃至低于 ORCDF，主表会同时输给自家旧版和外部 SOTA。修正：承诺改写为可兑现形式——"**在与最强基线整体持平（差距 <0.3–0.5 AUC 点，配显著性检验）的前提下，在 none_seen/低覆盖切片显著领先且提供逐概念可辨识诊断**"。主文用帕累托图（整体 AUC × none_seen AUC）呈现，把 v1 全适配器版作为"拟合力上界、诊断力下界"的参照点画进去，主动解释差距来源（A4/A6 消融数字支撑），而不是等审稿人发现。

实验与资产总览：

| 编号 | 主题 | 新代码量（经核查修正后） | 审稿修正严重度 |
|---|---|---|---|
| M1 | 覆盖分层性能退化 | 基本零新代码（KaNCD 列 uncertain） | 中 |
| M2 | 行为污染探针 | 新组装脚本约 200 行 | 高 |
| M3 | 历史遮蔽应力测试 | 小补丁约 20–30 行（切片输出） | 高 |
| M4 | UKC 表征两难诊断 | 新小脚本（复用提取逻辑） | 高 |
| E1 | 多数据集主表 | 零新代码（多数据集图重建为非零工作量） | 高（基线集） |
| E2 | 学生-知识点留出协议（核心） | 零新代码（切换建图来源）＋DisKCD 复现约数百行 | 中 |
| E3 | 覆盖切片胜出与低覆盖校准 | 零新代码 | （继承 M1/E1） |
| E4 | 留出知识点 DOA | 新写 evaluate_doa_holdout.py 约 150 行 | 中 |
| E5 | 门控行为机制分析 | 零新代码（表征图需新写；依赖 v2 诊断接口） | 中 |
| E6 | 遮蔽鲁棒性＋污染抑制 | 几乎零增量（复用 M2/M3） | 高（继承 M2/M3） |
| A1 | 融合模式四档 | 旗标已实现 | — |
| A2 | 个性化传播 on/off（最关键单一消融） | 待模块一 | 中 |
| A3 | 目标感知读出 on/off | 待模块二 | 低 |
| A4 | 适配器归因＋B0 基线 | 可行（原"7 全开"表述 wrong） | — |
| A5 | 图泄漏量化 | 可行（先修 num_concepts 维度脚枪） | 低 |
| A6 | 单调约束＋有界 g/s on/off | gs_mode 需扩展 | — |
| A7 | 门控固定 vs 学习 | "固定常数 w"档需新增约 10 行 | — |
| A8 | 图来源与层数 | 多图已支持；2 层待模块一 | — |

### 5.2 动机实验（M1–M4）

#### M1 覆盖分层性能退化（已发表基线的普遍失效证明）

- **设计**：在 assist09/assist17/nips34（可加 junyi）标准随机划分上，用 PyEdmine 跑 DINA/IRT/MIRT/NCD/RCD/HierCDF/HyperCD（KaNCD 只需在 MODEL_SCRIPTS 加一行 kancd.py），外加 SCD、SVGCD；将测试集交互按"目标习题知识点被该生训练历史覆盖的比例"切成 zero/low(<0.5)/partial/full 四桶，报告每桶 AUC/ACC/Brier/ECE、桶内样本数与 coverage_gap = AUC(full) − AUC(low)。预期所有耦合基线 zero 桶 AUC 接近随机或大幅低于整体，coverage_gap 显著为正且跨数据集一致。
- **证据作用**：证明"对未测知识点的诊断失效"是耦合 CDM 的普遍现象而非本模型的假想问题；zero/low 桶就是论文定义的问题切片，后续主实验在同一切片上验证修复，形成闭环；整体 AUC 与 zero 桶 AUC 的悬殊差距同时支撑"报告指标被高覆盖样本主导"。
- **资产状态（已核查）**：
  - confirmed：`scripts/evaluate_coverage_slice.py:110-119` 桶定义与设计完全一致（另有 no_concepts 桶）、154–168 行每桶全指标＋count、204–208 行 coverage_gap 与 low_coverage_brier/ece 列；`scripts/pyedmine_coverage_slice.py:196-230` 桶定义与本模型侧逐字相同（同口径可比），124–179 行加载 PyEdmine best_valid 检查点并按 source test.csv 行序对齐校验（173–174 行 label 一致性断言）；`scripts/scd_baselines.py`、`scripts/svgcd_baselines.py` 在位。本模型侧基本零新代码。
  - uncertain：KaNCD 不在 `pyedmine_cd_baselines.py` 的 MODEL_SCRIPTS（28–36 行）中，"加一行"取决于远端 PyEdmine（/home/xph/jwc/pyedmine）examples/cognitive_diagnosis/train/ 下是否已有 kancd.py——本机无法验证；assist09_holdout 的 source_dir 指向远端 `/tmp/assist09_holdout_seed2024`，易失需重新生成。
- **审稿修正（严重度：中）**：覆盖桶不是随机分配的——zero/low 桶天然富集冷门概念、难题、低活跃学生和小样本，桶间 AUC 差可能大部分由样本构成（难度分布、标签基率、桶内学生数）解释而非"诊断失效"；桶内 AUC 在样本少、基率极端时本身高方差甚至无定义。三项加固：(1) 每桶报告样本数、标签基率、bootstrap 置信区间；(2) 匹配分析——按习题难度与学生总作答量分层匹配后再比桶间差，或跑 logistic 回归以覆盖率为解释变量、控制难度/活跃度，报告覆盖率系数；(3) 用同一学生的跨桶配对比较（within-student contrast）代替跨桶混合比较，消除学生能力混淆。

#### M2 行为污染探针（噪声沿图扩散污染未测估计）

- **设计**：取严格留出划分中的 holdout 学生，冻结已训好的图扩散基线（RCD，经 PyEdmine）与本模型的耦合变体，不重训。按比例 p∈{0.1, 0.2, 0.4} 翻转其训练集中 TKC 上的作答标签（分两组：与留出 UKC 在概念图上相邻的已测点 vs 图上距离远的已测点），用污染后的历史重建模型输入（RCD 重建 u-e 图，本模型重建 history tensors），测量：(a) 留出 UKC 上掌握度/预测概率的漂移 mean|Δprob| 与污染前后掌握度的 Spearman 相关；(b) 留出真实标签上的 AUC 下降。对照组是纯结构预测器（ukc_only 分支或概念难度群体先验），其漂移按构造为 0。预期耦合模型漂移显著且"相邻注入"漂移远大于"远距注入"。
- **证据作用**：把"行为污染"从修辞变成可测量——耦合/图扩散模型让原始作答噪声（猜测/失误）沿概念图传入未测知识点估计，漂移的邻接依赖性直接证明污染路径就是图传播。
- **资产状态（已核查，confirmed）**：`scripts/evaluate_history_hiding_stress.py` 的 `mask_train_history_interactions`/`build_hidden_bundle` 接受任意修改后的 DataFrame（把"删行"改成"翻转 label"即可复用重建链路：翻转 label 不改变 TKC/UKC 掩码、只改 response_matrix→只进 TKC 分支，`data/pipeline.py:201-215`、`hetero_propagation.py:130,136`，故 ukc_only 下 student_state 不变、漂移按构造为 0）；RCD 侧 `pyedmine_rcd_history_hiding.py:98` 的 `write_pyedmine_cd_file` 写 int(label) 同样支持翻转；`scripts/split_student_concept_holdout.py` 提供未测知识点的真值标签；`data/concept_graph.py` 提供选相邻/远距 TKC 的邻接矩阵。需新写约 200 行的 `evaluate_behavior_pollution_probe.py` 组装脚本（估计合理）。**注意**：对照组必须关闭 pairwise/history-prior 等直接消费 response_matrix 的适配器，否则漂移不为 0。
- **审稿修正（严重度：高）**：核心指标 mean|Δprob| 在逻辑上与修复 P1 **直接冲突**——修复 P1 的全部目的就是让 UKC 估计随学生作答变化，一个完成 P1 的模型在此探针下必然产生漂移；漂移本身不区分"响应真实信号"与"被噪声污染"。当前 v1 漂移为 0 只因它学生无关，**这是缺陷不是美德**。E6 预期"解耦模型漂移显著小于 RCD"没有任何机制保证：TKC 状态由同样含猜测/失误噪声的作答训练，"净化"只是一个未定义的形容词。修正：把指标改为可区分信号与噪声的对照式设计——对同一学生分别注入 (a) 随机标签翻转（纯噪声）和 (b) 构造性的真实能力反转（交换高低分学生的作答模式，即真实信号），报告"信号响应度/噪声敏感度"之比；或直接报告注入噪声后留出 UKC 作答上的 AUC 降幅（任务级鲁棒性）而非表征漂移。同时给"净化"一个可操作定义（例如"UKC 估计对单条作答扰动的敏感度上界由 TKC 聚合态的维度与置信度加权决定"）并实测该敏感度。

#### M3 历史遮蔽应力测试（记忆而非推断的证明）

- **设计**：评测时随机隐藏每个学生 r∈{0.2, 0.4, 0.6, 0.8} 的训练历史（3 个种子），重建模型输入后直接评测（不重训），画 ΔAUC–遮蔽率曲线；对象为 RCD（历史以图输入，可遮蔽）与本模型各变体；对 NCD/KaNCD 类嵌入模型则陈述互补事实——其学生向量与历史输入无关、遮蔽后完全不变，即根本无法对证据变化做推断，只能靠重训"背"出参数。额外报告遮蔽后 low/zero 覆盖切片的 AUC。预期依赖历史记忆的模型 ΔAUC 陡降。
- **证据作用**：证明基线的性能建立在对个体历史作答模式的记忆之上而非对知识结构的推断——耦合把"记忆"与"诊断"纠缠在一起；同时为 E6（解耦模型退化更平缓）提供同协议对照。
- **资产状态（已核查，confirmed，一处下调）**：本模型侧 `scripts/evaluate_history_hiding_stress.py`——44–45 行默认 hide-ratios=0.2,0.4,0.6,0.8、3 个种子；181–192 行逐学生随机隐藏；195–219 行 build_hidden_bundle 用 `data/pipeline.build_history_tensors` 重建全部 history 张量；52–57 行 keep-exercise-evidence 控制；输出 delta_auc。RCD 侧 `scripts/pyedmine_rcd_history_hiding.py`——113–144 行重建 k_from_e/e_from_k/u_from_e/e_from_u 四图、147–172 行符号链接冻结检查点、纯评测。**唯一缺口**：设计中"遮蔽后 low/zero 覆盖切片 AUC"两个脚本都不输出（只有整体指标），需约 20–30 行小补丁——"零新代码"下调为"小补丁"。
- **审稿修正（严重度：高）**：两处硬伤——(1) 对 NCD/KaNCD 类模型，M3 自己承认"遮蔽后学生向量完全不变、无法探测"，"记忆"指控对这类模型只能靠陈述而非测量，审稿人会说这是对转导式参数化的贬义重命名而非实验结论；(2) 对 RCD，遮蔽历史后 ΔAUC 陡降只证明"模型使用历史作为输入"，任何基于证据的推断（包括理想的诊断模型）在证据被删后都会退化——"退化陡"≠"记忆"，"退化缓"也可能只说明模型更接近学生无关的群体先验（**v1 的 UKC 分支恰是如此——退化平缓可能是 P1 缺陷的伪装**）。M3/E6 的"曲线更平缓=更好"读法存在方向性混淆。修正：主张重写为可测量版本——"转导式 CDM 缺乏对证据变化的即时响应能力（归纳性缺失）"；对嵌入类模型改测"遮蔽后重训练的成本与新学生归纳协议（ICDM, WWW 2024）下的表现"；遮蔽曲线同时报告绝对 AUC 与遮蔽后 zero/low 覆盖桶 AUC——解耦模型的卖点必须是"**遮蔽后绝对性能更高且仍保有学生间区分度（用 MND 或学生间预测方差佐证）**"，而不是单纯"降得慢"。

#### M4 未测知识点表征两难诊断（无个性化 vs 被污染）

- **设计**：在留出划分上，选取覆盖模式相近但已测表现相反的学生对（已测知识点正确率 top/bottom 四分位），提取各模型对同一批未测知识点的掌握度估计：(a) 计算学生间估计的方差与两两余弦相似度——当前本模型 UKC 分支预期方差≈0、相似度≈1（证明"学生无关"一角，代码层面 ukc_states 仅差一个 0/1 掩码）；(b) 计算基线（RCD）未测估计与留出真实标签的相关——结合 M2 证明其个性化差异主要由噪声驱动（"被污染"一角）。输出一张"个性化程度 × 留出预测力"二维散点图，耦合基线与结构先验各占一个失效角。
- **证据作用**：原设计中为全文论证的枢纽——未测知识点估计要么无个性化（结构先验/本模型现状），要么被行为噪声污染（图扩散基线），现有方法无法同时做到"个性化且干净"，从而唯一地引出 TKC→UKC 个性化净化传播这一解法。
- **资产状态（已核查，confirmed，一处出入）**：`scripts/evaluate_gate_diagnostic.py:113-159` 的 `_interaction_gate_frame` 直接调 `tower.propagation` 提取 tkc_states/ukc_states（新脚本可照抄）；`scripts/split_student_concept_holdout.py` 提供学生对与留出标签。**出入**：`scripts/visualize_ukc.py:14-35` 只画 CSV 层面的每生 TKC/UKC 数量统计，并非学习到的"表征图"，表征散点图需新写。需新写一个小脚本复用上述提取逻辑做学生间统计。
- **审稿修正（严重度：高）**：M4 把结论当前提（见 5.1 story 修正），且实验矩阵没有给出 KaNCD 不落空角时的退路。修正：M4 **改为探索性诊断图而非"枢纽证明"**，并预注册两种结果的解释——若 KaNCD 表现好，论文卖点转为"在低秩外推之上，结构先验与监督解耦带来的增量与可解释性（逐概念掌握度、DOA、校准）"；若表现差，才升格为失效证明。另据 missing 清单：M4 的方差/余弦是自造指标，应改用或并报 ORCDF 的 MND 以便与文献可比。

### 5.3 主实验（E1–E6）

#### E1 主表：多数据集整体性能（不牺牲总体的底线证明）

- **设计**：assist09/assist17/nips34（＋junyi 变体）标准划分，对比 DINA/IRT/MIRT/NCD/RCD/HierCDF/HyperCD/SCD/SVGCD 与解耦模型（含全部修复），统一 seed 协议（≥3 seeds 报 mean±std），按 valid AUC 选 checkpoint，报告 test AUC/ACC/RMSE/Brier/ECE。注意主表模型须使用 train-only 概念图与精简适配器配置（与消融 A4/A5 一致），避免带泄漏成绩。
- **证据作用**：证明解耦不以整体预测力为代价（必要的合法性论据）；ECE 列同时铺垫校准叙事。
- **资产状态（已核查）**：`scripts/pyedmine_cd_baselines.py`（MODEL_SCRIPTS 28–36 行、DATASET_SPECS 38–99 行 standard/holdout 全配、`prepare --kt-source {full,train}` 156–161 行，confirmed）、`scripts/scd_baselines.py`、`scripts/svgcd_baselines.py`、`scripts/train.py`＋`scripts/run_assist09_baseline.sh`、`scripts/evaluate.py`、`scripts/evaluate_checkpoint_average.py`。零新代码。**uncertain**：本模型在 assist17/nips34 上跑需显式传路径并为每数据集重建 transition_graph——`configs/defaults.py` 只为 assist_09 配了 concept_graph 默认，assist_17/junyi 指向 `../ConceptSkillCDM` 相对路径、nips34 无本模型侧默认（train.py 支持显式路径，非零工作量但无代码障碍）。
- **审稿修正（严重度：高）**：基线集与文献报告自相矛盾——文献报告明确列 KaNCD/KSCD/ORCDF 为必选基线（KaNCD 是头号隐式外推对手，ORCDF 是 KDD 2024 图式 SOTA 且是 train-only 构图规范的出处），但 E1 基线列表**三个最关键的对手全部缺席**，KaNCD 只在 M1 里被顺带提了一句"加一行"。审稿人拿到主表第一眼就会问"为什么没有 ORCDF 和 KaNCD"，引用了 ORCDF 的构图规范却不与其比较会显得刻意回避。修正：E1/E2/E3 的基线集统一为 **IRT/MIRT/DINA/NCD ＋ KaNCD/KSCD（隐式外推路线，核心对手）＋ RCD/HierCDF/ORCDF（图与结构路线）＋ SCD（稀疏切片）**，HyperCD/SVGCD 降为附录；确认 PyEdmine 是否内置 KaNCD/KSCD/ORCDF，若无官方实现则用作者开源代码接入统一划分，并在附录报告复现数字与原文的偏差。另见 5.1 中"不牺牲整体"承诺的改写（帕累托图呈现）。

#### E2 严格学生-知识点留出协议（核心主实验）

- **设计**：用 `split_student_concept_holdout.py` 对每个数据集生成 70/10/20 留出划分（50% 合格学生做严格知识点留出，split_summary.json 已输出 none_seen/partial/all_seen 占比）；所有模型在留出 train 上训练，概念图从 train 重建（禁用默认的图拷贝：配 `--no-copy-transition-graph`＋`build_assist09_transition_graph` 改跑 train.csv；RCD 用 `prepare --kt-source train`）。报告 test 整体与 none_seen 子集（学生从未做过该题任一知识点）的 AUC/ACC/Brier/ECE。预期解耦模型恰在 none_seen 子集上领先幅度最大。
- **证据作用**：对"能诊断未测知识点"这一核心命题的直接检验——none_seen 子集就是 UKC 预测任务本身；与 M1 的 zero 桶同口径，形成"问题在此→修复在此"的对仗。
- **资产状态（已核查，confirmed）**：`scripts/split_student_concept_holdout.py:31-39` 默认 target_eval_ratio=0.30 / test_ratio_within_eval=2/3（≈70/10/20）、`--holdout-student-frac` 默认 0.5、合格阈值 30 交互/5 知识点；194–214＋331–332 行 none_seen/partial_seen/all_seen 计数与占比写入 split_summary.json；378 行输出 `student_concept_holdout_assignments.csv`（含每生 holdout_concepts，E4 直接可用）。`pyedmine_cd_baselines.py` 的 *_holdout 数据键已配置；`evaluate_coverage_slice.py` 与 `pyedmine_coverage_slice.py` 支持 none_seen≈zero 桶切片。零新代码（只需切换建图来源）。
- **审稿修正（严重度：中）**：文献报告说未测概念协议上"必须复现或至少对比 DisKCD"，实验矩阵却**没有任何 DisKCD 的 runner、复现计划或工时估算**——它是预印本，大概率无官方开源代码，五个数据集里三个（JAD/SDP/Math2）是私有线下课数据。"必选对比"在执行层面是空头支票，论文提交时只能退化为口头引用，而这正是新颖性最脆弱的一环。修正：现在就确认 DisKCD 是否放码；若无，按其论文描述在 ASSIST09/Junyi（它用过的两个公开集）上做一个诚实标注的"我们的复现"（异构图 TKC→UKC 消息传递核心即可，不必复刻全部工程细节），附录给出复现细节与偏差声明；同时把它的"UKC 相关习题留出"协议数字与本项目协议对齐报告。这项工作量（约数百行，可基于 RCD 的图代码改）应排进实验矩阵而非留白。

#### E3 覆盖切片胜出与低覆盖校准

- **设计**：标准划分上，同 M1 的四桶切片评测解耦模型 vs 全部基线；主指标：(a) zero/low 桶 AUC 绝对提升；(b) coverage_gap 收窄幅度；(c) 限定 zero/low 桶（UKC 主导样本）的 ECE 与 Brier。评测脚本已直接输出 low_coverage_ece/low_coverage_brier 列。
- **证据作用**：证明解耦精准修复 M1 暴露的失效切片而非平均涨点；低覆盖 ECE 表明 UKC 驱动的预测在置信度上也是诚实的——这是诊断系统（而非排序系统）的关键性质。
- **资产状态（已核查，confirmed）**：`scripts/evaluate_coverage_slice.py`、`scripts/pyedmine_coverage_slice.py`（两侧桶定义逐字相同，同口径可比）。零新代码。
- **审稿修正**：继承 M1 的成分混淆加固（每桶样本数/基率/bootstrap 置信区间、匹配分析、within-student contrast）与 E1 的基线集修正（加 KaNCD/KSCD/ORCDF）。

#### E4 留出知识点 DOA（未测概念的个性化诊断力）

- **设计**：定义留出 DOA：在 E2 划分下，对每个被留出的知识点 k，掌握度 M_{·,k} 来自从未见过该生 k 上作答的模型；用留出 test 中 k 上的真实正确率排序学生对 (a,b)，DOA_UKC(k) = Σ I(M_ak>M_bk)·I(acc_ak>acc_bk) / Σ I(acc_ak≠acc_bk)，对知识点取加权平均；同时报告已测知识点 DOA 作对照。基线的掌握度取各自的天然 proficiency 向量（NCD/KaNCD/RCD 均有）；本模型的逐知识点掌握度由目标知识点感知读出（P2 修复）产出；另加"概念难度群体先验"对照证明高 DOA 不是难度效应。预期：已测 DOA 各模型接近，未测 DOA 基线跌向 0.5 而解耦模型显著高。
- **证据作用**：论文的招牌可解释性结果——解耦后的 UKC 估计能在从未观测过的知识点上正确排序学生，同时反驳 M4 两难的两个角（有个性化、且不靠污染）。
- **资产状态（已核查）**：`split_student_concept_holdout.py`（留出标签与 assignments.csv）、`evaluate_gate_diagnostic.py` 的状态提取模式、`pyedmine_coverage_slice.py` 的 `--prediction-dir` 预测导出（33、309–310 行，confirmed）。**DOA 指标全仓库尚无实现**（grep 确认），需新写 `evaluate_doa_holdout.py`（核心约 150 行，估计合理）。前置依赖：本模型侧依赖 v2 才有的逐概念掌握度头（模块二）。
- **审稿修正（严重度：中）**：留出 DOA 用留出测试作答的正确率给学生对定序，但严格留出后每个 (学生, 概念) 对在测试集里往往只有 1–3 条作答，正确率是 0/0.5/1 的粗粒度噪声估计；学生对的"真实偏序"本身错误率很高，DOA 的天花板被真值噪声压低，各模型可能挤在 0.5–0.6 区间无法区分——招牌实验可能给不出显著差异。此外基线的"天然 proficiency 向量"对留出概念的取值语义各异（KaNCD 是低秩外推值，NCD 是从未被监督的自由参数），直接比较存在解释争议。修正：设定最小作答数阈值（如每对 (s,k) ≥3 条留出作答才参与定序）并报告参与配对数；对 DOA 做学生对级 bootstrap 置信区间与模型间配对显著性检验；补充一个连续版指标（掌握度与留出正确率的 per-concept Spearman）作为鲁棒性检查；对 NCD 类基线在正文注明其留出概念掌握度"未受监督"的语义，并以 KaNCD 为主要对照。

#### E5 门控行为机制分析（自适应融合确在工作）

- **设计**：用 `evaluate_gate_diagnostic.py` 输出各学生覆盖率分箱下的 mean tkc_weight、corr(coverage, tkc_weight)、effective_tkc_share；预期 w_u（UKC 权重）随覆盖率单调下降、相关系数显著；配 2–3 个低覆盖学生案例（UKC 贡献主导且预测正确）与表征可视化。
- **证据作用**：机制层证据——模型在且仅在证据稀缺时求助 UKC 分支，门控是功能性的而非装饰性的；支撑可解释性叙事并与消融 A7 呼应。
- **资产状态（已核查，confirmed，一处出入）**：`scripts/evaluate_gate_diagnostic.py`——113–159 行状态提取、178–247 行分箱聚合、271–280 行四种相关系数、162–175 行双塔平均，完整实现，零新代码。**出入**：`visualize_ukc.py` 只画 CSV 层面的每生 TKC/UKC 数量统计，学习表征的可视化需新写。
- **审稿修正（严重度：中，来自"评测脚本分发"攻击）**：该脚本直接前向提取 propagation 内部状态；P1 修复后 `ukc_states` 形状从 (K,D) 变为逐学生 (S,K,D)，全量学生提取的内存假设与张量形状假设同时失效，且 E5 依赖 v2 的逐概念掌握度头。须改为消费 v2 的稳定诊断接口（`export_mastery`/`export_branch_states`，见 3.2 第四步）。

#### E6 历史遮蔽鲁棒性 + 污染抑制（M2/M3 协议下的修复验证）

- **设计**：与 M3 同协议——解耦模型（及 fusion 消融）与 RCD 的 ΔAUC–遮蔽率曲线，另报低覆盖切片 ΔAUC，预期解耦模型退化最平缓（结构路径接管）；与 M2 同协议——对解耦模型注入同等标签噪声，测未测知识点掌握度漂移与留出 AUC 降幅，预期显著小于 RCD——因为噪声只能经过被认知状态平滑"净化"后的 TKC→UKC 通道传播。
- **证据作用**：闭合两条动机证据链——M3 暴露的记忆依赖被结构推断替代，M2 暴露的行为污染被净化传播抑制。
- **资产状态**：`evaluate_history_hiding_stress.py`、`pyedmine_rcd_history_hiding.py`、M2 的新探针脚本（复用）。几乎零增量代码（含 M3 提到的切片输出小补丁）。
- **审稿修正（严重度：高，继承 M2/M3）**："解耦模型漂移显著小于 RCD"没有任何机制保证——TKC 状态由同样含猜测/失误噪声的作答训练，"净化"是未定义的形容词；**如果实测漂移不比 RCD 小，M2/E6 这条证据链整体反噬**。须按 M2 修正改用"信号响应度/噪声敏感度之比"或任务级 AUC 降幅；按 M3 修正同时报告绝对 AUC 与学生间区分度（MND/学生间预测方差），避免"退化平缓=更好"的方向性混淆（v1 学生无关的 UKC 分支恰会伪装成鲁棒）。

### 5.4 消融实验（A1–A8）

#### A1 融合模式

- **设计**：`--student-fusion-mode` adaptive / tkc_only / ukc_only / mean——tkc_only 应在 zero/low 覆盖桶崩溃、ukc_only 量化纯结构信息量、adaptive 应全面优于 mean，证明双分支＋自适应门控均必要。
- **资产状态**：confirmed——`train.py:152-157` 与 `hetero_propagation.py:37` 四档旗标已实现。

#### A2 TKC→UKC 个性化传播 on/off（P1 修复）

- **设计**：关闭即退回当前学生无关 UKC；预期开启后 E2 none_seen AUC 与 E4 未测 DOA 显著提升、而 M2 探针漂移仍保持低位（净化通道不引入污染）。**这是全文最关键的单一消融**。
- **资产状态**：待模块一实现后加旗标。
- **审稿修正（严重度：中，继承模块一新颖性攻击）**：A2 必须报告**证据置信度加权单独开关**的增益，证明组合式贡献的每个成分非装饰（见 6.2 模块一）。

#### A3 目标知识点感知读出 on/off（P2 修复）

- **设计**：对比全局池化读出，预期未测 DOA 与逐知识点诊断输出质量提升；同时验证其对整体 AUC 无损。
- **资产状态**：待模块二实现。若 AUC 回撤超预算，用 A3 消融数字在附录正面呈现"拟合力换可辨识性"的交易（见 6.2 模块二审稿修正）。

#### A4 适配器归因（P4）

- **设计**：全开 vs 全关 vs 逐个开启；并把 concept_evidence_prior / history_evidence_logit_prior 等计数统计单独拎成一个"平滑计数先验"独立基线 B0（logistic 回归级）——若该基线已能拿到大部分增益，则相应适配器从主模型中剔除，确保论文归因给解耦而非计数统计。
- **资产状态**：**wrong（原表述需修正）**——"现 run_assist09_baseline.sh 配置 = 7 个全开"与脚本不符：`run_assist09_baseline.sh:34-46` 实开 6 个可选适配器，`concept_evidence_prior_residual` 与 `history_evidence_logit_prior_residual` 本就不在主线配置里（加上常开的 `cognitive_difficulty_adapter` 才是 7 个生效模块）。实验本身可行：全部适配器有独立 flag（`train.py:173-302`）。

#### A5 图泄漏量化（P3）

- **设计**：概念转移图用全量数据构建 vs 仅 train 构建（`build_assist09_transition_graph.py` 改喂 train.csv；holdout 划分去掉默认图拷贝即 `--no-copy-transition-graph`；RCD 侧用 `pyedmine_cd_baselines.py prepare --kt-source train` vs full）——在标准与留出两种划分上报告两版图的指标差，量化泄漏虚高幅度；主表统一采用 train-only 图。
- **资产状态**：confirmed——`data/concept_graph.py:37-54` 泄漏确认、builder `--interactions` 可指 train.csv、`configs/defaults.py:10` 证实默认吃泄漏图。**维度脚枪**：builder 的 num_concepts 从所给文件的最大知识点 id 推断（35–43 行），而模型的 K 来自 Q_matrix（`data/mappings.py:32-38`）——若某知识点只出现在 valid/test，train-only 图会比 K 小导致 K×K 矩阵乘维度错误，需给 builder 加 `--num-concepts` 或按 Q 矩阵补零（所有后续实验的前置修复）。
- **审稿修正（严重度：低）**：泄漏效应须与图稀疏化效应分离——泄漏效应 = 全量图增益 − 同等密度随机降采样图增益（见 3.2 第二步）。

#### A6 读出单调性约束＋有界 guess/slip on/off（P5）

- **设计**：预期 AUC 微降或持平，但 ECE 与 DOA 改善、g/s 参数落入可解释区间；用 gs_mode 旗标扩展。
- **资产状态**：gs_mode 现仅 constant/conditional 两档（confirmed），可按提案扩展。

#### A7 门控固定 vs 学习

- **设计**：student_gate_prior_alpha/beta 固定先验、常数 w、学习式门控三档对比，与 E5 的机制分析互证自适应门控的必要性。
- **资产状态**：confirmed 但有出入——`student_gate_prior_alpha/beta` 仅用于初始化门的 bias（`hetero_propagation.py:75-76`，`train.py:144` 帮助文本明言 "training remains adaptive"），训练中门仍会学；**"固定常数 w=先验"这一档并不存在**，需新增一个冻结门或 fixed 模式（约 10 行）；mean 档只提供 w=0.5。

#### A8 图来源与层数

- **设计**：transition / prerequisite / similarity 图（graph_mode 已支持多图输入）及传播 1 层 vs 2 层，作为次要敏感性分析（P6）。
- **资产状态**：前半 confirmed——graph_mode single/dual 已实现（dual 用 prerequisite＋similarity 门控融合，`hetero_propagation.py:104-127,190-203`），builder 输出全部三种图（`build_assist09_transition_graph.py:48-53`），`run_assist09_baseline.sh` 明确锁 single。后半依赖：当前只有单层传播，**2 层消融必须等模块一的多层实现落地**——与提案 priority 排序一致，非现成资产。

---

## 6. 模块设计

### 6.1 单薄性判决（thinness_verdict）

结论：核心确实单薄，但正确的处方不是"加模块"，而是"**换模块**"。具体判断分三层：

1. **核心贡献层过薄**：`models/hetero_propagation.py` 277 行，去掉参数解析后有效建模逻辑约 120–150 行——只有一层线性消息传递，无非线性、无 LayerNorm、无多层堆叠；更致命的是 UKC 分支只有一行（第 156 行），对同一 TKC/UKC 划分的所有学生输出完全相同，连 TKC 分支的图邻居消息（126–127 行）也是学生无关的。即"薄"不只是代码量问题，而是**现有模块没有实现论文声称的机制**（个性化诊断未测知识点）。
2. **表面厚度层是虚胖**：`models/decoupled_cdm.py` 1204 行，约 250 行是适配器超参校验样板（实测 115–228 行约 114 行，量级对），7 个可选 zero-init 残差适配器多为平滑计数统计的打补丁，且 393 行的 `cognitive_difficulty_adapter` 是无条件常开的第 8 个适配器；782–896 行的读出残差对 tkc/ukc 状态做了 detach（849–850 行），只是读出层贴膏药、梯度无法回流塑造传播——这恰恰是承认 P1 存在的证据。
3. **审稿视角**：CD 领域（NCDM/RCD/KaNCD 一线工作）并不奖励模块数量，反而严厉惩罚无原则的 A+B+C 堆叠；当前模型的问题是"薄在不该薄的地方（核心机制），厚在不该厚的地方（计数补丁）"。正确路线：删掉 6–7 个适配器，换成 2–3 个直接服务于 TKC/UKC 解耦叙事的原理性机制——模块总数下降、可引用贡献上升。

### 6.2 新增模块（4 个）

#### 模块一：TKC→UKC 学生条件化多层传播（核心新机制，替换现有 UKC 分支）

- **目的**：直接修复 P1——让 UKC 表征随学生的实际作答证据变化，实现论文核心主张"诊断未测知识点"；并取代 detach 版读出残差补丁（decoupled_cdm.py 782–896 行），让梯度真正回流训练传播函数。
- **设计**（张量草图，S=学生数, K=知识点数, D=维度）：
  - 第 0 层 Z⁰ = M_tkc ⊙ tkc_states ∈ (S,K,D)，即现有 TKC 分支输出作为证据源。
  - 第 l 层向 UKC 节点 k 的消息：m_{s,k}^{(l)} = Σ_{k'∈N(k)} â_{s,kk'}·W_l·z_{s,k'}^{(l-1)}，边权按学生归一化：â_{s,kk'} = A_{kk'}·m^tkc_{s,k'}·c_{s,k'} / Σ_{k''}(·)，置信度 c_{s,k'} = log1p(n_{s,k'})/log1p(cap) 来自 student_concept_evidence 的尝试次数——证据少的 TKC 节点传出的消息权重低。
  - UKC 节点融合：u_{s,k} = γ_{s,k}·Σ_l λ^l·h^{(l)}_{s,k} + (1−γ_{s,k})·u^static_k，γ 由 [收到的证据质量, 跳数衰减 λ^l, 静态先验消息] 经 zero-init 门控产生（保证初始等价于现有静态 UKC，训练稳定）。
  - 关键约束保留：UKC 节点不接受任何直接作答标签，只接收从 TKC 传播来的状态——"无直接行为监督"的故事线不破。L 取 1–2 层，L>2 有过平滑风险。
- **集成点（已核查，confirmed）**：三种训练模式都把全量 history 张量传入 forward，子集化发生在 forward 内部（`decoupled_cdm.py:353-358` torch.unique＋return_inverse → `hetero_propagation.py:94-100` index_select），minibatch 模式下 (S',K,K) 天然受控（S'=批内唯一学生数）；输出用 state_target_student_ids 回索引，per-student 逐知识点状态已是既有约定。**两个必改点**：(1) `student_concept_evidence` 已在 propagation.forward 签名中（91 行）但当前完全未使用、且未随 student_indices 做 index_select（94–100 行只选 4 个张量）——证据置信度加权必须补上，否则批模式下索引错位；(2) full_batch 模式传播覆盖全体学生（`engine.py:773-786` 无学生子集化），(S,K,K) 带梯度中间张量是主要内存开销，assist09 规模可行，更大数据集须用已实现的 `student_recompute_minibatch`（`--student-batch-size`）或稀疏化。
- **成本与风险**：中。实现——重写 forward 加 L 步循环＋einsum，约 100–150 行，顺带删掉 782–896 行读出残差。风险——多层传播可能不稳定（靠 zero-init 门控＋跳数衰减兜底）；**若 holdout 切片指标不涨则核心叙事证伪，这是必须尽早跑的实验而非风险规避对象**。
- **审稿修正（严重度：中，新颖性评级下调）**：原"高新颖性"评级偏高——逐边不确定性/置信度加权的消息传递在 CD 里已被 ISG-CD（KDD 2025）占位（响应边的猜测/失误不确定性建模），"结构影响以学生已测掌握度为条件"已被 HierCDF（KDD 2022）占位（父概念掌握度条件化），"TKC 向 UKC 的图传播"已被 DisKCD 占位。模块一真正剩下的可辩护增量是三者的组合加上"UKC 侧无监督参数"的约束——组合式贡献在 SIGIR/KDD 会被压级，除非留出实验的增益幅度替它说话。修正：把贡献重心从"机制新颖"挪到"**问题＋协议＋机制约束**"三位一体——主张的核心是"监督解耦这一归纳偏置及其专用评测协议（分层学生-概念留出＋留出 DOA＋MND）"，机制只是该偏置的一种实现；相关工作里主动引 ISG-CD/HierCDF/DisKCD 并各用一句话说明为何它们的机制不满足"无直接监督"约束，化被动为主动；消融 A2 必须报告置信度加权单独开关的增益。

#### 模块二：目标知识点感知读出＋逐知识点掌握度头（替换全局均值池化匹配）

- **目的**：修复 P2——当前 forward 只用全局池化的 student_state 与 q_repr 做 NCF 匹配，逐知识点状态算完即弃，模型退化为矩阵分解，无法输出逐知识点掌握度、无法算 DOA。此模块也是评测模块一效果的前提。
- **设计**：
  - 掌握度头：m_{s,k} = σ(w_m^T h_{s,k} + b) ∈ (S,K)，h 取融合后的逐知识点状态（TKC 节点用 tkc_states，UKC 节点用模块一的 u_{s,k}）。
  - 目标感知读出：对目标题 e，用现有 topk-gather 模式（782 行残差里已有现成代码）取出其知识点集 C(e) 的状态与掌握度，逐知识点交互 x_c = f_mono(m_{t,c} − diff_{e,c})·disc_e（NCDM 式），再按知识点聚合：logit_e = Σ_c α_c·x_c，注意力权重 α_c 由 [证据置信度 c_{s,k_c}, 知识点来源标签 τ_c∈{TKC,UKC}] 产生。
  - 这自然给出"构成感知融合"：一道题一部分知识点走实测掌握度、一部分走推断掌握度，按题级证据构成加权——取代（或降级为初始化先验）现有的每学生全局标量 w_u（`hetero_propagation.py:161-166`），后者把学生对所有题的 TKC/UKC 混合比压成一个数，明显过粗。
- **集成点（已核查，confirmed）**：`evaluate_model` 只消费 output.probs 与 labels（`engine.py:86-162`），`DecoupledForwardOutput` 已有 None 默认可选字段先例（`decoupled_cdm.py:23-26`），加 mastery 字段零破坏；topk-gather 现成代码模式在 837–861 行。**两个注意点**：(1) `exercise_difficulty` 是每题标量（230 行），NCDM 式逐知识点难度需新参数——提案已自知；(2) 所有新模型旗标必须同步加进 `scripts/analyze_prediction_slices.py:67-145` 的 load_model kwargs 白名单（它逐项显式重建 kwargs），否则 coverage_slice/gate_diagnostic/history_stress 全部评测脚本无法回载新检查点——多文件例行改动，易漏。
- **成本与风险**：低-中。实现——gather＋掩码聚合，代码模式已存在。风险——池化 NCF 读出有记忆能力，换成结构化读出初期 AUC 可能小跌，需与模块三（单调性）一起调。
- **审稿修正（严重度：低）**：目标概念感知读出＋单调交互是 NCDM（AAAI 2020）的原始设计，按概念注意力聚合是 RCD（SIGIR 2021）的标准件，报告也承认"单独不可引用"；"题级 TKC/UKC 构成感知融合"与全局 w_u 的差别只是门控粒度，审稿人会视为超参层面的选择而非贡献。更实际的风险被轻描淡写：换掉池化 NCF 读出后 v2 失去交互记忆能力，整体 AUC 回撤可能不小；而"先做成并联可切换头"的建议与"消融要干净"的要求矛盾（并联头本身又是一个适配器）。修正：论文中把模块二**降为"实现选择"而非贡献点**，贡献清单只保留监督解耦机制与评测协议两项；开发期允许并联头做调试，但主表与消融只报纯结构化读出的配置；若 AUC 回撤超过预算，用 A3 消融数字在附录正面呈现"拟合力换可辨识性"的交易，而不是靠并联头找补。

#### 模块三：单调交互函数＋有界且认知隔离的猜测/失误层

- **目的**：修复 P5 的可辨识性问题——当前 `cognitive_match_mlp` 无单调性约束（掌握度升高预测概率可以下降，诊断解释失效）；guess/slip 是无界 sigmoid 且以 student_state 为输入（480–492 行），认知信号可从非认知通道泄漏，g/s 与掌握度不可分。
- **设计**：
  - 单调性：对以掌握度为输入的交互层用正权重重参数化 w = softplus(θ)（NCDM 做法），保证 ∂P/∂m_k ≥ 0，配合模块二的逐知识点读出才有意义。
  - 有界化：g = g_max·σ(·)，s = s_max·σ(·)，g_max = s_max ≈ 0.3。
  - 认知隔离：g/s 的条件输入去掉 student_state，只允许题目表征＋作答量统计（如该生总答题数），或退回每生标量；同时删除 nn.Embedding 版 guess_logit/slip_logit（248–249 行）——它们既破坏可辨识性又破坏归纳性。
- **新颖性**：低。全是领域标准做法，不可引用；但缺失它们是审稿硬伤（"你的掌握度输出有什么保证？"），属于必须补的地基而非贡献。
- **成本与风险**：低（约 30–50 行）。指标可能小幅下降（单调约束限制拟合能力）——用拟合度换可辨识性的正常交易，论文里正面写。

#### 模块四：TKC/UKC 一致性自监督正则（可选项中唯一值得做的，作为模块一的训练信号）

- **目的**：在不违反"UKC 无直接行为监督"的前提下，给 TKC→UKC 传播函数提供训练信号——随机把训练集中已测知识点临时降级为"伪 UKC"，要求经传播推断出的状态与该知识点的真实 TKC 状态（或其上的真实作答）一致。
- **设计**：每 batch 采样掩码 M_drop ⊂ TKC，令 m'_tkc = m_tkc ⊙ (1−M_drop)，用模块一以 m'_tkc 重算被掩知识点的传播态 ĥ_k；**弱版本**损失 L_cons = Σ_{k∈M_drop} c_k·||sg(h^TKC_k) − ĥ_k||²（sg=stop-gradient，防塌缩）；**强版本**：用 ĥ_k 经掌握度头预测被掩知识点上的真实作答做 BCE。总损失 L = L_bce + λ_c·L_cons，λ_c≈0.1 起调。本质是知识点级掩码重建，与"推断掌握度应在可验证处与实测一致"的故事同构。
- **新颖性**：中-高——作为模块一的配套训练目标可并入主贡献叙述（"掩码知识点重建目标训练跨分区传播"），比单独包装成对比学习模块诚实得多。
- **成本与风险**：低-中。每 batch 二次传播（计算约 +30–50%，可每 N 步做一次摊薄）；多一个权重超参，若与主 BCE 冲突需要调度。
- **⚠️ 监督矛盾警告（审稿修正，严重度：高）**：**模块四的强版本与全文核心主张正面矛盾**——"用被掩概念上的真实作答对传播出的 ĥ_k 做 BCE"就是对 UKC 通路的直接行为监督。一旦加入，论文反复强调的归纳偏置"UKC 不受任何直接行为监督"就只剩措辞游戏：传播函数本身被作答标签端到端训练，只是测试时该概念恰好无数据。审稿人会指出这与摘要主张自相矛盾，并且此时整个方法退化为"带图参数化的掩码填补目标"——恰好落回 KaNCD 低秩外推的目标函数，坐实"用图重新发明矩阵分解"的指控。**此前四份报告都没有发现这个矛盾。二选一并在论文里显式声明**：
  - (a) 只用弱版本（stop-gradient 状态一致性 L2，不接触标签），保住"无直接行为监督"的字面主张；
  - (b) 采用强版本但改写核心主张为"UKC 无逐概念的行为监督参数，监督只经由一个共享的、情景式（episodic）训练的传播函数流入"——这是可辩护的表述，且可借力 meta-learning/掩码重建的成熟叙事（类比 GraphMAE, KDD 2022 的掩码图自编码）。
  - 无论选哪个，**必须补一个"低秩掩码填补（KaNCD 目标）vs 图传播填补"的头对头消融**，证明图结构在同一目标下带来增益。
- **其余可选项明确不做**：覆盖率感知课程学习（收益难归因，纯增加混淆变量，是典型的模块堆叠减分项）；完整贝叶斯/变分不确定性建模（采样＋KL 成本高、故事线不需要，模块一里的确定性证据置信度 c_{s,k} 已够用且免费）；序列式归纳学生编码器（把问题改写成 KT，超出本文范围——删掉 g/s 学生嵌入已获得最小归纳性，其余写 future work）。

### 6.3 Cut 清单（从主线移除，hybrid 下的含义是"不迁移进 v2"，v1 冻结类一行不动）

1. `high_concept_logit_adapter`（多知识点题残差，396–414 行）：平滑计数打补丁，移出主线，并入"历史统计基线"对照组。
2. `pairwise_history_interaction_adapter`：同上，本质是历史共现计数特征，与解耦机制无关。
3. `interpretable_readout_expert_adapter`（3 专家混合）：名为可解释实为拟合力堆叠，审稿高危，删除。
4. `concept_evidence_readout_residual` 与 `concept_evidence_prior_residual`：两个知识点计数先验残差，降级为独立的"平滑计数逻辑回归"基线 B0——它们本身是很强的对照，留在主线里只会稀释归因。
5. `history_evidence_logit_prior_residual`（学生/题目/知识点历史正确率对数几率先验，约 15 个超参）：同上并入 B0 基线；它一个模块占了 init 里近百行校验样板。
6. `student_conditioned_ukc_readout_residual`（782–896 行）：被模块一取代后删除——它 detach 了传播状态且只在目标知识点全未测时激活，是承认 P1 的临时膏药。
7. `cognitive_difficulty_adapter`（393 行）：无条件常开、未被任何 flag 控制的隐藏适配器，是当前所有实验的隐形混淆变量，立即删除或并入读出主干并显式消融。
8. guess_logit/slip_logit 学生嵌入（248–249 行）与 guess_mlp/slip_mlp 的 student_state 输入、`gs_difficulty_adapter`：由模块三的有界隔离版取代。
9. 每学生全局标量 w_u 融合门（`hetero_propagation.py:52-56, 161-166`）：降级为消融项，主线换成模块二的题级构成感知融合。
10. init 中约 250 行适配器超参校验样板（117–228 行）：随 flag 删除自然消失，模型文件预计从 1204 行缩到 400 行以内。

### 6.4 优先级（第 0–5 步）

- **第 0 步（前置，非模块但决定一切结果有效性）**：按既定计划用仅训练集数据重建概念转移图（修 P3），否则后续所有模块实验都在泄漏图上做，结论作废。（另据审稿修正：**KaNCD 的 none_seen 数字同为第 0 优先级**——先跑再定叙事。）
- **第 1 步：清场立基线**——删除全部可选适配器和 `cognitive_difficulty_adapter`，跑"纯核心"（当前传播＋NCF 读出）与"B0 计数先验基线"（被降级的三个计数残差单独成模型），拿到干净的归因起点（修 P4）；预期纯核心指标会掉，这个缺口正是后续模块的价值证明空间。
- **第 2 步：模块二**（目标知识点感知读出＋掌握度头）——**先于模块一做**，因为它是评测模块一的量尺（DOA、holdout 切片的逐知识点指标都依赖它），且风险低、代码模式现成。
- **第 3 步：模块一**（TKC→UKC 学生条件化传播）——论文主贡献，尽早在概念留出切分上验证"UKC 个性化是否真的提升未测知识点预测"；**若不提升需要回到机制设计而不是加补丁**。
- **第 4 步：模块三**（单调性＋有界隔离 g/s）——在主结果方向确认后加，补齐可辨识性地基，接受小幅指标回撤并在论文中正面论述。
- **第 5 步：模块四**（一致性自监督正则）——作为模块一的增强训练信号最后加，单独消融其 λ_c（注意 6.2 的监督矛盾警告，先定弱/强版本与主张措辞）。
- **贯穿要求**：每步只改一个变量并保留上一步 checkpoint 对照；最终消融表应覆盖 L 跳数∈{0,1,2}、证据置信度加权开/关、题级融合 vs 全局 w_u、单调约束开/关——这张表本身就是对"模块是否单薄"最有说服力的回答。

---

## 7. 统计严谨性与缺失项（missing 清单）

### 7.1 审稿攻击索引（15 条，按严重度）

对抗性审稿共提出 15 条攻击，均已按结构要求合并进对应章节（标注"审稿修正"），此处为索引：

| 攻击对象 | 严重度 | 已合并至 |
|---|---|---|
| 模块四强版本与"UKC 无直接行为监督"正面矛盾 | 高 | 6.2 模块四 |
| M2 漂移指标与修复 P1 目标直接冲突（E6 连带） | 高 | 5.2 M2 / 5.3 E6 |
| "两难困境"把结论当前提，KaNCD 是反例 | 高 | 5.1 story / 5.2 M4 |
| DisKCD 差异化辩护建立在未核实断言上 | 高 | 4.2 |
| M3"记忆 vs 推断"论证逻辑的方向性混淆 | 高 | 5.2 M3 / 5.3 E6 |
| E1 主表基线缺 KaNCD/KSCD/ORCDF | 高 | 5.3 E1 |
| "engine 改动是机械的"低估：full_batch 不可用、锁死 minibatch | 中 | 3.2 第三步 |
| 模块一"证据置信度加权传播"可引用性偏高 | 中 | 6.2 模块一 / A2 |
| M1 覆盖分桶的成分混淆 | 中 | 5.2 M1 / 5.3 E3 |
| E4 留出 DOA 的统计有效性（真值噪声压低天花板） | 中 | 5.3 E4 |
| "评测脚本加 --model 分发即可复用"低估形状耦合 | 中 | 3.2 第四步 / 5.3 E5 |
| "不牺牲整体性能"承诺大概率兑现不了 | 中 | 5.1 story / 5.3 E1 |
| E2 对 DisKCD"必须对比"无落地方案 | 中 | 5.3 E2 |
| 模块二"题级构成感知融合"新颖性与并联头矛盾 | 低 | 6.2 模块二 |
| "第二步一天完成"未算图稀疏化连锁反应 | 低 | 3.2 第二步 |

### 7.2 缺失项清单（10 条）

以下 10 项来自对抗性审稿的 missing 清单，为四份分析报告之间未消化或全体遗漏的缺口：

1. **统计严谨性整体缺失**：实验矩阵仅在 E1 写"≥3 seeds"，文献报告要求 ≥5 种子，且全部六个主实验和八个消融都没有规定显著性检验方法。建议：每配置 5 种子，模型间用配对 t 检验或 Wilcoxon，切片指标用 bootstrap 置信区间，主表标注显著性。
2. **主表基线缺 KaNCD、KSCD、ORCDF**（文献报告定为必选，实验矩阵 E1 未列），以及 DisKCD 的可执行对比方案——四份报告之间存在未消化的自相矛盾。
3. **MND 指标（ORCDF, KDD 2024）未进入实验矩阵的指标集**：它是检测 UKC 表征学生维塌缩（P1）最直接的现成指标，修复前后的 MND 对比本应是核心图表；M4 的方差/余弦是自造指标，应改用或并报 MND 以便与文献可比。
4. **可辨识性 sanity 实验**（ID-CDF 式"相同响应模式→相同诊断"）在文献报告的协议建议里出现，但实验矩阵完全没有承接——修复 P5 的价值主张缺少直接证据。
5. **下游任务验证缺失**（ORCDF 的模拟 CAT 或基于 UKC 掌握度的补练推荐）：对"未测概念诊断"这种无直接标签的主张，文献报告已指出这是审稿人认可的间接验证手段，实验矩阵未安排。
6. **效率与扩展性报告缺失**：训练/推理时间、显存、以及个性化传播在 Junyi 量级的可行性数字——图式 CD 论文必被问扩展性（RCD 在 Junyi OOM 是公开梗），而 v2 的传播开销比 v1 更大，回避不了。
7. **基线调参公平性协议缺失**：PyEdmine 基线用默认超参而自家模型精调，是审稿标准攻击点；需规定统一的调参预算（如每模型相同的搜索空间大小与验证协议）并在附录公开。
8. **留出划分自身的随机性未控制**：`split_student_concept_holdout` 的学生选择与概念留出应至少 3 个划分种子，报告跨划分方差——否则 none_seen 子集的结论可能是单次划分的偶然。
9. **新学生归纳评测**（ICDM 协议，留出 20% 学生）在文献报告列为加分项，实验矩阵未承接；v1/v2 均为转导式，若完全回避，"归纳性"会成为 rebuttal 期无法补救的弱点。
10. **时间序切分敏感性分析**（回应 P6"历史非时序"）在协议建议中出现但实验矩阵未安排，至少应在一个数据集上做时间切分的稳健性检查。

---

## 8. 执行环境

### 8.1 远端服务器与数据依赖

- **本地仓库无任何数据 CSV**（`data/` 下只有代码）。assist09 数据、PyEdmine、SCD、SVGCD 环境**全部在远端主机 xph-pc**（`scripts/remote_exec.sh:7-9`），实验须先推分支、经 `remote_exec.sh` 远程执行。
- `scripts/scd_baselines.py:16-18` 硬编码 `/home/xph/jwc/SCD` 与对应 conda 环境；PyEdmine 位于远端 `/home/xph/jwc/pyedmine`。
- `configs/defaults.py` 只为 assist_09 配了 concept_graph 默认；assist_17/junyi 指向 `../ConceptSkillCDM` 相对路径、nips34 无本模型侧默认——E1 要求本模型在 assist17/nips34 上跑需显式传路径并为每个数据集重建 transition_graph（`train.py` 支持显式路径，非零工作量但无代码障碍）。
- **所有"零新代码"结论均以远端环境可用为前提**；S≈4k / K≈120 等规模数字本机无法核实（本地无数据文件）。

### 8.2 已知 blockers

可行性核查结论：**无致命阻断项**，但以下为必须先处理的硬依赖：

1. **远端依赖与易失产物**：全部实验依赖远端主机 xph-pc（数据、PyEdmine、SCD/SVGCD 环境均不在本机，`scripts/remote_exec.sh` 要求先推分支）；`assist09_holdout` 的基线 source_dir 指向远端 `/tmp/assist09_holdout_seed2024`，重启即失——E2/M2 前必须用 `split_student_concept_holdout.py` 重新生成到持久路径，并同步更新 `pyedmine_cd_baselines.py:47` 的 DATASET_SPECS。
2. **train-only 建图维度脚枪**：`build_assist09_transition_graph.py:35-43` 从所给交互文件推断 num_concepts，而模型 K 来自 Q_matrix（`data/mappings.py:32-38`）；若个别知识点不出现在 train.csv，train-only 图为 K'×K'（K'<K），`hetero_propagation.py:126-127` 的 K×K 矩阵乘直接报错或错位。**第 0 步（A5/E2 建图）前必须给 builder 加 num_concepts 参数或按 Q 矩阵维度补零——这是所有后续实验的前置修复**。
3. **新旗标四处同步**：模块一/二/三的每个新超参旗标必须同步改四处——`train.py` argparse、`DecoupledCDM.__init__`、训练 summary 写出、`scripts/analyze_prediction_slices.py:67-145` 的 load_model kwargs 白名单；漏掉最后一处会让 coverage_slice/gate_diagnostic/history_stress/checkpoint_average 全部评测脚本无法回载新检查点。
4. **KaNCD 基线**依赖远端 PyEdmine 是否自带 kancd.py（本机不可验证）；若无则 M1/E1 的 KaNCD 列需要在 PyEdmine 侧补训练脚本，**不是"加一行"**。
5. **full_batch 显存**：模块一在 full_batch 模式下对全体学生构建带梯度的 (S,K,K) 边张量（`engine.py:773-786` 无学生子集化），assist09 尚可、junyi 级别必须改用已有的 `student_recompute_minibatch` 模式或实现稀疏化；同时 propagation.forward 中 `student_concept_evidence` 目前未被使用、也未随 student_indices 子集化（`hetero_propagation.py:91, 94-100`），证据置信度加权实现时必须补上，否则批模式下索引错位。
6. **两处"零新代码"表述下调为"小补丁"**：M3 的"遮蔽后 low/zero 覆盖切片 AUC"当前两个 stress 脚本都不输出（只有整体指标，需约 20–30 行）；A4 的"7 适配器全开=官方脚本配置"不准确（`run_assist09_baseline.sh` 只开 6 个可选适配器，两个计数 prior 残差本就未启用）。

---

*本文档整合六份分析报告：文献与评估协议调研（part_literature）、实验矩阵设计（part_experiments）、模块架构提案（part_modules）、对抗性审稿（part_review）、代码可行性核查（part_feasibility）、代码存活率评估（survival）。所有 file:line 引用经逐条代码核查；标注"审稿修正"的内容来自对抗性审稿报告（按严重度 high/medium/low 标注）；标注 confirmed/uncertain/wrong 的资产状态来自可行性核查报告。*
