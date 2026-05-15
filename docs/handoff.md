# Handoff

这份文档保留“扩展交接”用途，用来说明当前主线、关键判断和关键文件。

它不是新会话默认第一入口。流程与执行约束以 Trellis 为准：`.trellis/workflow.md` 和 `.trellis/spec/backend/experiment-protocol.md`。只有需要当前主线细节、关键判断和关键文件时再读这份文档。

## 当前主线

- 当前工作重点是在稳定基线上继续做可解释的结构改动。
- 当前正式主线执行约束见 `.trellis/spec/backend/experiment-protocol.md`；这里补充结果口径、候选支线和行动判断。
- 数据:
  - [train.csv](../data/assist_09_ordered/train.csv)
  - [valid.csv](../data/assist_09_ordered/valid.csv)
  - [test.csv](../data/assist_09_ordered/test.csv)
- Q 矩阵:
  - [Q_matrix.csv](../data/assist_09_ordered/Q_matrix.csv)
- 图:
  - [propagation_graph.csv](../data/assist_09_ordered/transition_graph/propagation_graph.csv)

当前推荐结果口径:

- 从 `master` 提交 `868f20b` 起，训练输出会包含 `Brier/ECE/分桶校准`。
- 更早的历史结果文件通常只含 `AUC/ACC/RMSE`，需要在新代码下复跑才有校准指标。

- 当前主线基座结果目录:
  - `results/exp_tkc_exercise_aggregation/`
  - 三 seed 均值约 `test_auc = 0.7597`
- 当前正式主线结果目录:
  - 代码口径已吸收实验 70；详细三 seed 指标见 [070_student_conditioned_ukc_readout_sidecar.md](./experiments/070_student_conditioned_ukc_readout_sidecar.md)
  - 三 seed 均值:
    - `test_auc = 0.765517`
    - `test_acc = 0.729104`
    - `test_rmse = 0.427350`
    - `test_brier = 0.182628`
    - `test_ece = 0.049044`
- 当前正式主线相对实验 23 基座均值差:
  - `AUC +0.005827`
  - `ACC +0.004320`
  - `RMSE -0.004038`
  - `Brier -0.003468`
  - `ECE -0.013232`
- 当前冲刺目标:
  - `test_auc ~= 0.780`
  - `0.778` 可视为接近可接受
  - 相对当前正式主线约需 `AUC +0.0125` 到 `+0.0145`
  - 这个距离已经超出常规小 residual / sidecar 的边际收益，后续默认优先考虑 representation-level 大结构改动
- 详细背景先看 [model_improvement_plan.md](./model_improvement_plan.md) 的当前快照；若需要按实验号定位，再查 [experiment_index.jsonl](./experiment_index.jsonl) 或对应 detail doc。

当前正向训练策略支线:

- 分支: `exp/training-modes`
- 判断:
  - 它仍是当前更强的 calibration-oriented 训练协议候选，但在 `AUC/ACC` 上仍弱于当前正式主线。
  - 收益主要来自真正 mini-batch SGD 的整体校准/泛化改善，不是彻底解决 `concept_count=4+` 或 `none_seen` 校准问题。
  - 它属于纯训练工程优化，单次运行耗时显著高于当前默认 full-batch 口径；在模型结构仍需继续迭代时，暂不适合作为 `master` 默认训练协议。
  - 详细结果见 [037_recompute_minibatch_training.md](./experiments/037_recompute_minibatch_training.md)。
- 分支: `exp/full-target-exclusion-opt`
- 判断:
  - 这是当前 target-exclusion 训练口径的正式候选；`2026-05-02` 三 seed 复跑均值为 `AUC 0.765495`、`ACC 0.729256`、`RMSE 0.427883`、`Brier 0.183084`、`ECE 0.051331`。
  - 它相对实验 51 有稳定 `AUC` 正向；实验 70 合入后，单独 target-exclusion 口径不再明显强于当前主线。
  - 实验 71 已验证“实验 70 + target-exclusion”直接组合不是 clean win: `seed=2024` 只有 `AUC +0.000950`，但 `RMSE/Brier/ECE` 回撤，因此不默认扩 seed。
  - 原始实验 61 最大的问题是训练成本过高；工程优化后，远端 train-only `1 epoch` median runtime 已从约 `8.845s` 降到约 `2.002s`，不再因为成本直接降级。
  - 当前将它暂定为主线候选，但仍不作为 `master` 默认训练协议。
  - 详细结果见 [061_full_target_excluded_training_audit.md](./experiments/061_full_target_excluded_training_audit.md)。

当前结构主线补充:

- 实验 51 的 full-trigger 三专家 readout residual 已吸收到当前 `master`。
- 实验 70 的 student-conditioned UKC `none_seen` readout sidecar 已吸收到当前 `master`，它不替换 `TKC/UKC/student_state` 主状态，只作为 target-local cognitive-logit residual。
- 当前主线新增默认配置:
  - `--interpretable-readout-expert-adapter`
  - `--interpretable-readout-expert-count 3`
  - `--student-conditioned-ukc-readout-residual`
- `min_count>=2` 和 `min_count>=3` 的 targeted 变体没有吸收；它们都弱于 full-trigger 版本。
- 后续 `exp/clean-readout-routing` 的 `seen/unseen` gate 统计和 `top-k` 稀疏路由也未超过当前主线，因此实验 51 原版 full-trigger 仍是这条线的正式保留版本。
- 再后续的 `exp/readout-routing-soft-regularizer` 也未超过当前主线；soft routing regularizer 在单 seed 有轻微正信号，但三 seed 均值仍全面弱于实验 51 原版。
- 再复访的 `exp/q-conditioned-local-mastery-readout` 也未超过当前主线；即便把 local mastery 做成更接近主 readout 的逐概念打分再聚合版本，`seed=2024` 仍明显弱于实验 51 原版。
- 再新增的 `exp/difficulty-weighted-propagation` 也未超过当前主线；propagation 侧 difficulty weighting 只有极小 `AUC` 正向，但 `ACC/RMSE/Brier/ECE` 副作用明显。
- 再新增的 `exp/student-pairwise-ranking-loss` 也未超过当前主线；`weight=0.05/0.02` 都没有形成 overall 正向，ranking loss 目前不构成默认训练升级路径。

近期暂停的 CF 模型支线:

- 分支:
  - `exp/cf-residual`
  - `exp/cf-residual-recompute`
  - `exp/cf-residual-dim-sweep`
- 判断:
  - 这是当前最强 ranking-oriented 结构候选，但不是校准候选。
  - 已验证与实验 37 不是无损叠加，且扩大容量后的收益主要依赖当前 split 下的 ID-aware side channel。
  - 详细结果见 [038_040_cf_residual_family.md](./experiments/038_040_cf_residual_family.md)。
  - 当前决定: CF 支线暂停，不继续扩容，也不作为纯 CDM 主线推进；若论文需要，可作为 optional hybrid / ID-aware ablation 或 appendix 讨论。

## 已经定下来的判断

- ordered ASSIST09 + transition graph 是当前固定数据主线。
- 当前训练主线使用单图 `propagation_graph`，不是 `dual graph`。
- `20 epoch` 远远不够，结构比较默认应看 `300 epoch` 量级。
- `conditional g/s` 明显优于 `constant g/s`，应保留。
- `TKC/UKC` 结构传播参数解耦是当前最可靠的正向结构改动。
- `TKC` 行为消息里显式保留错题信号是有效的，当前正误双通道优于只看正确题。
- 将全局固定 `alpha/beta` 升级为学生自适应 `TKC/UKC` 融合 gate 后，三 seed 结果已经稳定优于旧主线。
- `_build_exercise_component` 不应先按学生全历史对 `TKC` 行为项做全局归一化；概念内聚合应避免让“历史越长，行为证据越弱”。
- 在实验 21 当前主线上修掉这一步后，三 seed 结果显著提升，这一步已经吸收到当前 `master`。
- 实验 33 的 zero-init cognitive difficulty adapter 在三 seed 上稳定优于实验 23 基座，且避开了实验 29 的 seed 崩盘。
- 诊断 1 表明主线的主要剩余误差集中在多知识点题和 `none_seen` 校准。
- 实验 34 证明“多知识点题 targeted residual + guess/slip difficulty residual”可以在三 seed 上同时改善 `AUC/ACC/RMSE/Brier/ECE`，这一步已经吸收到当前 `master`。
- 实验 37 证明 `recompute_minibatch bs=8192 lr=1e-4` 是当前最强的 calibration-oriented 训练协议候选；但它属于纯训练工程优化，运行成本更高，且在 `AUC/ACC` 上仍弱于当前主线，因此暂不推广为 `master` 默认训练口径。
- 实验 61 在 `2026-05-02` 的工程优化复跑后，已经从“训练成本过高的审计分支”升级为正式候选：`AUC` 稳定高于当前主线，`ACC/RMSE/Brier` 基本持平到 very small 正向，但 `ECE` 仍更差，因此当前只暂定为主线候选，不直接切成默认训练口径。详细指标见 [061_full_target_excluded_training_audit.md](./experiments/061_full_target_excluded_training_audit.md)。
- 实验 48 证明“显式历史概念统计”对多知识点题确有局部信号，但 original form 的三 seed overall 不稳定；这条路如再访，应改成更局部的 targeted trigger / mixture，而不是继续推进统一 residual。详细指标见 [048_history_concept_stats_residual.md](./experiments/048_history_concept_stats_residual.md)。
- 实验 49 证明 history-carrier pairwise interaction residual 能把“显式历史概念统计”稳定转成 overall 正收益，并且 `none_seen` 也形成 clean win；这是实验 51 之前的结构主线更新，已经吸收到 `master`。详细指标见 [049_history_carrier_pairwise_interaction.md](./experiments/049_history_carrier_pairwise_interaction.md)。
- 实验 50 证明 learned pair aggregation 没有额外收益，主线保留简单均值聚合。详细指标见 [050_weighted_pairwise_history_aggregation.md](./experiments/050_weighted_pairwise_history_aggregation.md)。
- 实验 51 证明 interpretable readout expert residual 已形成稳定的非 ID-aware 结构更新；当前最强版本是 full-trigger 三专家，targeted 硬 trigger 反而更弱，这一步已经吸收到当前 `master`。详细指标见 [051_interpretable_readout_expert_residual.md](./experiments/051_interpretable_readout_expert_residual.md)。
- 实验 70 证明 student-conditioned UKC 信号以 `none_seen` readout sidecar 形式可以稳定转成三 seed overall 正收益；这一步已经吸收到当前 `master`。详细指标见 [070_student_conditioned_ukc_readout_sidecar.md](./experiments/070_student_conditioned_ukc_readout_sidecar.md)。
- 实验 52 证明在实验 51 底座上继续加入更显式的 `seen/unseen` routing 统计或 `top-k` 稀疏路由，并没有带来更强结果；这条 selective routing follow-up 暂停。详细指标见 [052_clean_interpretable_readout_routing.md](./experiments/052_clean_interpretable_readout_routing.md)。
- 实验 53 证明在实验 51 底座上继续加入全局 soft routing regularizer，也没有形成稳定三 seed 增益；这条 routing-regularization follow-up 同样暂停。详细指标见 [053_soft_routing_regularizer.md](./experiments/053_soft_routing_regularizer.md)。
- 实验 54 证明 “Q-conditioned local mastery 主 readout” 即使按更贴近 CD 语义的逐概念打分方式重做，单 seed 仍弱于当前主线，且没有留下足够强的多知识点 clean win；这条 readout 复访也暂停。详细指标见 [054_q_conditioned_local_mastery_readout.md](./experiments/054_q_conditioned_local_mastery_readout.md)。
- 实验 55 证明 propagation 侧 difficulty weighting 虽然能带来极小 `AUC` 正向，但整体更像“排序微升换误差与校准恶化”的折中，且没有留下足够强的多知识点 clean win；这条 propagation weighting 复访也暂停。详细指标见 [055_difficulty_weighted_propagation.md](./experiments/055_difficulty_weighted_propagation.md)。
- 实验 56 证明 student-wise pairwise ranking loss 不能单独把当前主线推高；它最多带来轻微排序偏好变化，但 overall `AUC/ACC` 仍不如实验 51。详细指标见 [056_student_pairwise_ranking_loss.md](./experiments/056_student_pairwise_ranking_loss.md)。
- 实验 57 证明 single-graph multi-hop propagation 复访后仍只形成轻微排序波动，且全局、coverage-conditioned、`UKC-only`、`2-hop only` 版本都没有给出 clean overall 正向；这条 multi-hop propagation 复访暂停。详细指标见 [057_single_graph_multi_hop_propagation.md](./experiments/057_single_graph_multi_hop_propagation.md)。
- 诊断 2 表明实验 51 full-trigger readout expert residual 可能会压制部分后续语义更干净结构的边际表现: `q-conditioned local mastery` 在实验 49 底座三 seed `AUC +0.001856`，但在实验 51 底座三 seed `AUC -0.001522`；`difficulty-weighted propagation` 也从实验 49 底座的误差/校准均值正向，转成实验 51 底座的均值全面回撤。后续新结构若在最新主线上轻微负向但语义足够干净，应优先追加实验 49 底座交叉复验；详细指标见 [experiment_themes.md](./experiment_themes.md) 的“诊断 2”。
- 诊断 3 的更早底座广扫没有推翻诊断 2: `q-conditioned local mastery` 在 `B33/B34` 只是小正，到 `B49` 才三 seed 明显正向；`concept-conditioned propagation` 的 `B33` 单 seed 强正未复现；multi-hop、qrepr-score、history stats、ranking loss、clean routing 多数只是误差/校准折中或早底座也不成立。后续更应直接隔离实验 51 expert，而不是盲目回退到更早主线。详细指标见 [experiment_themes.md](./experiment_themes.md) 的“诊断 3”。
- 实验 72 的表示瓶颈探针没有给出 `0.77+` 排序突破: `B49 + q-conditioned local mastery + exp70 sidecar` 只改善 `ACC/RMSE/Brier/ECE` 但 `AUC -0.000389`；`target-conditioned student context` 叠加实验 70 后 `AUC -0.007201`。recency/sequence encoder 会改变数据与历史可见性口径，当前不作为同一轮结构验证继续推进。详细指标见 [072_representation_bottleneck_probes.md](./experiments/072_representation_bottleneck_probes.md)。
- 实验 73 的当前主线口径扫没有发现 `0.77+` 训练配置；相对当前主线，最好的候选是 `lr=7e-4 + early_stop=20 + scheduler_patience=5`，三 seed 均值 `AUC +0.000928`、`ACC +0.001313`、`RMSE -0.000077`、`Brier -0.000066`、`ECE +0.002872`。它可作为 accuracy/ranking-oriented 候选口径，但不是 clean 默认切换。详细指标见 [073_current_mainline_protocol_sweep.md](./experiments/073_current_mainline_protocol_sweep.md)。
- 近期若干 follow-up（如 hard-Q residual、propagation/readout 侧多知识点 residual、全局共享 `none_seen` calibration bias、exact-3 aggressive residual/readout）除实验 51 外，都只形成局部 slice 信号或 seed-sensitive 折中，不作为主线结构推进；若要复访，按实验号查 [experiment_index.jsonl](./experiment_index.jsonl)，再按需打开对应 detail doc。
- 实验 38-40 的 CF 支线已确认主要依赖 ID-aware side channel，不作为纯 CDM 主线推进；若论文需要，可作为 optional hybrid / appendix 讨论。
- 多知识点题按知识点数分摊在当前口径下相对实验 23 几乎持平，暂时不是必须优先合入的关键因素。
- `dual graph` 相关 CLI / 配置现在只应视为 legacy ablation 入口，不属于当前默认工作路径。
- `valid/test` 当前应复用 `train` 行为历史做传播输入，不能各自重建行为矩阵。
- 更激进的 scheduler patience 没有带来更好结果。
- 不要回到裸 Q 共现图重新做主基线判断，除非用户明确要求。
- 并行训练的日志文件名现在已经唯一化，不再共用同一个 `train_*.log`。

## 当前分支优先级

- `exp/*` 分支只作为实验代码和复验参考，不直接代表当前主线。
- 具体分支以 `git branch -a` 为准；路线定位先看 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的摘要。只有已知实验号、分支名或要按状态筛选时，再查 [experiment_index.jsonl](./experiment_index.jsonl)，必要时打开 `docs/experiments/` 下的 detail 文件。
- 若继续优化 calibration-oriented 训练协议，优先从 `exp/training-modes` 出发。
- 若继续复核当前主线训练口径，优先参考 `exp/mainline-protocol-sweep`；当前最强候选是 `lr=7e-4 + early_stop=20 + scheduler_patience=5`，但只作为候选，不替换 `master` 默认口径。
- 若继续比较 target-exclusion 训练口径或准备正式主线切换对比，优先从 `exp/full-target-exclusion-opt` 出发。
- 若继续做结构主线，在当前 Trellis-managed worktree 中默认从 `exp/trellis-trial` 伪主线或其后代切新 `exp/*` 分支；不要直接从 `master` 切分支。实验 51 与实验 70 的模型主线语义仍按当前台账理解。
- 其余近期 `exp/*` 路线大多已形成暂停或降级判断；若要复访，默认先按实验号查 `docs/experiment_index.jsonl` 的 `status/reason_tags/verdict`，再按需打开对应 detail，确认是否真的出现了新的 slice 假设或机制假设后再决定是否重开。

## 当前关键文件

- 数据与映射:
  - [data/readers.py](../data/readers.py)
  - [data/mappings.py](../data/mappings.py)
  - [data/pipeline.py](../data/pipeline.py)
- 图构建:
  - [scripts/preprocess_assist09_ordered.py](../scripts/preprocess_assist09_ordered.py)
  - [scripts/build_assist09_transition_graph.py](../scripts/build_assist09_transition_graph.py)
- 模型:
  - [models/decoupled_cdm.py](../models/decoupled_cdm.py)
  - [models/hetero_propagation.py](../models/hetero_propagation.py)
- 训练与配置:
  - [scripts/train.py](../scripts/train.py)
  - [scripts/evaluate.py](../scripts/evaluate.py)
  - [scripts/remote_exec.sh](../scripts/remote_exec.sh)
  - [scripts/run_assist09_baseline.sh](../scripts/run_assist09_baseline.sh)
  - [scripts/run_assist09_multiseed.sh](../scripts/run_assist09_multiseed.sh)
  - [trainers/engine.py](../trainers/engine.py)
  - [configs/defaults.py](../configs/defaults.py)
- 实验台账:
  - [model_improvement_plan.md](./model_improvement_plan.md)
  - [experiment_index.jsonl](./experiment_index.jsonl)
  - [experiments/](./experiments/)

## 后续实验规则

- 新假设仍应控制变量，但当前默认不再把小 residual / sidecar 当作主要推进节奏。
- 当前已进入单因素边际收益放缓的平台期；单因素小改默认只作为新假设准入或大结构假设的辅助验证，不再视为完整推进节奏。
- 默认允许少量测试已各自成立的正交组合；组合验证仍应严格限量，默认只测最强的 `1-2` 组候选，不做组合爆炸。
- 默认优先探索更大一级、真正改变表示瓶颈的模块改动，例如学生状态形成、target-conditioned history、受约束结构学习，或明确标注为 hybrid 的 side channel。
- 主线默认仍锁定当前超参数口径；但若是更大一级模块改动，且单次结果表现为“overall 未过门槛但目标 slice 有明显改善”，可额外允许一次很小的 rescue sweep，再决定是否淘汰。
- 探索性结构改动默认先从 `exp/trellis-trial` 伪主线或其后代切 `exp/<short-name>` 分支。
- 对 readout / `q_repr` / target-conditioned history / student-state 形成这类容易受实验 51 full-trigger expert 影响的 representation-level 改动，仍从 `exp/trellis-trial` 伪主线语义实现，但首轮实验设计默认至少包含 `B49 seed=2024` 与当前 `Exp70 seed=2024` 两格；不要只跑当前主线单格后直接下结论。
- 新结构默认先跑单次；单次值得继续时再补 `2-3` 个 seed。
- 若局部改动单次相对当前主线的 `AUC/ACC` 连 `1e-3` 量级都明显达不到，默认不优先扩 seed；若冲 `0.78` 的大结构单 seed 连 `AUC +0.002` 左右信号都没有，通常也不优先扩 seed，除非切片信号非常强。
- 若做更大一级结构改动，优先选择能直接作用于多知识点交互、学生状态形成或 propagation/readout 主干语义的模块；避免只在最终 logit 附近继续堆局部补丁。
- 这类 rescue sweep 默认只用于更大一级模块，不用于普通 sidecar / residual；范围也应严格收敛，优先只看 `learning_rate`、保守容量版本，或更局部的激活阈值。
- 即使允许更大一级改动，也仍应控制 full-batch 显存占用和训练成本，避免无约束扩主干。
- 新实验默认在 `exp/*` 分支上进行，确认成立后再整理回 `master`。
- 远端运行前，先把当前分支 `git push` 到 `origin`，再执行 `remote_exec.sh`。
- 实验 37 后续工程优化已做过一轮:
  - `weight_decay` / `grad clipping` 未带来值得保留的 `AUC/ACC` 增益
  - `cosine + warmup` 可带来极小的 `AUC/ACC` 提升，但量级不足以推动主线切换，且 `ECE` 变差
  - 训练模式支线暂不合入 `master`；如继续推进，优先做保持语义干净的模型改动，而不是继续打磨训练协议
- CF residual 支线暂停:
  - 不继续扫 `cf_dim`，不作为纯 CDM 主线推进
  - 后续默认回归实验 34/37 这类干净主线
