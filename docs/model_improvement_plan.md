# Model Improvement Archive

这份文档只保留实验台账用途，用来回答三件事:

- 当前 `master` 的正式主线是什么
- 哪些路线已经证明有效或无效
- 下一步默认该优先试什么

它不是新会话默认入口。新会话先读 [docs/session_bootstrap.md](./session_bootstrap.md)；只有在需要查历史实验、避免重复试错时再回来看这份档案。

## 如何使用

- 先看“当前快照”，确认主线、结果口径和近线候选。
- 要判断某条路线是否还值得继续时，看“已验证有效”和“已验证无效或已降级”。
- 要设计下一轮实验时，看“近线 follow-up”和“默认下一步”。
- 要按实验号定位时，直接搜索 `实验 <编号>`。
- 从实验 45 起，主文档优先保留决策索引；细节看 `docs/experiments/`，结构化检索先看 [experiment_index.jsonl](./experiment_index.jsonl)。

## 当前快照

- 当前 `master` 正式主线沿用 [docs/session_bootstrap.md](./session_bootstrap.md) 里的“当前主线”口径。
- 从这份归档的实验视角看，它对应实验 70 主线:
  - 以实验 34 为底座
  - 吸收实验 49 的 history-carrier pairwise interaction residual
  - 再吸收实验 51 的 interpretable readout expert residual
  - 再吸收实验 70 的 student-conditioned UKC `none_seen` readout sidecar
- 这里不再重复维护与 `session_bootstrap.md` 等价的数据、图、超参数和 adapter 开关清单；需要确认默认运行口径时，优先回看 `session_bootstrap.md`
- 当前主线三 seed 参考均值:
  - `test_auc = 0.765517`
  - `test_acc = 0.729104`
  - `test_rmse = 0.427350`
  - `test_brier = 0.182628`
  - `test_ece = 0.049044`
- 当前结果报告默认主看 `AUC/ACC`
- `RMSE/Brier/ECE/分桶校准` 默认作为次要指标
- 若目标是推进主线，默认希望 `AUC` 或 `ACC` 的改善至少达到 `1e-3` 量级；达不到时，通常需要很强的 slice 证据才值得继续

- 当前已吸收的最新结构更新:
  - 实验 70: student-conditioned UKC `none_seen` readout sidecar 已进入 `master` 默认主线；三 seed 相对实验 51 主线均值 `AUC +0.001628`，且 `ACC/RMSE/Brier/ECE` 均值也小幅正向

- 当前正向支线候选:
  - 实验 37
    - branch: `exp/training-modes`
    - 判断: 它仍是当前更强的 calibration-oriented 训练协议候选，但在 `AUC/ACC` 上仍弱于当前主线，不作为默认 `master` 训练口径
    - 补充: 这条线属于纯训练工程优化，单次运行耗时显著高于当前默认 full-batch 口径；在模型结构仍需继续迭代时，暂不优先合入主线
    - 详细指标见下文“实验 37”
  - 实验 61
    - branch: `exp/full-target-exclusion-opt`
    - 判断: 它仍是 ranking-oriented target-exclusion 训练候选，工程优化后运行成本已从“明显过高”降到“可接受”，但仍不作为默认 `master` 训练口径
    - 补充: `2026-05-02` 复跑三 seed 后，它相对实验 51 有稳定 `AUC` 正向；实验 71 已验证它与实验 70 的直接组合不是 clean win，不默认继续扩组合 seed
    - 详细指标见下文“实验 61”

- 暂停中的 CF 支线:
  - 实验 38 `exp/cf-residual`: ranking-oriented 候选，但依赖学生内随机 split 的 ID-aware side channel，不作为纯 CDM 主线
  - 实验 39 `exp/cf-residual-recompute`: 不是实验 37 与 38 的无损叠加，不建议主线化
  - 实验 40 `exp/cf-residual-dim-sweep`: 大容量收益主要来自 transductive ID side channel，整条线暂停
  - 详细指标见下文“实验 38-40”

- 最近失败或已暂停的 follow-up:
  - 实验 43-47: 都只形成局部 slice 信号、AUC/ACC 不成立或整体副作用明显，不继续扩线
  - 实验 48: 证明“显式历史概念统计”有局部价值，但 original form 的三 seed overall 不稳定，不作为主线结构推进
  - 实验 50: learned weighting 没有带来额外收益，主线保留简单均值聚合
  - 实验 52: 更干净的 interpretable readout routing 没能超过实验 51 原版 full-trigger，不继续沿这条 selective routing 扩线
  - 实验 53: softer routing regularizer 也没能超过实验 51 原版 full-trigger，不继续沿这条 routing regularization 扩线
  - 实验 54: Q-conditioned local mastery readout 复访后仍弱于实验 51 主线，不继续沿这条“local mastery 主 readout”扩线
  - 实验 55: difficulty-weighted propagation 只带来极小 AUC 正向，但 `ACC/RMSE/Brier/ECE` 副作用明显，不继续沿这条 propagation weighting 扩线
  - 实验 56: student-wise pairwise ranking loss 也没把 overall 指标做成，不继续沿这条 ranking-loss 训练线扩权重或扩 seed
  - 实验 57: single-graph multi-hop propagation 复访后仍只有轻微排序波动，未形成 clean overall 正向；全局、coverage-conditioned、`UKC-only` 与 `2-hop only` 变体都不继续扩线
  - 实验 59: parallel local context readout 分支在 smoke 阶段就触发数值不稳定，当前实现不再继续
  - 实验 67: learned multi-concept exercise attribution 机制上区别于实验 24 的静态分摊，但单 seed overall 明显弱于实验 51，且多知识点/`none_seen` 切片没有 clean win，不继续扩 seed
  - 实验 68: scale-preserving / high-count-only / incorrect-only attribution rescue 都没有恢复到实验 51；最强只是 `ECE` 小幅改善但 `AUC/ACC` 仍回撤，不继续沿 attribution 主聚合替换路线扩线
  - 实验 69: student-conditioned UKC imputation 没有解决 `none_seen` 校准，反而显著做坏 `none_seen` 的 `ACC/RMSE/ECE`，不扩 seed
  - 实验 71: 实验 70 主线 + 实验 61 target-exclusion 训练口径只带来单 seed `AUC +0.000950`，但 `RMSE/Brier/ECE` 回撤，不扩 seed
  - 详细指标见对应实验条目

## 已验证有效

下面只保留真正改变主线判断的实验。

### 实验 3. 论文式 transition graph

- 从早期 Q 共现图切到论文式有向转移图后，效果第一次明显优于随机附近基线。
- 结论: transition graph 是后续所有正式对比的图结构起点。

### 实验 6. 超参数扫描

- 固定 `learning_rate = 1e-3`、`concept_dim = 64`。
- 结论: 后续结构比较默认锁定这组超参数。

### 实验 7. `conditional g/s`

- 条件化 `guess/slip` 明显优于学生常数 `g/s`。
- 结论: `conditional g/s` 为主线固定配置。

### 实验 8. 长训

- `20 epoch` 不够，结构比较至少应看 `300 epoch`。
- 结论: 长训是正式比较默认协议。

### 实验 9. 多 seed

- 早期单图长训基线在 `seed in {2024, 2025, 2026}` 上波动很小。
- 结论: 主线比较不能只看单次最好值，要看多 seed 稳定性。

### 实验 11. `TKC/UKC` 结构传播参数解耦

- `TKC/UKC` 概念传播从共享参数改为独立参数后形成稳定增益。
- 结论: 这是当前主线最可靠的正向结构改动之一。

### 实验 12. `TKC` 正误双通道行为消息

- 显式保留错题证据后，三 seed 均值明显优于只看正确题。
- 结论: 错题信号不能丢。

### 实验 21. `TKC/UKC` 学生自适应融合 gate

- 固定 `alpha/beta` 升级为学生级自适应 gate 后，三 seed 稳定优于旧主线。
- 结论: 自适应 `TKC/UKC` 融合已成为默认配置。

### 实验 23. 修正 `TKC` 行为项全局二次缩小

- 将 `_build_exercise_component` 中“历史越长，当前行为证据越弱”的系统性缩放偏差改为概念内加权平均。
- 相对实验 21，三 seed `test_auc` 均值约 `+0.0095`。
- 结论: 这是当前主线的稳定基座。

### 实验 33. `cognitive_match` zero-init difficulty adapter

- 在认知 logits 外加 zero-init difficulty sidecar adapter。
- 相对实验 23，三 seed 同时改善 `AUC/ACC/RMSE/Brier/ECE`，并解决实验 29 的 seed 崩盘。
- 结论: 这是实验 34 前的关键一跳。

### 实验 34. high-concept logit adapter + `guess/slip` difficulty adapter

- 对 `concept_count >= 2` 的题增加 zero-init high-concept logit residual。
- 在 conditional `guess/slip` 分支增加 zero-init difficulty residual。
- 相对实验 23 三 seed 均值:
  - `AUC +0.001506`
  - `ACC +0.002772`
  - `RMSE -0.002218`
  - `Brier -0.001909`
  - `ECE -0.011134`
- 结论: 实验 34 是当前 `master` 正式主线。

### 实验 49. history-carrier pairwise interaction residual

- 动机: 不再直接从局部 `TKC/UKC` embedding 读多知识点交互，而是显式建模题相关概念对，并把更直接的历史概念统计作为 state carrier
- 分支: `exp/pairwise-history-carrier`
- 结构:
  - 保留实验 34 主线
  - 只对 `concept_count >= 2` 的题激活 zero-init pairwise interaction residual
  - 对每个题相关概念对共享 scorer
  - pairwise 输入为 `c_k / c_j / e_e` 加上逐概念历史统计 `accuracy / seen / log_attempt_count`
  - pair score 做均值聚合后加到 `cognitive_logits`
- 三 seed 相对实验 34:
  - `seed=2024`: `AUC +0.001199`, `ACC +0.003844`, `RMSE -0.001182`, `Brier -0.001014`, `ECE -0.001450`
  - `seed=2025`: `AUC +0.002073`, `ACC +0.001427`, `RMSE -0.000984`, `Brier -0.000844`, `ECE -0.000084`
  - `seed=2026`: `AUC +0.000248`, `ACC +0.001351`, `RMSE -0.000728`, `Brier -0.000624`, `ECE -0.002408`
- 三 seed 均值:
  - `AUC +0.001173`
  - `ACC +0.002207`
  - `RMSE -0.000965`
  - `Brier -0.000827`
  - `ECE -0.001314`
- 切片:
  - `seed=2024` 的 `concept_count=2/3/4+` 均明显改善，其中 `concept_count=3` 为 `AUC +0.030833`, `ACC +0.025471`, `RMSE -0.013033`
  - `none_seen` 也形成 clean win: `ACC +0.007841`, `RMSE -0.004422`, `Brier -0.003331`, `ECE -0.008649`
- 结论:
  - 这是第一条把“显式历史概念统计”稳定转成 overall 正收益的多知识点结构
  - 它不依赖 item ID side channel，也不是 exact-3 特判，语义与实现都足够干净
  - 实验 49 已吸收到 `master`

### 实验 37. true mini-batch recompute training

- 动机: 真正引入 mini-batch SGD 噪声，同时保留传播梯度。
- 分支: `exp/training-modes`
- 关键工程:
  - 增加 `training_mode`
  - `recompute_minibatch` 每个 mini-batch 重跑完整传播
  - `frozen_readout`、`alternating_frozen_readout` 已验证为负向
- 正向配置: `recompute_minibatch bs=8192 lr=1e-4`
- 三 seed 结果:
  - `seed=2024`: `AUC 0.760737`, `ACC 0.727873`, `RMSE 0.428030`, `Brier 0.183210`, `ECE 0.042740`
  - `seed=2025`: `AUC 0.764064`, `ACC 0.728216`, `RMSE 0.426850`, `Brier 0.182201`, `ECE 0.045369`
  - `seed=2026`: `AUC 0.761622`, `ACC 0.727550`, `RMSE 0.428146`, `Brier 0.183309`, `ECE 0.045434`
- 三 seed 均值:
  - `test_auc = 0.762141`
  - `test_acc = 0.727879`
  - `test_rmse = 0.427675`
  - `test_brier = 0.182906`
  - `test_ece = 0.044514`
- 相对实验 34 三 seed 均值:
  - `AUC +0.000945`
  - `ACC +0.000324`
  - `RMSE -0.001494`
  - `Brier -0.001280`
  - `ECE -0.006628`
- 切片:
  - `concept_count=3` 的 `AUC +0.015340`
  - `concept_count=4+` 仍退化: `AUC -0.010693`, `RMSE +0.003337`, `ECE +0.001965`
  - `none_seen` 的 `RMSE/Brier` 改善，但 `ECE +0.001255`
- 结论:
  - 这是当前最强正向支线候选。
  - 训练协议优化已接近收益上限；`cosine + warmup` 仅带来 `AUC +0.000471`, `ACC +0.000127`，同时 `ECE +0.002442`，不值得整包主线化。

## 已验证无效或已降级

这些路线默认不要回到主线，除非用户明确要求复访。

### 明显无效

- 实验 1: 原始 Q 共现图基线
  - 只证明最小闭环能跑，不适合作为长期基线
- 实验 2: 稀疏归一化共现图 + 可学习 `q_e`
  - 没形成稳定泛化收益
- 实验 4: `TKC` 标量融合 + `student + exercise` 偏置 `g/s`
  - 早期全量结果退到随机附近
- 实验 5: 早期 `TKC` gated fusion
  - 单改融合形式没有解决主要问题
- 实验 10: `dual graph`
  - 在 ASSIST09 上明显退化，保留为 legacy ablation
- 实验 13: `q_e` residual 融合
  - 弱于当时主线
- 实验 14: `TKC/UKC` 同时局部硬 mean
  - 明显退化
- 实验 15: `TKC` 局部 mean + `UKC` 全局 mean
  - 没能救回局部硬汇聚路线
- 实验 20: 直接叠加 `local UKC neighbor` 和 `UKC-only coverage gate`
  - 没有形成 `1 + 1 > 1`
- 实验 22: 在实验 21 主线底座上复验 `local UKC neighbor`
  - 低于该底座
- 实验 25: readout 侧 `TKC` item-aware residual
  - 明显低于正式主线
- 实验 27: `q_repr` 内部 item-aware Q pooling
  - 单次弱于当前主线
  - 复访 `exp/qrepr-item-aware-residual` 后，`seed=2024` 相对实验 34 为 `AUC -0.000404`, `ACC +0.001180`, `RMSE +0.000123`, `Brier +0.000106`, `ECE +0.001479`
- 实验 29: 直接扩展 `cognitive_match` 输入以读入 `difficulty`
  - `seed=2026` 稳定崩盘，zero-init 也没救回
- 实验 30: 行为正误融合 gate 加 concept-specific bias
  - AUC 持平但误差和校准变差
- 实验 31: `UKC` no-param graph propagation
  - AUC 仅微升，但 `ACC/RMSE/Brier/ECE` 全变差
- 实验 32: `UKC` directed symmetric-like normalization
  - 单 seed 不满足继续扩 seed 门槛

### 近线失败路线

- 实验 35: local readout adapter
  - 读取当前题 Q mask 对应的 `TKC+UKC` 局部概念状态均值
  - 单 seed 相对实验 34 在 `AUC/ACC/RMSE/Brier/ECE` 上整体更差
- 实验 36: student base ability
  - 增加 zero-init 学生全局能力 embedding `theta_u`
  - 呈现“局部有信号，但整体排序和误差不占优”的折中
- 实验 41: item discrimination / 2PL logit scale
  - `seed=2024` 相对实验 34: `AUC -0.008723`, `ACC -0.003635`, `RMSE +0.001466`, `Brier +0.001261`, `ECE -0.019051`
  - 明显是校准折中，不是排序收益
- 实验 42: Soft-Q / learnable Q residual
  - 多组 `scale/stay/sparse/offset/threshold` 都未在 `AUC/ACC/RMSE/Brier/ECE` 上整体优于实验 34
  - 更激进的缺失边冷启动还会明显加噪
- 实验 43: hard-Q constrained concept residual
  - `seed=2024` 相对实验 34: `AUC -0.000155`, `ACC +0.000533`, `RMSE -0.000112`, `Brier -0.000096`, `ECE -0.001931`
  - `concept_count=4+` 与 `none_seen` 的 ECE 仍变差
- 实验 44: local exercise student adapter
  - `min_count=2`: `AUC -0.001609`, `ACC +0.000533`, `RMSE +0.000377`, `Brier +0.000324`, `ECE +0.002203`
  - `min_count=3`: `AUC -0.000509`, `ACC +0.000228`, `RMSE +0.000530`, `Brier +0.000456`, `ECE +0.001742`
  - 虽改善 `3/4+` 多知识点切片，但不能转化为 overall `AUC` 正收益
### 近期实验索引（详情外置）

下面从实验 45 起只保留决策索引；当前可用细节迁到 `docs/experiments/`，机器可读入口见 `docs/experiment_index.jsonl`。

- 实验 45: propagation-side concept-conditioned exercise residual
  - 分支/详情: `exp/propagation-concept-residual`; [045_propagation_concept_conditioned_exercise_residual.md](./experiments/045_propagation_concept_conditioned_exercise_residual.md)
  - 判断: 单 seed 局部信号不足；不继续沿 propagation 侧共享 residual 扩线

- 实验 46: readout-side qrepr score residual
  - 分支/详情: `exp/qrepr-score-residual`; [046_readout_qrepr_score_residual.md](./experiments/046_readout_qrepr_score_residual.md)
  - 判断: 早期单 seed 小信号未转成稳定主线收益

- 实验 47: final-logit none-seen calibration bias
  - 分支/详情: `exp/none-seen-calibration-bias`; [047_none_seen_calibration_bias.md](./experiments/047_none_seen_calibration_bias.md)
  - 判断: `none_seen` 可诊断但共享 final-logit bias 过粗，overall 不 clean

- 实验 48: history concept stats residual
  - 分支/详情: `exp/history-concept-stats`; [048_history_concept_stats_residual.md](./experiments/048_history_concept_stats_residual.md)
  - 判断: 显式历史统计有局部信号，但 original form 三 seed overall 不稳

- 实验 50: weighted pairwise history aggregation
  - 分支/详情: `exp/learned-pair-aggregation`; [050_weighted_pairwise_history_aggregation.md](./experiments/050_weighted_pairwise_history_aggregation.md)
  - 判断: learned pair aggregation 没有超过简单均值，主线保留 mean aggregation

- 实验 51: interpretable readout expert residual
  - 分支/详情: `exp/interpretable-readout-expert`; [051_interpretable_readout_expert_residual.md](./experiments/051_interpretable_readout_expert_residual.md)
  - 判断: full-trigger 三专家形成稳定正向，已进入后续主线底座

- 实验 52: clean interpretable readout routing
  - 分支/详情: `exp/clean-readout-routing`; [052_clean_interpretable_readout_routing.md](./experiments/052_clean_interpretable_readout_routing.md)
  - 判断: seen/unseen gate 与 top-k routing 没有超过实验 51 原版

- 实验 53: soft routing regularizer for readout experts
  - 分支/详情: `exp/readout-routing-soft-regularizer`; [053_soft_routing_regularizer.md](./experiments/053_soft_routing_regularizer.md)
  - 判断: soft regularizer 单 seed 小信号未能三 seed 复现

- 实验 54: Q-conditioned local mastery readout revisit
  - 分支/详情: `exp/q-conditioned-local-mastery-readout`; [054_q_conditioned_local_mastery_readout.md](./experiments/054_q_conditioned_local_mastery_readout.md)
  - 判断: 语义更干净但在实验 51 底座弱，后续作为诊断 2 交叉复验线索

- 实验 55: difficulty-weighted propagation
  - 分支/详情: `exp/difficulty-weighted-propagation`; [055_difficulty_weighted_propagation.md](./experiments/055_difficulty_weighted_propagation.md)
  - 判断: 极小 AUC 信号换误差/校准回撤，不是 clean win

- 实验 56: student-wise pairwise ranking loss
  - 分支/详情: `exp/student-pairwise-ranking-loss`; [056_student_pairwise_ranking_loss.md](./experiments/056_student_pairwise_ranking_loss.md)
  - 判断: ranking loss 只改排序偏好，overall AUC/ACC 不过门槛

- 实验 57: single-graph multi-hop propagation revisit
  - 分支/详情: `exp/single-graph-multi-hop-propagation`; [057_single_graph_multi_hop_propagation.md](./experiments/057_single_graph_multi_hop_propagation.md)
  - 判断: multi-hop variants 只形成轻微排序波动，切片也不 clean

- 实验 24: 多知识点题按知识点数分摊
  - 分支/详情: `legacy`; [024_multi_concept_equal_attribution.md](./experiments/024_multi_concept_equal_attribution.md)
  - 判断: 多知识点等分只有约 `+0.0001`，不作为主线结构

- 实验 26: `guess/slip` logit 正则
  - 分支/详情: `legacy`; [026_guess_slip_logit_regularizer.md](./experiments/026_guess_slip_logit_regularizer.md)
  - 判断: `guess/slip` logit 正则可改善校准但牺牲少量 AUC

- 实验 59: parallel local context readout adapter
  - 分支/详情: `exp/parallel-local-context-readout`; [059_parallel_local_context_readout.md](./experiments/059_parallel_local_context_readout.md)
  - 判断: 实现数值不稳，smoke 出现 BCE 输入越界，不进入正式比较

- 实验 60: pairwise history target-exclusion audit
  - 分支/详情: `exp/target-exclusion-audit`; [060_pairwise_history_target_exclusion_audit.md](./experiments/060_pairwise_history_target_exclusion_audit.md)
  - 判断: 证明 pairwise history target leakage/mismatch 会影响解释压力，导向实验 61

- 实验 61: full target-excluded training audit
  - 分支/详情: `exp/full-target-exclusion-opt`; [061_full_target_excluded_training_audit.md](./experiments/061_full_target_excluded_training_audit.md)
  - 指标摘要: `AUC +0.001606`, `ACC +0.000305`, `ECE +0.001932`
  - 判断: 三 seed AUC 正向但 ECE 更差；ranking-oriented 候选，不默认切换

- 实验 62-65: constrained `guess/slip` 系列
  - 分支/详情: `exp/guess-slip-diagnostics family`; [062_065_constrained_guess_slip_family.md](./experiments/062_065_constrained_guess_slip_family.md)
  - 判断: 硬约束能压掉语义反转但整体指标不恢复，越强 budget 越易伤 none_seen

- 实验 66: evidence-aware TKC propagation
  - 分支/详情: `exp/evidence-aware-tkc`; [066_evidence_aware_tkc_propagation.md](./experiments/066_evidence_aware_tkc_propagation.md)
  - 判断: 机制有信号但三 seed 未复现，behavior-only rescue 也不 clean

- 实验 67-68: learned multi-concept exercise attribution
  - 分支/详情: `exp/learned-exercise-attribution`; [067_068_learned_multi_concept_exercise_attribution.md](./experiments/067_068_learned_multi_concept_exercise_attribution.md)
  - 判断: 动态归因直接替换主聚合会削弱行为证据并伤 none_seen 校准

- 实验 69: student-conditioned UKC imputation
  - 分支/详情: `exp/student-conditioned-ukc-imputation`; [069_student_conditioned_ukc_imputation.md](./experiments/069_student_conditioned_ukc_imputation.md)
  - 判断: 直接替换 UKC 主状态放大 none_seen 低估，导向实验 70 sidecar

- 实验 70: student-conditioned UKC readout sidecar
  - 分支/详情: `exp/student-conditioned-ukc-readout-sidecar`; [070_student_conditioned_ukc_readout_sidecar.md](./experiments/070_student_conditioned_ukc_readout_sidecar.md)
  - 指标摘要: 三 seed `AUC 0.765517`, `ACC 0.729104`, `ECE 0.049044`
  - 判断: student-conditioned UKC sidecar 三 seed overall 正向，已进入当前默认主线

- 实验 71: exp70 + full target-excluded training combo
  - 分支/详情: `exp/exp70-target-exclusion-combo`; [071_exp70_target_excluded_training_combo.md](./experiments/071_exp70_target_excluded_training_combo.md)
  - 判断: 实验 70 + target-exclusion 单 seed AUC 小正但误差/校准回撤，不扩 seed

- 实验 72: representation bottleneck probes
  - 分支/详情: `exp/representation-bottleneck-probes`; [072_representation_bottleneck_probes.md](./experiments/072_representation_bottleneck_probes.md)
  - 判断: B49 local mastery 只改善误差/校准不提 AUC；target-conditioned context 明显伤排序

- 实验 73: current mainline protocol sweep
  - 分支/详情: `exp/mainline-protocol-sweep`; [073_current_mainline_protocol_sweep.md](./experiments/073_current_mainline_protocol_sweep.md)
  - 指标摘要: `AUC +0.000928`, `ACC +0.001313`, `ECE +0.002872`
  - 判断: `lr=7e-4 + patience` 是最强候选但 ECE 变差，非 clean 默认切换

- 实验 74: `guess/slip` monotonic soft penalty
  - 分支/详情: `exp/gs-monotonic-penalty`; [074_gs_monotonic_penalty.md](./experiments/074_gs_monotonic_penalty.md)
  - 指标摘要: `AUC -0.000170`, `ACC +0.000565`, `ECE +0.000200`
  - 判断: `1e-4` 三 seed 不稳且两个 seed 语义反转；`1e-3` 伤 AUC/ACC

## 旧口径的历史参考

下面这些实验只说明某类信号曾经出现过，不能直接当作当前主线结论。

- 实验 16: `TKC` item-aware attention + `UKC` 全局 mean
  - 说明“题目条件局部选择性”在旧主线下曾有信号
- 实验 17: coverage-aware `UKC` 邻域约束
  - 说明 `UKC` 全局噪声在旧口径下确实是问题
- 实验 18: coverage-aware 全局融合
  - 与后续主线融合语义重叠，不单独优先
- 实验 19: 只对全局 `UKC` 做 coverage gate
  - 保留为旧口径次优参考即可

## 近线 follow-up

这部分只保留正文之外仍值得额外记住的元判断，不再重复逐实验结论。

### 实验 28. `guess/slip` 显式引入题目难度特征

- 做法: 在 `guess/slip` 条件输入中拼入 `difficulty.detach()`
- 结果: 三 seed 下 AUC 基本持平，`ECE/Brier` 有均值改善
- 判断: 可记为候选，但增量不够大，不是新主线

### 诊断 1. 实验 33 主线的 test prediction slices

- 多知识点题明显更难，且随 `concept_count` 增长单调退化
- `none_seen` 排序不差，但校准很差，更像系统性低估
- 中等历史长度、中等历史正确率样本也偏弱
- 当前 split 是按学生随机抽样，不是严格时间切分，因此不优先做 recency-aware student state
- 结论:
  - 下一步应优先针对“多知识点题表示/读出偏弱”做 targeted 修补
  - `none_seen` 更适合作为后续单独校准问题，而不是当前第一优先结构问题
  - exact-3 aggressive residual/readout 已单 seed 验证到头，不再继续深挖同类结构

### 诊断 2. 实验 51 底座压制交叉复验

- 动机:
  - 当前主线吸收链是实验 34 -> 49 -> 51 -> 70；近期很多语义更干净的 readout / propagation 结构都只带来 `0.001-0.002` 量级甚至更小的收益
  - 为验证是否存在“某个历史主线底座吸收后，导致后续语义干净结构难以发挥”，把两个在实验 51 底座上失败的结构放回更早底座做交叉复验
- 口径:
  - 训练口径仍为 ASSIST09 ordered + single propagation graph + `300 epoch` + `lr=1e-3` + `concept_dim=64`
  - 实验 49 底座保留 `high_concept_logit_adapter / pairwise_history_interaction_adapter / gs_difficulty_adapter`
  - 实验 51 底座在实验 49 基础上再打开 `interpretable_readout_expert_adapter`
  - 实验 70 的 `student_conditioned_ukc_readout_residual` 本轮不纳入矩阵，先隔离实验 51 这个嫌疑点
- `q-conditioned local mastery readout` 扩大复验:
  - 分支: `exp/q-conditioned-local-mastery-readout`
  - 输出: `results/base_suppression/q_local_mastery_matrix/`
  - 实验 34 底座 `seed=2024`: `AUC +0.000233`, `ACC +0.001808`, `RMSE -0.000747`, `Brier -0.000640`, `ECE +0.000123`
  - 实验 49 底座三 seed 差值:
    - `seed=2024`: `AUC +0.000942`, `ACC +0.000019`, `RMSE -0.000394`, `Brier -0.000337`, `ECE -0.000725`
    - `seed=2025`: `AUC +0.002195`, `ACC -0.000247`, `RMSE -0.000350`, `Brier -0.000300`, `ECE +0.001913`
    - `seed=2026`: `AUC +0.002430`, `ACC -0.000058`, `RMSE -0.000715`, `Brier -0.000612`, `ECE -0.000914`
    - 均值: `AUC +0.001856`, `ACC -0.000095`, `RMSE -0.000486`, `Brier -0.000416`, `ECE +0.000091`
  - 实验 51 底座三 seed 差值:
    - `seed=2024`: `AUC -0.002073`, `ACC -0.001998`, `RMSE +0.000278`, `Brier +0.000238`, `ECE -0.003823`
    - `seed=2025`: `AUC -0.001573`, `ACC -0.000438`, `RMSE +0.000115`, `Brier +0.000099`, `ECE -0.001287`
    - `seed=2026`: `AUC -0.000920`, `ACC -0.001503`, `RMSE +0.000182`, `Brier +0.000156`, `ECE -0.003454`
    - 均值: `AUC -0.001522`, `ACC -0.001313`, `RMSE +0.000192`, `Brier +0.000164`, `ECE -0.002855`
- `difficulty-weighted behavior propagation` 扩大复验:
  - 分支: `exp/difficulty-weighted-propagation`
  - 输出: `results/base_suppression/difficulty_weighted_matrix/`
  - 实验 34 底座 `seed=2024`: `AUC +0.000081`, `ACC +0.001731`, `RMSE +0.000034`, `Brier +0.000029`, `ECE -0.000443`
  - 实验 49 底座三 seed 均值: `AUC +0.000212`, `ACC +0.000881`, `RMSE -0.000620`, `Brier -0.000530`, `ECE -0.003367`
  - 实验 51 底座三 seed 均值: `AUC -0.000306`, `ACC -0.000926`, `RMSE +0.000609`, `Brier +0.000523`, `ECE +0.003129`
- 判断:
  - `q-conditioned local mastery` 已经形成强三 seed 反转: 在实验 49 底座上 `AUC` 三 seed 全正且均值达到 `+0.001856`，但在实验 51 底座上 `AUC` 三 seed 全负且均值为 `-0.001522`
  - `difficulty-weighted behavior propagation` 不是强 AUC 路线，但也呈现底座敏感: 在实验 49 底座上均值改善 `ACC/RMSE/Brier/ECE`，在实验 51 底座上均值则 `AUC/ACC/RMSE/Brier/ECE` 全部转坏
  - 因此“实验 51 full-trigger readout expert residual 可能压制部分后续语义干净结构”的判断，从单 seed 嫌疑升级为当前应默认纳入实验设计的风险
  - 当前证据不支持把实验 49 判为更强正式主线；实验 51 自身三 seed 仍有稳定 `AUC` 正向
  - 后续若新结构在最新主线上轻微负向但语义足够干净，应优先比较 `实验 49 底座` 与 `实验 51/70 底座` 的边际收益；只有在旧底座正向、新底座负向时，再考虑重设主线吸收顺序、隔离实验 51 residual、或做 distillation / residual isolation

### 诊断 3. 更早底座广覆盖扫描

- 动机:
  - 诊断 2 已证明实验 51 会压制部分后续结构；进一步验证是否还有更靠前的“错误底座”，或者是否存在只在更早底座才成立的结构
  - 本轮优先做 `seed=2024` 粗扫；只有出现强信号的格子才扩 seed
- 底座定义:
  - `B33-like`: 当前代码内置 `cognitive_difficulty_adapter`，不打开 `high_concept / gs_difficulty / pairwise / expert`
  - `B34`: `B33-like + high_concept_logit_adapter + gs_difficulty_adapter`
  - `B49`: `B34 + pairwise_history_interaction_adapter`
  - `B51`: `B49 + interpretable_readout_expert_adapter`
- `q-conditioned local mastery` 更早底座补点:
  - `B33-like`, `seed=2024`: `AUC +0.000655`, `ACC +0.000057`, `RMSE -0.000413`, `Brier -0.000356`, `ECE +0.000082`
  - 结合诊断 2: `B33/B34` 只是小正，`B49` 才放大成三 seed `AUC +0.001856`，`B51` 又反转成三 seed `AUC -0.001522`
  - 判断: 目前最清楚的链条不是“越早越好”，而是 `B49` 的 history carrier 给 local mastery 提供可发挥条件，实验 51 expert 又压住这条 readout 结构
- `difficulty-weighted behavior propagation` 更早底座补点:
  - `B33-like`, `seed=2024`: `AUC -0.000760`, `ACC +0.000228`, `RMSE +0.000276`, `Brier +0.000238`, `ECE -0.001231`
  - 结合诊断 2: `B49` 均值改善 `ACC/RMSE/Brier/ECE`，`B51` 均值全面回撤
  - 判断: 它也不是越早越好；主要说明实验 49 后 propagation-side scaling 至少不伤校准，而实验 51 后副作用放大
- `single-graph multi-hop propagation` 粗扫:
  - 分支: `exp/multi-hop-propagation`
  - `B33-like`: `AUC -0.000772`, `ACC +0.000628`, `RMSE +0.000416`, `Brier +0.000359`, `ECE +0.000310`
  - `B34`: `AUC +0.000241`, `ACC -0.000191`, `RMSE -0.000110`, `Brier -0.000094`, `ECE +0.000289`
  - `B49`: `AUC +0.000214`, `ACC -0.000285`, `RMSE -0.000279`, `Brier -0.000239`, `ECE -0.002063`
  - `B51`: `AUC +0.000187`, `ACC -0.002284`, `RMSE +0.000803`, `Brier +0.000687`, `ECE +0.004522`
  - 判断: 它在 `B34/B49/B51` 都只有极小 AUC 正向；实验 51 后副作用明显变大，但早底座也没有足够强的 clean gain，不扩 seed
- `qrepr-score residual` 粗扫:
  - 分支: `exp/qrepr-score-residual`
  - `B33-like`: `AUC +0.000296`, `ACC +0.001218`, `RMSE -0.000504`, `Brier -0.000434`, `ECE -0.000857`
  - `B34`: `AUC -0.000151`, `ACC +0.002036`, `RMSE -0.000985`, `Brier -0.000845`, `ECE -0.004865`
  - 判断: 这是误差/校准型小修补，不是 AUC 推进路线；不构成底座压制主证据
- `concept-conditioned propagation residual` 粗扫与扩 seed:
  - 分支: `exp/concept-conditioned-prop`
  - `B34`, `seed=2024`: `AUC +0.000109`, `ACC -0.000361`, `RMSE +0.000271`, `Brier +0.000233`, `ECE +0.002161`
  - `B33-like`, `seed=2024`: `AUC +0.001349`, `ACC +0.001998`, `RMSE -0.000924`, `Brier -0.000793`, `ECE -0.001454`
  - `B33-like` 扩三 seed 后均值: `AUC -0.000521`, `ACC -0.000070`, `RMSE -0.000410`, `Brier -0.000352`, `ECE -0.002821`
  - 判断: 单 seed 早底座强正没有复现；它仍是校准/误差折中，不是稳定可回退底座
- 显式历史统计 residual 与 local evidence:
  - 分支: `exp/local-concept-evidence-head`
  - `history-concept-stats`, `B33-like`: `AUC -0.001419`, `ACC +0.001123`, `RMSE -0.000712`, `Brier -0.000612`, `ECE -0.003306`
  - `history-concept-stats`, `B34`: `AUC -0.000658`, `ACC +0.002474`, `RMSE -0.000410`, `Brier -0.000352`, `ECE +0.000528`
  - `local-concept-evidence-head`, `B33-like`: 正式长训在 valid 阶段触发 BCE 输入越界的 CUDA device-side assert；不继续扫
  - 判断: 原始显式统计 residual 仍是 ACC/误差/校准折中，pairwise history carrier 的成功不应被简化为“统计 residual 越早越好”
- `student-pairwise-ranking-loss weight=0.02` 粗扫:
  - 分支: `exp/student-pairwise-ranking-loss`
  - `B49`: `AUC -0.001024`, `ACC -0.000096`, `RMSE +0.000249`, `Brier +0.000214`, `ECE -0.001026`
  - `B51`: `AUC -0.000780`, `ACC -0.000552`, `RMSE +0.000272`, `Brier +0.000233`, `ECE +0.000818`
  - 判断: ranking loss 早底座也不成立，不属于被实验 51 压住的结构
- clean readout routing 粗扫:
  - 分支: `exp/clean-readout-routing`
  - `B34 dense`: `AUC -0.001514`, `ACC -0.003502`, `RMSE +0.001018`, `Brier +0.000876`, `ECE +0.001853`
  - `B34 topk=2`: `AUC +0.000834`, `ACC -0.001827`, `RMSE +0.000578`, `Brier +0.000497`, `ECE +0.004667`
  - `B49 dense`: `AUC -0.001157`, `ACC -0.002836`, `RMSE +0.001590`, `Brier +0.001365`, `ECE +0.003502`
  - `B49 topk=2`: `AUC +0.000798`, `ACC -0.002455`, `RMSE +0.000201`, `Brier +0.000173`, `ECE +0.001169`
  - 判断: 更干净 routing 在早底座也不是 clean win；实验 51 的问题不是简单靠 `seen/unseen` gate 或 top-k routing 就能修复
- 总结:
  - 更早底座扫描没有推翻诊断 2；目前唯一强且稳定的底座反转仍是 `q-conditioned local mastery`: `B49` 三 seed 明显正向，`B51` 三 seed 明显负向
  - 其它结构大多属于三类: 早底座也不成立、只改善误差/校准而不推 AUC、或在实验 51 后副作用放大但早底座收益太小
  - 后续若要继续验证“错误底座”，优先不再盲目扩所有旧路线，而应围绕实验 51 做更直接的隔离实验: 降低/冻结/蒸馏/正交化 readout expert residual，或测试“B49 + 新 readout + 不吸收实验 51”的主线替代链

### 主题归纳 1. `none_seen` 与校准

- 实验 47 已说明: `none_seen` 可以单独当校准问题做，但“对所有样本共享的 final-logit bias”过于粗糙
- 即便输入里放入 target coverage / `concept_count` / `difficulty`，模型也可能拿 overall ECE 去换 `none_seen` 自身校准
- 如果后续还要回到 `none_seen`，优先考虑更局部、更显式的触发方式，而不是继续扩这类全局共享 bias

### 主题归纳 2. 实验 51 读出侧后续

- “可解释 gate + 专家 residual” 这条线是成立的，说明读出侧适度增容本身有真实信号
- 但实验 52/53/54 共同说明: 无论是更硬的 selective routing、更软的 gate regularization，还是 local-first 的 readout 复访，都还没有形成比实验 51 更强的 clean overall 增益
- 因此若后续还要继续挖实验 51，前提应是出现更明确的 targeted slice 假设或更局部的引导目标，而不是默认继续扫 routing / local mastery 近邻变体

### 主题归纳 3. propagation 与目标层 tweak

- 实验 55/56/57 共同说明: difficulty 前移、直接叠 ranking loss、single-graph multi-hop propagation 都更像轻度改变排序偏好，而不是 clean overall 增益
- 实验 66 进一步说明: 即便显式历史统计前移到 propagation 主干，若以共享 gate 方式大范围注入，也更容易得到脆弱的单 seed 正信号，而不是稳定提点
- 实验 67 进一步说明: 动态归因虽然比静态分摊语义更强，但直接替换 exercise-to-concept 主聚合容易削弱行为证据并伤害 `none_seen` 校准
- 实验 68 进一步排除了几个自然 rescue: 保持 message scale、只打高 concept-count、只学 incorrect blame 都没有恢复主线收益
- 这些方向的常见模式是 `AUC` 有时略正，但 `ACC/RMSE/Brier/ECE` 更容易回撤
- 因此 propagation 侧与目标层 tweak 的优先级应继续下调；若再回到这些方向，前提应是已有更明确的局部 slice 假设，或已有更强的结构正向底座

## 默认下一步

通用协作、运行与分支规则沿用 [docs/session_bootstrap.md](./session_bootstrap.md)；这里仅补充历史台账导出的默认优先级:

1. 仍从当前 `master` 主线出发；新假设优先单改一个结构因素，单次成立后再补 `2-3` 个 seed。
2. 当前处于单因素边际收益放缓的平台期；实验 70 已把 `none_seen` 学生条件化信号转成 overall 正收益，但多知识点题仍不是 clean win。
3. 允许少量测试“已各自成立”的正交组合，但默认只测最强的 `1-2` 组候选，不做组合爆炸；实验 70 + 实验 61 的直接组合已在实验 71 单 seed 验证为不 clean，不默认扩 seed。
4. 允许探索更大一级、真正改变表示瓶颈的模块；若单次结果表现为“overall 未过门槛但目标 slice 有明显改善”，可额外允许一次很小的 rescue sweep；普通 sidecar / residual 默认不进入这类 sweep。
5. 若继续沿实验 51/70 的 readout 底座推进，默认保留实验 51 full-trigger expert 与实验 70 `none_seen` sidecar；但诊断 2 已提示实验 51 可能压制部分后续 clean structure 的边际表现，因此新结构若在最新主线上表现为轻微负向、但语义足够干净，可优先追加一次实验 49 底座单 seed 交叉复验，再决定是否淘汰。
6. 对已经系统复访但未形成 clean overall gain 的 readout / propagation / ranking-loss 路线，默认不再高优先级继续；只有在出现明确新假设、且机制上明显区别于已失败版本时，才考虑重开。
7. 若用户明确要继续训练协议优化，当前优先候选是实验 37 与实验 61 两条支线；否则默认优先继续模型结构改动。
8. 判断是否值得继续时，默认主看 `AUC/ACC`，并优先寻找至少 `1e-3` 量级的改善；`RMSE/Brier/ECE` 与分桶校准默认只用于判断副作用。
