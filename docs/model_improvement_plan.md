# Model Improvement Ledger

这份文档只保留实验台账入口用途，用来回答三件事:

- 当前 `master` 的正式主线是什么
- 哪些路线已经证明有效或无效
- 下一步默认该优先试什么

它不是新会话默认入口。流程与执行约束以 Trellis 为准：`.trellis/workflow.md` 和 `.trellis/spec/backend/experiment-protocol.md`。只有在需要查历史实验、避免重复试错时再回来看这份台账。

## 台账定位

- 实验台账仍然需要维护；它是跨会话、跨分支、跨 agent 接力时避免重复试错的结构化记忆。
- 台账不追求成为人类通读的完整实验史；默认优先服务 AI/agent 的检索、路由和决策。
- `docs/experiment_index.jsonl` 是结构化状态索引；本文件是当前判断和高信号路线图；`docs/experiments/` 是按需展开的证据附录。
- 历史恢复允许不完整，但必须显式标注。`detail_status` 是可信度提示，不是装饰字段。
- 旧实验只需保留足够回答“现在是什么状态、为什么接受/拒绝/暂停、什么条件下值得复访”；只有主线组件、强候选和容易被重复试错的失败路线才需要补充更多细节。

## 如何使用

- 先看“当前快照”，确认主线、结果口径和近线候选。
- 要判断某条路线是否还值得继续时，先看“已验证有效”和“已验证无效或已降级”的摘要，再跳到对应 detail doc。
- 要设计下一轮实验时，看“近线 follow-up”和“默认下一步”。
- 要按实验号、分支名、状态或失败原因定位时，查 `docs/experiment_index.jsonl`；它是 agent-facing 路由表和状态索引。
- 只有复访早期失败路线时才打开 `docs/archive_legacy_experiments.md`；只有需要跨实验诊断和主题归纳时才打开 `docs/experiment_themes.md`。
- 主文档只保留当前主线、候选和决策索引；实验细节默认外置。

## 当前快照

- 当前 Trellis 伪主线执行约束见 `.trellis/spec/backend/experiment-protocol.md`；正式 `master` accepted state 仍按下方实验 70 参考理解。
- 从这份台账的实验视角看，当前 `exp/trellis-trial` 伪主线对应实验 78 候选底座:
  - 以实验 34 为底座
  - 吸收实验 49 的 history-carrier pairwise interaction residual
  - 再吸收实验 51 的 interpretable readout expert residual
  - 再吸收实验 70 的 student-conditioned UKC `none_seen` readout sidecar
  - 再吸收实验 76 的 deterministic concept evidence prior residual
  - 再吸收实验 78 的 concept evidence readout correction
- 这里不再重复维护 Trellis spec 中的数据、图、超参数和 adapter 开关清单；需要确认默认运行口径时，优先查看 `.trellis/spec/backend/experiment-protocol.md`
- `master` 正式主线仍以实验 70 三 seed 作为 accepted reference:
  - `test_auc = 0.765517`
  - `test_acc = 0.729104`
  - `test_rmse = 0.427350`
  - `test_brier = 0.182628`
  - `test_ece = 0.049044`
- 当前 `exp/trellis-trial` 伪主线已合入实验 78 候选:
  - `seed=2024 test_auc = 0.772562`
  - `seed=2024 test_acc = 0.730823`
  - `seed=2024 test_rmse = 0.424811`
  - `seed=2024 test_brier = 0.180464`
  - `seed=2024 test_ece = 0.052361`
- 当前结果报告默认主看 `AUC/ACC`
- `RMSE/Brier/ECE/分桶校准` 默认作为次要指标
- 当前冲刺目标是 `test_auc ~= 0.780`，`0.778` 可视为接近可接受；相对当前 `exp/trellis-trial` 伪主线 seed=2024 还需约 `AUC +0.0055` 到 `+0.0075`
- 这个距离明显大于近期小 residual / sidecar 的常见边际收益，后续默认优先探索 representation-level 大结构改动；局部小改只有在支撑大结构假设时才优先考虑

- 当前已吸收的最新结构更新:
  - 实验 70: student-conditioned UKC `none_seen` readout sidecar 已进入 `master` 默认主线；三 seed 相对实验 51 主线均值 `AUC +0.001628`，且 `ACC/RMSE/Brier/ECE` 均值也小幅正向
  - 实验 76: deterministic concept evidence prior 已进入 `exp/trellis-trial` 伪主线默认运行口径；单 seed 相对实验 70 seed=2024 `AUC +0.005061`
  - 实验 78: concept evidence readout correction 已进入 `exp/trellis-trial` 伪主线默认运行口径；正常学习 seeds `2024/2025/2027` 相对实验 76 matched baseline 均值 `AUC +0.002542`

- 当前正向支线候选:
  - 实验 76
    - branch: `exp/evidence-calibrated-behavior-gate`
    - 判断: `concept_evidence_prior_residual` 的 `min_count=1, seen_ratio=1.0, max_logit=0.5` 已合入 `exp/trellis-trial` 伪主线；相对实验 70 seed=2024，`AUC +0.005061`、`ACC +0.000247`、`RMSE -0.001588`、`Brier -0.001355`，但 `ECE +0.002199`
    - 补充: 收益来自确定性的 student-concept train-history mastery prior，不使用 CF 或 student-exercise ID side channel；下一步优先补 seed，而不是继续堆同类 residual
    - 详细指标见 `docs/experiments/076_interpretable_concept_evidence_residuals.md`
  - 实验 37
    - branch: `exp/training-modes`
    - 判断: 它仍是当前更强的 calibration-oriented 训练协议候选，但在 `AUC/ACC` 上仍弱于当前主线，不作为默认 `master` 训练口径
    - 补充: 这条线属于纯训练工程优化，单次运行耗时显著高于当前默认 full-batch 口径；在模型结构仍需继续迭代时，暂不优先合入主线
    - 详细指标见 `docs/experiments/037_recompute_minibatch_training.md`
  - 实验 61
    - branch: `exp/full-target-exclusion-opt`
    - 判断: 它仍是 ranking-oriented target-exclusion 训练候选，工程优化后运行成本已从“明显过高”降到“可接受”，但仍不作为默认 `master` 训练口径
    - 补充: `2026-05-02` 复跑三 seed 后，它相对实验 51 有稳定 `AUC` 正向；实验 71 已验证它与实验 70 的直接组合不是 clean win，不默认继续扩组合 seed
    - 详细指标见 `docs/experiments/061_full_target_excluded_training_audit.md`

- 暂停中的 CF 支线:
  - 实验 38 `exp/cf-residual`: ranking-oriented 候选，但依赖学生内随机 split 的 ID-aware side channel，不作为纯 CDM 主线
  - 实验 39 `exp/cf-residual-recompute`: 不是实验 37 与 38 的无损叠加，不建议主线化
  - 实验 40 `exp/cf-residual-dim-sweep`: 大容量收益主要来自 transductive ID side channel，整条线暂停
  - 详情: `docs/experiments/038_040_cf_residual_family.md`；结构化状态见 `docs/experiment_index.jsonl`

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
  - 实验 75: history-conditioned Q representation 虽然改善少量多知识点 slice 校准，但 single seed overall `AUC/ACC/RMSE/Brier` 回撤，且 `none_seen` 排序回撤，不扩 seed
  - 实验 76 的前两条可解释 evidence 结构已被拒绝: evidence-calibrated behavior gate 与 trainable target-local concept evidence readout 都没有形成 clean overall gain；保留的是 deterministic concept evidence prior 的 `min_count=1` 配置
  - 实验 78 扫描中，`lr=7e-4` 是 calibration rescue 但 AUC 不升；`prior_strength=1.0` 有排序信号但误差/校准副作用过大；`prior_strength=1.5` 与 `max_logit=0.6` 都不是 clean win
  - 详细指标见对应实验条目

## 已验证有效

下面只保留改变主线判断或协议判断的摘要；详细证据外置到 `docs/experiments/`。如果已经知道实验号、状态或分支名，再用 `docs/experiment_index.jsonl` 快速定位。

- 实验 3: 论文式 transition graph
  - 详情: `docs/experiments/003_transition_graph.md`
  - 判断: transition graph 成为后续正式图结构起点
- 实验 6: 超参数扫描
  - 详情: `docs/experiments/006_hyperparameter_sweep.md`
  - 判断: 锁定 lr=1e-3、concept_dim=64 作为结构比较默认超参
- 实验 7: `conditional g/s`
  - 详情: 当前只保留本节摘要
  - 判断: conditional guess/slip 固化为默认配置
- 实验 8: 长训
  - 详情: `docs/experiments/008_long_training_protocol.md`
  - 判断: 正式结构比较默认采用长训协议
- 实验 9: 多 seed
  - 详情: `docs/experiments/009_multiseed_protocol.md`
  - 判断: 多 seed 稳定性成为正式比较要求
- 实验 11: `TKC/UKC` 结构传播参数解耦
  - 详情: `docs/experiments/011_tkc_ukc_decoupled_propagation.md`
  - 判断: TKC/UKC 传播参数解耦是可靠正向结构改动
- 实验 12: `TKC` 正误双通道行为消息
  - 详情: `docs/experiments/012_tkc_correct_incorrect_messages.md`
  - 判断: 错题证据必须保留为 TKC 行为消息
- 实验 21: `TKC/UKC` 学生自适应融合 gate
  - 详情: `docs/experiments/021_student_adaptive_fusion_gate.md`
  - 判断: 学生级 TKC/UKC 自适应融合 gate 成为默认配置
- 实验 23: 修正 `TKC` 行为项全局二次缩小
  - 详情: `docs/experiments/023_tkc_behavior_scaling_fix.md`
  - 判断: 修正行为项全局二次缩小后形成稳定基座
- 实验 33: `cognitive_match` zero-init difficulty adapter
  - 详情: `docs/experiments/033_cognitive_difficulty_adapter.md`
  - 判断: zero-init difficulty adapter 解决实验 29 崩盘并改善五指标
- 实验 34: high-concept logit adapter + `guess/slip` difficulty adapter
  - 详情: `docs/experiments/034_high_concept_gs_difficulty_adapters.md`
  - 判断: 实验 49/51/70 前的正式主线底座
- 实验 49: history-carrier pairwise interaction residual
  - 详情: `docs/experiments/049_history_carrier_pairwise_interaction.md`
  - 判断: history-carrier pairwise interaction residual 已吸收到主线
- 实验 37: true mini-batch recompute training
  - 详情: `docs/experiments/037_recompute_minibatch_training.md`
  - 判断: 仍是 calibration-oriented 训练候选，但不作为默认训练口径

## 已验证无效或已降级

- 近期实验 24/26/45-74 的状态索引见下方“近期实验索引（详情外置）”；需要按状态、失败原因或分支名筛选时再查 `docs/experiment_index.jsonl`。
- 早期 legacy / downgraded 条目只在复访旧失败路线时看 `docs/archive_legacy_experiments.md`。
- 跨实验诊断和主题归纳只在设计新实验或判断底座压制等横向问题时看 `docs/experiment_themes.md`。

### 近期实验索引（详情外置）

下面从实验 45 起只保留决策索引；当前可用细节迁到 `docs/experiments/`，机器可读入口见 `docs/experiment_index.jsonl`。

- 实验 45: propagation-side concept-conditioned exercise residual
  - 分支/详情: `exp/propagation-concept-residual`; `docs/experiments/045_propagation_concept_conditioned_exercise_residual.md`
  - 判断: 单 seed 局部信号不足；不继续沿 propagation 侧共享 residual 扩线

- 实验 46: readout-side qrepr score residual
  - 分支/详情: `exp/qrepr-score-residual`; `docs/experiments/046_readout_qrepr_score_residual.md`
  - 判断: 早期单 seed 小信号未转成稳定主线收益

- 实验 47: final-logit none-seen calibration bias
  - 分支/详情: `exp/none-seen-calibration-bias`; `docs/experiments/047_none_seen_calibration_bias.md`
  - 判断: `none_seen` 可诊断但共享 final-logit bias 过粗，overall 不 clean

- 实验 48: history concept stats residual
  - 分支/详情: `exp/history-concept-stats`; `docs/experiments/048_history_concept_stats_residual.md`
  - 判断: 显式历史统计有局部信号，但 original form 三 seed overall 不稳

- 实验 50: weighted pairwise history aggregation
  - 分支/详情: `exp/learned-pair-aggregation`; `docs/experiments/050_weighted_pairwise_history_aggregation.md`
  - 判断: learned pair aggregation 没有超过简单均值，主线保留 mean aggregation

- 实验 51: interpretable readout expert residual
  - 分支/详情: `exp/interpretable-readout-expert`; `docs/experiments/051_interpretable_readout_expert_residual.md`
  - 判断: full-trigger 三专家形成稳定正向，已进入后续主线底座

- 实验 52: clean interpretable readout routing
  - 分支/详情: `exp/clean-readout-routing`; `docs/experiments/052_clean_interpretable_readout_routing.md`
  - 判断: seen/unseen gate 与 top-k routing 没有超过实验 51 原版

- 实验 53: soft routing regularizer for readout experts
  - 分支/详情: `exp/readout-routing-soft-regularizer`; `docs/experiments/053_soft_routing_regularizer.md`
  - 判断: soft regularizer 单 seed 小信号未能三 seed 复现

- 实验 54: Q-conditioned local mastery readout revisit
  - 分支/详情: `exp/q-conditioned-local-mastery-readout`; `docs/experiments/054_q_conditioned_local_mastery_readout.md`
  - 判断: 语义更干净但在实验 51 底座弱，后续作为诊断 2 交叉复验线索

- 实验 55: difficulty-weighted propagation
  - 分支/详情: `exp/difficulty-weighted-propagation`; `docs/experiments/055_difficulty_weighted_propagation.md`
  - 判断: 极小 AUC 信号换误差/校准回撤，不是 clean win

- 实验 56: student-wise pairwise ranking loss
  - 分支/详情: `exp/student-pairwise-ranking-loss`; `docs/experiments/056_student_pairwise_ranking_loss.md`
  - 判断: ranking loss 只改排序偏好，overall AUC/ACC 不过门槛

- 实验 57: single-graph multi-hop propagation revisit
  - 分支/详情: `exp/single-graph-multi-hop-propagation`; `docs/experiments/057_single_graph_multi_hop_propagation.md`
  - 判断: multi-hop variants 只形成轻微排序波动，切片也不 clean

- 实验 24: 多知识点题按知识点数分摊
  - 分支/详情: `legacy`; `docs/experiments/024_multi_concept_equal_attribution.md`
  - 判断: 多知识点等分只有约 `+0.0001`，不作为主线结构

- 实验 26: `guess/slip` logit 正则
  - 分支/详情: `legacy`; `docs/experiments/026_guess_slip_logit_regularizer.md`
  - 判断: `guess/slip` logit 正则可改善校准但牺牲少量 AUC

- 实验 59: parallel local context readout adapter
  - 分支/详情: `exp/parallel-local-context-readout`; `docs/experiments/059_parallel_local_context_readout.md`
  - 判断: 实现数值不稳，smoke 出现 BCE 输入越界，不进入正式比较

- 实验 60: pairwise history target-exclusion audit
  - 分支/详情: `exp/target-exclusion-audit`; `docs/experiments/060_pairwise_history_target_exclusion_audit.md`
  - 判断: 证明 pairwise history target leakage/mismatch 会影响解释压力，导向实验 61

- 实验 61: full target-excluded training audit
  - 分支/详情: `exp/full-target-exclusion-opt`; `docs/experiments/061_full_target_excluded_training_audit.md`
  - 指标摘要: `AUC +0.001606`, `ACC +0.000305`, `ECE +0.001932`
  - 判断: 三 seed AUC 正向但 ECE 更差；ranking-oriented 候选，不默认切换

- 实验 62-65: constrained `guess/slip` 系列
  - 分支/详情: `exp/guess-slip-diagnostics family`; `docs/experiments/062_065_constrained_guess_slip_family.md`
  - 判断: 硬约束能压掉语义反转但整体指标不恢复，越强 budget 越易伤 none_seen

- 实验 66: evidence-aware TKC propagation
  - 分支/详情: `exp/evidence-aware-tkc`; `docs/experiments/066_evidence_aware_tkc_propagation.md`
  - 判断: 机制有信号但三 seed 未复现，behavior-only rescue 也不 clean

- 实验 67-68: learned multi-concept exercise attribution
  - 分支/详情: `exp/learned-exercise-attribution`; `docs/experiments/067_068_learned_multi_concept_exercise_attribution.md`
  - 判断: 动态归因直接替换主聚合会削弱行为证据并伤 none_seen 校准

- 实验 69: student-conditioned UKC imputation
  - 分支/详情: `exp/student-conditioned-ukc-imputation`; `docs/experiments/069_student_conditioned_ukc_imputation.md`
  - 判断: 直接替换 UKC 主状态放大 none_seen 低估，导向实验 70 sidecar

- 实验 70: student-conditioned UKC readout sidecar
  - 分支/详情: `exp/student-conditioned-ukc-readout-sidecar`; `docs/experiments/070_student_conditioned_ukc_readout_sidecar.md`
  - 指标摘要: 三 seed `AUC 0.765517`, `ACC 0.729104`, `ECE 0.049044`
  - 判断: student-conditioned UKC sidecar 三 seed overall 正向，已进入当前默认主线

- 实验 71: exp70 + full target-excluded training combo
  - 分支/详情: `exp/exp70-target-exclusion-combo`; `docs/experiments/071_exp70_target_excluded_training_combo.md`
  - 判断: 实验 70 + target-exclusion 单 seed AUC 小正但误差/校准回撤，不扩 seed

- 实验 72: representation bottleneck probes
  - 分支/详情: `exp/representation-bottleneck-probes`; `docs/experiments/072_representation_bottleneck_probes.md`
  - 判断: B49 local mastery 只改善误差/校准不提 AUC；target-conditioned context 明显伤排序

- 实验 73: current mainline protocol sweep
  - 分支/详情: `exp/mainline-protocol-sweep`; `docs/experiments/073_current_mainline_protocol_sweep.md`
  - 指标摘要: `AUC +0.000928`, `ACC +0.001313`, `ECE +0.002872`
  - 判断: `lr=7e-4 + patience` 是最强候选但 ECE 变差，非 clean 默认切换

- 实验 74: `guess/slip` monotonic soft penalty
  - 分支/详情: `exp/gs-monotonic-penalty`; `docs/experiments/074_gs_monotonic_penalty.md`
  - 指标摘要: `AUC -0.000170`, `ACC +0.000565`, `ECE +0.000200`
  - 判断: `1e-4` 三 seed 不稳且两个 seed 语义反转；`1e-3` 伤 AUC/ACC

- 实验 75: history-conditioned Q representation
  - 分支/详情: `exp/history-conditioned-q-repr`; `docs/experiments/075_history_conditioned_q_representation.md`
  - 指标摘要: 单 seed `AUC -0.001620`, `ACC -0.000266`, `ECE -0.001822`
  - 判断: 多知识点 slice 有极小正向和校准改善，但 overall 排序回撤且 `none_seen` AUC 回撤，不扩 seed

- 实验 76: interpretable concept evidence residuals
  - 分支/详情: `exp/evidence-calibrated-behavior-gate`; `docs/experiments/076_interpretable_concept_evidence_residuals.md`
  - 指标摘要: 最佳单 seed `AUC +0.005061`, `ACC +0.000247`, `RMSE -0.001588`, `Brier -0.001355`, `ECE +0.002199`
  - 判断: deterministic concept evidence prior 的 `min_count=1` 是新的可解释 CDM 候选；behavior gate 和 trainable readout residual 子线已拒绝

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


## 默认下一步

通用协作、运行与分支规则沿用 `.trellis/spec/backend/experiment-protocol.md`；这里仅补充历史台账导出的默认优先级:

1. 当前 Trellis-managed worktree 以后从 `exp/trellis-trial` 伪主线或其后代出发；`master` 只作为模型主线语义和 accepted state 参考。默认目标仍是冲 `test_auc ~= 0.780`；`0.778` 可视为接近可接受。
2. 实验 78 已按用户要求进入伪主线默认口径；下一步从实验 78 伪主线或其后代继续自由探索，优先寻找新的可解释结构信号并做 matched multi-seed 验证，不要继续扩 CF/ID side channel。
3. 当前处于单因素边际收益放缓的平台期；实验 70 已把 `none_seen` 学生条件化信号转成 overall 正收益，实验 76 则说明单知识点 student-concept train-history mastery prior 能提供更大 ranking 信号。普通小改默认只作为新假设准入或大结构假设的辅助验证，不再视为完整推进节奏。
4. 允许少量测试“已各自成立”的正交组合，但默认只测最强的 `1-2` 组候选，不做组合爆炸；实验 70 + 实验 61 的直接组合已在实验 71 单 seed 验证为不 clean，不默认扩 seed。
5. 默认优先探索更大一级、真正改变表示瓶颈的模块，例如学生状态形成、target-conditioned history、受约束的图/Q 结构学习，或明确标注为 hybrid 的 side channel；普通 sidecar / residual 默认不进入这类 sweep。
6. 若继续沿实验 51/70 的 readout 底座推进，默认保留实验 51 full-trigger expert 与实验 70 `none_seen` sidecar；但诊断 2 已提示实验 51 可能压制部分后续 clean structure 的边际表现。对 readout / `q_repr` / target-conditioned history / student-state 形成这类容易受实验 51 full-trigger expert 影响的 representation-level 改动，仍从 `exp/trellis-trial` 伪主线或其后代实现，但首轮实验设计默认至少包含 `B49 seed=2024` 与当前 `Exp70 seed=2024` 两格；不要只跑当前主线单格后直接下结论。若其它新结构在最新主线上表现为轻微负向、但语义足够干净，可优先追加一次实验 49 底座单 seed 交叉复验，再决定是否淘汰。
7. 对已经系统复访但未形成 clean overall gain 的 readout / propagation / ranking-loss 路线，默认不再高优先级继续；只有在出现明确新假设、且机制上明显区别于已失败版本时，才考虑重开。
8. 若用户明确要继续训练协议优化，当前优先候选是实验 37 与实验 61 两条支线；否则默认优先继续模型结构改动。
9. 判断是否值得继续时，默认主看 `AUC/ACC`；局部推进仍需至少 `1e-3` 量级改善，冲 `0.78` 的大结构单 seed 若连 `AUC +0.002` 左右信号都没有，通常不优先扩 seed。`RMSE/Brier/ECE` 与分桶校准默认只用于判断副作用。
