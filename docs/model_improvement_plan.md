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
- 从这份台账的实验视角看，当前 `exp/trellis-trial` 伪主线是实验 70 结构底座再叠加实验 81 的 single-only concept-evidence readout:
  - 以实验 34 为底座
  - 吸收实验 49 的 history-carrier pairwise interaction residual
  - 再吸收实验 51 的 interpretable readout expert residual
  - 再吸收实验 70 的 student-conditioned UKC `none_seen` readout sidecar
  - 再吸收实验 81 的 `concept_evidence_readout_residual(min_count=1, max_count=1)`
- 这里不再重复维护 Trellis spec 中的数据、图、超参数和 adapter 开关清单；需要确认默认运行口径时，优先查看 `.trellis/spec/backend/experiment-protocol.md`
- `master` 正式主线仍以实验 70 三 seed 作为 accepted reference:
  - `test_auc = 0.765517`
  - `test_acc = 0.729104`
  - `test_rmse = 0.427350`
  - `test_brier = 0.182628`
  - `test_ece = 0.049044`
- 当前 `exp/trellis-trial` 默认口径已 promote 到实验 81:
  - `seed=2024 test_auc = 0.767478`
  - `seed=2024 test_acc = 0.734248`
  - `seed=2024 test_rmse = 0.425562`
  - `seed=2024 test_brier = 0.181103`
  - `seed=2024 test_ece = 0.046972`
- 当前结果报告默认主看 `AUC/ACC`
- `RMSE/Brier/ECE/分桶校准` 默认作为次要指标
- 当前冲刺目标是 `test_auc ~= 0.780`，`0.778` 可视为接近可接受；相对当前 `exp/trellis-trial` 伪主线 seed=2024 还需约 `AUC +0.0105` 到 `+0.0125`
- 用户已修正本轮停止口径: seed2024 是当前 exp81 伪主线最低项，不能只和它比；当前 high-water 是 seed2027 `AUC 0.772682`，`+0.004` 停止阈值是 `0.776682`
- 这个距离明显大于近期小 residual / sidecar 的常见边际收益，后续默认优先探索 representation-level 大结构改动；局部小改只有在支撑大结构假设时才优先考虑

- 当前已吸收的最新结构更新:
  - 实验 70: student-conditioned UKC `none_seen` readout sidecar 已进入 `master` 默认主线；三 seed 相对实验 51 主线均值 `AUC +0.001628`，且 `ACC/RMSE/Brier/ECE` 均值也小幅正向
  - 实验 76: deterministic concept evidence prior 曾进入 `exp/trellis-trial` 伪主线默认运行口径，但 `2026-05-16` 补 official multi-seed 后确认 `seed2026` 从实验 76 开始就会退化，现已回退
  - 实验 78: concept evidence readout correction 曾进入 `exp/trellis-trial` 伪主线默认运行口径，但它建立在已回退的实验 76 底座上，现已随实验 76 一并回退
  - 实验 79: single-concept scoped readout 已验证为低幅稳定化信号但不合入；正常学习 seeds `2024/2025/2027` 相对实验 78 matched baseline 均值仅 `AUC +0.000769`
  - 实验 80: `single-only readout + exact-3 target interaction qrepr` 在实验 78 底座上形成过历史正向候选，但那条证据只保留为回退底座上的事实记录
  - 实验 81: `single-only` concept-evidence readout 已在 experiment 70 当前底座上完成三 seed 重验证，并已 promote 到 `exp/trellis-trial` 默认口径；相对 experiment 70 official three-seed baseline mean `AUC +0.004617`、`ACC +0.004275`、`RMSE/Brier/ECE` 同向改善
  - 实验 82: experiment 80 的 exact-3 interaction 已 rebased 到当前 exp81 伪主线并完成 matched seed=2024 验证；`scale=0.25` 全面回撤，`scale=0.125` 也只剩 `AUC +0.000054` 且 `ACC/RMSE/Brier/ECE` 仍反向，`concept_count=3` slice 也没有 clean 改善，因此拒绝
  - 实验 83: `q-local` 直接叠回当前 exp81 伪主线三 seed 判负；去掉 `expert` 后它会在 matched family 恢复，但 inverse / partial coverage gate 只会退化成 no-expert 轨迹，soft gate 更差，因此这条线当前留下的是“应改 expert 作用形式/位置”的结构诊断，而不是新的主线候选
  - 实验 84: 沿实验 83 诊断继续改 expert 输出形式/位置，并复访 deterministic evidence prior 的 final-logit、single-only、eval-only 与 cognitive-scale rescue；seed2026 已系统性回撤，seed2027 又确认 `max_logit=0.25` 出现 `AUC -0.0115` 负尾，`0.1875` 仍明显负向，`0.125` 仅弱混合，因此这条线记录为 rejected diagnostic / not trial candidate，不再继续微调同类 prior residual
  - 实验 85: 把 target-local evidence 信号前移到 state / qrepr 后仍未形成 trial 候选；`target_conditioned_student_state` seed2024 直接大幅负向，`target_evidence_attention_qrepr exact3 scale0.125` 只有 seed2024 弱正，seed2025/2027 回撤，三 seed 均值 `AUC -0.000383`、`ACC -0.001351` 且误差/校准也反向，因此拒绝，不继续同形参数扫
  - 实验 86: 把 readout expert 改成 per-sample contrastive / common-mode removed 参数化后，seed2024 达到 `AUC +0.002000` 且误差/校准改善，但 seed2025/2026 都小幅回撤；三 seed 均值只剩 `AUC +0.000256`，`ACC -0.000818`、`ECE +0.000233`，因此拒绝，不继续近邻 common-mode removal sweep
  - 实验 87: 自由探索确认 `0.005` 级 seed2024 overall 信号仍来自 deterministic student-concept evidence prior，而不是新的 hybrid ID residual:
    - hybrid ID residual 最好只是 `AUC +0.000218` 且 `ACC -0.000780`，只可视为低幅误差/校准 cleanup，不推广
    - `concept_evidence_prior_residual max_logit=0.25` 在当前 exp81 伪主线 seed2024 复现 `AUC 0.773261`，相对参考 `+0.005783`
    - 但实验 84 已证明同族配置有 seed2026/seed2027 负尾，因此该结果只作为 admission signal；下一步若继续，应解决 evidence prior 的稳定性，而不是直接合入或继续微调 max_logit
  - 实验 88: 在 deterministic evidence prior 上增加 high-confidence / high-mastery 门控后，`conf0.75 abs0.50 max0.25` 能保留 seed2024 `AUC +0.005049`，并把 seed2027 raw `-0.011524` 负尾收敛到约 `-0.000889`；但 seed2026 仍约 `-0.001683`，三 seed 均值只有 `AUC +0.000826`，因此记录为 stabilization diagnostic，不合入 trial，不继续附近 threshold / max_logit 微扫
  - 实验 89: 继续测试 evidence prior 的 model-agreement gate、`train_only` 应用和 positive/negative direction scope；agreement margin1.0 虽有 seed2024 `AUC +0.005566`，但 ACC/RMSE/Brier/ECE 明显变差且 seed2027 不如实验 88，其他方向都没过 seed2024 阈值，因此不再继续 deterministic prior mask/scope 小改
  - 实验 90: 重新核对本轮增长信号口径后，raw deterministic prior max0.25 只是在 matched seed2024 上复现 `+0.005783`，相对 high-water seed2027 `0.772682` 仅 `+0.000579`，不满足修正后的 `+0.004` 停止条件
  - 实验 91: valid-trained hybrid stacker 首次真正越过修正 high-water 停止线；4 个 seed2024 checkpoint 预测加 train-history tabular features，经 hist-gradient combiner 得到 `test_auc 0.787288`，相对 high-water `+0.014606`，且 `ACC/RMSE/Brier/ECE` 同向明显改善；这不是默认 CDM 主线组件，需作为明确 hybrid 候选做多 seed 验证或再整合进 readout/objective
  - 实验 92: 把 experiment 91 的 train-history 信号压成固定等权 deterministic output-logit readout prior 后，在 corrected high-water seed2027 checkpoint 上达到 `test_auc 0.776813`，相对 `0.772682` 为 `+0.004132`，满足停止阈值且没有 valid-trained combiner；但 `RMSE/Brier/ECE` 明显回撤，且同一 prior 放进 cognitive logit 或从头训练都会退化，因此只作为可解释 readout-prior 诊断信号，不合入默认纯 cognitive CDM
  - 实验 93/94: 将同一 train-history evidence 改为 `loss_only` 训练目标，约束 cognitive logits 与固定 evidence prior 的标准化排序对齐；hot config `target_concept=0.44, alignment=0.08810` 在 seed2027 达到 `test_auc 0.776868`，相对 high-water `+0.004187`，但 seed2026 会崩溃到 `AUC 0.504198`。降低到 `alignment=0.05` 后，seeds 2024/2025/2026/2027 全部 AUC 正向，matched mean `AUC +0.004828`，且 `ACC/RMSE/Brier/ECE` 均值同向改善。实验 95 的 CF-risk ablation 进一步证明，去掉 student/exercise 直接项的 `cogonly` 配置更强，matched mean `AUC +0.005508`、`ECE -0.004141`；只保留 student/exercise 的 `cfonly` 配置仅 `AUC +0.002939` 且 `ECE +0.003901`。当前 trial runner 已改为 `cogonly`

- 当前正向支线候选:
  - 实验 95
    - branch: `exp/trellis-trial`
    - 判断: 当前最符合“纯 CDM”方向的 trial candidate；history evidence 只作为训练期 cognitive alignment loss，推理时不加 output-logit sidecar，也没有 valid-trained combiner。CF-risk ablation 后，trial runner 采用更干净的 `cogonly` 版本，而不是含 student/exercise 直接项的 full 版本
    - runner: `scripts/run_assist09_history_alignment_trial.sh`
    - trial config: student/exercise evidence weight `0.0`、target-concept `0.44`、global-concept/mastery `0.22`、alignment `0.05`
    - 四 seed matched mean: `AUC +0.005508`、`ACC +0.002883`、`RMSE -0.002690`、`Brier -0.002279`、`ECE -0.004141`
    - 对照: full 配置均值 `AUC +0.004828`；`cfonly` 配置均值只有 `AUC +0.002939` 且 `ECE +0.003901`
    - 限制: hot config `alignment=0.08810` 虽然 seed2027 单点过线，但 seed2026 崩溃；trial 只能用 lower-strength `0.05`，下一步可测试更平滑 loss 以扩大稳定窗口
    - 详细指标见 `docs/experiments/095_history_alignment_cf_risk_ablation.md`；父实验 93/94 的原始 alignment family 和 full trial validation 见 `docs/experiments/093_history_evidence_cognitive_alignment.md`
  - 实验 92
    - branch: `exp/evidence-prior-calibrated-readout`
    - 判断: 这是解释 experiment 91/93 信号来源的重要 readout-prior 诊断；固定 `equal0.22` train-history evidence output-logit prior 在 seed2027 high-water checkpoint 上 `test_auc 0.776813`，超过修正停止线 `0.776682`
    - 限制: 它作用在最终 output logit，而不是 mastery/cognitive logit；`RMSE/Brier/ECE` 回撤，不能作为默认主线或纯 cognitive CDM 组件推广
    - 详细指标见 `docs/experiments/092_history_evidence_output_logit_prior.md`
  - 实验 91
    - branch: `exp/evidence-prior-calibrated-readout`
    - 判断: 这是当前最强 `0.78+` 级增长信号，已经满足修正后的 high-water `+0.004` 停止条件；但它是 valid-trained hybrid stacker，不是 `scripts/run_assist09_baseline.sh` 默认模型结构
    - 补充: 下一步优先做多 seed hybrid stacker 验证，并决定保留为 optional hybrid evaluator，还是把同一组 train-history 特征转成可训练 readout/objective 机制
    - 详细指标见 `docs/experiments/091_corrected_high_water_hybrid_stacker.md`
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
  - 实验 79: single-concept scoped readout 的四 seed AUC/RMSE/Brier/ECE 均正向，但正常学习 seeds mean `AUC +0.000769`，信号不够明显，按低幅稳定化诊断记录，不合入伪主线
  - 实验 80 之前的几条单模块近邻已经判清:
    - `target_concept_interaction_qrepr` 单独最好点是 `exact-3, scale=0.25`，相对实验 78 seed=2024 `AUC +0.000423`
    - `no-expert` 与 `no-expert + exact-3` 都没有放大这条信号，因此当前不把“实验 51 expert 压制”作为这条候选的主结论
    - 真正形成 clean candidate 的是实验 79 single-only readout 与 exact-3 interaction 的组合，而不是任何一条单模块独立成立
  - 实验 81: 同一个 `single-only` readout 假设 rebased 到 experiment 70 后，不再是低幅稳定化信号，而是当前底座上的 clear multi-seed win，并已 promote 到当前 pseudo-mainline
  - 实验 82: 在当前 exp81 伪主线上复访 experiment 80 的 exact-3 interaction 后，matched baseline 证明 `scale=0.25` 直接负向、`scale=0.125` 也只有 near-neutral AUC 且 `ACC/RMSE/Brier/ECE` 继续回撤；`concept_count=3` 目标切片没有 clean 收益，因此不扩 seed、不再把这条 rebase 当作默认 follow-up
  - 实验 83: `q-local` 在当前 exp81 伪主线上三 seed 判负，但 no-expert matched family 会恢复；coverage gate 只会把模型退化成 no-expert 轨迹或重新带回负面影响，因此后续若继续这条线，默认改 `expert` 作用形式/位置，而不是继续调 coverage gate
  - 实验 84: expert bound 单 seed 有信号但 seed2025 反转；post-expert q-local seed2025 仍明显负向；deterministic evidence prior 在 seed2024 有强排序信号，但 seed2027 证明 `max_logit=0.25` 与 `0.1875` 存在不可接受负尾，`0.125` 也只是弱混合结果；不合入 trial 候选，后续不要继续围绕 `concept_evidence_prior_*` 微调
  - 实验 85: concept-evidence state adapter、target-conditioned student-state rewrite 与 target-evidence attention qrepr 都没有跨 seed 成立；尤其 state rewrite 会大幅破坏排序，exact3 attention 也只是弱单 seed 信号，不合入 trial，不继续同形参数扫
  - 实验 86: contrastive readout expert 证明“限制 expert common-mode additive capacity”有单 seed 诊断信号，但跨 seed 幅度不足且 ACC/ECE 有副作用；不合入 trial，不继续同类 centering / common-mode removal 小改
  - 实验 87: hybrid ID residual 不是突破路径；deterministic evidence prior `max_logit=0.25` 再次确认 seed2024 `AUC +0.005783` 的强 admission signal，但仍受实验 84 的跨 seed 负尾约束，不作为 trial promote 候选
  - 实验 88: evidence-prior high-confidence/high-mastery gate 是目前最好的稳定化诊断，能保留 seed2024 `+0.005` 且大幅收窄 seed2027 负尾；但三 seed 均值只有 `AUC +0.000826` 且坏 seeds 未转正，不合入 trial，不继续同类确定性 residual scope/threshold 小扫
  - 实验 89: evidence-prior agreement / train-only / direction-only 三类补救均未优于实验 88；尤其 agreement margin1.0 只是用误差和校准换 seed2024 AUC，seed2027 还略差，因此后续不要继续 deterministic prior mask/scope 小改
  - 实验 90: raw prior 的 matched-seed admission signal 不是修正 high-water breakthrough；不要再把 seed2024 低参考当作停止条件
  - 实验 92: deterministic output-logit readout prior 是非 hybrid 的过线诊断，但误差和校准回撤明显；后续若要继续纯 CDM，应把这组 train-history evidence 移入校准目标或可靠性门控 readout，而不是直接推广 output-logit prior
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
2. 实验 76 / 78 已因 official multi-seed 暴露 `seed2026` 失败模式而从伪主线默认口径回退；实验 81 已作为 exp70-based follow-up promote 到当前 `exp/trellis-trial`，实验 82 证明 experiment 80 的 exact-3 interaction rebase 到当前底座后不再成立，实验 83 又说明 `q-local` 的问题主要在当前 `expert` 吸收方式而不是 `q-local` 本身。实验 84-86 进一步说明 bounded / post-expert / state-qrepr 前移 / contrastive common-mode removal 都不是稳定 trial 候选。下一步默认不再继续这条 readout/qrepr/expert-output 小组合，而是围绕实验 81 补 `B49 seed=2024` 交叉复验，或在这个新底座上继续更大的结构假设。
3. 当前处于单因素边际收益放缓的平台期；实验 70 已把 `none_seen` 学生条件化信号转成 overall 正收益。实验 76 说明单知识点 student-concept train-history mastery prior 在单 seed 上有大 ranking 信号，但这条线当前只能作为历史候选或待重构假设，不能直接视作当前主线组件。普通小改默认只作为新假设准入或大结构假设的辅助验证，不再视为完整推进节奏。
4. 允许少量测试“已各自成立”的正交组合，但默认只测最强的 `1-2` 组候选，不做组合爆炸；实验 70 + 实验 61 的直接组合已在实验 71 单 seed 验证为不 clean，不默认扩 seed。
5. 默认优先探索更大一级、真正改变表示瓶颈的模块，例如学生状态形成、target-conditioned history、受约束的图/Q 结构学习，或明确标注为 hybrid 的 side channel；普通 sidecar / residual 默认不进入这类 sweep。
6. 若继续沿实验 51/70 的 readout 底座推进，默认保留实验 51 full-trigger expert 与实验 70 `none_seen` sidecar；但诊断 2 已提示实验 51 可能压制部分后续 clean structure 的边际表现。对 readout / `q_repr` / target-conditioned history / student-state 形成这类容易受实验 51 full-trigger expert 影响的 representation-level 改动，仍从 `exp/trellis-trial` 伪主线或其后代实现，但首轮实验设计默认至少包含 `B49 seed=2024` 与当前 `Exp70 seed=2024` 两格；不要只跑当前主线单格后直接下结论。实验 81 已先按 exp70-based three-seed strong gain promote，`B49` 交叉复验转为 promote 后的确认项。
7. 对已经系统复访但未形成 clean overall gain 的 readout / propagation / ranking-loss 路线，默认不再高优先级继续；只有在出现明确新假设、且机制上明显区别于已失败版本时，才考虑重开。
8. 若用户明确要继续训练协议优化，当前优先候选是实验 37 与实验 61 两条支线；否则默认优先继续模型结构改动。
9. 判断是否值得继续时，默认主看 `AUC/ACC`；局部推进仍需至少 `1e-3` 量级改善，冲 `0.78` 的大结构单 seed 若连 `AUC +0.002` 左右信号都没有，通常不优先扩 seed。`RMSE/Brier/ECE` 与分桶校准默认只用于判断副作用。
