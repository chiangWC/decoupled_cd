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

- 默认运行约束见 `.trellis/spec/backend/experiment-protocol.md`；实验台账只记录路线判断，不重复维护完整开关清单。
- `master` accepted reference: 实验 70，three-seed mean `test_auc = 0.765517`。
- `exp/trellis-trial` baseline reference: 实验 81，经 `scripts/run_assist09_baseline.sh` 跑；seed2024 `test_auc = 0.767478`。
- active pure-CDM single-checkpoint trial runner: 实验 104，`scripts/run_assist09_history_alignment_trial.sh`，双塔 `64x80` + `dual_cdm_branch_bce_weight=0.10`。四 seed AUC `0.778773/0.778250/0.778508/0.777948`，mean `0.778370`，stdev `0.000306`。
- unpromoted pure-CDM refinement candidate: 实验 106，在实验 104 底座上将 `dual_cdm_branch_bce_weight` 从 `0.10` 提升到 `0.18`。四 seed AUC `0.778890/0.778552/0.778256/0.778618`，mean `0.778579`，stdev `0.000226`，相对实验 104 mean `+0.000209`；当前 HEAD 未把它提升为 runner 默认。
- lightweight pure-CDM reference: 实验 105 复测 exp104 pre-dual single64 base，seed2024 `test_auc 0.778379`、`max_cuda_memory_allocated_gb 6.33`，历史四 seed mean `0.776736`；它满足 `7-8GB` 显存目标但不满足替代当前 active runner 的稳定性。实验 107 恢复单塔 probe CLI 后复测两条轻量 follow-up：dim64 `output_alignment=0.002` 在 seed2026 从 `0.775065` 降到 `0.774925`，`checkpoint_selection_window=3` 在 seed2026 仅 `+0.000017` 且 seed2024 从 `0.778379` 降到 `0.778300`。实验 108 又继续排掉 soft difficulty regularization、pairwise/high-concept trigger threshold 收紧、cognitive-alignment residual focusing、validation `brier` checkpoint selection，并确认 `concept_evidence_prior_train_start_epoch=170` 仍只是 seed2026 `+0.000019` / seed2024 `-0.000167` 的低信号 tradeoff。因此继续保留 single64 作为轻量参考，不扩这些 follow-up 家族。shared-branch single-tower seed2024 最好 `0.776504`、`6.61GB`，single72 seeds 2024/2026 为 `0.776216/0.775486`、`6.83GB`，均不替代当前 active runner。
- 104 之后的实验主因: 双塔 `64x80` 虽然过 `0.778`，但 peak CUDA 约 `11.76GB`，提升相对成本有限。用户当前更希望后续 Codex 探索 `~7GB` 显存水平下更稳定的 pure-CDM 信号或更高峰值，而不是继续把 104/106 当作无限加重的默认主线。
- active pure-CDM checkpoint-average evaluator candidate: 实验 102，经 `scripts/evaluate_checkpoint_average.py` 跑；固定 probability average of experiment 95 + experiment 100 checkpoints，四 seed AUC `0.779011/0.778059/0.778969/0.779448`，mean `0.778872`，没有 valid-trained combiner 或 hybrid tabular side-channel。用户已明确不接受它作为当前 single-run/default-training 复现答案；它只保留为上界诊断和 evaluator 候选。
- 当前结果报告默认主看 `AUC/ACC`；`RMSE/Brier/ECE/分桶校准` 为次要指标。
- 当前 practical sprint target 已按用户口径调整为 `test_auc >= 0.778`；`0.780` 仍是 desirable headroom。当前 active runner 实验 104 已过线；实验 106 是已记录但未提升为默认的 pure-CDM refinement；实验 102 fixed checkpoint average mean `0.778872` 仍是 pure-CDM evaluator 上界；实验 99 hybrid stacker 三 seed mean `0.786910` 仍是最高绝对值但不是默认 CDM 路线。
- 后续若继续 heavy pure-CDM 论文路线，默认从当前 active runner 实验 104 出发；若采用实验 106，需要先明确 promote runner/default contract。若继续用户当前更关心的 low-VRAM 路线，应以实验 105 single64 为轻量参考，优先提出新的 `~7GB` 机制级假设，而不是继续实验 105-108 已判负的小旋钮。Heavy-route ablation 可补 `64x64/64x80/80x80`、branch 单独 AUC、融合 AUC、branch BCE 邻域解释、以及 `cognitive/guess/slip` 组件语义稳定性。

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
  - 实验 93: 将同一 train-history evidence 改为 `loss_only` 训练目标，约束 cognitive logits 与固定 evidence prior 的标准化排序对齐；hot config `target_concept=0.44, alignment=0.08810` 在 seed2027 达到 `test_auc 0.776868`，相对 high-water `+0.004187`，且 `ACC/RMSE/Brier/ECE` 同向改善。这证明 train-history evidence 可以作为纯 CDM training objective 的认知层信号，但单 seed 不足以 promote
  - 实验 94: 对实验 93 做 matched multi-seed validation；hot `alignment=0.08810` 在 seed2026 崩溃到 `AUC 0.504198`，因此拒绝 hot config。lower-strength full target `alignment=0.05` 在 seeds 2024/2025/2026/2027 全部 AUC 正向，matched mean `AUC +0.004828`，且 `ACC/RMSE/Brier/ECE` 均值同向改善，成为第一版稳定 trial candidate，但仍含 student/exercise direct history terms
  - 实验 95: CF-risk ablation 进一步证明，去掉 student/exercise 直接项的 `cogonly` 配置更强，matched mean `AUC +0.005508`、`ECE -0.004141`；只保留 student/exercise 的 `cfonly` 配置仅 `AUC +0.002939` 且 `ECE +0.003901`。当前 trial runner 已改为 `cogonly`
  - 实验 96: 在实验 95 `cogonly` 目标上把 standardized MSE alignment 换成 `standardized_smooth_l1` 或 `correlation` loss；seed2027 都只到 `test_auc ~= 0.7747`，低于实验 95 seed2027 `0.776163` 和 corrected stop threshold `0.776682`，因此拒绝，不扩 seed，不继续同 target/loss-shape 小扫
  - 实验 97: 在实验 95 `cogonly` 目标上继续测试 reliability-weighted alignment 与 target-only target construction；最佳 pure-CDM 点 `confidence_power=0.5, floor=0.2` 只到 seed2027 `test_auc 0.776264`，比实验 95 seed2027 高 `+0.000101` 但仍低于 corrected threshold。当前分支同时复现实验 91 hybrid hist-gradient stacker，seed2024 `test_auc 0.787288`、相对 high-water `+0.014606`，作为当前明确 hybrid growth signal；pure-CDM micro-sweep 暂停
  - 实验 98: 将纯 CDM runner 相关实现迁移到 `exp/pure-cdm-runner-integration`，补齐 cog-only trainable fusion、pairwise rank alignment 与 reliability-weighted alignment runner。新增路线没有超过实验 95：best fusion seed2027 `0.775913`，rank alignment 最好 `0.772090`，reliability best seed2027 `0.776264` 仍低于 `0.776682`，且四 seed 扩展在 seed2026 崩溃到 `0.504202`；保留实验 95 runner
  - 实验 99: 在 `exp/auc-078-exploration` 上完成 experiment 91/97 hist-gradient hybrid stacker 的 controlled three-seed validation；seeds 2024/2025/2026 分别为 `0.787288/0.788060/0.785382`，三 seed mean `0.786910`，全部超过用户修正目标 `0.778`。这是当前最强增长信号，但仍是 valid-trained hybrid evaluator，不是默认 CDM 或 pure-CDM runner promotion
  - 实验 100: 按用户要求回到 pure CDM runner/default promotion 路线，在 `exp/pure-cdm-default-promotion` 上新增 loss-only history evidence output-alignment objective 和 opt-in runner。`concept_dim=80 + output_alignment=0.004` 给出 pure-CDM seed2027 `test_auc 0.778122`，seed2026 `0.778100`，但四 seed mean 只有 `0.777030`，相对实验 95 mean 约 `+0.000751`，且 seeds 2024/2025 回撤；不修改 default runner，保留为局部增长信号和复现实验路径
  - 实验 101: 继续围绕实验 100 做 pure-CDM default-promotion follow-up。output-alignment confidence weighting 未超过 unweighted seed2027 `0.778122`；cognitive alignment 降到 `0.035/0.040/0.045` 或升到 `0.060` 都未修复 seed2024/2025；Adam `weight_decay=1e-5/3e-5/1e-4` 在 seeds 2024/2025 明显压垮 AUC；dim76 seed2024 近随机；`lr=7e-4/1.5e-3 + early_stop=20 + scheduler_patience=5`、cog-only linear readout、exercise difficulty init 也仍低于实验 95。因此这些支线均判负，不做 default promotion
  - 实验 102: 新增 `scripts/evaluate_checkpoint_average.py`，验证 experiment 95 cog-only checkpoint 与 experiment 100 dim80+output_alignment=0.004 checkpoint 的 prediction-only probability average。四 seed AUC `0.779011/0.778059/0.778969/0.779448`，mean `0.778872`，相对实验 95 mean `+0.002593`、相对实验 100 mean `+0.001841`，修复 seed2024/2025 tail 且保留 seed2026/2027 headroom。该路线没有 valid-trained combiner 或 hybrid features，是当前最强 pure-CDM runner/evaluator 候选；但它是两 checkpoint inference runner，不直接修改 `scripts/run_assist09_baseline.sh`
  - 实验 103: 用户拒绝把实验 102 fixed checkpoint average 当作 default-training answer 后，继续单 checkpoint pure CDM runner 探索。`late170to230` cognitive alignment anneal `0.05 -> 0.0881` 在 seeds 2024/2025/2026/2027 达到 AUC `0.778242/0.776913/0.775059/0.776280`，mean `0.776623`，相对实验 95 mean `+0.000345` 且四 seed 全正；但仍低于 `0.778` target 和实验 102 fixed-average mean，不作为已完成 default promotion。capacity dim72/80、rank alignment、target/global 配比偏移、multi-head readout、direct cognitive prior、concept calibrated readout、SWA 均未形成更好 single-checkpoint 默认候选
  - 实验 104: 在 `exp/pure-cdm-default-promotion` 上新增单 checkpoint 双塔 `DecoupledCDMEnsemble` 并合入 `exp/trellis-trial`。主塔 `concept_dim=64`、副塔 `concept_dim=80`，模型内概率平均；`dual_cdm_branch_bce_weight=0.10` 直接监督两个 tower 的 branch 输出。四 seed AUC `0.778773/0.778250/0.778508/0.777948`，mean `0.778370`、stdev `0.000306`，相对实验 95 mean `+0.002091`，也高于 pre-dual single-checkpoint base mean `0.776736`。这是当前 HEAD active runner 对应的 pure-CDM default-training 候选
  - 实验 105: 针对 exp104 显存成本复探 single-tower 轻量路线。exp104 pre-dual single64 seed2024 复测 `AUC 0.778379`、`max CUDA 6.33GB`，但历史四 seed mean `0.776736`；新增 shared-branch single-tower seed2024 最好只有 `0.776504`、`6.61GB`，single72 seeds 2024/2026 为 `0.776216/0.775486`、`6.83GB`。因此 single64 只作为轻量参考，shared-branch 与 dim72 均不替代实验 104；全程未使用非纯 CDM 或 checkpoint average
  - 实验 106: 继续从 exp104 双塔出发，只扫 pure-CDM branch BCE 权重。`0.18` 四 seed AUC `0.778890/0.778552/0.778256/0.778618`，mean `0.778579`、stdev `0.000226`，高于 exp104 `0.778370` 且更稳；`0.15` 给出最高峰值 `0.779046`，`0.20` mean `0.778561` 但 seed2026 尾部更弱。当前 HEAD 未把它提升为 runner 默认
  - 实验 107: 恢复 `scripts/train.py` 的单塔 alignment probe CLI，并在 exp105 single64 轻量基线上重测两条低成本 follow-up。`history_evidence_output_alignment_weight=0.002` 在 seed2026 为 `0.774925`，低于 single64 base `0.775065`；`checkpoint_selection_window=3` 在 seed2026 仅到 `0.775082`，但 seed2024 从 `0.778379` 降到 `0.778300`。因此不继续 dim64 output alignment 微扫，也不继续 checkpoint smoothing，single64 只保留为轻量参考
  - 实验 108: 在 exp105 single64 轻量基线上继续测试新的 low-VRAM pure-CDM follow-up。soft difficulty regularization `w=0.004/min_count=3/max_abs_logit=0.15/strength=4/cap=64` 在 seed2026 为 `0.774668`；`pairwise-history-interaction-min-count=3` 为 `0.773786`；`high-concept-logit-min-count=3` 与 cognitive-alignment residual focusing 都在 seed2026 近随机崩盘；`checkpoint-selection-metric=brier` 也降到 `0.773786`。唯一不崩的 schedule 改动是 `concept_evidence_prior_train_start_epoch=170`，但 seed2026 仅 `0.775084`、seed2024 从 `0.778379` 降到 `0.778212`。因此不要继续这些轻量 follow-up 家族，下一步需要新的机制级想法

- 当前正向支线候选:
  - 实验 106
    - branch/source: `exp/lightweight-single-tower-cdm`
    - 判断: 已记录的 single-run/single-checkpoint pure-CDM refinement 候选；不依赖 fixed checkpoint average、valid-trained combiner 或 hybrid tabular side-channel；当前 HEAD 未提升为 runner 默认
    - runner: `scripts/run_assist09_history_alignment_trial.sh`
    - config: 实验 104 双塔 `64x80` + late-window/train-only prior 底座，branch BCE 从 `0.10` 试验性提升到 `0.18`
    - 关键指标: seed2024 `0.778890`、seed2025 `0.778552`、seed2026 `0.778256`、seed2027 `0.778618`，mean `0.778579`，stdev `0.000226`
    - 限制: 仍是双塔路线，max CUDA peak 约 `11.76GB`，未解决实验 105 的轻量显存诉求；轻量 fallback 仍是 single64，但四 seed mean 不够
    - 详细指标见 `docs/experiments/106_dual_cdm_branch_bce_refinement.md`
  - 实验 104
    - branch/source: 已合入 `exp/trellis-trial`，trial commit `305c1dd`；探索来源为 `exp/pure-cdm-default-promotion`
    - 判断: 被实验 106 的 branch BCE `0.18` refinement 取代；仍是双塔结构基座和 ablation 对照
    - runner: `scripts/run_assist09_history_alignment_trial.sh`
    - config: 历史实验 104 底座为实验 103 late-window + train-only concept prior，并开启 dual CDM ensemble、secondary concept dim 80、branch BCE 0.10
    - 关键指标: seed2024 `0.778773`、seed2025 `0.778250`、seed2026 `0.778508`、seed2027 `0.777948`，mean `0.778370`
    - 限制: 解释性低于单塔 CDM，因为最终输出是两个 CDM tower 的结构化平均；但每个 tower 仍保留 `cognitive/guess/slip` 分解，且 branch BCE 让 branch-level 输出本身可监督。论文前必须补 `weight/capacity/branch` ablation
    - 详细指标见 `docs/experiments/104_single_checkpoint_dual_cdm_ensemble.md`
  - 实验 103
    - branch: `exp/pure-cdm-default-promotion`
    - 判断: 实验 104 前的 single-checkpoint baseline；late-window anneal 四 seed mean `0.776623`，四 seed 全正但低于 `0.778`，现在作为 104 的底座和 ablation 对照
    - 详细指标见 `docs/experiments/103_pure_cdm_late_alignment_promotion.md`
  - 实验 102
    - branch: `exp/pure-cdm-default-promotion`
    - 判断: 当前最强 pure-CDM runner/evaluator 候选；固定 probability average of experiment 95 cog-only checkpoint + experiment 100 dim80 output-alignment checkpoint，四 seed 全过 `0.778`。但用户当前要求继续 single-run/default CDM promotion，不接受它作为默认训练复现答案
    - runner/evaluator: `scripts/evaluate_checkpoint_average.py`
    - 关键指标: seed2024 `0.779011`、seed2025 `0.778059`、seed2026 `0.778969`、seed2027 `0.779448`，mean `0.778872`
    - 限制: 这是两 checkpoint inference runner，不是单 checkpoint default-training promotion；不使用 valid-trained combiner，也不使用 train-history tabular side-channel。暂不改 `scripts/run_assist09_baseline.sh`
    - follow-up: 若用户接受 checkpoint-average runner 路线，下一步应决定是否补正式 runner wrapper / seed2027-only reproduction docs / accepted default-evaluator semantics；若用户坚持单模型默认训练，则仍需新结构机制
    - 详细指标见 `docs/experiments/102_pure_cdm_checkpoint_average_runner.md`
  - 实验 100
    - branch: `exp/pure-cdm-default-promotion`
    - 判断: 当前最接近 `0.778` 的纯 CDM runner/default-promotion probe，但不满足默认晋升稳定性。dim80 加弱 output-alignment 在 seed2026/2027 暴露 headroom，最佳 seed2027 `AUC 0.778122`；四 seed mean `0.777030` 仅小幅高于实验 95，且 2024/2025 regression 明确
    - runner: historical probe was `scripts/run_assist09_history_output_alignment_trial.sh`; this wrapper is no longer part of the active CLI surface after exp104 cleanup
    - 限制: 不改 `scripts/run_assist09_baseline.sh`，也不替换实验 95 trial runner；实验 101 已判负 confidence weighting、weight decay、training-protocol、capacity interpolation、linear readout、exercise difficulty init 等 follow-up，后续只有在新结构机制能消除 2024/2025 tail 时才值得复访
    - 详细指标见 `docs/experiments/100_pure_cdm_default_promotion.md`
    - follow-up 判负见 `docs/experiments/101_pure_cdm_default_promotion_followups.md`
  - 实验 99
    - branch: `exp/auc-078-exploration`
    - 判断: 当前最强 `0.778+` controlled growth signal；hist-gradient hybrid stacker over four checkpoint predictions plus train-history tabular features 在 seeds 2024/2025/2026 全部过线，三 seed mean `AUC 0.786910`，`ACC/RMSE/Brier/ECE` 也强正
    - 关键指标: seed2024 `0.787288`、seed2025 `0.788060`、seed2026 `0.785382`；seed2026 即使一个成员 `raw_prior_max05` 崩溃到 `0.502923`，stacker 仍过 `0.778`
    - 限制: valid-trained combiner + train-history tabular side-channel，明确属于 hybrid evaluator；不能报告为 `scripts/run_assist09_baseline.sh` 默认模型，也不能当作纯 CDM promotion
    - follow-up: 可补 seed2027，或另开集成任务把同一 train-history feature family 转成模型侧 readout/objective；不要继续实验 95 同 target 的 pure-CDM micro-sweep，除非有新的机制假设
    - 详细指标见 `docs/experiments/099_hybrid_stacker_multiseed_078.md`
  - 实验 95
    - branch: `exp/trellis-trial`
    - 判断: 当前最符合“纯 CDM”方向的 trial candidate；history evidence 只作为训练期 cognitive alignment loss，推理时不加 output-logit sidecar，也没有 valid-trained combiner。CF-risk ablation 后，trial runner 采用更干净的 `cogonly` 版本，而不是含 student/exercise 直接项的 full 版本
    - runner: `scripts/run_assist09_history_alignment_trial.sh`
    - trial config: student/exercise evidence weight `0.0`、target-concept `0.44`、global-concept/mastery `0.22`、alignment `0.05`
    - 四 seed matched mean: `AUC +0.005508`、`ACC +0.002883`、`RMSE -0.002690`、`Brier -0.002279`、`ECE -0.004141`
    - 对照: full 配置均值 `AUC +0.004828`；`cfonly` 配置均值只有 `AUC +0.002939` 且 `ECE +0.003901`
    - 限制: hot config `alignment=0.08810` 虽然 seed2027 单点过线，但 seed2026 崩溃；trial 只能用 lower-strength `0.05`，下一步可测试更平滑 loss 以扩大稳定窗口
    - follow-up: 实验 96 已测试单纯替换 smooth/correlation loss，实验 97/98 已测试 reliability weighting、target-only target construction、cog-only trainable fusion 与 rank alignment；这些 probe 都没有越过 corrected threshold，且 reliability 四 seed 扩展触发 seed2026 崩溃。后续不要继续同 target/loss-shape/简单置信度权重/小型 readout 微扫，应改更大的 representation-level 消费方式或转向明确 hybrid 验证
    - 详细指标见 `docs/experiments/095_history_alignment_cf_risk_ablation.md`；父实验 93 的原始 alignment family 见 `docs/experiments/093_history_evidence_cognitive_alignment.md`，实验 94 的 full trial validation 见 `docs/experiments/094_history_alignment_trial_validation.md`
  - 实验 92
    - branch: `exp/evidence-prior-calibrated-readout`
    - 判断: 这是解释 experiment 91/93 信号来源的重要 readout-prior 诊断；固定 `equal0.22` train-history evidence output-logit prior 在 seed2027 high-water checkpoint 上 `test_auc 0.776813`，超过修正停止线 `0.776682`
    - 限制: 它作用在最终 output logit，而不是 mastery/cognitive logit；`RMSE/Brier/ECE` 回撤，不能作为默认主线或纯 cognitive CDM 组件推广
    - 详细指标见 `docs/experiments/092_history_evidence_output_logit_prior.md`
  - 实验 91
    - branch: `exp/evidence-prior-calibrated-readout`
    - 判断: 这是 experiment 99 的 parent signal；实验 97 已在 `exp/smooth-cognitive-alignment` 分支复现同一 hist-gradient hybrid result（seed2024 `test_auc 0.787288`），实验 99 已完成 seeds 2024/2025/2026 三 seed validation。但它是 valid-trained hybrid stacker，不是 `scripts/run_assist09_baseline.sh` 默认模型结构
    - 补充: 后续若继续这条线，应以 experiment 99 为当前 evidence，决定保留为 optional hybrid evaluator，还是把同一组 train-history 特征转成可训练 readout/objective 机制
    - 详细指标见 `docs/experiments/091_corrected_high_water_hybrid_stacker.md` 与 `docs/experiments/097_autonomous_growth_signal_exploration.md`
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
  - 实验 96: 在实验 95 cogonly target 上替换 smooth L1 / correlation alignment loss 后，seed2027 均只到约 `0.7747`，不如当前 experiment 95 MSE runner；不要继续同一 target 的 loss-shape 微扫
  - 实验 98: 纯 CDM runner 集成后的新增增强路线已判清：cog-only fusion 与 rank alignment 均低于实验 95，reliability weighting 虽 seed2027 有 `+0.000101` 微正但 seed2026 崩溃到 `0.504202`；不要推广这些 runner 为新的纯 CDM trial candidate
  - 实验 100: 纯 CDM output-alignment/default-promotion probe 已给出局部 `0.778+` 单 seed 信号，但四 seed 不稳；不要把 dim80 或 output alignment 直接设为默认 runner，不继续附近 weight 小扫
  - 实验 101: 实验 100 后续的 confidence weighting、cognitive-alignment 近邻权重、Adam weight decay、dim76 插值、lr/patience 协议、cog-only linear readout、exercise difficulty init 都未修复弱 seed；不要继续这些 follow-up，除非先提出能解释 2024/2025 tail 的结构性机制
  - 实验 103: late-window anneal 是当前 single-checkpoint 最好点，但边界也判清：`165->225` 与 `170->220` 更稳但均值低，`175->235` 与 `170->240` 伤校准或 seed2027；rank alignment 只改善二级指标并压 AUC；target-heavy/global-heavy prior mix 不稳定，global-heavy 触发 seed2026 随机崩溃；dim72/80 capacity 线仍伤 seed2024。因此后续不要继续同类窗口/配比小扫，除非提出新的 representation-level 机制
  - 实验 105: shared-branch single-tower 与 single72 capacity probe 均低于 exp104 pre-dual single64 reference，不替代实验 104；若显存是硬约束，可用 single64 reference，但不要把 shared-branch 或 dim72 设为长期默认路线
  - 实验 107: 恢复单塔 probe 控制后，dim64 `output_alignment=0.002` 和 `checkpoint_selection_window=3` 仍未形成值得扩四 seed 的路线；不要继续这两条单塔微调，除非先提出新的机制级理由
  - 实验 108: soft difficulty regularization、pairwise/high-concept trigger threshold 收紧、cognitive-alignment residual focusing、validation `brier` checkpoint selection 都已在 exp105 single64 轻量基线上判负；`concept_evidence_prior_train_start_epoch=170` 也只是 seed2026 极小正向而 seed2024 转负。不要继续这些轻量 rescue follow-up，除非先提出新的机制级解释
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

1. 当前 Trellis-managed worktree 以后从 `exp/trellis-trial` 伪主线或其后代出发；`master` 只作为模型主线语义和 accepted state 参考。当前 practical sprint target 是 `test_auc >= 0.778`，active runner 实验 104 已过线但显存重；实验 106 的 single-checkpoint pure-CDM mean `0.778579` 是未提升为默认的 refinement 候选；实验 110 的 `dual64x80 + branchBCE=0.18 + recompute_minibatch batch=65536 lr=3e-4` 是当前最好的 `~7GB` 级 pure-CDM 候选，四 seed mean `0.778321`、peak CUDA `6.19GB`，几乎贴住实验 104 且显存低很多。实验 109 的 `dual64x32 + branchBCE=0.18` 是较早的 full-batch 低成本双塔候选，四 seed mean `0.777674`、peak CUDA `8.85GB`。用户当前更希望继续找 `~7GB` 显存水平下更稳定或更高峰值的 pure-CDM 信号。实验 99 的 hybrid stacker 三 seed mean `0.786910` 仍是最高绝对值但不是默认 CDM 路线；`0.780` 仍可作为 desirable headroom。
2. 实验 76 / 78 已因 official multi-seed 暴露 `seed2026` 失败模式而从伪主线默认口径回退；实验 81 已作为 exp70-based follow-up promote 到当前 `exp/trellis-trial`，实验 82 证明 experiment 80 的 exact-3 interaction rebase 到当前底座后不再成立，实验 83 又说明 `q-local` 的问题主要在当前 `expert` 吸收方式而不是 `q-local` 本身。实验 84-86 进一步说明 bounded / post-expert / state-qrepr 前移 / contrastive common-mode removal 都不是稳定 trial 候选。下一步默认不再继续这条 readout/qrepr/expert-output 小组合，而是围绕实验 81 补 `B49 seed=2024` 交叉复验，或在这个新底座上继续更大的结构假设。
3. 当前处于单因素边际收益放缓的平台期；实验 70 已把 `none_seen` 学生条件化信号转成 overall 正收益。实验 76 说明单知识点 student-concept train-history mastery prior 在单 seed 上有大 ranking 信号，但这条线当前只能作为历史候选或待重构假设，不能直接视作当前主线组件。普通小改默认只作为新假设准入或大结构假设的辅助验证，不再视为完整推进节奏。
4. 允许少量测试“已各自成立”的正交组合，但默认只测最强的 `1-2` 组候选，不做组合爆炸；实验 70 + 实验 61 的直接组合已在实验 71 单 seed 验证为不 clean，不默认扩 seed。
5. 默认优先探索更大一级、真正改变表示瓶颈的模块，例如学生状态形成、target-conditioned history、受约束的图/Q 结构学习，或明确标注为 hybrid 的 side channel；普通 sidecar / residual 默认不进入这类 sweep。
6. 若继续沿实验 51/70 的 readout 底座推进，默认保留实验 51 full-trigger expert 与实验 70 `none_seen` sidecar；但诊断 2 已提示实验 51 可能压制部分后续 clean structure 的边际表现。对 readout / `q_repr` / target-conditioned history / student-state 形成这类容易受实验 51 full-trigger expert 影响的 representation-level 改动，仍从 `exp/trellis-trial` 伪主线或其后代实现，但首轮实验设计默认至少包含 `B49 seed=2024` 与当前 `Exp70 seed=2024` 两格；不要只跑当前主线单格后直接下结论。实验 81 已先按 exp70-based three-seed strong gain promote，`B49` 交叉复验转为 promote 后的确认项。
7. 对已经系统复访但未形成 clean overall gain 的 readout / propagation / ranking-loss 路线，默认不再高优先级继续；只有在出现明确新假设、且机制上明显区别于已失败版本时，才考虑重开。
8. 若用户明确要继续训练协议优化，当前优先候选是实验 37 与实验 61 两条支线；否则默认优先继续模型结构改动。
9. 判断是否值得继续时，默认主看 `AUC/ACC`；局部推进仍需至少 `1e-3` 量级改善，冲 `0.78` 的大结构单 seed 若连 `AUC +0.002` 左右信号都没有，通常不优先扩 seed。`RMSE/Brier/ECE` 与分桶校准默认只用于判断副作用。
