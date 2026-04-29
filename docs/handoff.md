# Handoff

这份文档保留“扩展交接”用途，用来说明当前主线、关键判断和关键文件。

它不是新会话默认第一入口。新会话默认先读 [docs/session_bootstrap.md](./session_bootstrap.md)，只有在需要更多项目上下文时再读这份文档。

## 当前主线

- 当前工作重点是在稳定基线上继续做可解释的结构改动。
- 数据:
  - [train.csv](../data/assist_09_ordered/train.csv)
  - [valid.csv](../data/assist_09_ordered/valid.csv)
  - [test.csv](../data/assist_09_ordered/test.csv)
- Q 矩阵:
  - [Q_matrix.csv](../data/assist_09_ordered/Q_matrix.csv)
- 图:
  - [propagation_graph.csv](../data/assist_09_ordered/transition_graph/propagation_graph.csv)
- 默认配置:
  - `learning_rate = 1e-3`
  - `concept_dim = 64`
  - `gs_mode = conditional`
  - `graph_mode = single`
  - `TKC/UKC` 结构传播参数独立
  - `TKC` 行为消息使用正误双通道 + gated fusion
  - `TKC/UKC` 学生级融合使用自适应 gate
  - `high_concept_logit_adapter = true`
  - `high_concept_logit_min_count = 2`
  - `pairwise_history_interaction_adapter = true`
  - `pairwise_history_interaction_min_count = 2`
  - `gs_difficulty_adapter = true`
  - 长训比较默认看 `300 epoch`
  - 实验报告默认主看 `AUC/ACC`
  - `RMSE/Brier/ECE/分桶校准` 默认作为次要指标，用于判断校准与误差副作用
  - 若目标是推进主线，默认希望 `AUC` 或 `ACC` 的改善至少达到 `1e-3` 量级

当前推荐结果口径:

- 从 `master` 提交 `868f20b` 起，训练输出会包含 `Brier/ECE/分桶校准`。
- 更早的历史结果文件通常只含 `AUC/ACC/RMSE`，需要在新代码下复跑才有校准指标。

- 当前主线基座结果目录:
  - `results/exp_tkc_exercise_aggregation/`
  - 三 seed 均值约 `test_auc = 0.7597`
- 当前正式主线结果目录:
  - `results/exp_pairwise_history_carrier/`
  - 三 seed 均值:
    - `test_auc = 0.762369`
    - `test_acc = 0.729763`
    - `test_rmse = 0.428205`
    - `test_brier = 0.183360`
    - `test_ece = 0.049828`
- 当前正式主线相对实验 23 基座均值差:
  - `AUC +0.002679`
  - `ACC +0.004979`
  - `RMSE -0.003183`
  - `Brier -0.002736`
  - `ECE -0.012448`
- 详细背景见 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的实验 34、实验 49 和实验 50。

当前正向训练策略支线:

- 分支:
  - `exp/training-modes`
- 关键配置:
  - `training_mode = recompute_minibatch`
  - `batch_size = 8192`
  - `learning_rate = 1e-4`
- 三 seed 均值:
  - `test_auc = 0.762141`
  - `test_acc = 0.727879`
  - `test_rmse = 0.427675`
  - `test_brier = 0.182906`
  - `test_ece = 0.044514`
- 相对实验 49 三 seed 均值:
  - `AUC -0.000228`
  - `ACC -0.001884`
  - `RMSE -0.000530`
  - `Brier -0.000454`
  - `ECE -0.005314`
- 判断:
  - 它仍是当前更强的 calibration-oriented 训练协议候选，但在 `AUC/ACC` 上已弱于实验 49 主线。
  - 收益主要来自真正 mini-batch SGD 的整体校准/泛化改善，不是彻底解决 `concept_count=4+` 或 `none_seen` 校准问题。
  - 后续若继续优化训练策略，默认从 `exp/training-modes` 出发。

近期暂停的 CF 模型支线:

- 分支:
  - `exp/cf-residual`
- 关键配置:
  - `cf_logit_residual = true`
  - `cf_dim = 16`
  - final-logit 学生-题目 MF residual，不注入 `cognitive_logits`
- 三 seed 均值:
  - `test_auc = 0.764671`
  - `test_acc = 0.730315`
  - `test_rmse = 0.428093`
  - `test_brier = 0.183264`
  - `test_ece = 0.051171`
- 相对实验 34 三 seed 均值:
  - `AUC +0.003475`
  - `ACC +0.002759`
  - `RMSE -0.001077`
  - `Brier -0.000923`
  - `ECE +0.000029`
- 判断:
  - 这是当前最强 ranking-oriented 结构候选，但不是校准候选。
  - 相对实验 37，`AUC/ACC` 更强，但 `RMSE/Brier/ECE` 更弱，尤其 `ECE +0.006657`。
  - 后续已验证和实验 37 的 `recompute_minibatch bs=8192 lr=1e-4` 不是无损叠加；容量扩展到 `cf_dim=64/128` 后 AUC 可到 `0.794/0.812`，但主要依赖当前学生内随机 split 的 transductive ID side channel。
  - 当前决定: CF 支线暂停，不继续扩容，也不作为纯 CDM 主线推进；若论文需要，可作为 optional hybrid / ID-aware ablation 或 appendix 讨论。

最近叠加验证:

- 分支:
  - `exp/cf-residual-recompute`
- 关键配置:
  - `cf_logit_residual = true`
  - `cf_dim = 16`
  - `training_mode = recompute_minibatch`
  - `batch_size = 8192`
  - `learning_rate = 1e-4`
- 三 seed 均值:
  - `test_auc = 0.762828`
  - `test_acc = 0.726852`
  - `test_rmse = 0.428070`
  - `test_brier = 0.183244`
  - `test_ece = 0.047508`
- 判断:
  - 相对实验 34: `AUC +0.001633`, `RMSE -0.001100`, `Brier -0.000942`, `ECE -0.003634`, 但 `ACC -0.000704`
  - 相对实验 37: `AUC +0.000687`, 但 `ACC/RMSE/Brier/ECE` 都更弱，其中 `ECE +0.002994`
  - 相对实验 38: `ECE -0.003663`，但 `AUC -0.001842`, `ACC -0.003463`
  - 结论: 不是无损叠加，暂不建议作为默认主线；如果继续 CF 路线，优先试更低容量或正则化，而不是直接合入

近期已吸收的 follow-up:

- 实验 33:
  - `cognitive_match` zero-init difficulty adapter
  - 先把实验 29 的“难度条件信号”改造成稳定 sidecar 形式
- 实验 34:
  - 对 `concept_count >= 2` 增加 high-concept logit residual
  - 在 conditional `guess/slip` 分支增加 difficulty residual
  - 这一步把实验 33 的诊断 1 follow-up 正式吸收到 `master`

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
- 实验 37 证明在当前主线结构不变的前提下，`recompute_minibatch bs=8192 lr=1e-4` 三 seed 同时改善 `AUC/ACC/RMSE/Brier/ECE`；但代码仍留在 `exp/training-modes`，尚未推广为 `master` 默认训练协议。
- 实验 38 证明 final-logit 学生-题目 MF residual 在当前学生内随机 split 下能显著改善 `AUC/ACC`，但不改善 `ECE`；它是 ranking-oriented 后门，不应混作纯 CDM 解释通道。
- 实验 39 证明 `cf_logit_residual cf_dim=16` 与 `recompute_minibatch bs=8192 lr=1e-4` 不是无损叠加；它形成 AUC/ECE 折中，但弱于实验 37 的校准，也弱于实验 38 的 AUC/ACC。
- 实验 40 证明扩大 CF residual 容量能显著抬高当前 split 的 AUC，但新增收益主要由 ID-aware residual 主导；该支线已暂停，后续回归实验 34/37 这类干净主线。
- 实验 43 证明 hard-Q constrained concept residual 会被 gate 使用，但在 `seed=2024` 上只是 `AUC -0.000155` 换 `ACC/RMSE/Brier/ECE` 小幅改善，且 `concept_count=4+` / `none_seen` 的 ECE 仍变差；不扩 seed，不作为主线结构推进。
- 实验 45 `exp/concept-conditioned-prop` 在 propagation 侧引入 exercise-to-concept concept-conditioned zero-init residual 后，`seed=2024` 相对当前主线 baseline 仅 `AUC +0.000109`，但 `ACC -0.000362`、`RMSE +0.000271`、`Brier +0.000233`、`ECE +0.002160`；`concept_count=2` 有轻微正向，但 `3/4+` 与 `none_seen` 仍不是干净收益，不扩 seed，不作为主线结构推进。
- 实验 46 `exp/qrepr-score-residual` 在 readout 侧引入 multi-concept-only zero-init exercise-conditioned Q-pooling score residual 后，`seed=2024` 相对当前主线 baseline 为 `AUC +0.000664`，但 `ACC -0.001370`、`RMSE +0.000185`、`Brier +0.000159`、`ECE +0.001519`；`concept_count=2/3` 的 AUC 有提升，但 `4+`、`none_seen`、`partial_seen` 仍出现明显副作用，不扩 seed，不作为主线结构推进。
- 实验 47 `exp/none-seen-calibration-bias` 在 final logit 侧加入只看 target coverage / `concept_count` / `difficulty` 的 zero-init calibration bias 后，`seed=2024` 相对当前主线 baseline 为 `AUC -0.000945`、`ACC -0.000381`、`RMSE +0.000438`、`Brier +0.000376`、`ECE -0.001393`；overall `ECE` 虽略降，但 `none_seen` 的 `ACC/RMSE/Brier/ECE` 全部变差，不扩 seed，不作为主线结构推进。
- 实验 48 `exp/history-concept-stats-adapter` 做了三 seed 复验:
  - 结构: 显式构造学生-概念历史正确率 / 覆盖率统计，按题相关概念聚合 `mean/min/gap/seen_ratio`，以 zero-init residual 形式接到 `cognitive_logits`
  - 三 seed 均值相对当前主线 baseline: `AUC -0.000366`, `ACC +0.001389`, `RMSE -0.000139`, `Brier -0.000120`, `ECE +0.000637`
  - `seed=2024` 的 `concept_count=2/3/4+` 切片明显改善，但 `seed=2025` 的多知识点切片出现不稳定反转，`none_seen` 也没有被修好
  - 结论: 这条路证明“显式历史概念统计”对多知识点题确有局部信号，但 overall 不稳定，不作为主线结构推进；若后续再访，优先改成更局部的 targeted trigger，而不是对所有 `concept_count>=2` 题统一加 residual
- 实验 49 `exp/pairwise-history-carrier` 做了三 seed 复验:
  - 结构: 保留实验 34 主线，其上新增只对 `concept_count>=2` 激活的 zero-init pairwise interaction residual
  - carrier 不再读局部 `TKC/UKC` embedding，而是在线构造逐概念历史统计: `accuracy / seen / log_attempt_count`
  - 对题相关概念对共享 scorer，pair score 做均值聚合后直接加到 `cognitive_logits`
  - 三 seed 均值相对实验 34 主线: `AUC +0.001173`, `ACC +0.002207`, `RMSE -0.000965`, `Brier -0.000827`, `ECE -0.001314`
  - `seed=2024` 切片:
    - `concept_count=2`: `AUC +0.011842`, `ACC +0.012003`, `RMSE -0.004802`
    - `concept_count=3`: `AUC +0.030833`, `ACC +0.025471`, `RMSE -0.013033`, `ECE -0.007842`
    - `concept_count=4+`: `AUC +0.018086`, `ACC +0.033058`, `RMSE -0.004449`
    - `none_seen`: `ACC +0.007841`, `RMSE -0.004422`, `ECE -0.008649`
  - 结论: 这是当前最强的结构主线更新，已经吸收到 `master`
- 实验 50 `exp/pairwise-history-weighted-agg` 做了单 seed follow-up:
  - 做法: 在实验 49 的 history-carrier pairwise residual 上，把 pair score 聚合从固定均值改成 learned weighting
  - `seed=2024` 相对实验 49: `AUC -0.000441`, `ACC -0.000305`, `RMSE -0.000120`, `Brier -0.000103`, `ECE +0.000002`
  - 结论: learned weighting 没有带来额外收益，主线保留简单均值聚合
- `exp/multi-concept-interaction` 已验证更激进的 exact-3 多知识点 residual/readout:
  - simpler `exact-3 readout` 仍是这条线上最平衡的版本: `AUC 0.761332`, `ACC 0.727283`, `RMSE 0.429016`, `Brier 0.184055`, `ECE 0.049997`
  - `tri_concept_readout_adapter` 能明显抬高 `concept_count=3` 的 AUC，但 calibration 代价过大，不值得继续扩 seed
  - `tri_concept_interaction_adapter` 能把 `concept_count=3` 和 `partial_seen` 局部指标继续做强，并基本抹平 `4+` 退化，但 overall 仍不如 simpler `exact-3 readout`
  - 结论: 这条 exact-3 aggressive residual/readout family 已到头，停止继续
- 多知识点题按知识点数分摊在当前口径下相对实验 23 几乎持平，暂时不是必须优先合入的关键因素。
- `dual graph` 相关 CLI / 配置现在只应视为 legacy ablation 入口，不属于当前默认工作路径。
- `valid/test` 当前应复用 `train` 行为历史做传播输入，不能各自重建行为矩阵。
- 更激进的 scheduler patience 没有带来更好结果。
- `dual graph` 在 `assist_09` 上明显退化，默认不要当主线。
- 不要回到裸 Q 共现图重新做主基线判断，除非用户明确要求。
- 并行训练的日志文件名现在已经唯一化，不再共用同一个 `train_*.log`。

## 当前实验分支说明

- `exp/*` 分支只作为实验代码和复验参考，不直接代表当前主线。
- 具体分支以 `git branch -a` 为准；每条路线的定位和结果以 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的详细条目与 D 部分快速索引为准。
- `exp/training-modes` 是当前优先继续的训练策略支线:
  - 已实现 `full_batch`、`target_accumulation`、`frozen_readout`、`alternating_frozen_readout`、`recompute_minibatch`
  - 已验证正向配置是 `recompute_minibatch bs=8192 lr=1e-4`
  - 已验证负向配置包括 `frozen_readout`、`alternating_frozen_readout`、`recompute_minibatch lr=1e-3`、`recompute_minibatch bs=4096 lr=3e-4`
  - 后续优化可优先试更细的学习率、调度器、早停策略或 weight decay，而不是再回到 frozen readout
- `exp/cf-residual` / `exp/cf-residual-dim-sweep` 当前暂停:
  - 已实现默认关闭的 `--cf-logit-residual`
  - 已验证 `cf_dim=16/64/128` 能提高当前学生内随机 split 的 ranking 指标
  - 已判断大容量收益主要依赖 transductive ID side channel，不作为纯 CDM 主线推进
- `exp/cf-residual-recompute` 是叠加验证支线:
  - 已验证 `cf_dim=16 + recompute_minibatch bs=8192 lr=1e-4`
  - 结果不是无损叠加，暂不建议把这组配置主线化
- `exp/multi-concept-interaction` 当前已收尾:
  - 已实现并验证 `multi_concept_interaction_adapter`、`multi_concept_readout_adapter`、`tri_concept_readout_adapter`、`tri_concept_interaction_adapter`
  - 单 seed 最平衡结果是 simpler `exact-3 readout`，相对实验 34 同 seed: `AUC +0.000028`, `ACC +0.001637`, `RMSE -0.000460`, `Brier -0.000395`, `ECE -0.001145`
  - 两个 tri-concept aggressive 版本都只带来局部 slice 改善，不能稳定转化为更优 overall，多 seed 价值不足
  - 这条支线暂停，不再继续追加 exact-3 aggressive 结构
- `exp/concept-conditioned-prop` 当前已做单 seed 判断:
  - propagation 侧 `exercise -> concept` concept-conditioned zero-init residual 在 `seed=2024` 上只带来 `AUC +0.000109`
  - `concept_count=2` 有小幅正向，但 `concept_count=3/4+` 与 `none_seen` 没形成可扩线的 clean win
  - 这条支线暂停，不继续扩 seed
- `exp/qrepr-score-residual` 当前已做单 seed 判断:
  - readout 侧 static Q pooling + multi-concept-only zero-init exercise-conditioned score residual 在 `seed=2024` 上带来 `AUC +0.000664`
  - `concept_count=2/3` 有局部 AUC 信号，但 `ACC` 不升，且 `4+`、`none_seen`、`partial_seen` 明显变差
  - 这条支线暂停，不继续扩 seed
- `exp/none-seen-calibration-bias` 当前已做单 seed 判断:
  - final-logit zero-init calibration bias 在 `seed=2024` 上只换来 `ECE -0.001393`
  - `none_seen` 本身没有被修好，反而出现 `ACC/RMSE/Brier/ECE` 一起变差
  - 这条支线暂停，不继续扩 seed
- `exp/history-concept-stats-adapter` 当前已做三 seed 判断:
  - `seed=2024` 的多知识点切片很强，但 `seed=2025/2026` 没能把这种局部收益稳定转成更优 overall
  - 三 seed 均值是 `AUC` 小降、`ACC/RMSE/Brier` 小幅改善、`ECE` 小幅变差
  - 这条支线暂停，不继续扩 seed；若再访，优先尝试更显式的 targeted trigger / mixture，而不是统一 residual

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

## 后续实验规则

- 默认一次只改一个结构因素。
- 若单因素已经给出明确的 overall 正向信号，或两个因素分别给出可解释且互补的 slice 信号，可少量做双因素组合验证。
- 双因素验证仍应严格限量，默认只测最强的 `1-2` 组候选，不做组合爆炸。
- 探索性结构改动默认先从最新 `master` 切 `exp/<short-name>` 分支。
- 新结构默认先跑单次；单次值得继续时再补 `2-3` 个 seed。
- 若单次相对当前主线的 `AUC/ACC` 连 `1e-3` 量级都明显达不到，默认不优先扩 seed，除非用户明确要求或切片信号非常强。
- 优先考虑更轻量的传播侧改动。
- 避免显著增加 full-batch 显存占用的主干改动。
- 新实验默认在 `exp/*` 分支上进行，确认成立后再整理回 `master`。
- 远端运行前，先把当前分支 `git push` 到 `origin`，再执行 `remote_exec.sh`。
- 实验 37 后续工程优化已做过一轮:
  - `weight_decay` / `grad clipping` 未带来值得保留的 `AUC/ACC` 增益
  - `cosine + warmup` 可带来极小的 `AUC/ACC` 提升，但量级不足以推动主线切换，且 `ECE` 变差
  - 训练模式支线暂不合入 `master`；如继续推进，优先做保持语义干净的模型改动，而不是继续打磨训练协议
- CF residual 支线暂停:
  - 不继续扫 `cf_dim`，不作为纯 CDM 主线推进
  - 后续默认回归实验 34/37 这类干净主线
