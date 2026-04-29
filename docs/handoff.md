# Handoff

这份文档保留“扩展交接”用途，用来说明当前主线、关键判断和关键文件。

它不是新会话默认第一入口。新会话默认先读 [docs/session_bootstrap.md](./session_bootstrap.md)，只有在需要更多项目上下文时再读这份文档。

## 当前主线

- 当前工作重点是在稳定基线上继续做可解释的结构改动。
- 当前正式主线配置与 [docs/session_bootstrap.md](./session_bootstrap.md) 的“当前主线”一致；这里只补充结果口径、候选支线和行动判断。
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
  - 代码口径已吸收实验 51；详细三 seed 指标以 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的实验 51 为准
  - 三 seed 均值:
    - `test_auc = 0.763889`
    - `test_acc = 0.728951`
    - `test_rmse = 0.427952`
    - `test_brier = 0.183143`
    - `test_ece = 0.049399`
- 当前正式主线相对实验 23 基座均值差:
  - `AUC +0.004199`
  - `ACC +0.004167`
  - `RMSE -0.003436`
  - `Brier -0.002953`
  - `ECE -0.012877`
- 详细背景见 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的实验 34、实验 49、实验 50 和实验 51。

当前正向训练策略支线:

- 分支: `exp/training-modes`
- 判断:
  - 它仍是当前更强的 calibration-oriented 训练协议候选，但在 `AUC/ACC` 上仍弱于当前正式主线。
  - 收益主要来自真正 mini-batch SGD 的整体校准/泛化改善，不是彻底解决 `concept_count=4+` 或 `none_seen` 校准问题。
  - 它属于纯训练工程优化，单次运行耗时显著高于当前默认 full-batch 口径；在模型结构仍需继续迭代时，暂不适合作为 `master` 默认训练协议。
  - 详细结果以 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的实验 37 为准。
  - 后续若继续优化训练策略，默认从 `exp/training-modes` 出发。

当前结构主线补充:

- 实验 51 的 full-trigger 三专家 readout residual 已吸收到当前 `master`。
- 当前主线新增默认配置:
  - `--interpretable-readout-expert-adapter`
  - `--interpretable-readout-expert-count 3`
- `min_count>=2` 和 `min_count>=3` 的 targeted 变体没有吸收；它们都弱于 full-trigger 版本。
- 后续 `exp/clean-readout-routing` 的 `seen/unseen` gate 统计和 `top-k` 稀疏路由也未超过当前主线，因此实验 51 原版 full-trigger 仍是这条线的正式保留版本。

近期暂停的 CF 模型支线:

- 分支:
  - `exp/cf-residual`
  - `exp/cf-residual-recompute`
  - `exp/cf-residual-dim-sweep`
- 判断:
  - 这是当前最强 ranking-oriented 结构候选，但不是校准候选。
  - 已验证与实验 37 不是无损叠加，且扩大容量后的收益主要依赖当前 split 下的 ID-aware side channel。
  - 详细结果以 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的实验 38-40 为准。
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
- 实验 48 证明“显式历史概念统计”对多知识点题确有局部信号，但 original form 的三 seed overall 不稳定；这条路如再访，应改成更局部的 targeted trigger / mixture，而不是继续推进统一 residual。详细指标见 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的实验 48。
- 实验 49 证明 history-carrier pairwise interaction residual 能把“显式历史概念统计”稳定转成 overall 正收益，并且 `none_seen` 也形成 clean win；这是实验 51 之前的结构主线更新，已经吸收到 `master`。详细指标见 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的实验 49。
- 实验 50 证明 learned pair aggregation 没有额外收益，主线保留简单均值聚合。详细指标见 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的实验 50。
- 实验 51 证明 interpretable readout expert residual 已形成稳定的非 ID-aware 结构更新；当前最强版本是 full-trigger 三专家，targeted 硬 trigger 反而更弱，这一步已经吸收到当前 `master`。详细指标见 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的实验 51。
- 实验 52 证明在实验 51 底座上继续加入更显式的 `seen/unseen` routing 统计或 `top-k` 稀疏路由，并没有带来更强结果；这条 selective routing follow-up 暂停。详细指标见 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的实验 52。
- 近期若干 follow-up（如 hard-Q residual、propagation/readout 侧多知识点 residual、全局共享 `none_seen` calibration bias、exact-3 aggressive residual/readout）除实验 51 外，都只形成局部 slice 信号或 seed-sensitive 折中，不作为主线结构推进；细节统一以 [docs/model_improvement_plan.md](./model_improvement_plan.md) 为准。
- 实验 38-40 的 CF 支线已确认主要依赖 ID-aware side channel，不作为纯 CDM 主线推进；若论文需要，可作为 optional hybrid / appendix 讨论。
- 多知识点题按知识点数分摊在当前口径下相对实验 23 几乎持平，暂时不是必须优先合入的关键因素。
- `dual graph` 相关 CLI / 配置现在只应视为 legacy ablation 入口，不属于当前默认工作路径。
- `valid/test` 当前应复用 `train` 行为历史做传播输入，不能各自重建行为矩阵。
- 更激进的 scheduler patience 没有带来更好结果。
- 不要回到裸 Q 共现图重新做主基线判断，除非用户明确要求。
- 并行训练的日志文件名现在已经唯一化，不再共用同一个 `train_*.log`。

## 当前分支优先级

- `exp/*` 分支只作为实验代码和复验参考，不直接代表当前主线。
- 具体分支以 `git branch -a` 为准；每条路线的定位和结果以 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的详细条目与 D 部分快速索引为准。
- 若继续优化训练协议，优先从 `exp/training-modes` 出发；它是当前唯一仍值得继续的训练策略支线。
- 若继续做结构主线，默认直接从最新 `master` 切新 `exp/*` 分支；实验 51 代码已吸收到主线，不需要回到 `exp/interpretable-readout-experts` 继续堆改动。
- `exp/clean-readout-routing` 已形成暂停判断，不作为当前优先继续线。
- `exp/cf-residual*` 当前暂停，不作为纯 CDM 主线推进。
- `exp/multi-concept-interaction`、`exp/concept-conditioned-prop`、`exp/qrepr-score-residual`、`exp/none-seen-calibration-bias`、`exp/history-concept-stats-adapter` 当前都已形成暂停判断；如需复访，先以 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的对应实验条目为准。

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

- 默认先只改一个结构因素，先把单因素证据立住。
- 若单因素已经给出明确的 overall 正向信号，或两个因素彼此正交、分别给出可解释且互补的证据，可少量做双因素组合验证。
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
