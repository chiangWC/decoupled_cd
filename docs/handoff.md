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
  - 长训比较默认看 `300 epoch`
  - 实验报告默认同时看 `AUC/ACC/RMSE` 和 `Brier/ECE/分桶校准`

当前推荐结果口径:

- 从 `master` 提交 `868f20b` 起，训练输出会包含 `Brier/ECE/分桶校准`。
- 更早的历史结果文件通常只含 `AUC/ACC/RMSE`，需要在新代码下复跑才有校准指标。

- 当前主线多 seed 结果目录:
  - `results/exp_tkc_exercise_aggregation/`
  - `test_auc` 均值约 `0.7597`
- 当前主线同口径校准重跑目录:
  - `results/exp_master_calibration_rerun/`
  - 三 seed 均值: `test_auc = 0.759690`, `test_ece = 0.062276`
- 单次最好结果和完整指标见 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的实验 23 与实验 28。

近期 follow-up:

- `exp/tkc-exercise-aggregation-qnorm`
  - 远端结果目录:
    - `results/exp_tkc_exercise_aggregation_qnorm/`
  - 三个 seed 的 `test_auc` 均值约 `0.7598`
  - 相比当前主线 `results/exp_tkc_exercise_aggregation/` 仅增约 `+0.0001`
- `exp/gs-difficulty-aware`
  - 远端结果目录:
    - `results/exp_gs_difficulty_aware/`
  - 三 seed 下 AUC 基本持平，ECE 有改善；暂记为可合入候选，不是已定新主线。
  - 详细指标见 [docs/model_improvement_plan.md](./model_improvement_plan.md) 的实验 28。

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

- 一次只改一个结构因素。
- 探索性结构改动默认先从最新 `master` 切 `exp/<short-name>` 分支。
- 新结构默认先跑单次；单次值得继续时再补 `2-3` 个 seed。
- 优先考虑更轻量的传播侧改动。
- 避免显著增加 full-batch 显存占用的主干改动。
- 新实验默认在 `exp/*` 分支上进行，确认成立后再整理回 `master`。
- 远端运行前，先把当前分支 `git push` 到 `origin`，再执行 `remote_exec.sh`。
