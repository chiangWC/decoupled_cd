# Handoff

这份文档保留“扩展交接”用途，用来说明当前主线、关键判断和关键文件。

它不是新会话默认第一入口。新会话默认先读 [docs/session_bootstrap.md](/home/jameschiang/work/decoupled_cd/docs/session_bootstrap.md)，只有在需要更多项目上下文时再读这份文档。

## 当前主线

- 当前工作重点是在稳定基线上继续做可解释的结构改动。
- 数据:
  - [train.csv](/home/jameschiang/work/decoupled_cd/data/assist_09_ordered/train.csv)
  - [valid.csv](/home/jameschiang/work/decoupled_cd/data/assist_09_ordered/valid.csv)
  - [test.csv](/home/jameschiang/work/decoupled_cd/data/assist_09_ordered/test.csv)
- Q 矩阵:
  - [Q_matrix.csv](/home/jameschiang/work/decoupled_cd/data/assist_09_ordered/Q_matrix.csv)
- 图:
  - [propagation_graph.csv](/home/jameschiang/work/decoupled_cd/data/assist_09_ordered/transition_graph/propagation_graph.csv)
- 默认配置:
  - `learning_rate = 1e-3`
  - `concept_dim = 64`
  - `gs_mode = conditional`
  - `graph_mode = single`
  - `TKC/UKC` 结构传播参数独立
  - `TKC` 行为消息使用正误双通道 + gated fusion
  - `TKC/UKC` 学生级融合使用自适应 gate
  - 长训比较默认看 `300 epoch`

当前推荐结果口径:

- 单次最好结果:
  - 远端结果文件:
    - `results/exp_adaptive_tkc_ukc_gate/assist_09_tkc_dual_channel_seed2025_300ep.json`
  - `best_val_auc = 0.757129`
  - `best_epoch = 300`
  - `test_auc = 0.751710`
  - `test_acc = 0.723020`
  - `test_rmse = 0.435074`
- 多 seed 结果:
  - 远端结果目录:
    - `results/exp_adaptive_tkc_ukc_gate/`
  - `test_auc` 均值约 `0.7502`

## 已经定下来的判断

- ordered ASSIST09 + transition graph 是当前固定数据主线。
- 当前训练主线使用单图 `propagation_graph`，不是 `dual graph`。
- `20 epoch` 远远不够，结构比较默认应看 `300 epoch` 量级。
- `conditional g/s` 明显优于 `constant g/s`，应保留。
- `TKC/UKC` 结构传播参数解耦是当前最可靠的正向结构改动。
- `TKC` 行为消息里显式保留错题信号是有效的，当前正误双通道优于只看正确题。
- 将全局固定 `alpha/beta` 升级为学生自适应 `TKC/UKC` 融合 gate 后，三 seed 结果已经稳定优于旧主线。
- `dual graph` 相关 CLI / 配置现在只应视为 legacy ablation 入口，不属于当前默认工作路径。
- `valid/test` 当前应复用 `train` 行为历史做传播输入，不能各自重建行为矩阵。
- 更激进的 scheduler patience 没有带来更好结果。
- `dual graph` 在 `assist_09` 上明显退化，默认不要当主线。
- 不要回到裸 Q 共现图重新做主基线判断，除非用户明确要求。
- 并行训练的日志文件名现在已经唯一化，不再共用同一个 `train_*.log`。

## 当前实验分支说明

- 当前已有的 `exp/*` 分支都是历史上“离主线还差一点”的次优路线，不是当前正式主线。
- 这些分支可作为复验参考，但不要直接把它们视为与当前 `master` 等价的候选主线。
- 现有分支:
  - `exp/adaptive-tkc-ukc-gate`
  - `exp/local-ukc-neighbor-fusion`
  - `exp/tkc-item-aware-attention`
  - `exp/ukc-coverage-beta-gate`
- 这些路线对应的背景、定位和相对主线的关系，参考 [docs/model_improvement_plan.md](/home/jameschiang/work/decoupled_cd/docs/model_improvement_plan.md) 的 D 部分快速索引。

## 当前关键文件

- 数据与映射:
  - [data/readers.py](/home/jameschiang/work/decoupled_cd/data/readers.py)
  - [data/mappings.py](/home/jameschiang/work/decoupled_cd/data/mappings.py)
  - [data/pipeline.py](/home/jameschiang/work/decoupled_cd/data/pipeline.py)
- 图构建:
  - [scripts/preprocess_assist09_ordered.py](/home/jameschiang/work/decoupled_cd/scripts/preprocess_assist09_ordered.py)
  - [scripts/build_assist09_transition_graph.py](/home/jameschiang/work/decoupled_cd/scripts/build_assist09_transition_graph.py)
- 模型:
  - [models/decoupled_cdm.py](/home/jameschiang/work/decoupled_cd/models/decoupled_cdm.py)
  - [models/hetero_propagation.py](/home/jameschiang/work/decoupled_cd/models/hetero_propagation.py)
- 训练与配置:
  - [scripts/train.py](/home/jameschiang/work/decoupled_cd/scripts/train.py)
  - [scripts/evaluate.py](/home/jameschiang/work/decoupled_cd/scripts/evaluate.py)
  - [scripts/remote_exec.sh](/home/jameschiang/work/decoupled_cd/scripts/remote_exec.sh)
  - [scripts/run_assist09_baseline.sh](/home/jameschiang/work/decoupled_cd/scripts/run_assist09_baseline.sh)
  - [scripts/run_assist09_multiseed.sh](/home/jameschiang/work/decoupled_cd/scripts/run_assist09_multiseed.sh)
  - [trainers/engine.py](/home/jameschiang/work/decoupled_cd/trainers/engine.py)
  - [configs/defaults.py](/home/jameschiang/work/decoupled_cd/configs/defaults.py)

## 后续实验规则

- 一次只改一个结构因素。
- 默认先跑 `2-3` 个 seed 再判断改动是否成立。
- 优先考虑更轻量的传播侧改动。
- 避免显著增加 full-batch 显存占用的主干改动。
- 新实验默认在 `exp/*` 分支上进行，确认成立后再整理回 `master`。
- 远端运行前，先把当前分支 `git push` 到 `origin`，再执行 `remote_exec.sh`。
