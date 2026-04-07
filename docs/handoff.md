# Handoff

这份文档是新会话的轻量入口，目标是用最少上下文接手当前项目。

## 默认阅读顺序

默认只需要先读两份:

1. [docs/workflow.md](/home/jameschiang/work/decoupled_cd/docs/workflow.md)
2. [docs/handoff.md](/home/jameschiang/work/decoupled_cd/docs/handoff.md)

其余文档按需再读:

- [README_spec.md](/home/jameschiang/work/decoupled_cd/README_spec.md)
  - 需要修改模型语义、核对 Step 1-4 定义时再读
- [docs/model_improvement_plan.md](/home/jameschiang/work/decoupled_cd/docs/model_improvement_plan.md)
  - 需要设计新实验、确认哪些改动已经做过时再读
- [docs/transition_graph_notes.md](/home/jameschiang/work/decoupled_cd/docs/transition_graph_notes.md)
  - 需要修改构图逻辑或检查图文件来源时再读
- [docs/reuse_plan.md](/home/jameschiang/work/decoupled_cd/docs/reuse_plan.md)
  - 只有在做工程重构、目录整理或参考项目复用时再读
- [docs/environment.md](/home/jameschiang/work/decoupled_cd/docs/environment.md)
  - 只是命令速查页

## 当前主线

当前不是工程闭环问题，而是在稳定基线上继续做可解释的结构改动。

当前推荐基线:

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
  - 长训比较默认看 `300 epoch`

当前推荐结果口径:

- 单次最好结果:
  - [assist_09_eval_history_fix_retrain_seed2024_300ep_gpu1.json](/home/jameschiang/work/decoupled_cd/results/assist_09_eval_history_fix_retrain_seed2024_300ep_gpu1.json)
  - `best_val_auc = 0.744711`
  - `best_epoch = 299`
  - `test_auc = 0.738442`
  - `test_acc = 0.711260`
  - `test_rmse = 0.441660`
- 多 seed 结果:
  - [assist_09_eval_history_fix_retrain_seed2024_300ep_gpu1.json](/home/jameschiang/work/decoupled_cd/results/assist_09_eval_history_fix_retrain_seed2024_300ep_gpu1.json)
  - [assist_09_eval_history_fix_retrain_seed2025_300ep_gpu1.json](/home/jameschiang/work/decoupled_cd/results/assist_09_eval_history_fix_retrain_seed2025_300ep_gpu1.json)
  - [assist_09_eval_history_fix_retrain_seed2026_300ep_gpu2.json](/home/jameschiang/work/decoupled_cd/results/assist_09_eval_history_fix_retrain_seed2026_300ep_gpu2.json)
  - `test_auc` 均值约 `0.7381`

## 已经定下来的判断

- ordered ASSIST09 + transition graph 是当前固定数据主线。
- 当前训练主线使用单图 `propagation_graph`，不是 `dual graph`。
- `20 epoch` 远远不够，结构比较默认应看 `300 epoch` 量级。
- `conditional g/s` 明显优于 `constant g/s`，应保留。
- `TKC/UKC` 结构传播参数解耦是当前最可靠的正向结构改动。
- `valid/test` 当前应复用 `train` 行为历史做传播输入，不能各自重建行为矩阵。
- 更激进的 scheduler patience 没有带来更好结果。
- `dual graph` 在 `assist_09` 上明显退化，默认不要当主线。
- 不要回到裸 Q 共现图重新做主基线判断，除非用户明确要求。

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

## 当前推荐启动方式

- 单次正式基线:
  - `bash scripts/remote_exec.sh bash scripts/run_assist09_baseline.sh`
- 多 seed 正式基线:
  - `bash scripts/remote_exec.sh bash scripts/run_assist09_multiseed.sh`
- 运行前仍然先:
  - `bash scripts/sync_to_remote.sh`
- 默认约定:
  - 所有项目代码都在远端主机上运行，包括训练、评估、测试和 smoke test。
  - 本地默认只做代码修改、阅读文档和同步。

## 推荐给新会话的开场提示

```text
当前项目目录是 /home/jameschiang/work/decoupled_cd。

请先阅读：
1. docs/workflow.md
2. docs/handoff.md

当前主线是：
- ordered ASSIST09
- data/assist_09_ordered/transition_graph/propagation_graph.csv
- learning_rate=1e-3
- concept_dim=64
- gs_mode=conditional
- graph_mode=single
- TKC/UKC 结构传播参数独立
- 结构比较默认看 300 epoch

当前规则：
- 默认一次只改一个结构因素
- 不要把 dual graph 当当前主线
- 如需比较新结构，默认先跑 2-3 个 seed
- 所有项目代码都在远端主机上运行；先同步并激活 decoupled_cd 环境
```
