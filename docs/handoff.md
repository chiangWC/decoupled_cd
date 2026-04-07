# Handoff

这份文档用于让新的会话或新的账号快速接手当前项目。

## 先读这些文件

接手时按下面顺序阅读:

1. [README_spec.md](/home/jameschiang/work/decoupled_cd/README_spec.md)
2. [docs/reuse_plan.md](/home/jameschiang/work/decoupled_cd/docs/reuse_plan.md)
3. [docs/model_improvement_plan.md](/home/jameschiang/work/decoupled_cd/docs/model_improvement_plan.md)
4. [docs/transition_graph_notes.md](/home/jameschiang/work/decoupled_cd/docs/transition_graph_notes.md)
5. [docs/environment.md](/home/jameschiang/work/decoupled_cd/docs/environment.md)

## 环境

- 本地项目目录: `/home/jameschiang/work/decoupled_cd`
- 远端项目目录: `/home/xph/jwc/research/decoupled_cd`
- conda 环境: `decoupled_cd`
- 本地环境入口:

```bash
cd /home/jameschiang/work/decoupled_cd
./scripts/enter_env.sh
```

- 远端只用于运行代码。
- 如需在远端执行项目命令，先激活环境:

```bash
ssh xph-pc 'cd ~/jwc/research/decoupled_cd && conda activate decoupled_cd && <your-command>'
```

## 当前项目状态

当前项目已经具备:

- Step 1: `E_u / TKC_u / UKC_u`
- Step 2: `TKC / UKC` 分离传播
- Step 3: 答题概率建模
- Step 4: BCE 反向优化
- `train / valid / test` 评估链路
- GPU 全量训练能力

当前不是工程闭环问题，而是继续在稳定基线上做可解释的模型改动。

## 已实现的工程层

- 数据读取与映射:
  - [data/readers.py](/home/jameschiang/work/decoupled_cd/data/readers.py)
  - [data/mappings.py](/home/jameschiang/work/decoupled_cd/data/mappings.py)
  - [data/pipeline.py](/home/jameschiang/work/decoupled_cd/data/pipeline.py)
- 图构建:
  - [data/q_matrix.py](/home/jameschiang/work/decoupled_cd/data/q_matrix.py)
  - [data/concept_graph.py](/home/jameschiang/work/decoupled_cd/data/concept_graph.py)
- 模型:
  - [models/hetero_propagation.py](/home/jameschiang/work/decoupled_cd/models/hetero_propagation.py)
  - [models/decoupled_cdm.py](/home/jameschiang/work/decoupled_cd/models/decoupled_cdm.py)
- 训练与评估:
  - [trainers/engine.py](/home/jameschiang/work/decoupled_cd/trainers/engine.py)
  - [scripts/train.py](/home/jameschiang/work/decoupled_cd/scripts/train.py)
  - [scripts/evaluate.py](/home/jameschiang/work/decoupled_cd/scripts/evaluate.py)
- 配置与工具:
  - [configs/defaults.py](/home/jameschiang/work/decoupled_cd/configs/defaults.py)
  - [utils/device.py](/home/jameschiang/work/decoupled_cd/utils/device.py)
  - [utils/logging.py](/home/jameschiang/work/decoupled_cd/utils/logging.py)
  - [utils/metrics.py](/home/jameschiang/work/decoupled_cd/utils/metrics.py)
  - [utils/io.py](/home/jameschiang/work/decoupled_cd/utils/io.py)
  - [utils/seed.py](/home/jameschiang/work/decoupled_cd/utils/seed.py)

## ASSIST09 相关数据

### 原始数据

- 原始文件:
  - `/home/xph/jwc/MRCogD/data/assist-09/meta-data/skill_builder_data_corrected_collapsed.csv`
- 原始数据中存在 `order_id`
- 原始文件本身没有按 `order_id` 排序

### 有序版处理脚本

- 脚本:
  - [scripts/preprocess_assist09_ordered.py](/home/jameschiang/work/decoupled_cd/scripts/preprocess_assist09_ordered.py)
- 作用:
  - 先按 `user_id + order_id` 排序
  - 再保留每个学生对同一题的第一次作答
  - 再生成:
    - `data.csv`
    - `Q_matrix.csv`
    - `train.csv`
    - `valid.csv`
    - `test.csv`

有序版输出目录:

- [data/assist_09_ordered](/home/jameschiang/work/decoupled_cd/data/assist_09_ordered)

## 当前推荐基线

当前推荐基线不是裸 Q 共现图，而是:

- 数据:
  - [data/assist_09_ordered/train.csv](/home/jameschiang/work/decoupled_cd/data/assist_09_ordered/train.csv)
  - [data/assist_09_ordered/valid.csv](/home/jameschiang/work/decoupled_cd/data/assist_09_ordered/valid.csv)
  - [data/assist_09_ordered/test.csv](/home/jameschiang/work/decoupled_cd/data/assist_09_ordered/test.csv)
- Q 矩阵:
  - [data/assist_09_ordered/Q_matrix.csv](/home/jameschiang/work/decoupled_cd/data/assist_09_ordered/Q_matrix.csv)
- 图:
  - [data/assist_09_ordered/transition_graph/propagation_graph.csv](/home/jameschiang/work/decoupled_cd/data/assist_09_ordered/transition_graph/propagation_graph.csv)
- 默认配置:
  - `learning_rate = 1e-3`
  - `concept_dim = 64`
  - `gs_mode = conditional`
  - `TKC/UKC` 结构传播参数独立
  - 长训基线按 `300 epoch` 看

当前推荐结果口径:

- 结果文件:
  - [assist_09_tkc_ukc_separate_300ep.json](/home/jameschiang/work/decoupled_cd/results/assist_09_tkc_ukc_separate_300ep.json)
- 最优 checkpoint:
  - [assist_09_tkc_ukc_separate_300ep_best.pt](/home/jameschiang/work/decoupled_cd/results/assist_09_tkc_ukc_separate_300ep_best.pt)
- 指标:
  - `best_val_auc = 0.721520`
  - `best_epoch = 203`
  - `test_auc = 0.714303`
  - `test_acc = 0.695675`
  - `test_rmse = 0.449215`
- 多 seed:
  - [assist_09_tkc_ukc_separate_seed2024_300ep.json](/home/jameschiang/work/decoupled_cd/results/multiseed/assist_09_tkc_ukc_separate_seed2024_300ep.json)
  - [assist_09_tkc_ukc_separate_seed2025_300ep.json](/home/jameschiang/work/decoupled_cd/results/multiseed/assist_09_tkc_ukc_separate_seed2025_300ep.json)
  - [assist_09_tkc_ukc_separate_seed2026_300ep.json](/home/jameschiang/work/decoupled_cd/results/multiseed/assist_09_tkc_ukc_separate_seed2026_300ep.json)
  - `test_auc` 均值约 `0.7120`

## 图构建说明

论文式转移图脚本:

- [scripts/build_assist09_transition_graph.py](/home/jameschiang/work/decoupled_cd/scripts/build_assist09_transition_graph.py)

图文件说明文档:

- [docs/transition_graph_notes.md](/home/jameschiang/work/decoupled_cd/docs/transition_graph_notes.md)

当前构图结果:

- 知识点数: `123`
- 先修边数: `1161`
- 相似边数: `2492`

## 当前模型主干

当前主干不是最早的“简单点积 + 学生常数 g/s”版本，而是:

- Step 1:
  - 从交互中构造 `student_exercise_mask / TKC / UKC`
- Step 2:
  - `TKC` 分支同时接收
    - 题目-行为消息
    - 概念邻接消息
  - `UKC` 分支只接收概念邻接消息
  - `TKC` 与 `UKC` 的结构传播参数现已解耦，不再共享同一套概念传播变换
  - 学生状态由 `alpha * mean(TKC) + beta * mean(UKC)` 融合
- Step 3:
  - 题目表示 `q_e` 由 Q 掩码下的概念 gated pooling 与题目独立 embedding 融合得到
  - 认知概率不是简单点积，而是通过一个小匹配头计算
  - `g/s` 使用 `conditional` 模式，即依赖学生状态和题目表示，而不是纯学生常数
- Step 4:
  - BCE 训练
  - `train/valid/test` 评估
  - early stopping
  - best checkpoint 保存
  - `ReduceLROnPlateau` 调度框架

相关主文件:

- [models/decoupled_cdm.py](/home/jameschiang/work/decoupled_cd/models/decoupled_cdm.py)
- [models/hetero_propagation.py](/home/jameschiang/work/decoupled_cd/models/hetero_propagation.py)
- [trainers/engine.py](/home/jameschiang/work/decoupled_cd/trainers/engine.py)

## 已做过的关键实验结论

### 1. 图构建

- 原始 Q 共现图可训练，但效果弱
- 论文式 transition graph 明显更值得保留
- ordered ASSIST09 + transition graph 已经是当前固定基线

### 2. 超参数与训练长度

- `learning_rate = 1e-3`
- `concept_dim = 64`
- `gs_mode = conditional`

这套组合明显优于之前默认值。更关键的是，训练长度影响非常大:

- `20 epoch` 仍然远未训满
- `100 epoch` 后效果大幅上升
- `200 epoch` 继续提升
- `300 epoch` 达到当前最好结果

### 3. `conditional g/s` 是有效改动

- 在调优后超参数下，`conditional g/s` 明显优于 `constant g/s`
- 因此当前正式基线应保留 `conditional g/s`

### 4. scheduler 已接入，但更激进的 patience 没带来更好结果

- 当前训练引擎已支持 checkpoint 与 `ReduceLROnPlateau`
- 把 scheduler patience 调到更激进后，结果略差于原始单图 `300 epoch` 基线
- 因此 scheduler 不是当前主提升来源
- 当前正式主线应以 [assist_09_tkc_ukc_separate_300ep.json](/home/jameschiang/work/decoupled_cd/results/assist_09_tkc_ukc_separate_300ep.json) 为准

### 5. 自动 GPU 选择问题已修复

- 之前 `--device auto` 看起来会错误选到满卡
- 根因不是 `utils/device.py`，而是 [configs/defaults.py](/home/jameschiang/work/decoupled_cd/configs/defaults.py) 里默认把 `gpus` 锁死成了 `"0"`
- 现在默认 `gpus = None`
- 所以 `--device auto` 会在所有可见卡里选空闲最多的设备

### 6. 双图分开传播是负结果

- 当前代码支持:
  - `graph_mode = single`
  - `graph_mode = dual`
- `dual` 模式会分别读取:
  - `prerequisite_graph`
  - `similarity_graph`
- 但在 `assist_09` 上，双图 300 epoch 结果明显退化:
  - [assist_09_dual_graph_300ep.json](/home/jameschiang/work/decoupled_cd/results/assist_09_dual_graph_300ep.json)
  - `test_auc = 0.501716`
- 结论:
  - 双图接口可保留作实验开关
  - 但默认主线应继续使用单图 `propagation_graph`

### 7. `TKC/UKC` 结构传播参数解耦是正结果

- 当前最好结构改动是:
  - 保持单图 `propagation_graph`
  - 保持 `conditional g/s`
  - 将 `TKC` 与 `UKC` 的概念结构传播参数从共享改成独立
- 单次最好结果:
  - [assist_09_tkc_ukc_separate_300ep.json](/home/jameschiang/work/decoupled_cd/results/assist_09_tkc_ukc_separate_300ep.json)
  - `test_auc = 0.714303`
- 多 seed 也稳定优于旧基线
- 这条改动应视为当前新的候选主线

## 下一步建议

优先顺序:

1. 保持当前单图主线不变:
   - `propagation_graph`
   - `conditional g/s`
   - `TKC/UKC` 结构传播参数独立
2. 后续实验继续采用单变量比较:
   - 一次只改一个结构因素
   - 默认先跑 `2-3` 个 seed 再判断是否成立
3. 如需继续改结构:
   - 优先考虑更轻量的传播侧改动
   - 避免显著增加 full-batch 显存占用的主干改动

当前不建议:

- 回到裸 Q 共现图
- 忽略 `300 epoch` 基线重新讨论结构优劣
- 把 dual graph 误当成当前默认主线
- 同时大改多处主干

## 推荐给下一个会话的开场提示

可以直接把下面这段发给新的会话:

```text
当前项目目录是 /home/jameschiang/work/decoupled_cd。

请先阅读：
1. README_spec.md
2. docs/reuse_plan.md
3. docs/model_improvement_plan.md
4. docs/transition_graph_notes.md
5. docs/handoff.md
6. docs/environment.md
7. docs/workflow.md

当前推荐基线是：
- ordered ASSIST09
- data/assist_09_ordered/transition_graph/propagation_graph.csv
- learning_rate=1e-3
- concept_dim=64
- gs_mode=conditional
- TKC/UKC 结构传播参数独立
- 当前正式最好结果见 results/assist_09_tkc_ukc_separate_300ep.json

当前下一步：
- 在这条新主线上继续做单变量结构改动
- 不要把 dual graph 误当成当前主线
- 如需比较新结构，默认先跑 2-3 个 seed
- 如需去远端跑代码，先激活 decoupled_cd 环境
```
