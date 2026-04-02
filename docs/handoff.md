# Handoff

这份文档用于让新的 Codex 会话或新的账号快速接手当前项目。

## 先读这些文件

接手时按下面顺序阅读:

1. [README_spec.md](/home/xph/jwc/research/decoupled_cd/README_spec.md)
2. [docs/reuse_plan.md](/home/xph/jwc/research/decoupled_cd/docs/reuse_plan.md)
3. [docs/model_improvement_plan.md](/home/xph/jwc/research/decoupled_cd/docs/model_improvement_plan.md)
4. [docs/transition_graph_notes.md](/home/xph/jwc/research/decoupled_cd/docs/transition_graph_notes.md)
5. [docs/environment.md](/home/xph/jwc/research/decoupled_cd/docs/environment.md)

## 环境

- 项目目录: `/home/xph/jwc/research/decoupled_cd`
- conda 环境: `decoupled_cd`
- 推荐启动方式:

```bash
cd /home/xph/jwc/research/decoupled_cd
./scripts/start_codex.sh
```

## 当前项目状态

当前项目已经具备:

- Step 1: `E_u / TKC_u / UKC_u`
- Step 2: `TKC / UKC` 分离传播
- Step 3: 答题概率建模
- Step 4: BCE 反向优化
- `train / valid / test` 评估链路
- GPU 全量训练能力

当前不是工程闭环问题，而是模型效果问题。

## 已实现的工程层

- 数据读取与映射:
  - [data/readers.py](/home/xph/jwc/research/decoupled_cd/data/readers.py)
  - [data/mappings.py](/home/xph/jwc/research/decoupled_cd/data/mappings.py)
  - [data/pipeline.py](/home/xph/jwc/research/decoupled_cd/data/pipeline.py)
- 图构建:
  - [data/q_matrix.py](/home/xph/jwc/research/decoupled_cd/data/q_matrix.py)
  - [data/concept_graph.py](/home/xph/jwc/research/decoupled_cd/data/concept_graph.py)
- 模型:
  - [models/hetero_propagation.py](/home/xph/jwc/research/decoupled_cd/models/hetero_propagation.py)
  - [models/decoupled_cdm.py](/home/xph/jwc/research/decoupled_cd/models/decoupled_cdm.py)
- 训练与评估:
  - [trainers/engine.py](/home/xph/jwc/research/decoupled_cd/trainers/engine.py)
  - [scripts/train.py](/home/xph/jwc/research/decoupled_cd/scripts/train.py)
  - [scripts/evaluate.py](/home/xph/jwc/research/decoupled_cd/scripts/evaluate.py)
- 配置与工具:
  - [configs/defaults.py](/home/xph/jwc/research/decoupled_cd/configs/defaults.py)
  - [utils/device.py](/home/xph/jwc/research/decoupled_cd/utils/device.py)
  - [utils/logging.py](/home/xph/jwc/research/decoupled_cd/utils/logging.py)
  - [utils/metrics.py](/home/xph/jwc/research/decoupled_cd/utils/metrics.py)
  - [utils/io.py](/home/xph/jwc/research/decoupled_cd/utils/io.py)

## ASSIST09 相关数据

### 原始数据

- 原始文件:
  - `/home/xph/jwc/MRCogD/data/assist-09/meta-data/skill_builder_data_corrected_collapsed.csv`
- 原始数据中存在 `order_id`
- 原始文件本身没有按 `order_id` 排序

### 有序版处理脚本

- 脚本:
  - [scripts/preprocess_assist09_ordered.py](/home/xph/jwc/research/decoupled_cd/scripts/preprocess_assist09_ordered.py)
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

- [data/assist_09_ordered](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered)

## 当前推荐基线

当前推荐基线不是裸 Q 共现图，而是:

- 数据:
  - [data/assist_09_ordered/train.csv](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/train.csv)
  - [data/assist_09_ordered/valid.csv](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/valid.csv)
  - [data/assist_09_ordered/test.csv](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/test.csv)
- Q 矩阵:
  - [data/assist_09_ordered/Q_matrix.csv](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/Q_matrix.csv)
- 图:
  - [data/assist_09_ordered/transition_graph/propagation_graph.csv](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/transition_graph/propagation_graph.csv)

## 图构建说明

论文式转移图脚本:

- [scripts/build_assist09_transition_graph.py](/home/xph/jwc/research/decoupled_cd/scripts/build_assist09_transition_graph.py)

图文件说明文档:

- [docs/transition_graph_notes.md](/home/xph/jwc/research/decoupled_cd/docs/transition_graph_notes.md)

当前构图结果:

- 知识点数: `123`
- 先修边数: `1161`
- 相似边数: `2492`

## 已做过的关键实验结论

### 原始共现图基线

- 能训练
- 效果接近随机略高

### 稀疏归一化共现图 + 可学习 `q_e`

- 能训练
- 没有形成稳定提升

### 论文式转移图

- 当前最值得保留
- 在 `assist_09` 上取得目前更好的 `test_auc`

最新一版完整实验结果:

- [assist_09_transition_graph_20ep.json](/home/xph/jwc/research/decoupled_cd/results/assist_09_transition_graph_20ep.json)

## 下一步建议

优先做:

1. `TKC` 两路消息融合
   - 把 `exercise/behavior` 消息
   - 和 `concept neighbor` 消息
   - 从简单相加改成可学习融合
2. 再考虑升级 `g / s`
   - 从学生常数升级到 `student + exercise` 偏置

当前不建议:

- 继续只靠增加 epoch
- 回退到裸 Q 共现图
- 同时大改多个模块

## 推荐给下一个会话的开场提示

可以直接把下面这段发给新的 Codex:

```text
当前项目目录是 /home/xph/jwc/research/decoupled_cd。

请先阅读：
1. README_spec.md
2. docs/reuse_plan.md
3. docs/model_improvement_plan.md
4. docs/transition_graph_notes.md
5. docs/handoff.md
6. docs/environment.md

当前推荐基线是：
- ordered ASSIST09
- data/assist_09_ordered/transition_graph/propagation_graph.csv

当前下一步：
- 优先修改 TKC 两路消息融合
- 然后再考虑升级 g/s
```
