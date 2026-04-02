# Transition Graph Notes

本文档说明 [assist_09_ordered/transition_graph](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/transition_graph) 目录下各文件的含义，以及当前训练主线实际使用的图文件。

## 目录说明

目录路径:

- [assist_09_ordered/transition_graph](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/transition_graph)

该目录中的文件由 [build_assist09_transition_graph.py](/home/xph/jwc/research/decoupled_cd/scripts/build_assist09_transition_graph.py) 生成，构图输入是有序版未切分数据:

- [data.csv](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/data.csv)

## 文件含义

### `correct_matrix.csv`

文件:

- [correct_matrix.csv](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/transition_graph/correct_matrix.csv)

含义:

- 对应论文中的准确率矩阵 `C`
- `C_{i,j}` 表示:
  - 学生在正确回答知识点 `i` 后
  - 紧接着也正确回答知识点 `j` 的条件概率
- 对角线被置为 `0`

这是最原始的方向性统计矩阵。

### `transition_scores.csv`

文件:

- [transition_scores.csv](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/transition_graph/transition_scores.csv)

含义:

- 对 `correct_matrix` 做 min-max 归一化后的结果
- 对应论文里归一化之后的 `T_{ij}`
- 数值范围在 `0~1`
- 数值越大，表示 `i -> j` 的关系越强

这是二值化前的连续关系分数矩阵。

### `transition_binary.csv`

文件:

- [transition_binary.csv](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/transition_graph/transition_binary.csv)

含义:

- 对 `transition_scores` 做阈值过滤后的二值矩阵
- 阈值使用:
  - `threshold = mean(transition_scores)^3`
- 若 `T_{i,j} > threshold`，则该位置为 `1`
- 否则为 `0`

这是论文式“关系存在/不存在”的基础矩阵。

### `prerequisite_graph.csv`

文件:

- [prerequisite_graph.csv](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/transition_graph/prerequisite_graph.csv)

含义:

- 从 `transition_binary` 中提取出的单向边
- 若:
  - `T_{i,j} = 1`
  - 且 `T_{j,i} = 0`
- 则保留 `i -> j`

因此它表示“方向性更强”的关系，可解释为先修关系图。

### `similarity_graph.csv`

文件:

- [similarity_graph.csv](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/transition_graph/similarity_graph.csv)

含义:

- 从 `transition_binary` 中提取出的双向边
- 若:
  - `T_{i,j} = 1`
  - 且 `T_{j,i} = 1`
- 则认为 `i` 和 `j` 为相似关系

因此它表示“双向强关联”的知识点关系图。

### `propagation_graph.csv`

文件:

- [propagation_graph.csv](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/transition_graph/propagation_graph.csv)

含义:

- 这是当前训练主线真正使用的图
- 它由:
  - `prerequisite_graph`
  - `similarity_graph`
  合并得到
- 之后补 self-loop
- 最后做行归一化

所以它不是论文原文里的单个中间矩阵，而是为了当前 Step 2 传播实现额外构造出的传播矩阵。

## 当前训练主线使用哪个文件

当前训练主线默认接入的是:

- [propagation_graph.csv](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/transition_graph/propagation_graph.csv)

原因:

- `prerequisite_graph` 只保留单向关系，信息不够完整
- `similarity_graph` 只保留双向关系，也不完整
- `propagation_graph` 兼容了两类关系，更适合当前 Step 2 的统一传播实现

## 统计摘要

摘要文件:

- [summary.json](/home/xph/jwc/research/decoupled_cd/data/assist_09_ordered/transition_graph/summary.json)

当前构图结果:

- 知识点数: `123`
- 先修边数: `1161`
- 相似边数: `2492`

## 备注

当前实现对多知识点题目的处理方式是:

- 若相邻两次正确作答分别对应知识点集合 `A` 和 `B`
- 则对所有 `i in A, j in B, i != j` 都累加一次转移计数

这是一种可执行近似，能够支持当前项目继续推进，但不一定与原论文对多知识点题目的具体实现完全一致。
