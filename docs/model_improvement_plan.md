# Model Improvement Plan

这份文档记录当前模型改进工作的状态、已经完成的实验，以及下一轮应优先推进的方向。它不是正式规范，作用是避免后续继续重复已经验证过的改动。

## 当前状态

当前项目已经具备:

- Step 1-4 的最小可运行闭环
- `train / valid / test` 评估链路
- GPU 全量训练能力
- `assist_09` 上多轮完整对比实验
- 当前已有 `200 / 300 epoch` 长训结果

当前问题不再是“工程跑不通”，而是“需要在合理超参数下判断结构改动是否真的有效”。

## 已完成实验

### 1. 原始 Q 共现图基线

特点:

- `N(k)` 由 Q 矩阵裸共现图构造
- `q_e` 使用题目所需知识点 embedding 的简单平均

结果:

- `assist_09` 20 epoch 全量训练可稳定运行
- 指标略高于随机，但提升有限

### 2. 稀疏归一化共现图 + 可学习 `q_e`

改动:

- `N(k)` 改为归一化共现 + top-k 稀疏图
- `q_e` 改为可学习加权汇聚 + MLP

实验结论:

- 训练稳定
- 没有形成稳定泛化收益
- `test_auc` 只有微小变化，`val_auc` 反而更差

结论:

- 这轮改动不能作为当前主基线

### 3. 论文式转移图

改动:

- 基于有序版 ASSIST09 数据
- 使用论文风格的 `C / T` 关系构建方法
- 导出:
  - `prerequisite_graph`
  - `similarity_graph`
  - `propagation_graph`

当前结果:

- 先修边数: `1161`
- 相似边数: `2492`
- 在 `assist_09` 上 20 epoch 全量训练取得当前最好 `test_auc`

结论:

- 这版图比前两版更值得作为后续改造基线

### 4. `TKC` 标量融合 + `student + exercise` 偏置 `g/s`

改动:

- `TKC` 的行为消息和邻接消息加入独立可学习标量权重
- `g / s` 从学生常数改为 `student + exercise` 偏置

实验结论:

- 训练稳定
- 20 epoch 全量对比没有带来提升
- `test_auc` 下降到随机附近，明显不如论文式转移图基线

结论:

- 简单标量融合不够
- 下一轮如果继续改 `TKC`，应改成更细粒度的 gated fusion，而不是只加两个全局标量

### 5. `TKC` gated fusion

改动:

- 保留论文式转移图基线
- 将 `TKC` 分支的行为消息和邻接消息改为 gated fusion
- `g / s` 回退到学生级常数，避免和融合策略混在一起

实验结论:

- 训练稳定
- 20 epoch 全量对比仍未超过论文式转移图基线
- `test_auc` 仍在 `0.503` 左右，`val_auc` 低于基线

结论:

- 当前 `TKC` 路径的主要问题可能不只是融合形式
- 下一轮不应继续只在 `TKC` 融合层面微调

### 6. 超参数扫描

改动:

- 基于当前工作树做 `learning_rate x concept_dim` 扫描
- 扫描范围:
  - `learning_rate in {1e-3, 3e-4, 1e-4}`
  - `concept_dim in {16, 32, 64}`
- `concept_dim = 128` 在当前 full-batch GPU 路径下稳定 OOM

实验结论:

- 最优组合是 `learning_rate = 1e-3`, `concept_dim = 64`
- 当前最好结果:
  - `best_val_auc = 0.537669`
  - `test_auc = 0.527329`
- 文件:
  - `results/hparam_sweeps/assist09_transition_lr_1e-3_dim_64.json`

结论:

- 之前“结构改动无效”的判断，部分受固定超参数影响
- 后续结构比较必须固定在更合理的超参数上进行

### 7. `g / s` 公平对比

改动:

- 在统一超参数下比较:
  - 学生常数 `g / s`
  - 条件化 `g_{u,e} / s_{u,e}`

实验结论:

- `conditional g/s` 明显优于 `constant g/s`
- 当前正式基线保留 `conditional g/s`

结论:

- 这部分结构改动在合理超参数下是成立的

### 8. 长训与训练策略

改动:

- 保持:
  - ordered ASSIST09
  - transition graph
  - `learning_rate = 1e-3`
  - `concept_dim = 64`
  - `conditional g/s`
- 新增:
  - best checkpoint 保存
  - `ReduceLROnPlateau` 调度框架
- 依次完成:
  - `100 epoch`
  - `200 epoch`
  - `300 epoch`

实验结论:

- `100 epoch`:
  - `test_auc = 0.666546`
- `200 epoch`:
  - `test_auc = 0.704830`
- `300 epoch`:
  - `best_val_auc = 0.716939`
  - `best_epoch = 243`
  - `test_auc = 0.709411`
  - best checkpoint:
    - `results/assist_09_current_worktree_300ep_best.pt`

结论:

- 当前模型之前远未训满
- 训练轮数对效果影响极大
- `300 epoch` 相比 `200 epoch` 仍有提升，但边际收益已经变小
- 当前最优正式结果应以 `300 epoch` 版本为准

## 当前推荐基线

后续模型改动应默认建立在下面这组设置上:

- 数据: `data/assist_09_ordered/`
- Q 矩阵: `data/assist_09_ordered/Q_matrix.csv`
- 知识图: `data/assist_09_ordered/transition_graph/propagation_graph.csv`
- `learning_rate = 1e-3`
- `concept_dim = 64`
- `gs_mode = conditional`
- 长训推荐上限: `300 epoch`
- 训练入口: [train.py](/home/xph/jwc/research/decoupled_cd/scripts/train.py)

## 当前主要瓶颈

在当前调优后基线下，主要问题已经收敛到以下几点:

- `128` 维无法直接用于当前 full-batch GPU 路径
- 当前最好结果已经依赖较长训练，后续需要更正式的训练策略管理
- 当前 `q_e` 与认知主干的进一步改动仍需谨慎，因为已有一次负结果
- learning-rate scheduler 已接入，但当前这轮长训中尚未真正触发降学习率

## 下一轮优先级

### 1. 先锁定当前正式最好基线

目标:

- 以当前最好结果作为后续所有比较的参照

建议:

- 固定:
  - `learning_rate = 1e-3`
  - `concept_dim = 64`
  - ordered ASSIST09 + transition graph
- `gs_mode = conditional`
- 采用当前 `300 epoch` 最优 checkpoint 作为正式基线

原因:

- 现在训练充分程度已经被证明是决定性变量

### 2. 再考虑训练策略微调

目标:

- 提升长训效率与后段收敛质量

建议:

- 保持初始 `learning_rate = 1e-3`
- 让 scheduler 更积极一些
  - 例如减小 `lr_scheduler_patience`
- 再尝试更长训练或更稳的 early stopping

原因:

- 当前 300 epoch 仍有提升，但增幅变小

### 3. 然后再回头审视 `q_e`

目标:

- 让题目表示和学生状态的匹配更有区分性

建议:

- 检查当前 `q_e` 的 gated pooling 是否过弱
- 重新考虑是否引入题目独立 embedding 与 Q 表示融合

原因:

- 当前认知概率主干仍然是最值得继续增强的位置

## 当前不建议优先做的事

- 不建议继续单纯增加 epoch
- 不建议回到裸 Q 共现图
- 不建议在未固定超参数前同时大改 `N(k)`、`q_e`、`g/s` 和训练目标

原因:

- 当前已经有可训练、可评估、可比较的基线
- 现在最需要的是可解释的单变量改动

## 下一步执行建议

建议按下面顺序推进:

1. 固定当前 `300 epoch` 最优结果为正式基线
2. 如需继续提升，先调训练策略而不是立刻改结构
3. 若训练策略收益见顶，再重新设计 `q_e`
