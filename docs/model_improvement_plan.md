# Model Improvement Plan

这份文档记录当前模型改进工作的状态、已经完成的实验，以及下一轮应优先推进的方向。它不是正式规范，作用是避免后续继续重复已经验证过的改动。

这是一份“实验台账”，不是新会话默认必读文档。只有在设计实验、核对历史结论、避免重复试错时再读。

## 当前状态

当前项目已经具备:

- Step 1-4 的最小可运行闭环
- `train / valid / test` 评估链路
- GPU 全量训练能力
- `assist_09` 上多轮完整对比实验
- 当前已有稳定的 `300 epoch` 长训结果
- 当前已有多 seed 复现实验结果

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

### 9. 多 seed 复现

改动:

- 在当前正式单图基线上加入显式 `seed` 控制
- 跑 `seed in {2024, 2025, 2026}` 的 `300 epoch` 对比

实验结论:

- 单图基线三组 `test_auc` 分别为:
  - `0.709478`
  - `0.707565`
  - `0.706154`
- 均值约 `0.7077`
- 波动很小，说明当前单图基线是稳定的

结论:

- 当前结果不是单次偶然值
- 后续结构比较应默认和这组多 seed 基线比较，而不是只看单次最好结果

### 10. 双图分开传播

改动:

- 保留单图 `propagation_graph`
- 新增可选 `graph_mode = dual`
- 在 `dual` 模式下分别读取:
  - `prerequisite_graph`
  - `similarity_graph`
- 在传播层分别做两路邻接传播，再门控融合

实验结论:

- `300 epoch` 全量对比明显退化
- `best_val_auc = 0.508805`
- `test_auc = 0.501716`
- `best_epoch = 3`

结论:

- 这条实现路线当前不成立
- 双图接口可以保留作实验开关，但不应作为正式主线

### 11. `TKC/UKC` 结构传播参数解耦

改动:

- 在单图 `propagation_graph` 基线上
- 将原先共享的概念结构传播变换拆成:
  - `tkc_concept_to_concept`
  - `ukc_concept_to_concept`
- 即:
  - `TKC` 的结构传播参数独立
  - `UKC` 的结构传播参数独立

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.744711`
  - `test_auc = 0.738442`
- `seed=2025`:
  - `best_val_auc = 0.742114`
  - `test_auc = 0.737970`
- `seed=2026`:
  - `best_val_auc = 0.743833`
  - `test_auc = 0.737942`
- 新结构 `test_auc` 均值约 `0.7381`
- 这组结果建立在“`valid/test` 复用 `train` 行为历史输入”的修复后评估口径上

结论:

- 这是当前最可靠的正向结构改动

### 12. `TKC` 正误双通道行为消息

改动:

- 保留单图 `propagation_graph`
- 保留 `conditional g/s`
- 保留 `TKC/UKC` 结构传播参数独立
- 将 `TKC` 行为消息从“只看正确题”改成:
  - 正确题通道
  - 错误题通道
  - 行为侧 gated fusion
- 同时将按知识点聚合的实现改成更省显存的索引式写法，避免双通道在 full-batch 下 OOM

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.748738`
  - `test_auc = 0.744609`
- `seed=2025`:
  - `best_val_auc = 0.749699`
  - `test_auc = 0.745276`
- `seed=2026`:
  - `best_val_auc = 0.754099`
  - `test_auc = 0.747414`
- 新结构 `test_auc` 均值约 `0.7458`
- 相比上一版评估修复后基线均值 `0.7381`，提升约 `+0.0077`

结论:

- “错题信号被丢掉”是当前实现里的真实问题，不只是理论担忧
- `TKC` 行为消息应显式保留错误作答证据
- 当前候选主线应更新为“单图 + conditional g/s + `TKC/UKC` 独立结构传播参数 + `TKC` 正误双通道”
- 它符合 `TKC/UKC` 分离建模的理论设定

### 13. `q_e` 轻量 residual 融合

改动:

- 保留单图 `propagation_graph`
- 保留 `conditional g/s`
- 保留 `TKC/UKC` 结构传播参数独立
- 保留 `TKC` 正误双通道行为消息
- 在 `q_repr` 里给 `Q` pooling 表示和题目独立 embedding 增加一条轻量 gated residual 融合

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.744008`
  - `test_auc = 0.737416`
- 对比当前主线:
  - `best_val_auc = 0.748738`
  - `test_auc = 0.744609`

结论:

- 这条 `q_e` residual 方案没有带来收益，应视为负结果
- 当前不建议再直接把题目独立 embedding 以残差形式叠回 `q_repr`

### 14. 题目条件局部汇聚: `TKC/UKC` 同时局部 mean

改动:

- 保留单图 `propagation_graph`
- 保留 `conditional g/s`
- 保留 `TKC/UKC` 结构传播参数独立
- 保留 `TKC` 正误双通道行为消息
- 将认知预测从全局 `student_state` 改成题目条件局部汇聚
- 对目标题 `q_e` 相关概念分别做:
  - `TKC` 局部 mean
  - `UKC` 局部 mean
- 再用
  - `alpha * local_tkc + beta * local_ukc`
  进入 `P_cog`
- `guess/slip` 仍保留原来的全局学生向量

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.737040`
  - `test_auc = 0.731763`
  - 文件:
    - `results/assist_09_item_conditioned_seed2024_300ep.json`
- `seed=2025`:
  - `best_val_auc = 0.736479`
  - `test_auc = 0.731650`
  - 文件:
    - `results/assist_09_item_conditioned_seed2025_300ep.json`
- 两个 seed 的 `test_auc` 均值约 `0.7317`

结论:

- “预测前不要直接把概念状态全部 mean 掉”这个方向本身值得研究
- 但最简单的“`TKC/UKC` 同时局部 mean”会明显伤害当前主线效果
- 问题不只是训练没收敛；`300 epoch` 下仍显著低于当前主线

### 15. 题目条件局部汇聚: `TKC` 局部 mean + `UKC` 全局 mean

改动:

- 保留上面的题目条件预测思路
- 只让 `TKC` 对目标题相关概念做局部 mean
- `UKC` 回退为原来的全局支持项 mean

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.735676`
  - `test_auc = 0.731123`
  - 文件:
    - `results/assist_09_tkc_local_ukc_global_seed2024_300ep.json`

结论:

- 把 `UKC` 保持为全局支持项并没有救回这条路线
- 说明问题不只是 `UKC` 不该局部化，更可能是“局部硬 mean 汇聚”本身过强、过早

### 16. 题目条件局部汇聚: `TKC` item-aware attention + `UKC` 全局 mean

改动:

- 继续保留:
  - `TKC` 只在目标题相关概念内做局部选择
  - `UKC` 作为全局支持项
- 将 `TKC` 的局部硬 mean 改成由 `q_repr` 条件化的 item-aware attention
- 即:
  - 用题目表示对相关概念 `TKC` 状态打分
  - 再做加权汇聚

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.743627`
  - `test_auc = 0.739887`
  - 文件:
    - `results/assist_09_tkc_attention_ukc_global_seed2024_300ep.json`
- `seed=2025`:
  - `best_val_auc = 0.741752`
  - `test_auc = 0.736367`
  - 文件:
    - `results/assist_09_tkc_attention_ukc_global_seed2025_300ep.json`
- 两个 seed 的 `test_auc` 均值约 `0.7381`

结论:

- 相比局部硬 mean，这条 attention 版本显著更合理，也明显救回了性能
- 它说明“题目条件下的概念选择性”不是错误方向，问题主要在于汇聚方式过硬
- 但当前这版仍稳定低于正式主线均值 `0.7458`
- 因此它应被视为“稳定次优结构”，可以保留为后续参考，但当前不应替代正式主线

## 当前推荐基线

后续模型改动应默认建立在下面这组设置上:

- 数据: `data/assist_09_ordered/`
- Q 矩阵: `data/assist_09_ordered/Q_matrix.csv`
- 知识图: `data/assist_09_ordered/transition_graph/propagation_graph.csv`
- `learning_rate = 1e-3`
- `concept_dim = 64`
- `gs_mode = conditional`
- `TKC/UKC` 结构传播参数独立
- `TKC` 行为消息正误双通道
- 长训推荐上限: `300 epoch`
- 训练入口: [train.py](/home/xph/jwc/research/decoupled_cd/scripts/train.py)

当前推荐结果口径:

- 多 seed `test_auc` 均值约 `0.7458`
- 单次最好结果:
  - `best_val_auc = 0.754099`
  - `test_auc = 0.747414`
  - 文件:
    - `results/assist_09_tkc_dual_channel_seed2026_300ep_gpu2.json`

## 当前主要瓶颈

在当前调优后基线下，主要问题已经收敛到以下几点:

- `128` 维无法直接用于当前 full-batch GPU 路径
- 当前最好结果已经依赖较长训练，后续需要更正式的训练策略管理
- 当前 `q_e` 与认知主干的进一步改动仍需谨慎，因为已有一次显存受限的负结果
- learning-rate scheduler 已接入，但当前这轮长训中尚未真正触发降学习率
- 双图分开传播当前明显退化，不应再作为短期主方向

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

1. 固定当前“单图 + `conditional g/s` + `TKC/UKC` 独立结构传播参数 + `TKC` 正误双通道”版本为正式候选基线
2. 在这条新主线上补更多 seed 或整理正式汇总
3. 若继续做结构改动，优先保持单图路径不动，只做单变量比较
