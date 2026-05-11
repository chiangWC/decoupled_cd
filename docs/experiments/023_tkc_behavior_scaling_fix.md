# Experiment 23: 修正 `TKC` 行为项的全局二次缩小

改动:

- 以实验 21 的当前正式主线为底座:
  - 单图 `propagation_graph`
  - `conditional g/s`
  - `TKC/UKC` 结构传播参数独立
  - `TKC` 正误双通道行为消息
  - 学生级自适应 `TKC/UKC` 融合 gate
- 检查 `_build_exercise_component` 后确认当前实现会先按学生全历史对 `weighted_exercises` 做一次全局归一化，再在概念维度上再除一次 `concept_weights`。
- 这会让学生历史越长，`TKC` 行为证据越容易被系统性压小，与 Step 2 中“`TKC` 保留行为信号”的语义不一致。
- 将该聚合改为概念内加权平均:
  - 不再先按学生全历史做全局归一化
  - 直接在每个概念内按该概念实际命中的行为权重做平均
- 同时补了最小回归测试，防止“无关历史变长会压小当前概念行为项”的问题回归。

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.763858`
  - `best_epoch = 175`
  - `test_auc = 0.760568`
  - 文件:
    - `results/exp_tkc_exercise_aggregation/assist_09_tkc_dual_channel_seed2024_300ep.json`
- `seed=2025`:
  - `best_val_auc = 0.764494`
  - `best_epoch = 180`
  - `test_auc = 0.759346`
  - 文件:
    - `results/exp_tkc_exercise_aggregation/assist_09_tkc_dual_channel_seed2025_300ep.json`
- `seed=2026`:
  - `best_val_auc = 0.764509`
  - `best_epoch = 185`
  - `test_auc = 0.759156`
  - 文件:
    - `results/exp_tkc_exercise_aggregation/assist_09_tkc_dual_channel_seed2026_300ep.json`
- 三个 seed 的 `test_auc` 均值约 `0.7597`。
- 相比实验 21 当前正式主线均值 `0.7502`，提升约 `+0.0095`。

结论:

- 这不是“实现风格差异”，而是当前 `TKC` 行为聚合里的真实缩放偏差。
- 修掉这一步后，三 seed 提升幅度明显且一致，是目前最强的待整理候选。
- 如果后续要把近期正向结果整理回 `master`，这条改动应优先于其它 follow-up 被吸收。
