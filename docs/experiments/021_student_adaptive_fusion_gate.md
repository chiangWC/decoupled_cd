# Experiment 21: `TKC/UKC` 学生自适应融合 gate

改动:

- 保留:
  - 单图 `propagation_graph`
  - `conditional g/s`
  - `TKC/UKC` 结构传播参数独立
  - `TKC` 正误双通道行为消息
- 将原先全局固定的:
  - `alpha * tkc_mean + beta * ukc_mean`
  替换为学生级自适应融合。
- 融合 gate 输入为:
  - `coverage`
  - `tkc_mean`
  - `ukc_mean`
- 最终形式为:
  - `w_u * tkc_mean + (1 - w_u) * ukc_mean`

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.754493`
  - `best_epoch = 278`
  - `test_auc = 0.749272`
  - 文件:
    - `results/exp_adaptive_tkc_ukc_gate/assist_09_tkc_dual_channel_seed2024_300ep.json`
- `seed=2025`:
  - `best_val_auc = 0.757129`
  - `best_epoch = 300`
  - `test_auc = 0.751710`
  - 文件:
    - `results/exp_adaptive_tkc_ukc_gate/assist_09_tkc_dual_channel_seed2025_300ep.json`
- `seed=2026`:
  - `best_val_auc = 0.756685`
  - `best_epoch = 288`
  - `test_auc = 0.749739`
  - 文件:
    - `results/exp_adaptive_tkc_ukc_gate/assist_09_tkc_dual_channel_seed2026_300ep.json`
- 三个 seed 的 `test_auc` 均值约 `0.7502`。
- 相比实验 12 的旧主线均值 `0.7458`，提升约 `+0.0044`。

结论:

- 全局固定 `alpha/beta` 的融合方式过于粗糙，学生级自适应 gate 能更好利用覆盖率差异。
- 这次提升不是单 seed 偶然值，而是三 seed 一致提升。
- 这条线已经足够取代实验 12，成为当前正式主线。
