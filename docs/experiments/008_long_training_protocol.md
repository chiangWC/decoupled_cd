# Experiment 8: 长训与训练策略

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

- `100 epoch`: `test_auc = 0.666546`
- `200 epoch`: `test_auc = 0.704830`
- `300 epoch`:
  - `best_val_auc = 0.716939`
  - `best_epoch = 243`
  - `test_auc = 0.709411`
  - best checkpoint:
    - `results/assist_09_current_worktree_300ep_best.pt`

结论:

- 当前模型之前远未训满。
- 结构比较默认应看 `300 epoch` 量级，而不是 `20 epoch`。
