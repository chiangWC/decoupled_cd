# Experiment 11: `TKC/UKC` 结构传播参数解耦

改动:

- 在单图 `propagation_graph` 基线上，将原先共享的概念结构传播变换拆成:
  - `tkc_concept_to_concept`
  - `ukc_concept_to_concept`

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
- `test_auc` 均值约 `0.7381`。
- 这组结果建立在“`valid/test` 复用 `train` 行为历史输入”的修复后评估口径上。

结论:

- 这是当前最可靠的正向结构改动之一。
