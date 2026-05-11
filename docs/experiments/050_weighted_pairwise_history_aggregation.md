# Experiment 50: weighted pairwise history aggregation

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 50: weighted pairwise history aggregation
  - 分支: `exp/pairwise-history-weighted-agg`
  - 做法: 在实验 49 上把 pair score 聚合从固定均值改成 learned weighting，其余结构不变
  - `seed=2024` 相对实验 49:
    - `AUC -0.000441`
    - `ACC -0.000305`
    - `RMSE -0.000120`
    - `Brier -0.000103`
    - `ECE +0.000002`
  - 结论:
    - learned weighting 没有提供额外收益
    - 这说明当前增益主要来自“history carrier + pairwise scorer”本身，而不是更复杂的 pair aggregator
    - 主线保留简单均值聚合，不继续扩 seed
