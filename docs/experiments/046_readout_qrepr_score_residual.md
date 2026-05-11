# Experiment 46: readout-side qrepr score residual

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 46: readout-side qrepr score residual
  - 分支: `exp/qrepr-score-residual`
  - 做法: 保留 static `q_pool_gate`，仅对 `concept_count >= 2` 的题增加 zero-init exercise-conditioned Q-pooling score residual；按 batch chunk 分块计算 additive score，避免显式物化完整 `B x K x D`
  - `seed=2024` 相对当前主线 baseline: `AUC +0.000664`, `ACC -0.001370`, `RMSE +0.000185`, `Brier +0.000159`, `ECE +0.001519`
  - 切片:
    - `concept_count=2`: `AUC +0.002694`, `ECE -0.003136`，但 `ACC -0.006002`
    - `concept_count=3`: `AUC +0.007144`, `RMSE -0.002483`, `ECE -0.002975`，但 `ACC -0.016611`
    - `concept_count=4+`: `ACC +0.024793`, `ECE -0.008125`，但 `AUC -0.012312`, `RMSE +0.006886`
    - `none_seen`: `AUC +0.001804`, 但 `RMSE +0.008018`, `ECE +0.017540`
    - `partial_seen`: `AUC -0.001127`, `RMSE +0.010477`, `ECE +0.009486`
  - 结论: 相比实验 45，这条 readout 侧窄变体更接近目标瓶颈，但仍然是“局部排序改善换整体与校准副作用”的折中；不扩 seed，不纳入主线
