# Experiment 47: final-logit none-seen calibration bias

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 47: final-logit none-seen calibration bias
  - 分支: `exp/none-seen-calibration-bias`
  - 做法: 在最终概率输出前增加 zero-init calibration residual；输入只看 target concept coverage、`concept_count` 和 `difficulty`，不改 TKC/UKC propagation 语义
  - `seed=2024` 相对当前主线 baseline: `AUC -0.000945`, `ACC -0.000381`, `RMSE +0.000438`, `Brier +0.000376`, `ECE -0.001393`
  - 切片:
    - `none_seen`: `AUC +0.000401`, 但 `ACC -0.005428`, `RMSE +0.003811`, `Brier +0.002895`, `ECE +0.011709`
    - `partial_seen`: `AUC +0.004038`, `ACC +0.003030`, 但 `RMSE +0.000513`, `Brier +0.000418`, `ECE +0.015319`
    - `all_seen`: `AUC -0.000886`, `ACC -0.000237`, `RMSE +0.000340`, `Brier +0.000293`, `ECE -0.001199`
    - `concept_count=1`: `AUC -0.001279`, `ACC -0.000754`, `RMSE +0.000597`, `Brier +0.000510`, `ECE -0.001459`
    - `concept_count=2`: `AUC +0.001413`, `ACC +0.002801`, `RMSE -0.000603`, `Brier -0.000529`, `ECE -0.000978`
    - `concept_count=3`: `AUC -0.002885`, `ACC -0.008859`, `RMSE +0.000545`, `Brier +0.000520`, `ECE -0.000689`
    - `concept_count=4+`: `AUC -0.005805`, `ACC +0.000000`, `RMSE +0.002798`, `Brier +0.002615`, `ECE +0.001669`
  - 结论: 这类“全局共享 final-logit calibration bias”太钝。虽然 overall `ECE` 略降，但没有解决 `none_seen`，反而把目标切片的 `ACC/RMSE/Brier/ECE` 一起做坏；不扩 seed，不纳入主线
