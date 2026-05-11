# Experiment 45: propagation-side concept-conditioned exercise residual

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 45: propagation-side concept-conditioned exercise residual
  - 分支: `exp/concept-conditioned-prop`
  - 做法: 对 `concept_count >= 2` 的题，在 propagation 的 `exercise -> concept` 消息上增加 concept-conditioned zero-init residual；按 Q 非零边现算，避免显式构造稠密 `E x K x D`
  - `seed=2024` 相对当前主线 baseline: `AUC +0.000109`, `ACC -0.000362`, `RMSE +0.000271`, `Brier +0.000233`, `ECE +0.002160`
  - 切片:
    - `concept_count=2`: `AUC +0.001875`, `RMSE -0.000777`, `ECE -0.002424`
    - `concept_count=3`: `AUC +0.003068`, 但 `ACC -0.003322`, `RMSE +0.001202`, `ECE +0.006053`
    - `concept_count=4+`: `ACC +0.024793`, `ECE -0.016021`, 但 `AUC -0.002352`, `RMSE +0.004048`
    - `none_seen`: `AUC +0.003574`, 但 `RMSE +0.001578`, `ECE +0.001645`
  - 结论: 这是“局部切片有信号但整体不成立”的传播侧修补；不扩 seed，不纳入主线
