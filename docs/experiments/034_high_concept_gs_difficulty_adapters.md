# Experiment 34: high-concept logit adapter + `guess/slip` difficulty adapter

- 对 `concept_count >= 2` 的题增加 zero-init high-concept logit residual。
- 在 conditional `guess/slip` 分支增加 zero-init difficulty residual。
- 相对实验 23 三 seed 均值:
  - `AUC +0.001506`
  - `ACC +0.002772`
  - `RMSE -0.002218`
  - `Brier -0.001909`
  - `ECE -0.011134`
- 结论: 实验 34 是当前 `master` 正式主线。
