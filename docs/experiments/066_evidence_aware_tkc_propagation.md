# Experiment 66: evidence-aware TKC propagation

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 66: evidence-aware TKC propagation
  - 分支: `exp/evidence-aware-tkc`
  - 做法:
    - 把显式历史统计从 readout residual 前移到 propagation 主干，构造逐 `student-concept` 的 `attempt/correct/incorrect/accuracy/log_attempt/seen` 特征
    - 分别注入 `behavior fusion gate`、`TKC behavior vs graph prior` gate、`student-level TKC/UKC fusion` gate，并补了可拆分的子开关
    - 数据侧新增真实 `student_exercise_count_tensor`，保留重复作答次数
  - 工程验证:
    - 远端单测 `python -m unittest tests.test_hetero_propagation tests.test_history_visibility tests.test_training_modes tests.test_decoupled_cdm` 通过
  - 结果:
    - full 版本 `seed=2024`: `AUC 0.762905`, `ACC 0.729300`, `RMSE 0.427547`, `Brier 0.182796`, `ECE 0.043839`
    - 拆分后最强单 seed 是 `behavior-only`: `AUC 0.765136`, `ACC 0.732193`, `RMSE 0.426751`, `Brier 0.182117`, `ECE 0.046792`
    - 但多 seed 后没有复现 clean win:
      - 直接替换式 `behavior-only` 三 seed 均值: `AUC 0.760512`, `ACC 0.728298`, `RMSE 0.429400`, `Brier 0.184390`, `ECE 0.050343`
      - residual 化 `behavior-only` 三 seed 均值: `AUC 0.761936`, `ACC 0.729047`, `RMSE 0.429112`, `Brier 0.184139`, `ECE 0.053518`
    - 相对实验 51 当前主线三 seed 均值，两版都没有形成 overall 正向；residual 版虽然更稳，但 `ECE` 还更差
  - 结论:
    - 这条线在机制上成立，且可确认有效信号主要来自 `correct/incorrect behavior fusion gate`
    - `reliability` 和 `student fusion` 不是主增益源，单独或组合开启都没有形成稳定提升
    - 当前不进入主线候选；若后续再访，应只做更局部的 behavior gate 调节，例如只改 bias / temperature，或只作用于 `concept_count>=2` / low-evidence concept
