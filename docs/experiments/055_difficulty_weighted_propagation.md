# Experiment 55: difficulty-weighted propagation

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 55: difficulty-weighted propagation
  - 分支: `exp/difficulty-weighted-propagation`
  - 提交: `b9704ec`
  - 做法:
    - 在 propagation 的 `correct/incorrect` 两条 `exercise -> concept` 历史证据前，各自增加独立的 zero-init multiplicative scaling
    - scaling 输入读取 `difficulty + concept_count + detached exercise_embedding`
    - `correct_weight = base_correct * multiplier_correct`
    - `incorrect_weight = base_incorrect * multiplier_incorrect`
    - `multiplier = 2 * sigmoid(raw_scale)`，因此零初始化时严格退化回当前主线
  - 结果:
    - smoke:
      - `max_rows=2000`, `epoch=1` 能正常训练并产出 summary
    - `seed=2024`:
      - `AUC 0.764173`
      - `ACC 0.724124`
      - `RMSE 0.430202`
      - `Brier 0.185073`
      - `ECE 0.061324`
  - 相对实验 51 `seed=2024`:
    - `AUC +0.000122`
    - `ACC -0.004548`
    - `RMSE +0.001907`
    - `Brier +0.001636`
    - `ECE +0.010659`
  - 切片:
    - `concept_count=2`: `AUC 0.750349`, `ACC 0.712457`, `ECE 0.070721`
    - `concept_count=3`: `AUC 0.705193`, `ACC 0.673311`, `ECE 0.102832`
    - `concept_count=4+`: `AUC 0.747586`, `ACC 0.669421`, `ECE 0.091158`
    - `none_seen`: `AUC 0.810439`, `ACC 0.769602`, `ECE 0.151766`
  - 结论:
    - 这条线确实证明“把 difficulty 前移到 propagation”会改变排序行为，单 seed `AUC` 有极小正向；但代价过大，`ACC/RMSE/Brier/ECE` 全部明显回撤，切片也没有形成足够干净的多知识点收益
    - 不继续沿这条 difficulty-weighted propagation 扩线
    - 若后续还要 revisit propagation weighting，优先考虑更局部、更学生条件化的证据强度，而不是当前这种按题目全局共享的 difficulty multiplier
