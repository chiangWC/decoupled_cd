# Experiment 51: interpretable readout expert residual

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 51: interpretable readout expert residual
  - 分支: `exp/interpretable-readout-experts`
  - 做法:
    - 在 `cognitive_logits` 外增加 zero-init readout expert residual
    - gate 只读取 `concept_count / difficulty / dispersion / coverage` 这四类可解释量
    - expert 侧读取 detached readout features，不读学生/题目 ID，不改 propagation 主干语义
  - 最强配置:
    - `--interpretable-readout-expert-adapter`
    - `--interpretable-readout-expert-count 3`
  - `seed=2024` 相对实验 49 control:
    - `AUC +0.001548`
    - `ACC -0.000818`
    - `RMSE +0.000000`
    - `Brier +0.000001`
    - `ECE +0.000973`
  - targeted 变体:
    - `min_count>=2` 的三专家版本能让 `concept_count=2/3` 切片转正，但 overall `AUC -0.000151`，不如 full-trigger
    - `min_count>=3` 的两专家/三专家版本都未超过 full-trigger 单次结果
  - full-trigger 三 seed 结果:
    - `seed=2024`: `AUC 0.764051`, `ACC 0.728672`, `RMSE 0.428295`, `Brier 0.183437`, `ECE 0.050665`
    - `seed=2025`: `AUC 0.764711`, `ACC 0.728387`, `RMSE 0.428095`, `Brier 0.183265`, `ECE 0.051829`
    - `seed=2026`: `AUC 0.762904`, `ACC 0.729795`, `RMSE 0.427466`, `Brier 0.182727`, `ECE 0.045703`
  - full-trigger 三 seed 均值相对实验 49:
    - `AUC +0.001520`
    - `ACC -0.000812`
    - `RMSE -0.000253`
    - `Brier -0.000217`
    - `ECE -0.000429`
  - 结论:
    - 这条线已经从单 seed 信号变成稳定的结构候选，当前是最强的非 ID-aware follow-up
    - 它的代价是小幅 `ACC` 回撤，但 `AUC` 增益已经达到继续保留的门槛
    - 第一版 full-trigger 收益主要来自 `concept_count=1 / all_seen`，没有自然学成“只服务高 concept-count”的干净专家分工
    - 如果后续继续做 selective routing，应建立在这条 full-trigger 正向底座上，而不是直接退回更硬的 `3+` trigger
    - 这一步现已吸收到当前 `master`
