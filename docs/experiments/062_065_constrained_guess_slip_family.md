# Experiment 62-65: constrained `guess/slip` 系列

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 62-65: constrained `guess/slip` 系列
  - 分支:
    - `exp/guess-slip-diagnostics`
    - `exp/exp61-guess-slip-constraint`
    - `exp/decoupled-gs-budget`
    - `exp/uncertainty-conditioned-gs-budget`
  - 核心发现:
    - 旧 `guess/slip` 使用独立 `sigmoid`，实际 checkpoint 中大量样本出现 `guess + slip > 1`，会导致 `dp / dcognitive < 0` 的语义反转
    - 三元 softmax、budget/split sigmoid、uncertainty-conditioned mixture 都能把 `ratio(guess_plus_slip > 1)` 压到 `0`
    - 但约束后整体指标没有恢复；越强的 non-cognitive budget 补偿越容易伤害 `none_seen`
  - 代表结果:
    - 实验 62 三元 softmax 相对实验 51 同 seed: `AUC +0.000615`, `ACC -0.001960`, `RMSE +0.000631`, `Brier +0.000541`, `ECE +0.003486`
    - 实验 63 叠到 exp61 后相对 exp61 opt 同 seed: `AUC -0.000656`, `ACC -0.003368`, `RMSE +0.001579`, `Brier +0.001354`, `ECE +0.005701`
    - 实验 64 budget/split 相对实验 51 同 seed: `AUC -0.007257`, `ACC -0.001598`, `RMSE +0.003836`, `Brier +0.003300`, `ECE +0.003622`
    - 实验 65 uncertainty-conditioned mixture 相对实验 51 同 seed: `AUC -0.005558`, `ACC -0.001427`, `RMSE +0.002958`, `Brier +0.002542`, `ECE +0.005545`
  - 结论:
    - 语义诊断成立，诊断工具值得保留；但这些参数化改动不作为主线候选
    - 后续若再访，不能只继续调 scalar budget，应引入更显式的可解释状态、expert routing 或配套训练补偿
