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
  - 诊断细节:
    - 实验 51 同口径旧 checkpoint 已真实触发语义反转，不只是理论风险
    - `seed=2025` test: `guess_plus_slip_mean=1.945274`, `p95=1.999303`, `ratio(>1)=0.999638`
    - `seed=2026` test: `guess_plus_slip_mean=1.872332`, `p95=1.992325`, `ratio(>1)=0.999486`
    - 三元 softmax 约束后，实验 62 正式 `seed=2024` test: `guess_plus_slip_mean=0.101998`, `max=0.998407`, `p95=0.554550`, `ratio(>1)=0.0`
    - budget/split 约束后，实验 64 正式 `seed=2024` test: `guess_plus_slip_mean=0.139092`, `p95=0.946891`, `ratio(>1)=0.0`
    - uncertainty-conditioned mixture 后，实验 65 正式 `seed=2024` test: `guess_plus_slip_mean=0.295582`, `p95=0.905642`, `ratio(>1)=0.0`
  - 代表结果:
    - 实验 62 三元 softmax 相对实验 51 同 seed: `AUC +0.000615`, `ACC -0.001960`, `RMSE +0.000631`, `Brier +0.000541`, `ECE +0.003486`
    - 实验 63 叠到 exp61 后相对 exp61 opt 同 seed: `AUC -0.000656`, `ACC -0.003368`, `RMSE +0.001579`, `Brier +0.001354`, `ECE +0.005701`
    - 实验 64 budget/split 相对实验 51 同 seed: `AUC -0.007257`, `ACC -0.001598`, `RMSE +0.003836`, `Brier +0.003300`, `ECE +0.003622`
    - 实验 65 uncertainty-conditioned mixture 相对实验 51 同 seed: `AUC -0.005558`, `ACC -0.001427`, `RMSE +0.002958`, `Brier +0.002542`, `ECE +0.005545`
  - 机制判断:
    - 实验 62 的三元 softmax 把非认知预算压得太低，说明旧性能部分依赖无约束 `guess/slip` 的额外自由度
    - 实验 64 解耦总量和 split 后，预算只从 `0.1020` 回升到 `0.1391`，没有恢复实验 51 的量级，且 AUC 明显下滑
    - 实验 65 用显式 `budget + fallback` 和 uncertainty-conditioned residual 把预算补到 `0.2956`，但 `none_seen` 明显变坏: 实验 51 `none_seen ECE 0.109546`，实验 65 `none_seen ECE 0.147647`
    - 因此失败不是“某个约束参数化太硬”这一层问题，而是 constrained non-cognitive 分支缺少能替代旧自由度的可解释状态或训练补偿
  - 结论:
    - 语义诊断成立，诊断工具值得保留；但这些参数化改动不作为主线候选
    - 后续若再访，不能只继续调 scalar budget，应引入更显式的可解释状态、expert routing 或配套训练补偿
