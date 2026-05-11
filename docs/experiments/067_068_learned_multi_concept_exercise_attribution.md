# Experiment 67-68: learned multi-concept exercise attribution

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 67-68: learned multi-concept exercise attribution
  - 分支: `exp/learned-exercise-attribution`
  - 做法: 在 propagation 的 correct/incorrect exercise message 聚合处，用 `student-conditioned / response-conditioned / history-conditioned` scorer 为多知识点题动态归因；后续 rescue 测了 scale-preserving、`concept_count=4+`、incorrect-only 版本
  - 代表结果:
    - 原版相对实验 51 同 seed: `AUC -0.005409`, `ACC -0.005043`, `RMSE +0.002845`, `Brier +0.002444`, `ECE +0.001614`
    - scale-preserving rescue 相对实验 51 同 seed: `AUC -0.003315`, `ACC -0.000837`, `RMSE +0.000937`, `Brier +0.000803`, `ECE -0.002072`
    - `min4` 与 incorrect-only 也未恢复主线；`4+` 只有小幅 mixed signal，样本数 `363`，不足以抵消 overall 回撤
  - 结论:
    - 动态归因机制区别于实验 24 的静态分摊，但直接替换 propagation 主聚合会削弱行为证据，并显著伤害 `none_seen` 校准
    - 不继续沿 attribution 主聚合替换路线 rescue；若以后必须复访，只应作为 additive residual / calibration sidecar，而不是替换主聚合
