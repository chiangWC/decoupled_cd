# Experiment 38-40: CF residual family

> Compressed note for the paused collaborative-filtering residual branch family. The available archive only preserves the decision-level summary.

- 实验 38 `exp/cf-residual`: ranking-oriented 候选，但依赖学生内随机 split 的 ID-aware side channel，不作为纯 CDM 主线。
- 实验 39 `exp/cf-residual-recompute`: 不是实验 37 与 38 的无损叠加，不建议主线化。
- 实验 40 `exp/cf-residual-dim-sweep`: 大容量收益主要来自 transductive ID side channel，整条线暂停。

结论: CF residual family 保留为暂停支线；除非明确接受 transductive ID side channel，否则不进入主线吸收链。
