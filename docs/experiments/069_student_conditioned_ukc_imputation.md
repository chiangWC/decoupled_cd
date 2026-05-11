# Experiment 69: student-conditioned UKC imputation

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 69: student-conditioned UKC imputation
  - 分支: `exp/student-conditioned-ukc-imputation`
  - 做法: 对未测 concept 用图邻接已测 TKC states 聚合 student-specific prior，并与静态 UKC graph prior 融合，直接改 UKC 状态形成
  - 结果:
    - `seed=2024`: `AUC 0.762949`, `ACC 0.726693`, `RMSE 0.429449`, `Brier 0.184427`, `ECE 0.054935`
    - 相对实验 51 同 seed: `AUC -0.001102`, `ACC -0.001979`, `RMSE +0.001154`, `Brier +0.000990`, `ECE +0.004270`
    - `none_seen` 排序略变好但校准明显变坏: `AUC +0.002737`, `ACC -0.041616`, `RMSE +0.026330`, `ECE +0.064790`
  - 结论:
    - 直接替换 UKC 主状态会放大 `none_seen` under-confidence，不扩 seed
    - 这个负结果导向实验 70 的设计: 不替换主状态，只做 target-local readout sidecar
