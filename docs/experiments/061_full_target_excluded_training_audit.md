# Experiment 61: full target-excluded training audit

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 61: full target-excluded training audit
  - 分支: `exp/full-target-exclusion-audit` -> `exp/full-target-exclusion-opt`
  - 做法: 训练 loss 路径开启 `--exclude-target-from-train-history`，同时对 propagation history 与 pairwise history 扣除当前 target；`valid/test` 仍复用 `train` history
  - 关键工程: 原始 full target-exclusion 路径已优化为 baseline + 局部 delta 更新，train-only `1 epoch` median runtime 从约 `8.845s` 降到约 `2.002s`，语义保持一致
  - 优化后三 seed 均值:
    - `AUC 0.765495`
    - `ACC 0.729256`
    - `RMSE 0.427883`
    - `Brier 0.183084`
    - `ECE 0.051331`
  - 相对实验 51 三 seed 均值:
    - `AUC +0.001606`
    - `ACC +0.000305`
    - `RMSE -0.000069`
    - `Brier -0.000059`
    - `ECE +0.001932`
  - 结论:
    - 完整 target exclusion 的 `AUC` 正向在三 seed 上稳定复现，说明训练/测试 history mismatch 不是纯方法学噪声
    - 工程障碍已解除，当前保留为 ranking-oriented 训练候选，但因 `ECE` 仍更差，不直接吸收到 `master`
    - 实验 71 已验证它和实验 70 直接组合不是 clean win；若未来明确只追求 `AUC`，可作为 ablation 或候选训练口径
