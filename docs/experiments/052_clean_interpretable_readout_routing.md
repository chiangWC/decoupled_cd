# Experiment 52: clean interpretable readout routing

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 52: clean interpretable readout routing
  - 分支: `exp/clean-readout-routing`
  - 做法:
    - 以实验 51 的 full-trigger 三专家 residual 为底座
    - gate 输入从 `concept_count / difficulty / dispersion / coverage` 扩成 `concept_count / seen_count / unseen_count / difficulty / dispersion / coverage`
    - 额外测试可选 `top-k` 稀疏路由，希望得到更干净的 selective routing，而不是继续用硬 `min_count` trigger
  - 结果:
    - smoke:
      - `max_rows=2000`, `epoch=1`, `topk=2` 能正常训练并写出 checkpoint / summary
    - `seed=2024`, dense:
      - `AUC 0.761346`, `ACC 0.726655`, `RMSE 0.429885`, `Brier 0.184801`, `ECE 0.053194`
    - `seed=2024`, `topk=2`:
      - `AUC 0.763301`, `ACC 0.727036`, `RMSE 0.428496`, `Brier 0.183609`, `ECE 0.050861`
  - 结论:
    - `topk=2` 虽然比 dense 好，但仍弱于实验 51 原版 full-trigger；相对实验 49 control 也只是保住了小幅 `AUC` 正向，`ACC/RMSE/Brier/ECE` 全部回撤
    - 不继续沿这条 routing 设计扩 seed；实验 51 原版 full-trigger 仍然是这条线应保留的最强基线
    - 若后续还要 revisit selective routing，优先考虑更软的路由约束或训练正则，而不是显式 top-k 稀疏化
