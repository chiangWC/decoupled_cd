# Experiment 57: single-graph multi-hop propagation revisit

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 57: single-graph multi-hop propagation revisit
  - 分支: `exp/multi-hop-propagation`
  - 做法:
    - 保留 single-graph 主线，不回到 dual graph
    - 依次复访全局 `2/3-hop` residual、coverage-conditioned residual、`UKC-only` residual，以及 `2-hop only` 简化版
    - 所有变体都保持“零初始化时退化回实验 51 主线”这一约束
  - 结果:
    - 最好的 overall 只达到 `seed=2024: AUC 0.764542`
    - 但对应 `ACC 0.725475 / RMSE 0.428501 / Brier 0.183613 / ECE 0.053370`
    - 更轻的 `2-hop only` 版本也只是 `AUC 0.764388 / ACC 0.728387 / ECE 0.053974`
  - 结论:
    - 这条线反复呈现“很小的 AUC 正向，换来 ACC 或校准回撤”的模式；`none_seen` 与 `4+` 多知识点题也没有形成足够干净的收益
    - 不继续沿 propagation 主干做 multi-hop mixing 扩线
    - 若以后再访，应只在更明确的局部 slice 假设下做 targeted readout / mixture，而不是继续修改 propagation 主干

### 语义更干净，但不值得主线吸收
