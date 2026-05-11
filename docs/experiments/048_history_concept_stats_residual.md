# Experiment 48: history concept stats residual

> Migrated from `docs/model_improvement_plan.md` in `exp/experiment-doc-index-pilot`. This preserves the currently available compressed detail; it may not include older uncompressed notes from git history.

- 实验 48: history concept stats residual
  - 分支: `exp/local-concept-evidence-head`
  - 做法:
    - 先做离线诊断，验证“逐概念历史正确率 / 最弱概念”在原始数据上确有强信号，但当前 `TKC/UKC` 局部状态本身几乎不含可直接 readout 的同类信号
    - 后续不再从局部 embedding 读证据，改为显式构造学生-概念历史统计:
      - 为 data bundle 增加学生-题目历史作答次数矩阵
      - 对 target 题相关概念聚合 `mean_acc_seen / min_acc_seen / gap / seen_ratio`
      - 以 zero-init residual 形式接到 `cognitive_logits`
  - 三 seed 相对当前主线 baseline:
    - `seed=2024`: `AUC -0.000600`, `ACC +0.003673`, `RMSE -0.000824`, `Brier -0.000707`, `ECE -0.000457`
    - `seed=2025`: `AUC +0.000748`, `ACC +0.000038`, `RMSE +0.000376`, `Brier +0.000323`, `ECE +0.004516`
    - `seed=2026`: `AUC -0.001245`, `ACC +0.000457`, `RMSE +0.000030`, `Brier +0.000025`, `ECE -0.002147`
  - 三 seed 均值差:
    - `AUC -0.000366`
    - `ACC +0.001389`
    - `RMSE -0.000139`
    - `Brier -0.000120`
    - `ECE +0.000637`
  - 切片:
    - `seed=2024` 的 `concept_count=2/3/4+` 明显改善，尤其 `4+`: `AUC +0.026060`, `ACC +0.033058`, `RMSE -0.011397`, `ECE -0.015181`
    - 但 `seed=2025` 的切片不稳定: `concept_count=3` 仍强正向，`concept_count=2/4+` 的 `RMSE/Brier/ECE` 反而转差，`none_seen` 继续变坏
    - `none_seen` 在已看的 seed 上没有形成 clean win，仍不是这条结构的受益点
  - 结论:
    - 这条“显式历史概念统计 residual”证明了多知识点题的确能从更直接的历史概念统计里获益，但收益主要体现在局部 slice，不足以稳定转化为更优 overall
    - 相比当前主线，它更像 `AUC` 与 `ACC/RMSE/Brier` 之间的 seed-sensitive 折中，不作为主线结构推进
    - 如果后续再回到这条思路，优先考虑把它作为 targeted auxiliary / mixture trigger，而不是对所有 `concept_count>=2` 题统一加 residual
