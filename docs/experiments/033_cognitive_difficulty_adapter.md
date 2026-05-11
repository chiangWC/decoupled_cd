# Experiment 33: `cognitive_match` 的 zero-init difficulty adapter

改动:

- 以当前 `master` 为底座:
  - 单图 `propagation_graph`
  - `conditional g/s`
  - `TKC/UKC` 结构传播参数独立
  - `TKC` 正误双通道行为消息
  - 学生级自适应 `TKC/UKC` 融合 gate
  - 报告包含 `Brier/ECE/分桶校准`
- 保留主线认知 logits:
  - `self.cognitive_match_mlp(match_inputs).squeeze(-1) - difficulty`
- 在所有现有模块之后新增 sidecar adapter:
  - `cognitive_difficulty_adapter = MLP(concept_dim * 4 + 1 -> concept_dim -> 1)`
  - 输入为 `cat([match_inputs.detach(), difficulty.detach()])`
  - 输出层权重和 bias 均初始化为 0
- 最终认知 logits:
  - `base_cognitive_logits - difficulty + cognitive_difficulty_adapter(adapter_inputs)`
- 目的: 复访实验 29 中“认知匹配读到难度条件”的正向信号，同时避免直接扩展 `cognitive_match_mlp` 输入维度导致的初始化扰动和 `seed=2026` 稳定崩盘。

实验结果:

- 分支:
  - `exp/cog-difficulty-adapter`
- `seed=2024`:
  - `best_val_auc = 0.763090`
  - `best_epoch = 178`
  - `test_auc = 0.760849`
  - `test_acc = 0.721783`
  - `test_rmse = 0.431065`
  - `test_brier = 0.185817`
  - `test_ece = 0.057483`
  - 文件:
    - `results/exp_cog_difficulty_adapter/assist_09_cog_difficulty_adapter_seed2024_300ep.json`
- `seed=2025`:
  - `best_val_auc = 0.764293`
  - `best_epoch = 186`
  - `test_auc = 0.759360`
  - `test_acc = 0.725323`
  - `test_rmse = 0.430594`
  - `test_brier = 0.185411`
  - `test_ece = 0.054023`
  - 文件:
    - `results/exp_cog_difficulty_adapter/assist_09_cog_difficulty_adapter_seed2025_300ep.json`
- `seed=2026`:
  - `best_val_auc = 0.765794`
  - `best_epoch = 172`
  - `test_auc = 0.760797`
  - `test_acc = 0.727816`
  - `test_rmse = 0.429176`
  - `test_brier = 0.184192`
  - `test_ece = 0.050316`
  - 文件:
    - `results/exp_cog_difficulty_adapter/assist_09_cog_difficulty_adapter_seed2026_300ep.json`
- 三 seed 均值:
  - `test_auc = 0.760335`
  - `test_acc = 0.724974`
  - `test_rmse = 0.430278`
  - `test_brier = 0.185140`
  - `test_ece = 0.053941`
- 当前 `master` 三 seed 均值:
  - `test_auc = 0.759690`
  - `test_acc = 0.724784`
  - `test_rmse = 0.431388`
  - `test_brier = 0.186096`
  - `test_ece = 0.062276`
- 相对当前 `master` 的三 seed 均值差:
  - `test_auc = +0.000645`
  - `test_acc = +0.000190`
  - `test_rmse = -0.001109`
  - `test_brier = -0.000955`
  - `test_ece = -0.008335`

结论:

- 该 adapter 避开了实验 29 的 `seed=2026` 崩盘；`seed=2026` 反而是三组里提升最大的 seed。
- 三 seed 均值在 `AUC/ACC/RMSE/Brier/ECE` 上全部优于当前 `master`，其中 `ECE` 改善最明显。
- `seed=2024/2025` 的 ACC 略低于对应主线，但三 seed 均值仍略高；且 RMSE/Brier/ECE 三个概率质量指标三 seed 均改善。
- 当前建议合入 `master`，作为新的认知匹配主线。
