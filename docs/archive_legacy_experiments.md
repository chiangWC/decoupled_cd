# Legacy And Downgraded Experiment Archive

This file holds compressed legacy and downgraded experiment notes moved out of `docs/model_improvement_plan.md`.

Open it only when revisiting an older failed route or checking why a legacy idea should not be retried. For normal orientation, use `docs/model_improvement_plan.md`; for known experiment IDs or status filtering, use `docs/experiment_index.jsonl`.

## 已验证无效或已降级

这些路线默认不要回到主线，除非用户明确要求复访。

### 明显无效

- 实验 1: 原始 Q 共现图基线
  - 只证明最小闭环能跑，不适合作为长期基线
- 实验 2: 稀疏归一化共现图 + 可学习 `q_e`
  - 没形成稳定泛化收益
- 实验 4: `TKC` 标量融合 + `student + exercise` 偏置 `g/s`
  - 早期全量结果退到随机附近
- 实验 5: 早期 `TKC` gated fusion
  - 单改融合形式没有解决主要问题
- 实验 10: `dual graph`
  - 在 ASSIST09 上明显退化，保留为 legacy ablation
- 实验 13: `q_e` residual 融合
  - 弱于当时主线
- 实验 14: `TKC/UKC` 同时局部硬 mean
  - 明显退化
- 实验 15: `TKC` 局部 mean + `UKC` 全局 mean
  - 没能救回局部硬汇聚路线
- 实验 20: 直接叠加 `local UKC neighbor` 和 `UKC-only coverage gate`
  - 没有形成 `1 + 1 > 1`
- 实验 22: 在实验 21 主线底座上复验 `local UKC neighbor`
  - 低于该底座
- 实验 25: readout 侧 `TKC` item-aware residual
  - 明显低于正式主线
- 实验 27: `q_repr` 内部 item-aware Q pooling
  - 单次弱于当前主线
  - 复访 `exp/qrepr-item-aware-residual` 后，`seed=2024` 相对实验 34 为 `AUC -0.000404`, `ACC +0.001180`, `RMSE +0.000123`, `Brier +0.000106`, `ECE +0.001479`
- 实验 29: 直接扩展 `cognitive_match` 输入以读入 `difficulty`
  - `seed=2026` 稳定崩盘，zero-init 也没救回
- 实验 30: 行为正误融合 gate 加 concept-specific bias
  - AUC 持平但误差和校准变差
- 实验 31: `UKC` no-param graph propagation
  - AUC 仅微升，但 `ACC/RMSE/Brier/ECE` 全变差
- 实验 32: `UKC` directed symmetric-like normalization
  - 单 seed 不满足继续扩 seed 门槛

### 近线失败路线

- 实验 35: local readout adapter
  - 读取当前题 Q mask 对应的 `TKC+UKC` 局部概念状态均值
  - 单 seed 相对实验 34 在 `AUC/ACC/RMSE/Brier/ECE` 上整体更差
- 实验 36: student base ability
  - 增加 zero-init 学生全局能力 embedding `theta_u`
  - 呈现“局部有信号，但整体排序和误差不占优”的折中
- 实验 41: item discrimination / 2PL logit scale
  - `seed=2024` 相对实验 34: `AUC -0.008723`, `ACC -0.003635`, `RMSE +0.001466`, `Brier +0.001261`, `ECE -0.019051`
  - 明显是校准折中，不是排序收益
- 实验 42: Soft-Q / learnable Q residual
  - 多组 `scale/stay/sparse/offset/threshold` 都未在 `AUC/ACC/RMSE/Brier/ECE` 上整体优于实验 34
  - 更激进的缺失边冷启动还会明显加噪
- 实验 43: hard-Q constrained concept residual
  - `seed=2024` 相对实验 34: `AUC -0.000155`, `ACC +0.000533`, `RMSE -0.000112`, `Brier -0.000096`, `ECE -0.001931`
  - `concept_count=4+` 与 `none_seen` 的 ECE 仍变差
- 实验 44: local exercise student adapter
  - `min_count=2`: `AUC -0.001609`, `ACC +0.000533`, `RMSE +0.000377`, `Brier +0.000324`, `ECE +0.002203`
  - `min_count=3`: `AUC -0.000509`, `ACC +0.000228`, `RMSE +0.000530`, `Brier +0.000456`, `ECE +0.001742`
  - 虽改善 `3/4+` 多知识点切片，但不能转化为 overall `AUC` 正收益

近期实验 24/26/45-74 的索引留在 [model_improvement_plan.md](./model_improvement_plan.md) 和 [experiment_index.jsonl](./experiment_index.jsonl)，不在本 legacy 归档重复维护。
