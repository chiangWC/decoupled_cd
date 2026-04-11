# Model Improvement Archive

这份文档只保留“实验台账”用途，用来回答三件事:

- 当前 `master` 的正式主线是什么
- 哪些路线已经证明有效或无效
- 下一步默认该优先试什么

它不是新会话默认入口。新会话先读 [docs/session_bootstrap.md](./session_bootstrap.md)；只有在需要查历史实验、避免重复试错时再回来看这份档案。

## 如何使用

- 先看“当前快照”，确认主线、结果口径和近线候选。
- 需要判断某条路线是否还值得继续时，看“已验证有效”和“已验证无效/降级”。
- 需要设计下一轮实验时，看“当前诊断与下一步”。
- 需要精确文件路径或完整上下文时，再去看对应结果目录或 `git log`，不要把这份文档当成长篇实验报告。

## 当前快照

- 当前 `master` 正式主线口径是实验 34:
  - ordered ASSIST09
  - `transition_graph/propagation_graph.csv`
  - `graph_mode = single`
  - `learning_rate = 1e-3`
  - `concept_dim = 64`
  - `gs_mode = conditional`
  - `TKC/UKC` 结构传播参数独立
  - `TKC` 行为消息使用正误双通道
  - `TKC/UKC` 学生级融合使用自适应 gate
  - 已修正 `TKC` 行为项的全局二次缩小
  - 吸收实验 33 的 zero-init cognitive difficulty adapter
  - `high_concept_logit_adapter = true`
  - `high_concept_logit_min_count = 2`
  - `gs_difficulty_adapter = true`
- 当前主线结果目录:
  - `results/exp_high_concept_logit_adapter/`
- 当前主线三 seed 参考均值:
  - `test_auc = 0.761196`
  - `test_acc = 0.727556`
  - `test_rmse = 0.429170`
  - `test_brier = 0.184187`
  - `test_ece = 0.051142`
- 当前结果报告默认同时看:
  - `AUC/ACC/RMSE`
  - `Brier/ECE/分桶校准`

主线形成的最后两步:

- 实验 33:
  - 把实验 29 中不稳定的“认知分支读难度”改成 zero-init sidecar adapter
- 实验 34:
  - 对多知识点题增加 high-concept logit residual
  - 对 conditional `guess/slip` 增加 difficulty residual
  - 相对实验 23 基座三 seed 均值:
    - `AUC +0.001506`
    - `ACC +0.002772`
    - `RMSE -0.002218`
    - `Brier -0.001909`
    - `ECE -0.011134`

## 已验证有效

下面只保留真正改变主线判断的实验。

### 实验 3. 论文式 transition graph

- 从早期 Q 共现图切到论文式有向转移图后，效果第一次明显优于随机附近基线。
- 结论: transition graph 是后续所有正式对比的图结构起点。

### 实验 6. 超参数扫描

- 在合理训练口径下固定了 `learning_rate = 1e-3`、`concept_dim = 64`。
- 结论: 后续结构比较默认锁定这组超参数。

### 实验 7. `conditional g/s`

- 条件化 `guess/slip` 明显优于学生常数 `g/s`。
- 结论: `conditional g/s` 为主线固定配置。

### 实验 8. 长训与训练策略

- 证明 `20 epoch` 远远不够，结构比较至少应看 `300 epoch`。
- 结论: 长训是正式比较的默认协议。

### 实验 9. 多 seed 复现

- 早期单图长训基线在 `seed in {2024, 2025, 2026}` 上波动很小。
- 结论: 主线比较不能只看单次最好值，要看多 seed 稳定性。

### 实验 11. `TKC/UKC` 结构传播参数解耦

- 将 `TKC/UKC` 的概念传播从共享参数改为独立参数后，形成稳定增益。
- 结论: 这是当前主线最可靠的正向结构改动之一。

### 实验 12. `TKC` 正误双通道行为消息

- 显式保留错题证据后，三 seed 均值明显优于只看正确题的版本。
- 结论: “错题信号不能丢”是主线级判断。

### 实验 21. `TKC/UKC` 学生自适应融合 gate

- 将固定 `alpha/beta` 升级为学生级自适应 gate 后，三 seed 稳定优于旧主线。
- 结论: 自适应 `TKC/UKC` 融合已成为默认配置。

### 实验 23. 修正 `TKC` 行为项全局二次缩小

- 找到 `_build_exercise_component` 中“历史越长，当前行为证据越弱”的系统性缩放偏差，并改为概念内加权平均。
- 相对实验 21，三 seed `test_auc` 均值提升约 `+0.0095`。
- 结论: 这是当前主线的稳定基座，但已被实验 33/34 继续向前推进。

### 实验 33. `cognitive_match` zero-init difficulty adapter

- 不直接扩展 `cognitive_match_mlp` 输入，而是在认知 logits 外加 zero-init difficulty sidecar adapter。
- 相对实验 23 基座，三 seed 同时改善 `AUC/ACC/RMSE/Brier/ECE`，并解决了实验 29 的 seed 崩盘问题。
- 结论: 这是实验 34 之前的关键前一跳。

### 实验 34. high-concept logit adapter + `guess/slip` difficulty adapter

- 对 `concept_count >= 2` 的题增加 zero-init high-concept logit residual。
- 在 conditional `guess/slip` 分支增加 zero-init difficulty residual。
- 相对实验 23 基座，三 seed 均值改善:
  - `AUC +0.001506`
  - `ACC +0.002772`
  - `RMSE -0.002218`
  - `Brier -0.001909`
  - `ECE -0.011134`
- 结论: 实验 34 是当前 `master` 正式主线。

## 已验证无效或已降级

这些路线默认不要再回到主线，除非用户明确要求复访。

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
  - 在 ASSIST09 上明显退化，保留为 legacy ablation 即可
- 实验 13: `q_e` residual 融合
  - 弱于当时主线，不建议继续
- 实验 14: `TKC/UKC` 同时局部硬 mean
  - 明显退化
- 实验 15: `TKC` 局部 mean + `UKC` 全局 mean
  - 没能救回局部硬汇聚路线
- 实验 20: 直接叠加 `local UKC neighbor` 和 `UKC-only coverage gate`
  - 没有形成 `1 + 1 > 1`
- 实验 22: 在实验 21 主线底座上复验 `local UKC neighbor`
  - 低于该底座，不建议继续
- 实验 25: readout 侧 `TKC` item-aware residual
  - 工程可跑，但效果明显低于正式主线
- 实验 27: `q_repr` 内部 item-aware Q pooling
  - 单次弱于当前主线，不建议扩 seed
- 实验 29: 直接扩展 `cognitive_match` 输入以读入 `difficulty`
  - `seed=2026` 稳定崩盘，zero-init 也没救回
- 实验 30: 行为正误融合 gate 加 concept-specific bias
  - AUC 持平但误差和校准变差
- 实验 31: `UKC` no-param graph propagation
  - AUC 仅微升，但 `ACC/RMSE/Brier/ECE` 全变差
- 实验 32: `UKC` directed symmetric-like normalization
  - 单 seed 已不满足继续扩 seed 的门槛
- 实验 35: local readout adapter
  - 做法: zero-init cognitive logit sidecar 读取当前题目 Q mask 对应的 `TKC+UKC` 局部概念状态均值
  - 工程备注: 朴素 `[N, K, dim]` target gather 会 OOM，实验分支改为 sparse Q `index_add` 聚合
  - `seed=2024` 相对实验 34 同 seed:
    - `test_auc -0.004031`
    - `test_acc -0.000856`
    - `test_rmse +0.001213`
    - `test_brier +0.001043`
    - `test_ece +0.000762`
  - `concept_count=2/3` 切片也退化；`4+` 仅在 `RMSE/Brier` 小幅改善但样本少且 `ECE` 明显变差
  - 结论: 不扩 seed，不建议合入主线
- 实验 36: student base ability
  - 做法: 在 cognitive logit 分支增加 zero-init 学生全局能力 embedding `theta_u`
  - `seed=2024` 相对实验 34 同 seed:
    - `test_auc -0.002491`
    - `test_acc +0.001827`
    - `test_rmse +0.000491`
    - `test_brier +0.000422`
    - `test_ece -0.000231`
  - 切片上有信号:
    - `none_seen` 的 `RMSE/Brier/ECE` 改善，但 `AUC` 下降
    - `student_history_count=51-100` 与 `6-20` 改善，`101+` 与 `21-50` 的误差变差
    - `concept_count=2/3/4+` 的 `RMSE/Brier` 改善，但主量级样本 `concept_count=1` 退化
  - 结论: exact 版本整体代价大于收益，不扩 seed；后续若复访，应考虑更强约束或只作为 targeted calibration sidecar

### 语义更干净，但不值得主线吸收

- 实验 24: 多知识点题按知识点数分摊
  - 相对实验 23 三 seed 仅约 `+0.0001`
  - 可保留为语义更干净的实现参考，但不是关键收益来源
- 实验 26: `guess/slip` logit 正则
  - 可改善校准，但会牺牲少量 AUC
  - 不建议作为默认主线

## 旧口径的历史参考

下面这些实验主要用于说明“某类信号曾经出现过”，不能直接当作当前主线结论。

### 实验 16. `TKC` item-aware attention + `UKC` 全局 mean

- 说明“题目条件局部选择性”在旧主线口径下曾有信号。
- 结论: 可作为历史参考，但不能直接外推到当前主线。

### 实验 17. coverage-aware `UKC` 邻域约束

- 说明 `UKC` 全局噪声在旧口径下确实是问题。
- 结论: 这类局部化约束值得记住，但后续在新主线上的复验并未胜出。

### 实验 18. coverage-aware 全局融合

- 说明 coverage 作为融合条件在旧口径下有一定信号。
- 结论: 与后续主线的融合语义有重叠，不单独作为当前优先方向。

### 实验 19. 只对全局 `UKC` 做 coverage gate

- 在旧口径下接近最佳，但仍未成为当前主线的直接祖先。
- 结论: 保留为“旧口径次优参考”即可。

## 近线 follow-up

这部分只保留当前最值得记住的候选和最近的诊断。

### 实验 28. `guess/slip` 显式引入题目难度特征

- 做法: 在 `guess/slip` 条件输入中拼入 `difficulty.detach()`
- 结果: 三 seed 下 AUC 基本持平，`ECE/Brier` 有均值改善
- 判断: 可合入候选，但增量不够大，暂不是已定新主线

### 诊断 1. 实验 33 主线的 test prediction slices

关键观察:

- 多知识点题明显更难，而且随 `concept_count` 增长单调退化
- `none_seen` 样本排序不差，但校准很差，说明更像系统性低估
- 中等历史长度、中等历史正确率样本也偏弱

关于 recency:

- 当前 split 是按学生随机抽样，不是严格时间切分
- 因此在现有协议下不建议优先做 recency-aware student state

诊断结论:

- 下一步应优先针对“多知识点题表示/读出偏弱”做 targeted 修补
- `none_seen` 更适合作为后续单独校准问题，而不是当前第一优先结构问题

## 默认下一步

如果没有用户明确指定路线，默认按下面优先级思考:

1. 先从当前 `master` 主线出发，只改一个结构因素。
2. 优先考虑轻量、zero-init、可回退的 sidecar / residual 改动。
3. 优先修多知识点题的 targeted residual / readout 问题，而不是回到早期 `dual graph`、裸 Q 图或大幅改主干。
4. 新结构先跑单次；单次值得继续时再补 `2-3` 个 seed。
5. 判断是否值得合入时，默认同时看 `AUC/ACC/RMSE/Brier/ECE`，不要只看 AUC。

## 快速索引

当前主线形成链路:

- 实验 3: transition graph
- 实验 6: 合理超参数
- 实验 7: `conditional g/s`
- 实验 8: 长训 `300 epoch`
- 实验 9: 多 seed 稳定性
- 实验 11: `TKC/UKC` 解耦
- 实验 12: `TKC` 正误双通道
- 实验 21: 学生自适应 `TKC/UKC` gate
- 实验 23: 修正 `TKC` 行为项全局二次缩小
- 实验 33: zero-init cognitive difficulty adapter
- 实验 34: high-concept logit adapter + `gs_difficulty_adapter`

当前值得记住的候选:

- 实验 28: `guess/slip` 读入难度，校准改善

当前默认不建议继续的代表路线:

- 实验 10: `dual graph`
- 实验 13: `q_e` residual
- 实验 14/15: 局部硬汇聚
- 实验 22: 在新主线底座上复验 `local UKC neighbor`
- 实验 25: readout 侧 `TKC` residual
- 实验 27: `q_repr` item-aware Q pooling
- 实验 29: 直接扩展 `cognitive_match` 输入
- 实验 31/32: `UKC` 图传播轻量替换
- 实验 35: local readout adapter
- 实验 36: student base ability
