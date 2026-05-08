# Model Improvement Archive

这份文档只保留实验台账用途，用来回答三件事:

- 当前 `master` 的正式主线是什么
- 哪些路线已经证明有效或无效
- 下一步默认该优先试什么

它不是新会话默认入口。新会话先读 [docs/session_bootstrap.md](./session_bootstrap.md)；只有在需要查历史实验、避免重复试错时再回来看这份档案。

## 如何使用

- 先看“当前快照”，确认主线、结果口径和近线候选。
- 要判断某条路线是否还值得继续时，看“已验证有效”和“已验证无效或已降级”。
- 要设计下一轮实验时，看“近线 follow-up”和“默认下一步”。
- 要按实验号定位时，直接搜索 `实验 <编号>`。

## 当前快照

- 当前 `master` 正式主线沿用 [docs/session_bootstrap.md](./session_bootstrap.md) 里的“当前主线”口径。
- 从这份归档的实验视角看，它对应实验 51 主线:
  - 以实验 34 为底座
  - 吸收实验 49 的 history-carrier pairwise interaction residual
  - 再吸收实验 51 的 interpretable readout expert residual
- 这里不再重复维护与 `session_bootstrap.md` 等价的数据、图、超参数和 adapter 开关清单；需要确认默认运行口径时，优先回看 `session_bootstrap.md`
- 当前主线三 seed 参考均值:
  - `test_auc = 0.763889`
  - `test_acc = 0.728951`
  - `test_rmse = 0.427952`
  - `test_brier = 0.183143`
  - `test_ece = 0.049399`
- 当前结果报告默认主看 `AUC/ACC`
- `RMSE/Brier/ECE/分桶校准` 默认作为次要指标
- 若目标是推进主线，默认希望 `AUC` 或 `ACC` 的改善至少达到 `1e-3` 量级；达不到时，通常需要很强的 slice 证据才值得继续

- 当前正向支线候选:
  - 实验 37
    - branch: `exp/training-modes`
    - 判断: 它仍是当前更强的 calibration-oriented 训练协议候选，但在 `AUC/ACC` 上仍弱于当前主线，不作为默认 `master` 训练口径
    - 补充: 这条线属于纯训练工程优化，单次运行耗时显著高于当前默认 full-batch 口径；在模型结构仍需继续迭代时，暂不优先合入主线
    - 详细指标见下文“实验 37”
  - 实验 61
    - branch: `exp/full-target-exclusion-opt`
    - 判断: 它是当前更强的 ranking-oriented target-exclusion 训练候选，工程优化后运行成本已从“明显过高”降到“可接受”，暂定为主线候选，但仍不作为默认 `master` 训练口径
    - 补充: `2026-05-02` 复跑三 seed 后，`AUC` 稳定高于当前主线，`ACC/RMSE/Brier/ECE` 只形成 very small mixed deltas，因此更适合作为正式候选保留，而不是直接替换当前默认训练定义
    - 详细指标见下文“实验 61”

- 暂停中的 CF 支线:
  - 实验 38 `exp/cf-residual`: ranking-oriented 候选，但依赖学生内随机 split 的 ID-aware side channel，不作为纯 CDM 主线
  - 实验 39 `exp/cf-residual-recompute`: 不是实验 37 与 38 的无损叠加，不建议主线化
  - 实验 40 `exp/cf-residual-dim-sweep`: 大容量收益主要来自 transductive ID side channel，整条线暂停
  - 详细指标见下文“实验 38-40”

- 最近失败或已暂停的 follow-up:
  - 实验 43-47: 都只形成局部 slice 信号、AUC/ACC 不成立或整体副作用明显，不继续扩线
  - 实验 48: 证明“显式历史概念统计”有局部价值，但 original form 的三 seed overall 不稳定，不作为主线结构推进
  - 实验 50: learned weighting 没有带来额外收益，主线保留简单均值聚合
  - 实验 52: 更干净的 interpretable readout routing 没能超过实验 51 原版 full-trigger，不继续沿这条 selective routing 扩线
  - 实验 53: softer routing regularizer 也没能超过实验 51 原版 full-trigger，不继续沿这条 routing regularization 扩线
  - 实验 54: Q-conditioned local mastery readout 复访后仍弱于实验 51 主线，不继续沿这条“local mastery 主 readout”扩线
  - 实验 55: difficulty-weighted propagation 只带来极小 AUC 正向，但 `ACC/RMSE/Brier/ECE` 副作用明显，不继续沿这条 propagation weighting 扩线
  - 实验 56: student-wise pairwise ranking loss 也没把 overall 指标做成，不继续沿这条 ranking-loss 训练线扩权重或扩 seed
  - 实验 57: single-graph multi-hop propagation 复访后仍只有轻微排序波动，未形成 clean overall 正向；全局、coverage-conditioned、`UKC-only` 与 `2-hop only` 变体都不继续扩线
  - 实验 59: parallel local context readout 分支在 smoke 阶段就触发数值不稳定，当前实现不再继续
  - 详细指标见对应实验条目

## 已验证有效

下面只保留真正改变主线判断的实验。

### 实验 3. 论文式 transition graph

- 从早期 Q 共现图切到论文式有向转移图后，效果第一次明显优于随机附近基线。
- 结论: transition graph 是后续所有正式对比的图结构起点。

### 实验 6. 超参数扫描

- 固定 `learning_rate = 1e-3`、`concept_dim = 64`。
- 结论: 后续结构比较默认锁定这组超参数。

### 实验 7. `conditional g/s`

- 条件化 `guess/slip` 明显优于学生常数 `g/s`。
- 结论: `conditional g/s` 为主线固定配置。

### 实验 8. 长训

- `20 epoch` 不够，结构比较至少应看 `300 epoch`。
- 结论: 长训是正式比较默认协议。

### 实验 9. 多 seed

- 早期单图长训基线在 `seed in {2024, 2025, 2026}` 上波动很小。
- 结论: 主线比较不能只看单次最好值，要看多 seed 稳定性。

### 实验 11. `TKC/UKC` 结构传播参数解耦

- `TKC/UKC` 概念传播从共享参数改为独立参数后形成稳定增益。
- 结论: 这是当前主线最可靠的正向结构改动之一。

### 实验 12. `TKC` 正误双通道行为消息

- 显式保留错题证据后，三 seed 均值明显优于只看正确题。
- 结论: 错题信号不能丢。

### 实验 21. `TKC/UKC` 学生自适应融合 gate

- 固定 `alpha/beta` 升级为学生级自适应 gate 后，三 seed 稳定优于旧主线。
- 结论: 自适应 `TKC/UKC` 融合已成为默认配置。

### 实验 23. 修正 `TKC` 行为项全局二次缩小

- 将 `_build_exercise_component` 中“历史越长，当前行为证据越弱”的系统性缩放偏差改为概念内加权平均。
- 相对实验 21，三 seed `test_auc` 均值约 `+0.0095`。
- 结论: 这是当前主线的稳定基座。

### 实验 33. `cognitive_match` zero-init difficulty adapter

- 在认知 logits 外加 zero-init difficulty sidecar adapter。
- 相对实验 23，三 seed 同时改善 `AUC/ACC/RMSE/Brier/ECE`，并解决实验 29 的 seed 崩盘。
- 结论: 这是实验 34 前的关键一跳。

### 实验 34. high-concept logit adapter + `guess/slip` difficulty adapter

- 对 `concept_count >= 2` 的题增加 zero-init high-concept logit residual。
- 在 conditional `guess/slip` 分支增加 zero-init difficulty residual。
- 相对实验 23 三 seed 均值:
  - `AUC +0.001506`
  - `ACC +0.002772`
  - `RMSE -0.002218`
  - `Brier -0.001909`
  - `ECE -0.011134`
- 结论: 实验 34 是当前 `master` 正式主线。

### 实验 49. history-carrier pairwise interaction residual

- 动机: 不再直接从局部 `TKC/UKC` embedding 读多知识点交互，而是显式建模题相关概念对，并把更直接的历史概念统计作为 state carrier
- 分支: `exp/pairwise-history-carrier`
- 结构:
  - 保留实验 34 主线
  - 只对 `concept_count >= 2` 的题激活 zero-init pairwise interaction residual
  - 对每个题相关概念对共享 scorer
  - pairwise 输入为 `c_k / c_j / e_e` 加上逐概念历史统计 `accuracy / seen / log_attempt_count`
  - pair score 做均值聚合后加到 `cognitive_logits`
- 三 seed 相对实验 34:
  - `seed=2024`: `AUC +0.001199`, `ACC +0.003844`, `RMSE -0.001182`, `Brier -0.001014`, `ECE -0.001450`
  - `seed=2025`: `AUC +0.002073`, `ACC +0.001427`, `RMSE -0.000984`, `Brier -0.000844`, `ECE -0.000084`
  - `seed=2026`: `AUC +0.000248`, `ACC +0.001351`, `RMSE -0.000728`, `Brier -0.000624`, `ECE -0.002408`
- 三 seed 均值:
  - `AUC +0.001173`
  - `ACC +0.002207`
  - `RMSE -0.000965`
  - `Brier -0.000827`
  - `ECE -0.001314`
- 切片:
  - `seed=2024` 的 `concept_count=2/3/4+` 均明显改善，其中 `concept_count=3` 为 `AUC +0.030833`, `ACC +0.025471`, `RMSE -0.013033`
  - `none_seen` 也形成 clean win: `ACC +0.007841`, `RMSE -0.004422`, `Brier -0.003331`, `ECE -0.008649`
- 结论:
  - 这是第一条把“显式历史概念统计”稳定转成 overall 正收益的多知识点结构
  - 它不依赖 item ID side channel，也不是 exact-3 特判，语义与实现都足够干净
  - 实验 49 已吸收到 `master`

### 实验 37. true mini-batch recompute training

- 动机: 真正引入 mini-batch SGD 噪声，同时保留传播梯度。
- 分支: `exp/training-modes`
- 关键工程:
  - 增加 `training_mode`
  - `recompute_minibatch` 每个 mini-batch 重跑完整传播
  - `frozen_readout`、`alternating_frozen_readout` 已验证为负向
- 正向配置: `recompute_minibatch bs=8192 lr=1e-4`
- 三 seed 结果:
  - `seed=2024`: `AUC 0.760737`, `ACC 0.727873`, `RMSE 0.428030`, `Brier 0.183210`, `ECE 0.042740`
  - `seed=2025`: `AUC 0.764064`, `ACC 0.728216`, `RMSE 0.426850`, `Brier 0.182201`, `ECE 0.045369`
  - `seed=2026`: `AUC 0.761622`, `ACC 0.727550`, `RMSE 0.428146`, `Brier 0.183309`, `ECE 0.045434`
- 三 seed 均值:
  - `test_auc = 0.762141`
  - `test_acc = 0.727879`
  - `test_rmse = 0.427675`
  - `test_brier = 0.182906`
  - `test_ece = 0.044514`
- 相对实验 34 三 seed 均值:
  - `AUC +0.000945`
  - `ACC +0.000324`
  - `RMSE -0.001494`
  - `Brier -0.001280`
  - `ECE -0.006628`
- 切片:
  - `concept_count=3` 的 `AUC +0.015340`
  - `concept_count=4+` 仍退化: `AUC -0.010693`, `RMSE +0.003337`, `ECE +0.001965`
  - `none_seen` 的 `RMSE/Brier` 改善，但 `ECE +0.001255`
- 结论:
  - 这是当前最强正向支线候选。
  - 训练协议优化已接近收益上限；`cosine + warmup` 仅带来 `AUC +0.000471`, `ACC +0.000127`，同时 `ECE +0.002442`，不值得整包主线化。

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
- 实验 45: propagation-side concept-conditioned exercise residual
  - 分支: `exp/concept-conditioned-prop`
  - 做法: 对 `concept_count >= 2` 的题，在 propagation 的 `exercise -> concept` 消息上增加 concept-conditioned zero-init residual；按 Q 非零边现算，避免显式构造稠密 `E x K x D`
  - `seed=2024` 相对当前主线 baseline: `AUC +0.000109`, `ACC -0.000362`, `RMSE +0.000271`, `Brier +0.000233`, `ECE +0.002160`
  - 切片:
    - `concept_count=2`: `AUC +0.001875`, `RMSE -0.000777`, `ECE -0.002424`
    - `concept_count=3`: `AUC +0.003068`, 但 `ACC -0.003322`, `RMSE +0.001202`, `ECE +0.006053`
    - `concept_count=4+`: `ACC +0.024793`, `ECE -0.016021`, 但 `AUC -0.002352`, `RMSE +0.004048`
    - `none_seen`: `AUC +0.003574`, 但 `RMSE +0.001578`, `ECE +0.001645`
  - 结论: 这是“局部切片有信号但整体不成立”的传播侧修补；不扩 seed，不纳入主线
- 实验 46: readout-side qrepr score residual
  - 分支: `exp/qrepr-score-residual`
  - 做法: 保留 static `q_pool_gate`，仅对 `concept_count >= 2` 的题增加 zero-init exercise-conditioned Q-pooling score residual；按 batch chunk 分块计算 additive score，避免显式物化完整 `B x K x D`
  - `seed=2024` 相对当前主线 baseline: `AUC +0.000664`, `ACC -0.001370`, `RMSE +0.000185`, `Brier +0.000159`, `ECE +0.001519`
  - 切片:
    - `concept_count=2`: `AUC +0.002694`, `ECE -0.003136`，但 `ACC -0.006002`
    - `concept_count=3`: `AUC +0.007144`, `RMSE -0.002483`, `ECE -0.002975`，但 `ACC -0.016611`
    - `concept_count=4+`: `ACC +0.024793`, `ECE -0.008125`，但 `AUC -0.012312`, `RMSE +0.006886`
    - `none_seen`: `AUC +0.001804`, 但 `RMSE +0.008018`, `ECE +0.017540`
    - `partial_seen`: `AUC -0.001127`, `RMSE +0.010477`, `ECE +0.009486`
  - 结论: 相比实验 45，这条 readout 侧窄变体更接近目标瓶颈，但仍然是“局部排序改善换整体与校准副作用”的折中；不扩 seed，不纳入主线
- 实验 47: final-logit none-seen calibration bias
  - 分支: `exp/none-seen-calibration-bias`
  - 做法: 在最终概率输出前增加 zero-init calibration residual；输入只看 target concept coverage、`concept_count` 和 `difficulty`，不改 TKC/UKC propagation 语义
  - `seed=2024` 相对当前主线 baseline: `AUC -0.000945`, `ACC -0.000381`, `RMSE +0.000438`, `Brier +0.000376`, `ECE -0.001393`
  - 切片:
    - `none_seen`: `AUC +0.000401`, 但 `ACC -0.005428`, `RMSE +0.003811`, `Brier +0.002895`, `ECE +0.011709`
    - `partial_seen`: `AUC +0.004038`, `ACC +0.003030`, 但 `RMSE +0.000513`, `Brier +0.000418`, `ECE +0.015319`
    - `all_seen`: `AUC -0.000886`, `ACC -0.000237`, `RMSE +0.000340`, `Brier +0.000293`, `ECE -0.001199`
    - `concept_count=1`: `AUC -0.001279`, `ACC -0.000754`, `RMSE +0.000597`, `Brier +0.000510`, `ECE -0.001459`
    - `concept_count=2`: `AUC +0.001413`, `ACC +0.002801`, `RMSE -0.000603`, `Brier -0.000529`, `ECE -0.000978`
    - `concept_count=3`: `AUC -0.002885`, `ACC -0.008859`, `RMSE +0.000545`, `Brier +0.000520`, `ECE -0.000689`
    - `concept_count=4+`: `AUC -0.005805`, `ACC +0.000000`, `RMSE +0.002798`, `Brier +0.002615`, `ECE +0.001669`
  - 结论: 这类“全局共享 final-logit calibration bias”太钝。虽然 overall `ECE` 略降，但没有解决 `none_seen`，反而把目标切片的 `ACC/RMSE/Brier/ECE` 一起做坏；不扩 seed，不纳入主线
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

- 实验 50: weighted pairwise history aggregation
  - 分支: `exp/pairwise-history-weighted-agg`
  - 做法: 在实验 49 上把 pair score 聚合从固定均值改成 learned weighting，其余结构不变
  - `seed=2024` 相对实验 49:
    - `AUC -0.000441`
    - `ACC -0.000305`
    - `RMSE -0.000120`
    - `Brier -0.000103`
    - `ECE +0.000002`
  - 结论:
    - learned weighting 没有提供额外收益
    - 这说明当前增益主要来自“history carrier + pairwise scorer”本身，而不是更复杂的 pair aggregator
    - 主线保留简单均值聚合，不继续扩 seed

- 实验 51: interpretable readout expert residual
  - 分支: `exp/interpretable-readout-experts`
  - 做法:
    - 在 `cognitive_logits` 外增加 zero-init readout expert residual
    - gate 只读取 `concept_count / difficulty / dispersion / coverage` 这四类可解释量
    - expert 侧读取 detached readout features，不读学生/题目 ID，不改 propagation 主干语义
  - 最强配置:
    - `--interpretable-readout-expert-adapter`
    - `--interpretable-readout-expert-count 3`
  - `seed=2024` 相对实验 49 control:
    - `AUC +0.001548`
    - `ACC -0.000818`
    - `RMSE +0.000000`
    - `Brier +0.000001`
    - `ECE +0.000973`
  - targeted 变体:
    - `min_count>=2` 的三专家版本能让 `concept_count=2/3` 切片转正，但 overall `AUC -0.000151`，不如 full-trigger
    - `min_count>=3` 的两专家/三专家版本都未超过 full-trigger 单次结果
  - full-trigger 三 seed 结果:
    - `seed=2024`: `AUC 0.764051`, `ACC 0.728672`, `RMSE 0.428295`, `Brier 0.183437`, `ECE 0.050665`
    - `seed=2025`: `AUC 0.764711`, `ACC 0.728387`, `RMSE 0.428095`, `Brier 0.183265`, `ECE 0.051829`
    - `seed=2026`: `AUC 0.762904`, `ACC 0.729795`, `RMSE 0.427466`, `Brier 0.182727`, `ECE 0.045703`
  - full-trigger 三 seed 均值相对实验 49:
    - `AUC +0.001520`
    - `ACC -0.000812`
    - `RMSE -0.000253`
    - `Brier -0.000217`
    - `ECE -0.000429`
  - 结论:
    - 这条线已经从单 seed 信号变成稳定的结构候选，当前是最强的非 ID-aware follow-up
    - 它的代价是小幅 `ACC` 回撤，但 `AUC` 增益已经达到继续保留的门槛
    - 第一版 full-trigger 收益主要来自 `concept_count=1 / all_seen`，没有自然学成“只服务高 concept-count”的干净专家分工
    - 如果后续继续做 selective routing，应建立在这条 full-trigger 正向底座上，而不是直接退回更硬的 `3+` trigger
    - 这一步现已吸收到当前 `master`

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

- 实验 53: soft routing regularizer for readout experts
  - 分支: `exp/readout-routing-soft-regularizer`
  - 提交: `910b9c8`
  - 做法:
    - 以实验 51 的 full-trigger 三专家 residual 为底座
    - 暴露 gate probability，并在训练时加入轻量 routing regularizer
    - regularizer 形式为 `conditional_entropy - marginal_entropy`
    - 本轮只测试 `--interpretable-readout-expert-routing-mi-weight 0.05`
  - 结果:
    - smoke:
      - `max_rows=2000`, `epoch=1`, `mi_weight=0.05` 能正常训练并产出 summary
    - `seed=2024`:
      - `AUC 0.764348`
      - `ACC 0.729148`
      - `RMSE 0.427924`
      - `Brier 0.183119`
      - `ECE 0.050730`
    - `seed=2025`:
      - `AUC 0.764564`
      - `ACC 0.726008`
      - `RMSE 0.429125`
      - `Brier 0.184148`
      - `ECE 0.055972`
    - `seed=2026`:
      - `AUC 0.761720`
      - `ACC 0.729148`
      - `RMSE 0.427652`
      - `Brier 0.182886`
      - `ECE 0.041885`
  - 三 seed 均值相对实验 51:
    - `AUC -0.000345`
    - `ACC -0.000850`
    - `RMSE +0.000282`
    - `Brier +0.000241`
    - `ECE +0.000130`
  - 结论:
    - `seed=2024` 虽然略优于实验 51 同 seed，但 `seed=2025/2026` 没有复现，三 seed 均值回到全面弱于当前主线
    - 不继续沿这条 soft routing regularizer 扩线；实验 51 原版 full-trigger 仍然是这条线应保留的最强基线
    - 若后续还要 revisit routing，优先考虑更局部的软引导或更明确的 slice 目标，而不是继续围绕同一种全局 gate regularizer 小步扫参

- 实验 54: Q-conditioned local mastery readout revisit
  - 分支: `exp/q-conditioned-local-mastery-readout`
  - 提交: `aeb42b7`
  - 做法:
    - 不再只读全局 `student_state`
    - 对每个目标交互，只 gather 该题 Q mask 命中的概念
    - 对每个目标概念，基于 `tkc_state / ukc_state / concept_embedding / seen_flag / difficulty` 共享打分
    - 用几何均值式的 concept aggregation 形成题目级 local mastery logit
    - 在原有 `cognitive_logits` 外加 zero-init gate: `old_logit + gate * local_mastery_logit`
  - 工程验证:
    - 初版按 “交互数 x 全概念数” 展开局部状态导致远端正式训练 OOM
    - 改成只对命中概念做 gather 后，单测与 smoke 恢复正常
  - 结果:
    - smoke:
      - `max_rows=2000`, `epoch=1` 能正常训练并产出 summary
    - `seed=2024`:
      - `AUC 0.761978`
      - `ACC 0.726674`
      - `RMSE 0.428573`
      - `Brier 0.183675`
      - `ECE 0.046842`
  - 相对实验 51 `seed=2024`:
    - `AUC -0.002073`
    - `ACC -0.001998`
    - `RMSE +0.000278`
    - `Brier +0.000238`
    - `ECE -0.003823`
  - 切片:
    - `concept_count=2`: `AUC 0.749767`, `ACC 0.717925`, `ECE 0.060700`
    - `concept_count=3`: `AUC 0.698639`, `ACC 0.682171`, `ECE 0.082560`
    - `concept_count=4+`: `AUC 0.731486`, `ACC 0.680441`, `ECE 0.125002`
    - `none_seen`: `AUC 0.808718`, `ACC 0.800362`, `ECE 0.112099`
  - 结论:
    - 这次复访已经不是实验 35 那种“局部概念状态均值 residual”，而是更接近主 readout 的逐概念打分再聚合版本；即便如此，overall `AUC/ACC` 仍明显不成立，切片上也没有出现足够强的多知识点 clean win
    - 不继续沿这条 local mastery main-readout 设计扩线
    - 若后续再访，必须带着更强的概念交互假设或更明确的聚合归纳偏置，而不是再重复“逐概念打分 + 简单聚合”框架

- 实验 55: difficulty-weighted propagation
  - 分支: `exp/difficulty-weighted-propagation`
  - 提交: `b9704ec`
  - 做法:
    - 在 propagation 的 `correct/incorrect` 两条 `exercise -> concept` 历史证据前，各自增加独立的 zero-init multiplicative scaling
    - scaling 输入读取 `difficulty + concept_count + detached exercise_embedding`
    - `correct_weight = base_correct * multiplier_correct`
    - `incorrect_weight = base_incorrect * multiplier_incorrect`
    - `multiplier = 2 * sigmoid(raw_scale)`，因此零初始化时严格退化回当前主线
  - 结果:
    - smoke:
      - `max_rows=2000`, `epoch=1` 能正常训练并产出 summary
    - `seed=2024`:
      - `AUC 0.764173`
      - `ACC 0.724124`
      - `RMSE 0.430202`
      - `Brier 0.185073`
      - `ECE 0.061324`
  - 相对实验 51 `seed=2024`:
    - `AUC +0.000122`
    - `ACC -0.004548`
    - `RMSE +0.001907`
    - `Brier +0.001636`
    - `ECE +0.010659`
  - 切片:
    - `concept_count=2`: `AUC 0.750349`, `ACC 0.712457`, `ECE 0.070721`
    - `concept_count=3`: `AUC 0.705193`, `ACC 0.673311`, `ECE 0.102832`
    - `concept_count=4+`: `AUC 0.747586`, `ACC 0.669421`, `ECE 0.091158`
    - `none_seen`: `AUC 0.810439`, `ACC 0.769602`, `ECE 0.151766`
  - 结论:
    - 这条线确实证明“把 difficulty 前移到 propagation”会改变排序行为，单 seed `AUC` 有极小正向；但代价过大，`ACC/RMSE/Brier/ECE` 全部明显回撤，切片也没有形成足够干净的多知识点收益
    - 不继续沿这条 difficulty-weighted propagation 扩线
    - 若后续还要 revisit propagation weighting，优先考虑更局部、更学生条件化的证据强度，而不是当前这种按题目全局共享的 difficulty multiplier

- 实验 56: student-wise pairwise ranking loss
  - 分支: `exp/student-pairwise-ranking-loss`
  - 提交: `7c90eb9`
  - 做法:
    - 不改模型结构，只在 BCE 外叠加同学生内的 pairwise logistic ranking loss
    - 对每个学生，把 train 交互拆成正样本集合和负样本集合，约束 `score_pos > score_neg`
    - ranking score 使用最终预测概率的 `logit(prob)`
    - 只测试 `weight=0.05` 和 `weight=0.02`
  - 工程验证:
    - 训练集约 `19.4` 万交互，总 student-wise 正负配对约 `922` 万
    - 单测通过，1 epoch smoke 通过，训练耗时与当前 full-batch 主线同量级，没有出现不可接受的额外成本
  - 结果:
    - `weight=0.05`, `seed=2024`:
      - `AUC 0.763793`
      - `ACC 0.726370`
      - `RMSE 0.428940`
      - `Brier 0.183989`
      - `ECE 0.053809`
    - `weight=0.02`, `seed=2024`:
      - `AUC 0.763271`
      - `ACC 0.728120`
      - `RMSE 0.428567`
      - `Brier 0.183670`
      - `ECE 0.051483`
  - 相对实验 51 `seed=2024`:
    - `weight=0.05`:
      - `AUC -0.000258`
      - `ACC -0.002302`
      - `RMSE +0.000645`
      - `Brier +0.000552`
      - `ECE +0.003144`
    - `weight=0.02`:
      - `AUC -0.000780`
      - `ACC -0.000552`
      - `RMSE +0.000272`
      - `Brier +0.000233`
      - `ECE +0.000818`
  - 切片:
    - `weight=0.02`, `concept_count=2`: `AUC 0.750245`, `ACC 0.718592`, `ECE 0.065355`
    - `weight=0.02`, `concept_count=3`: `AUC 0.704542`, `ACC 0.687708`, `ECE 0.083009`
    - `weight=0.02`, `concept_count=4+`: `AUC 0.741598`, `ACC 0.680441`, `ECE 0.081981`
    - `weight=0.02`, `none_seen`: `AUC 0.809586`, `ACC 0.803378`, `ECE 0.106477`
  - 结论:
    - 这条线确实会把验证 AUC 往上推一点，但在 test 上没有超过实验 51 主线；`0.05` 和 `0.02` 都没形成 overall 正向，切片上也没有出现足够强的多知识点 clean win
    - 不继续沿这条 ranking-loss 训练线扩权重或扩 seed
    - 若后续还要 revisit ranking-oriented 目标，应优先建立在某个已经有明确结构正向的底座之上，而不是单独把 ranking loss 当成默认下一步

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

- 实验 24: 多知识点题按知识点数分摊
  - 相对实验 23 三 seed 仅约 `+0.0001`
- 实验 26: `guess/slip` logit 正则
  - 可改善校准，但会牺牲少量 AUC

- 实验 59: parallel local context readout adapter
  - 分支: `exp/parallel-local-context-readout`
  - 做法:
    - 保留实验 51 当前主线全部配置
    - 不再把 local context 回写主 `student_state`
    - 改为从目标题相关概念的 `TKC/UKC` 局部状态构造 target-conditioned attention，再走一条并联 readout residual 分支，直接加到 `cognitive_logits`
  - 工程验证:
    - 远端 `python -m unittest tests.test_decoupled_cdm` 通过
    - 但 `2 epoch + max_rows=5000` smoke 在验证阶段触发 CUDA device-side assert；根因表现为 BCE 输入超出 `[0, 1]`
    - 修过一轮输入维度与 `nan_to_num`/无效行归零后，smoke 仍然不稳定
  - 结论:
    - 这不是“指标略差但可继续调参”的情况，而是当前实现本身在数值上就不稳，尚未达到可比较 overall 指标的最小门槛
    - 不继续沿这版 parallel local context readout 实现扩线，也不进入 rescue sweep
    - 若以后还要 revisit 更大一级 local-context 模块，优先先加显式幅度约束或更保守的 mixture 结构，再决定是否值得进入正式比较

- 实验 60: pairwise history target-exclusion audit
  - 分支: `exp/target-exclusion-audit`
  - 提交:
    - `831da8b`: 加入审计开关、分布打印与 pairwise target exclusion
  - 动机:
    - 审计 train / valid / test 的历史口径是否存在关键 mismatch
    - 在不改模型结构的前提下，先验证“只对 pairwise history residual 做 target exclusion”是否带来 clean 正收益
  - 工程诊断:
    - 当前默认训练仍是 full-batch；`--batch-size` 只被记录，不参与实际优化步切分
    - 在 ASSIST09 `train.csv` 上，`(stu_id, exer_id)` 没有重复，因此这次 pairwise target exclusion 对 train history 的扣除是精确的，不是近似版本
    - `train_model` 末尾确实会恢复 best checkpoint；审计 run 中 `restored_val_auc == best_val_auc`
  - 口径确认:
    - 训练 bundle 允许 target 留在 history
    - `valid/test` bundle 强制只复用 `train` history
    - 因此训练/测试 propagation history mismatch 是当前代码中的显式事实，不是推测
  - 结果:
    - baseline `seed=2024`:
      - `AUC 0.764051`
      - `ACC 0.728672`
      - `RMSE 0.428295`
      - `Brier 0.183437`
      - `ECE 0.050665`
    - pairwise target-exclusion `seed=2024`:
      - `AUC 0.764082`
      - `ACC 0.726883`
      - `RMSE 0.429038`
      - `Brier 0.184073`
      - `ECE 0.054048`
  - 相对 baseline:
    - `AUC +0.000030`
    - `ACC -0.001789`
    - `RMSE +0.000743`
    - `Brier +0.000637`
    - `ECE +0.003383`
  - 切片:
    - `concept_count=3`: `AUC -0.008054`, `ACC -0.009967`, `RMSE +0.003870`
    - `concept_count=4+`: `AUC -0.008402`, `ACC -0.005510`, `RMSE +0.004269`
    - `none_seen`: `AUC -0.002347`, `ACC -0.009047`, `RMSE +0.000828`
    - `partial_seen` 有局部正信号，但只有 `330` 条样本，不足以改变 overall 判断
  - 额外观察:
    - 开启 target exclusion 后，test 上 `guess_plus_slip` 均值从 `0.244109` 升到 `0.320332`
    - 这说明只削弱 pairwise history 自举信号后，模型明显把更多解释压力转移到了 `guess/slip` 分支
  - 结论:
    - “history mismatch 存在”这件事已经坐实，但“只修 pairwise residual”没有带来 clean 正收益；这次负结果也不能直接推出“完整 leave-one-out 一定无效”，因为 propagation 训练口径仍保持 target-visible
    - 先把这次审计结论固化为负向证据，不继续沿 pairwise-only target exclusion 扩 seed
    - 当前更值得单独处理的是 `--batch-size` 名义生效、实际无效的训练工程问题；若以后一定要把这条线彻底判死，只应再做一次“pairwise + propagation 同时 target-excluded”的单 seed 最终判定实验

- 实验 61: full target-excluded training audit
  - 分支: `exp/full-target-exclusion-audit`
  - 提交:
    - `4fa628d`: 加入 propagation + pairwise 同时 target-excluded 的训练路径与单测
    - `bb32578`: 去掉 exclusion 训练时多余的全局 propagation 前向，修复首轮 OOM
    - `ca14a6d`: 收紧 target-conditioned propagation 内部 chunk
    - `cf8c47f`: 把 full-batch exclusion 改成单 optimizer step 的梯度累积，实现完整数据可运行
  - 动机:
    - 对实验 60 留下的未决问题做最终复验
    - 验证“训练时同时去掉 propagation history 与 pairwise history 的 target self-inclusion”后，是否能得到比 pairwise-only exclusion 更干净的收益
  - 做法:
    - 保持当前实验 51 主线结构与超参数不变
    - 只在 train loss 路径上开启 `--exclude-target-from-train-history`
    - 对每个 train target，预测时把该条 `(stu_id, exer_id, label)` 自身从 target-conditioned propagation history 与 pairwise history 统计中扣除
    - `valid/test` 仍固定复用 `train` history，不改 evaluation 口径
  - 工程备注:
    - 直接在 full split 上做单次前向会 OOM，因此最终实现改成“full-batch 语义 + chunked gradient accumulation”: 仍然每个 epoch 只做 `1` 次 optimizer step，但 target-conditioned 前向按子批次累积梯度
    - 完整数据 `1 epoch` smoke 已在远端跑通；`300 epoch` 正式单 seed 也已跑通
  - 三 seed 结果:
    - `seed=2024`, `best_epoch=192`:
      - `AUC 0.765170`
      - `ACC 0.728539`
      - `RMSE 0.428619`
      - `Brier 0.183714`
      - `ECE 0.053582`
    - `seed=2025`, `best_epoch=197`:
      - `AUC 0.766096`
      - `ACC 0.726674`
      - `RMSE 0.428169`
      - `Brier 0.183329`
      - `ECE 0.053421`
    - `seed=2026`, `best_epoch=182`:
      - `AUC 0.764803`
      - `ACC 0.730157`
      - `RMSE 0.427572`
      - `Brier 0.182818`
      - `ECE 0.048511`
  - 三 seed 均值:
    - `AUC 0.765356`
    - `ACC 0.728457`
    - `RMSE 0.428120`
    - `Brier 0.183287`
    - `ECE 0.051838`
  - 相对实验 51 当前主线三 seed 均值:
    - `AUC +0.001467`
    - `ACC -0.000494`
    - `RMSE +0.000168`
    - `Brier +0.000144`
    - `ECE +0.002439`
  - 相对实验 51 同 seed:
    - `seed=2024`: `AUC +0.001119`, `ACC -0.000133`, `RMSE +0.000324`, `Brier +0.000277`, `ECE +0.002917`
    - `seed=2025`: `AUC +0.001385`, `ACC -0.001713`, `RMSE +0.000074`, `Brier +0.000064`, `ECE +0.001592`
    - `seed=2026`: `AUC +0.001899`, `ACC +0.000362`, `RMSE +0.000106`, `Brier +0.000091`, `ECE +0.002808`
  - 结论:
    - 完整 target exclusion 的 `AUC` 正向在三 seed 上稳定复现，说明“训练/测试 propagation history mismatch”不是纯方法学噪声；但它仍没有形成 clean overall win，更像稳定的 ranking-oriented 训练口径
    - 暂不直接吸收到 `master`
    - 若后续目标明确偏向 `AUC`，这条线可以作为正式候选保留；若主线仍坚持 `AUC/ACC` 与校准并重，则当前不继续沿这条训练口径扩线
  - `2026-05-02` 工程优化复跑:
    - 优化分支: `exp/full-target-exclusion-opt`
    - 关键提交:
      - `92cae0f`: 让 target-excluded history stats cache 成为可复用路径
      - `dc7cbee`: 把 target-excluded propagation 改成 baseline + 局部 delta 更新
      - `501612d`: full-batch exclusion 按 epoch 复用 propagation reference
      - `041f902`: target-exclusion inner chunk 跟外层训练批次对齐
    - 工程结果:
      - 在远端 `xph-pc`、完整 `assist_09_ordered/train.csv`、`epochs=1`、只测 train 不测 eval 的 benchmark 下，原始审计分支 `fe0c114` 的 median runtime 约 `8.845s`
      - 同口径下，优化分支 `041f902` 的 median runtime 约 `2.002s`
      - 端到端训练耗时约 `4.4x` 加速，降幅约 `77%`
      - benchmark 中 `train_loss` 保持一致，说明这轮工程优化没有改变训练语义
    - 正式三 seed 复跑结果:
      - `seed=2024`, `best_epoch=184`:
        - `AUC 0.765258`
        - `ACC 0.729643`
        - `RMSE 0.427945`
        - `Brier 0.183137`
        - `ECE 0.050238`
      - `seed=2025`, `best_epoch=206`:
        - `AUC 0.766181`
        - `ACC 0.726903`
        - `RMSE 0.428576`
        - `Brier 0.183677`
        - `ECE 0.056064`
      - `seed=2026`, `best_epoch=178`:
        - `AUC 0.765046`
        - `ACC 0.731222`
        - `RMSE 0.427128`
        - `Brier 0.182439`
        - `ECE 0.047690`
    - 三 seed 均值:
      - `AUC 0.765495`
      - `ACC 0.729256`
      - `RMSE 0.427883`
      - `Brier 0.183084`
      - `ECE 0.051331`
    - 相对实验 51 当前主线三 seed 均值:
      - `AUC +0.001606`
      - `ACC +0.000305`
      - `RMSE -0.000069`
      - `Brier -0.000059`
      - `ECE +0.001932`
    - 相对实验 61 原始审计版三 seed 均值:
      - `AUC +0.000139`
      - `ACC +0.000799`
      - `RMSE -0.000237`
      - `Brier -0.000203`
      - `ECE -0.000507`
    - 更新结论:
      - exp61 的主要工程障碍已经解除，优化后复跑结果也没有变坏；`AUC` 正向依旧稳定，且 `ACC/RMSE/Brier` 已从原始审计版的轻微负向回到基本持平或 very small 正向
      - 将 `exp/full-target-exclusion-opt` 暂定为主线候选保留，但当前不直接把它吸收到 `master` 默认训练口径
      - 若后续要做正式主线切换比较，这条线应作为与实验 37 并列的训练候选，而不是继续视作仅供归档的 ranking-only 审计分支

- 实验 62: constrained `guess/slip` probability budget
  - 分支: `exp/guess-slip-diagnostics`
  - 提交:
    - `e8b04fe`: 增加 `guess/slip` 分布诊断脚本
    - `f0f43fd`: 将 `guess/slip` 改为共享概率预算的耦合参数化，并补单测
  - 动机:
    - 当前输出层使用 `p = (1 - slip) * cognitive + guess * (1 - cognitive)`
    - 若 `guess + slip > 1`，则 `dp / dcognitive = 1 - guess - slip < 0`，会出现“认知越高，最终答对概率反而越低”的语义反转
  - 诊断:
    - 旧实现里 `guess_probs` 与 `slip_probs` 是彼此独立的 `sigmoid`，没有任何 `guess + slip <= 1` 约束
    - 对实验 51 同口径的已有 checkpoint 做只读审计后发现，这个问题已经真实发生，而不是纯理论风险
    - `exp_interpretable_readout_experts_seed2025` test split:
      - `guess_plus_slip_mean = 1.945274`
      - `guess_plus_slip_max = 1.999997`
      - `guess_plus_slip_p95 = 1.999303`
      - `ratio(guess_plus_slip > 1) = 0.999638`
    - `exp_interpretable_readout_experts_seed2026` test split:
      - `guess_plus_slip_mean = 1.872332`
      - `guess_plus_slip_max = 1.999906`
      - `guess_plus_slip_p95 = 1.992325`
      - `ratio(guess_plus_slip > 1) = 0.999486`
  - 做法:
    - 保持 `guess_logit / slip_logit` 两条分支和所有上游输入不变
    - 只把最终 `guess/slip` 概率映射改成三元 softmax:
      - `guess = softmax([guess_logit, slip_logit, 0])[0]`
      - `slip = softmax([guess_logit, slip_logit, 0])[1]`
    - 这样可显式保证 `guess >= 0`、`slip >= 0` 且 `guess + slip <= 1`
  - 验证:
    - 远端单测 `python -m unittest tests.test_decoupled_cdm` 通过
    - 旧 checkpoint 在新前向下重新审计后，`ratio(guess_plus_slip > 1)` 已回到 `0`
    - `exp_guess_slip_constraint_seed2024` 训练后 test split:
      - `guess_plus_slip_mean = 0.101998`
      - `guess_plus_slip_max = 0.998407`
      - `guess_plus_slip_p95 = 0.554550`
      - `ratio(guess_plus_slip > 1) = 0.0`
  - 单 seed 结果:
    - `seed=2024`, `best_epoch=181`:
      - `AUC 0.764666`
      - `ACC 0.726712`
      - `RMSE 0.428926`
      - `Brier 0.183978`
      - `ECE 0.054151`
  - 相对实验 51 同 seed baseline:
    - `AUC +0.000615`
    - `ACC -0.001960`
    - `RMSE +0.000631`
    - `Brier +0.000541`
    - `ECE +0.003486`
  - 结论:
    - 这次修复确实清除了输出层的语义违例，且不是只在极少数样本上起作用；但在当前形式下，它更像“修正 non-cognitive 分支后改变了排序偏好”，没有形成 clean overall win
    - 先保留这条线的机制结论和诊断工具，不直接吸收到 `master`
    - 若后续继续推进这类语义修复，更合理的下一步不是直接扩 seed，而是考虑给 constrained `guess/slip` 增加更有表达力的参数化或配套训练补偿，再看能否保住 `ACC/Brier/ECE`

- 实验 63: constrained `guess/slip` on top of exp61
  - 分支: `exp/exp61-guess-slip-constraint`
  - 提交:
    - `c9e7a98`: 在 `exp61` 基座上把 `guess/slip` 改为共享概率预算的耦合参数化，并补单测
  - 动机:
    - 验证“约束 `guess + slip <= 1`”是否能和 experiment 61 的 full target-exclusion 训练口径形成互补
    - 重点看它能否在保留 exp61 的 ranking 收益同时，缓解 exp61 原本的 `ACC/Brier/ECE` 代价
  - 做法:
    - 保持 exp61 的 target-exclusion 训练语义、缓存优化和其余主线结构不变
    - 只把最终 `guess/slip` 概率映射改成三元 softmax:
      - `guess = softmax([guess_logit, slip_logit, 0])[0]`
      - `slip = softmax([guess_logit, slip_logit, 0])[1]`
    - 从而显式保证 `guess >= 0`、`slip >= 0` 且 `guess + slip <= 1`
  - 验证:
    - 远端单测 `python -m unittest tests.test_decoupled_cdm` 通过
    - `epochs=1, max_rows=2000` 的 target-exclusion smoke 已跑通
    - 正式 `seed=2024` 训练后，test split 上:
      - `guess_plus_slip_mean = 0.124655`
      - `guess_plus_slip_max = 0.998527`
      - `guess_plus_slip_p95 = 0.604326`
      - `ratio(guess_plus_slip > 1) = 0.0`
  - 单 seed 结果:
    - `seed=2024`, `best_epoch=186`:
      - `AUC 0.764602`
      - `ACC 0.726275`
      - `RMSE 0.429523`
      - `Brier 0.184490`
      - `ECE 0.055939`
  - 相对 exp61 优化版同 seed (`exp61_opt_seed2024`):
    - `AUC -0.000656`
    - `ACC -0.003368`
    - `RMSE +0.001579`
    - `Brier +0.001354`
    - `ECE +0.005701`
  - 结论:
    - 在 exp61 口径上，这个约束确实修掉了输出层语义违例，但没有和 target-exclusion 形成互补，反而让 `AUC/ACC/RMSE/Brier/ECE` 同时更差
    - 不继续在 exp61 基座上扩 seed
    - 若后续还要推进 constrained `guess/slip`，应把重点放在“如何补回表达能力或训练补偿”上，而不是简单叠加到已有 ranking-oriented 训练口径；当前这条组合不作为主线候选

- 实验 64: decoupled non-cognitive budget and guess/slip split
  - 分支: `exp/decoupled-gs-budget`
  - 提交:
    - `267886a`: 增加可切换的 `guess/slip` 概率参数化、补回分布诊断脚本并补单测
  - 动机:
    - 复访实验 62 的失败机制，验证问题是否主要来自“三元 softmax 把非认知总预算和 guess/slip 分配比例绑死”
    - 重点看在继续满足 `guess + slip <= 1` 的前提下，把“总量”和“分配”解耦后，能否补回实验 62 丢失的 `ACC/Brier/ECE`
  - 做法:
    - 保持实验 51 主线结构、训练协议和 `guess/slip` 上游输入不变
    - 把最终概率映射扩成可切换的 `gs_probability_mode`
    - 新增的 `budget_split_sigmoid` 形式为:
      - `m = sigmoid(budget_logit)`
      - `r = sigmoid(split_logit)`
      - `guess = m * r`
      - `slip = m * (1 - r)`
    - 从而显式保证 `guess >= 0`、`slip >= 0` 且 `guess + slip = m <= 1`
  - 验证:
    - 远端单测 `python -m unittest tests.test_decoupled_cdm` 通过
    - `epochs=1, max_rows=2000` 的 smoke 已跑通
    - 正式 `seed=2024` 训练后，test split 上:
      - `guess_mean = 0.032703`
      - `slip_mean = 0.106389`
      - `guess_plus_slip_mean = 0.139092`
      - `guess_plus_slip_max = 0.999999`
      - `guess_plus_slip_p95 = 0.946891`
      - `ratio(guess_plus_slip > 1) = 0.0`
    - 对照同口径 test split:
      - 实验 51 `guess_plus_slip_mean = 0.244109`
      - 实验 62 `guess_plus_slip_mean = 0.101998`
  - 单 seed 结果:
    - `seed=2024`, `best_epoch=182`:
      - `AUC 0.756794`
      - `ACC 0.727074`
      - `RMSE 0.432131`
      - `Brier 0.186737`
      - `ECE 0.054287`
  - 相对实验 62 同 seed:
    - `AUC -0.007872`
    - `ACC +0.000362`
    - `RMSE +0.003205`
    - `Brier +0.002759`
    - `ECE +0.000136`
  - 相对实验 51 同 seed baseline:
    - `AUC -0.007257`
    - `ACC -0.001598`
    - `RMSE +0.003836`
    - `Brier +0.003300`
    - `ECE +0.003622`
  - 结论:
    - 这次改动确实把 `guess/slip` 总预算从实验 62 的 `0.1020` 拉回到 `0.1391`，说明“共享 softmax 过度压缩非认知总量”这个机制判断并不是空的；但它没有把预算补回到实验 51 的量级，也没有把 overall 指标救回来
    - 不继续扩 seed
    - 若后续再访 constrained `guess/slip`，应优先考虑更直接补回总预算表达力的方案，例如显式 `null` 通道建模或额外训练补偿，而不是停留在当前这版两头 `sigmoid` 的 budget/split 重参数化；当前这条 follow-up 不作为主线候选

- 实验 65: uncertainty-conditioned constrained non-cognitive mixture
  - 分支: `exp/uncertainty-conditioned-gs-budget`
  - 提交:
    - `6f0c658`: 增加 uncertainty-conditioned `budget + fallback` 非认知 mixture、补回分布诊断脚本并补单测
  - 动机:
    - 在实验 64 的基础上再往前走一级，不再只改 `guess/slip` 参数化
    - 重点验证“把 constrained non-cognitive 分支直接重写成显式 `budget + fallback` mixture，并让 budget 读取 coverage / concept_count / dispersion / target history 统计”后，能否补回实验 62/64 丢掉的表达力
  - 做法:
    - 保持实验 51 主线结构与训练协议不变
    - 将最终非认知分支改写为:
      - `m = sigmoid(budget_logit)`
      - `r = sigmoid(fallback_logit)`
      - `guess = m * r`
      - `slip = m * (1 - r)`
      - `p = (1 - m) * p_cog + m * r`
    - 与实验 64 不同，`budget_logit` 不再只由单一旧 logit 承担，而是用 `guess_logit + slip_logit` 作为 base，再叠加读取以下特征的 residual:
      - `difficulty`
      - `concept_count`
      - `coverage`
      - `dispersion`
      - 目标题相关概念的 `mean_accuracy / mean_log_attempts`
      - `cognitive_uncertainty`
    - `fallback_logit` 同样使用 `guess_logit - slip_logit` 的 base 加 uncertainty-conditioned residual
  - 验证:
    - 远端单测 `python -m unittest tests.test_decoupled_cdm` 通过
    - `epochs=1, max_rows=2000` 的 smoke 已跑通
    - smoke `seed=2024` test split:
      - `guess_plus_slip_mean = 0.502403`
      - `guess_plus_slip_p95 = 0.852836`
      - `ratio(guess_plus_slip > 1) = 0.0`
    - 正式 `seed=2024` 训练后，test split 上:
      - `guess_mean = 0.162202`
      - `slip_mean = 0.133380`
      - `guess_plus_slip_mean = 0.295582`
      - `guess_plus_slip_max = 0.998884`
      - `guess_plus_slip_p95 = 0.905642`
      - `ratio(guess_plus_slip > 1) = 0.0`
    - 对照同口径 test split:
      - 实验 51 `guess_plus_slip_mean = 0.244109`
      - 实验 62 `guess_plus_slip_mean = 0.101998`
      - 实验 64 `guess_plus_slip_mean = 0.139092`
  - 单 seed 结果:
    - `seed=2024`, `best_epoch=158`:
      - `AUC 0.758493`
      - `ACC 0.727245`
      - `RMSE 0.431253`
      - `Brier 0.185979`
      - `ECE 0.056210`
  - 相对实验 64 同 seed:
    - `AUC +0.001699`
    - `ACC +0.000171`
    - `RMSE -0.000878`
    - `Brier -0.000758`
    - `ECE +0.001923`
  - 相对实验 51 同 seed baseline:
    - `AUC -0.005558`
    - `ACC -0.001427`
    - `RMSE +0.002958`
    - `Brier +0.002542`
    - `ECE +0.005545`
  - 切片:
    - `concept_count=4+`:
      - 实验 51 `AUC 0.739185`, `ACC 0.674931`, `RMSE 0.459865`, `ECE 0.082999`
      - 实验 65 `AUC 0.740835`, `ACC 0.680441`, `RMSE 0.459807`, `ECE 0.079169`
    - `none_seen`:
      - 实验 51 `AUC 0.812949`, `ACC 0.805187`, `RMSE 0.376444`, `ECE 0.109546`
      - 实验 65 `AUC 0.794008`, `ACC 0.766586`, `RMSE 0.397493`, `ECE 0.147647`
  - 结论:
    - 这次 stronger 方案确实把 constrained non-cognitive 总预算补回到比实验 51 更高的区间，也优于实验 62/64 的“预算被压瘪”形态；相对实验 64 也带来了小幅 recovery
    - 但它仍没有形成 enough overall recovery，更关键的是把 `none_seen` 显著做坏了；`concept_count=4+` 的改善也不足以支撑继续做 rescue sweep
    - 不继续扩 seed，也不进入 rescue sweep；若后续还要继续探索更强一级 constrained non-cognitive 模块，重点应转向“为 fallback 分支引入更显式的可解释状态或 expert routing”，而不是继续只围绕 scalar budget 做增强

- 实验 66: evidence-aware TKC propagation
  - 分支: `exp/evidence-aware-tkc`
  - 做法:
    - 把显式历史统计从 readout residual 前移到 propagation 主干，构造逐 `student-concept` 的 `attempt/correct/incorrect/accuracy/log_attempt/seen` 特征
    - 分别注入 `behavior fusion gate`、`TKC behavior vs graph prior` gate、`student-level TKC/UKC fusion` gate，并补了可拆分的子开关
    - 数据侧新增真实 `student_exercise_count_tensor`，保留重复作答次数
  - 工程验证:
    - 远端单测 `python -m unittest tests.test_hetero_propagation tests.test_history_visibility tests.test_training_modes tests.test_decoupled_cdm` 通过
  - 结果:
    - full 版本 `seed=2024`: `AUC 0.762905`, `ACC 0.729300`, `RMSE 0.427547`, `Brier 0.182796`, `ECE 0.043839`
    - 拆分后最强单 seed 是 `behavior-only`: `AUC 0.765136`, `ACC 0.732193`, `RMSE 0.426751`, `Brier 0.182117`, `ECE 0.046792`
    - 但多 seed 后没有复现 clean win:
      - 直接替换式 `behavior-only` 三 seed 均值: `AUC 0.760512`, `ACC 0.728298`, `RMSE 0.429400`, `Brier 0.184390`, `ECE 0.050343`
      - residual 化 `behavior-only` 三 seed 均值: `AUC 0.761936`, `ACC 0.729047`, `RMSE 0.429112`, `Brier 0.184139`, `ECE 0.053518`
    - 相对实验 51 当前主线三 seed 均值，两版都没有形成 overall 正向；residual 版虽然更稳，但 `ECE` 还更差
  - 结论:
    - 这条线在机制上成立，且可确认有效信号主要来自 `correct/incorrect behavior fusion gate`
    - `reliability` 和 `student fusion` 不是主增益源，单独或组合开启都没有形成稳定提升
    - 当前不进入主线候选；若后续再访，应只做更局部的 behavior gate 调节，例如只改 bias / temperature，或只作用于 `concept_count>=2` / low-evidence concept

## 旧口径的历史参考

下面这些实验只说明某类信号曾经出现过，不能直接当作当前主线结论。

- 实验 16: `TKC` item-aware attention + `UKC` 全局 mean
  - 说明“题目条件局部选择性”在旧主线下曾有信号
- 实验 17: coverage-aware `UKC` 邻域约束
  - 说明 `UKC` 全局噪声在旧口径下确实是问题
- 实验 18: coverage-aware 全局融合
  - 与后续主线融合语义重叠，不单独优先
- 实验 19: 只对全局 `UKC` 做 coverage gate
  - 保留为旧口径次优参考即可

## 近线 follow-up

这部分只保留正文之外仍值得额外记住的元判断，不再重复逐实验结论。

### 实验 28. `guess/slip` 显式引入题目难度特征

- 做法: 在 `guess/slip` 条件输入中拼入 `difficulty.detach()`
- 结果: 三 seed 下 AUC 基本持平，`ECE/Brier` 有均值改善
- 判断: 可记为候选，但增量不够大，不是新主线

### 诊断 1. 实验 33 主线的 test prediction slices

- 多知识点题明显更难，且随 `concept_count` 增长单调退化
- `none_seen` 排序不差，但校准很差，更像系统性低估
- 中等历史长度、中等历史正确率样本也偏弱
- 当前 split 是按学生随机抽样，不是严格时间切分，因此不优先做 recency-aware student state
- 结论:
  - 下一步应优先针对“多知识点题表示/读出偏弱”做 targeted 修补
  - `none_seen` 更适合作为后续单独校准问题，而不是当前第一优先结构问题
  - exact-3 aggressive residual/readout 已单 seed 验证到头，不再继续深挖同类结构

### 主题归纳 1. `none_seen` 与校准

- 实验 47 已说明: `none_seen` 可以单独当校准问题做，但“对所有样本共享的 final-logit bias”过于粗糙
- 即便输入里放入 target coverage / `concept_count` / `difficulty`，模型也可能拿 overall ECE 去换 `none_seen` 自身校准
- 如果后续还要回到 `none_seen`，优先考虑更局部、更显式的触发方式，而不是继续扩这类全局共享 bias

### 主题归纳 2. 实验 51 读出侧后续

- “可解释 gate + 专家 residual” 这条线是成立的，说明读出侧适度增容本身有真实信号
- 但实验 52/53/54 共同说明: 无论是更硬的 selective routing、更软的 gate regularization，还是 local-first 的 readout 复访，都还没有形成比实验 51 更强的 clean overall 增益
- 因此若后续还要继续挖实验 51，前提应是出现更明确的 targeted slice 假设或更局部的引导目标，而不是默认继续扫 routing / local mastery 近邻变体

### 主题归纳 3. propagation 与目标层 tweak

- 实验 55/56/57 共同说明: difficulty 前移、直接叠 ranking loss、single-graph multi-hop propagation 都更像轻度改变排序偏好，而不是 clean overall 增益
- 实验 66 进一步说明: 即便显式历史统计前移到 propagation 主干，若以共享 gate 方式大范围注入，也更容易得到脆弱的单 seed 正信号，而不是稳定提点
- 这些方向的常见模式是 `AUC` 有时略正，但 `ACC/RMSE/Brier/ECE` 更容易回撤
- 因此 propagation 侧与目标层 tweak 的优先级应继续下调；若再回到这些方向，前提应是已有更明确的局部 slice 假设，或已有更强的结构正向底座

## 默认下一步

通用协作、运行与分支规则沿用 [docs/session_bootstrap.md](./session_bootstrap.md)；这里仅补充历史台账导出的默认优先级:

1. 仍从当前 `master` 主线出发；新假设优先单改一个结构因素，单次成立后再补 `2-3` 个 seed。
2. 当前处于单因素边际收益放缓的平台期；结构主线仍优先解决多知识点题表示/读出偏弱，`none_seen` 默认视为后续单独处理的校准问题。
3. 允许少量测试“已各自成立”的正交组合，但默认只测最强的 `1-2` 组候选，不做组合爆炸；优先考虑结构改动和训练协议这类职责分离的组合。
4. 允许探索更大一级、真正改变表示瓶颈的模块；若单次结果表现为“overall 未过门槛但目标 slice 有明显改善”，可额外允许一次很小的 rescue sweep；普通 sidecar / residual 默认不进入这类 sweep。
5. 若继续沿实验 51 的 readout expert 底座推进，默认保留原版 full-trigger 作为基座；只有在出现更明确的 targeted slice 假设或更局部的引导目标时，才再访这条线。
6. 对已经系统复访但未形成 clean overall gain 的 readout / propagation / ranking-loss 路线，默认不再高优先级继续；只有在出现明确新假设、且机制上明显区别于已失败版本时，才考虑重开。
7. 若用户明确要继续训练协议优化，当前优先候选是实验 37 与实验 61 两条支线；否则默认优先继续模型结构改动。
8. 判断是否值得继续时，默认主看 `AUC/ACC`，并优先寻找至少 `1e-3` 量级的改善；`RMSE/Brier/ECE` 与分桶校准默认只用于判断副作用。
