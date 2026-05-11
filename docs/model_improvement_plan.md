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
- 从这份归档的实验视角看，它对应实验 70 主线:
  - 以实验 34 为底座
  - 吸收实验 49 的 history-carrier pairwise interaction residual
  - 再吸收实验 51 的 interpretable readout expert residual
  - 再吸收实验 70 的 student-conditioned UKC `none_seen` readout sidecar
- 这里不再重复维护与 `session_bootstrap.md` 等价的数据、图、超参数和 adapter 开关清单；需要确认默认运行口径时，优先回看 `session_bootstrap.md`
- 当前主线三 seed 参考均值:
  - `test_auc = 0.765517`
  - `test_acc = 0.729104`
  - `test_rmse = 0.427350`
  - `test_brier = 0.182628`
  - `test_ece = 0.049044`
- 当前结果报告默认主看 `AUC/ACC`
- `RMSE/Brier/ECE/分桶校准` 默认作为次要指标
- 若目标是推进主线，默认希望 `AUC` 或 `ACC` 的改善至少达到 `1e-3` 量级；达不到时，通常需要很强的 slice 证据才值得继续

- 当前已吸收的最新结构更新:
  - 实验 70: student-conditioned UKC `none_seen` readout sidecar 已进入 `master` 默认主线；三 seed 相对实验 51 主线均值 `AUC +0.001628`，且 `ACC/RMSE/Brier/ECE` 均值也小幅正向

- 当前正向支线候选:
  - 实验 37
    - branch: `exp/training-modes`
    - 判断: 它仍是当前更强的 calibration-oriented 训练协议候选，但在 `AUC/ACC` 上仍弱于当前主线，不作为默认 `master` 训练口径
    - 补充: 这条线属于纯训练工程优化，单次运行耗时显著高于当前默认 full-batch 口径；在模型结构仍需继续迭代时，暂不优先合入主线
    - 详细指标见下文“实验 37”
  - 实验 61
    - branch: `exp/full-target-exclusion-opt`
    - 判断: 它仍是 ranking-oriented target-exclusion 训练候选，工程优化后运行成本已从“明显过高”降到“可接受”，但仍不作为默认 `master` 训练口径
    - 补充: `2026-05-02` 复跑三 seed 后，它相对实验 51 有稳定 `AUC` 正向；实验 71 已验证它与实验 70 的直接组合不是 clean win，不默认继续扩组合 seed
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
  - 实验 67: learned multi-concept exercise attribution 机制上区别于实验 24 的静态分摊，但单 seed overall 明显弱于实验 51，且多知识点/`none_seen` 切片没有 clean win，不继续扩 seed
  - 实验 68: scale-preserving / high-count-only / incorrect-only attribution rescue 都没有恢复到实验 51；最强只是 `ECE` 小幅改善但 `AUC/ACC` 仍回撤，不继续沿 attribution 主聚合替换路线扩线
  - 实验 69: student-conditioned UKC imputation 没有解决 `none_seen` 校准，反而显著做坏 `none_seen` 的 `ACC/RMSE/ECE`，不扩 seed
  - 实验 71: 实验 70 主线 + 实验 61 target-exclusion 训练口径只带来单 seed `AUC +0.000950`，但 `RMSE/Brier/ECE` 回撤，不扩 seed
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

- 实验 62-65: constrained `guess/slip` 系列
  - 分支:
    - `exp/guess-slip-diagnostics`
    - `exp/exp61-guess-slip-constraint`
    - `exp/decoupled-gs-budget`
    - `exp/uncertainty-conditioned-gs-budget`
  - 核心发现:
    - 旧 `guess/slip` 使用独立 `sigmoid`，实际 checkpoint 中大量样本出现 `guess + slip > 1`，会导致 `dp / dcognitive < 0` 的语义反转
    - 三元 softmax、budget/split sigmoid、uncertainty-conditioned mixture 都能把 `ratio(guess_plus_slip > 1)` 压到 `0`
    - 但约束后整体指标没有恢复；越强的 non-cognitive budget 补偿越容易伤害 `none_seen`
  - 代表结果:
    - 实验 62 三元 softmax 相对实验 51 同 seed: `AUC +0.000615`, `ACC -0.001960`, `RMSE +0.000631`, `Brier +0.000541`, `ECE +0.003486`
    - 实验 63 叠到 exp61 后相对 exp61 opt 同 seed: `AUC -0.000656`, `ACC -0.003368`, `RMSE +0.001579`, `Brier +0.001354`, `ECE +0.005701`
    - 实验 64 budget/split 相对实验 51 同 seed: `AUC -0.007257`, `ACC -0.001598`, `RMSE +0.003836`, `Brier +0.003300`, `ECE +0.003622`
    - 实验 65 uncertainty-conditioned mixture 相对实验 51 同 seed: `AUC -0.005558`, `ACC -0.001427`, `RMSE +0.002958`, `Brier +0.002542`, `ECE +0.005545`
  - 结论:
    - 语义诊断成立，诊断工具值得保留；但这些参数化改动不作为主线候选
    - 后续若再访，不能只继续调 scalar budget，应引入更显式的可解释状态、expert routing 或配套训练补偿

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

- 实验 67-68: learned multi-concept exercise attribution
  - 分支: `exp/learned-exercise-attribution`
  - 做法: 在 propagation 的 correct/incorrect exercise message 聚合处，用 `student-conditioned / response-conditioned / history-conditioned` scorer 为多知识点题动态归因；后续 rescue 测了 scale-preserving、`concept_count=4+`、incorrect-only 版本
  - 代表结果:
    - 原版相对实验 51 同 seed: `AUC -0.005409`, `ACC -0.005043`, `RMSE +0.002845`, `Brier +0.002444`, `ECE +0.001614`
    - scale-preserving rescue 相对实验 51 同 seed: `AUC -0.003315`, `ACC -0.000837`, `RMSE +0.000937`, `Brier +0.000803`, `ECE -0.002072`
    - `min4` 与 incorrect-only 也未恢复主线；`4+` 只有小幅 mixed signal，样本数 `363`，不足以抵消 overall 回撤
  - 结论:
    - 动态归因机制区别于实验 24 的静态分摊，但直接替换 propagation 主聚合会削弱行为证据，并显著伤害 `none_seen` 校准
    - 不继续沿 attribution 主聚合替换路线 rescue；若以后必须复访，只应作为 additive residual / calibration sidecar，而不是替换主聚合

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

- 实验 70: student-conditioned UKC readout sidecar
  - 分支: `exp/student-conditioned-ukc-readout-sidecar`
  - 提交:
    - `9b73a37`: 增加 `--student-conditioned-ukc-readout-residual`
  - 动机:
    - 实验 69 说明直接替换 UKC 主状态会放大 `none_seen` 低估；但 `none_seen` 仍能从学生条件化的图邻接 TKC 证据中获益
    - 因此改成只读 readout sidecar: 不改 `TKC/UKC/student_state` 主状态，只在 `none_seen` 且存在图邻接 TKC 证据时，加一个 zero-init cognitive-logit residual
    - residual 输入包含 `student_state/q_repr`、图邻接 TKC 聚合得到的 student-conditioned UKC summary、静态 UKC summary、二者差异、difficulty、concept count、coverage 与邻接证据统计；输入均 detached
    - 关键修正: 当前 full-batch 训练历史包含目标本身，train 中真实 `none_seen=0`，sidecar 会完全学不到。因此训练态下仅对该 sidecar 使用 `student_concept_attempt_counts - target_q` 的 leave-target-out coverage proxy；评估态仍使用真实 train-history coverage
    - 工程上按 target chunk 计算 sidecar，避免全训练集一次性展开目标题 graph rows 导致 OOM
  - 工程验证:
    - 远端 `python -m unittest tests.test_decoupled_cdm tests.test_hetero_propagation tests.test_history_visibility tests.test_training_modes` 通过
    - 训练触发诊断: train leave-target-out proxy 下 `none_seen_proxy=5974`，其中 `5167` 个有邻接 TKC 证据；valid/test 的 eligible rate 约 `86%`
  - 结果:
    - `seed=2024`: `AUC 0.765368`, `ACC 0.728558`, `RMSE 0.427880`, `Brier 0.183082`, `ECE 0.052010`
    - `seed=2025`: `AUC 0.766528`, `ACC 0.729605`, `RMSE 0.426968`, `Brier 0.182302`, `ECE 0.049967`
    - `seed=2026`: `AUC 0.764655`, `ACC 0.729148`, `RMSE 0.427201`, `Brier 0.182501`, `ECE 0.045154`
  - 三 seed 均值:
    - `AUC 0.765517`
    - `ACC 0.729104`
    - `RMSE 0.427350`
    - `Brier 0.182628`
    - `ECE 0.049044`
  - 相对实验 51 当前主线三 seed 均值:
    - `AUC +0.001628`
    - `ACC +0.000153`
    - `RMSE -0.000602`
    - `Brier -0.000515`
    - `ECE -0.000355`
  - 切片观察:
    - `none_seen` 三 seed 均值: `AUC 0.817951`, `ACC 0.823482`, `RMSE 0.358966`, `ECE 0.062414`
    - `seed=2024` 相对实验 51 的 `none_seen`: `AUC +0.001171`, `ACC +0.013269`, `RMSE -0.015090`, `ECE -0.040188`
    - `concept_count=4+` 三 seed 均值: `AUC 0.746527`, `ACC 0.689624`, `RMSE 0.458434`, `ECE 0.089720`
    - `4+` 多知识点不是纯 clean win: AUC/ACC/RMSE 有正向，但 `seed=2024` 的 ECE 比实验 51 更差；样本数仅 `363`，后续不应为它单独扩大复杂度
  - 结论:
    - 这是当前第一条把 `none_seen` 学生条件化信号稳定转成三 seed overall 正收益的结构路线
    - 和实验 47 的区别在于它不是全局共享 final-logit bias；和实验 69 的区别在于它不替换 UKC 主状态，只作为 target-local readout sidecar
    - 已合入 `master` 并成为当前默认主线；后续探索默认从实验 70 口径出发，实验 70 + 实验 61 target-exclusion 的直接组合已由实验 71 判定为不 clean

- 实验 71: exp70 + full target-excluded training combo
  - 分支: `exp/exp70-target-exclusion-combo`
  - 提交:
    - `d58eec8`: 在实验 70 主线底座上合入实验 61 的 `--exclude-target-from-train-history` 训练路径，并让 target-exclusion reference 复用 sidecar 所需的只读 TKC/UKC states
  - 动机:
    - 实验 70 是当前最强 non-ID-aware 结构主线，实验 61 是 ranking-oriented 训练候选；二者职责相对分离，适合作为少量正交组合验证
  - 工程验证:
    - 远端 `python -m unittest tests.test_decoupled_cdm tests.test_hetero_propagation tests.test_history_visibility tests.test_training_modes` 通过
    - 远端 `epochs=1, max_rows=2000` smoke 通过
  - 结果:
    - `seed=2024`, `best_epoch=196`:
      - `AUC 0.766318`
      - `ACC 0.729129`
      - `RMSE 0.428091`
      - `Brier 0.183262`
      - `ECE 0.055403`
  - 相对实验 70 同 seed:
    - `AUC +0.000950`
    - `ACC +0.000571`
    - `RMSE +0.000211`
    - `Brier +0.000180`
    - `ECE +0.003393`
  - 相对实验 61 opt 同 seed:
    - `AUC +0.001060`
    - `ACC -0.000514`
    - `RMSE +0.000146`
    - `Brier +0.000125`
    - `ECE +0.005165`
  - 切片观察:
    - `none_seen`: `AUC 0.815324`, `ACC 0.823884`, `RMSE 0.357460`, `Brier 0.127778`, `ECE 0.069456`
    - 相对实验 70 同 seed，`none_seen` 的 `AUC/ACC/RMSE/Brier` 小幅正向，但 `ECE` 基本持平略差
    - `concept_count=4+`: `AUC 0.738085`, `ACC 0.663912`, `RMSE 0.461673`, `Brier 0.213142`, `ECE 0.087995`
    - 相对实验 70 同 seed，`4+` 的 `ECE` 改善，但 `AUC/ACC/RMSE/Brier` 明显回撤；该 slice 样本数仍只有 `363`
  - 结论:
    - 组合确实继续把排序往上推，但 `AUC` 增量未达到默认 `1e-3` 门槛，且整体 `RMSE/Brier/ECE` 副作用比收益更清楚
    - 不扩 seed，不把 target-exclusion 训练口径叠到实验 70 默认主线
    - 若未来明确只追求 `AUC`，可把它作为 ranking-oriented ablation；若继续推进主线，优先寻找新的结构假设，而不是继续扩实验 61 组合

- 实验 72: representation bottleneck probes
  - 分支: `exp/representation-bottleneck-modules`
  - 提交:
    - `a8fec59`: 在当前代码上补回两个显式可选模块，`q_conditioned_local_mastery_adapter` 与 `target_conditioned_student_context_adapter`；默认不开，不改变 `master` 默认行为
  - 动机:
    - 诊断 2/3 表明继续堆 final-logit 小 residual 很难冲到 `0.77+`
    - 本轮先验证两个“更大一级但仍不改数据/图/训练口径”的表示瓶颈改动:
      - 用 `q-conditioned local mastery` 替代实验 51 expert，测试 `B49 + q-local + exp70 none_seen sidecar` 的替代吸收链
      - 在当前实验 70 主线上加入 `target-conditioned student context`，测试把目标题相关局部 `TKC/UKC` context 写回主 readout state 是否能突破当前学生状态瓶颈
  - 工程验证:
    - 远端 `python -m unittest tests.test_decoupled_cdm tests.test_hetero_propagation tests.test_history_visibility tests.test_training_modes` 通过
    - 远端 `epochs=1, max_rows=2000` smoke 通过
  - `B49 + q-conditioned local mastery + exp70 sidecar`，关闭实验 51 expert:
    - `seed=2024`, `best_epoch=173`:
      - `AUC 0.764979`
      - `ACC 0.730157`
      - `RMSE 0.427213`
      - `Brier 0.182511`
      - `ECE 0.050572`
    - 相对实验 70 同 seed:
      - `AUC -0.000389`
      - `ACC +0.001599`
      - `RMSE -0.000667`
      - `Brier -0.000571`
      - `ECE -0.001438`
    - 切片:
      - `concept_count=4+`: `AUC 0.733136`, `ACC 0.677686`, `RMSE 0.463660`, `ECE 0.099115`
      - `none_seen`: `AUC 0.813291`, `ACC 0.825090`, `RMSE 0.356982`, `ECE 0.062702`
    - 判断:
      - 它是误差/校准型替代链，不是排序突破；`AUC` 未超过当前主线同 seed，且目标 `4+` slice 不强
      - 不扩 seed，不用它替代实验 51 expert
  - `target-conditioned student context` 叠加当前实验 70 主线:
    - `seed=2024`, `best_epoch=153`:
      - `AUC 0.758167`
      - `ACC 0.727074`
      - `RMSE 0.430651`
      - `Brier 0.185460`
      - `ECE 0.050105`
    - 相对实验 70 同 seed:
      - `AUC -0.007201`
      - `ACC -0.001484`
      - `RMSE +0.002771`
      - `Brier +0.002378`
      - `ECE -0.001905`
    - 判断:
      - 写回主 readout state 的版本仍然明显扰动排序与误差，即便加上实验 70 sidecar 也没有被救回来
      - 不扩 seed，也不做 min-count rescue
  - 关于 recency / sequence encoder:
    - 当前 ordered split 文件只有 `stu_id/exer_id/cpt_seq/label`，没有显式时间戳；可用的只是行顺序
    - 当前约定又要求 `valid/test` 复用 `train` 行为历史，且当前 split 不是严格时间切分
    - 因此直接做 recency-aware student state 会同时改变数据语义与历史可见性口径，当前不作为同一轮结构验证继续推进
  - 结论:
    - 本轮两个不改数据口径的大模块探针都没有提供冲 `0.77+` 的排序信号
    - 后续如果继续做表示瓶颈级改动，应先明确是否允许改变数据/历史可见性口径；否则优先不要再做“写回主状态”的 local context 变体

- 实验 73: current mainline protocol sweep
  - 分支: `exp/mainline-protocol-sweep`
  - 输出: `results/mainline_protocol_sweep/`
  - 口径:
    - 不改当前实验 70 主线结构，不改数据、图、`concept_dim`、`gs_mode`
    - 先用 `seed=2024` 扫 full-batch 学习率、patience/scheduler 组合与 recompute minibatch
    - 只给接近门槛或有明显 `ACC` 信号的配置补 `seed=2025/2026`
  - `seed=2024` 粗扫:
    - 默认 sanity 复现当前主线: `AUC 0.765368`, `ACC 0.728558`, `RMSE 0.427880`, `Brier 0.183082`, `ECE 0.052010`
    - `lr=3e-4`: `AUC 0.757601`, `ECE 0.038516`; 校准变好但明显欠排序
    - `lr=5e-4`: `AUC 0.766028`, `ACC 0.728406`, `ECE 0.057126`; AUC 小正但校准明显回撤
    - `lr=7e-4`: `AUC 0.765779`, `ACC 0.727283`, `ECE 0.061719`; 不如后续长 patience 版本
    - `lr=1.5e-3`: `AUC 0.766310`, `ACC 0.726864`, `ECE 0.057815`; 单 seed AUC 接近门槛但 ACC/ECE 副作用
    - `lr=2e-3`: `AUC 0.765703`, `ACC 0.726579`, `ECE 0.060059`; 高 lr 不继续提升
    - `lr=1e-3, early_stop=10/20, scheduler_patience=5`: 都选到 `epoch=184`，`AUC 0.765419`, `ACC 0.729529`
    - `lr=7e-4, early_stop=20, scheduler_patience=5`: `AUC 0.765558`, `ACC 0.730518`
    - `lr=1.5e-3, early_stop=20, scheduler_patience=5`: `AUC 0.766323`, `ACC 0.728273`
    - `recompute_minibatch bs=8192 lr=1e-4`: `AUC 0.762270`, `ECE 0.043387`; 仍是校准/误差型，不是 AUC 路线
    - `recompute_minibatch bs=8192 lr=3e-4`: `AUC 0.760713`
    - `recompute_minibatch bs=4096 lr=1e-4`: `AUC 0.761224`; 成本更高且无收益
  - 候选扩 seed:
    - `lr=1.5e-3, early_stop=20, scheduler_patience=5`:
      - `seed=2024`: `AUC +0.000955`, `ACC -0.000285`, `ECE +0.003699`
      - `seed=2025`: `AUC -0.001311`, `ACC -0.001675`, `ECE -0.000867`
      - `seed=2026`: `AUC -0.002270`, `ACC +0.001142`, `ECE -0.000027`
      - 三 seed 均值差: `AUC -0.000875`, `ACC -0.000273`, `RMSE +0.000329`, `Brier +0.000281`, `ECE +0.000935`
      - 判断: 单 seed AUC 小涨不稳定，不作为候选
    - `lr=7e-4, early_stop=20, scheduler_patience=5`:
      - `seed=2024`: `AUC +0.000190`, `ACC +0.001960`, `RMSE -0.000003`, `Brier -0.000003`, `ECE +0.001766`
      - `seed=2025`: `AUC +0.000505`, `ACC -0.000057`, `RMSE +0.000499`, `Brier +0.000426`, `ECE +0.004079`
      - `seed=2026`: `AUC +0.002088`, `ACC +0.002036`, `RMSE -0.000728`, `Brier -0.000622`, `ECE +0.002772`
      - 三 seed 均值差: `AUC +0.000928`, `ACC +0.001313`, `RMSE -0.000077`, `Brier -0.000066`, `ECE +0.002872`
      - 三 seed 均值约: `AUC 0.766445`, `ACC 0.730417`, `RMSE 0.427273`, `Brier 0.182562`, `ECE 0.051916`
      - 判断: 这是当前最好的训练口径候选，稳定小幅改善 `AUC/ACC`，但达不到 `0.77+`，且校准变差
  - 邻域 refine:
    - `lr=6e-4, early_stop=20, scheduler_patience=5`, `seed=2024`: `AUC +0.000351`, `ACC +0.002036`, `ECE +0.001995`
    - `lr=8e-4, early_stop=20, scheduler_patience=5`, `seed=2024`: `AUC -0.000456`, `ACC +0.001313`, `ECE +0.001521`
    - `lr=9e-4, early_stop=20, scheduler_patience=5`, `seed=2024`: `AUC -0.000097`, `ACC +0.000228`, `ECE +0.001380`
    - 判断: `6e-4/7e-4` 附近主要是 ACC 口径收益，AUC 没有更强局部峰
  - 结论:
    - 口径扫没有发现能把当前主线推到 `0.77+` 的训练配置
    - `lr=7e-4 + early_stop=20 + scheduler_patience=5` 可作为 accuracy/ranking-oriented 候选口径，但不是 clean 默认切换: `AUC/ACC` 小正，`ECE` 明显变差，训练更久
    - recompute minibatch 在当前实验 70 主线上仍不适合作为 AUC 推进路线；它最多是 calibration-oriented ablation

- 实验 74: `guess/slip` monotonic soft penalty
  - 分支: `exp/gs-monotonic-penalty`
  - 详情: [074_gs_monotonic_penalty.md](./experiments/074_gs_monotonic_penalty.md)
  - 三 seed 均值差: `AUC -0.000170`, `ACC +0.000565`, `RMSE -0.000012`, `Brier -0.000010`, `ECE +0.000200`
  - 关键失败原因: `1e-4` 在 `seed=2025/2026` 的 best checkpoint 几乎全量 `guess+slip>1`，软正则权重不足以稳定压住独立 sigmoid 退化解
  - 判断: 诊断脚本值得保留；`1e-4` 不作为主线候选，`1e-3` 因 AUC/ACC tradeoff 不扩 seed

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

### 诊断 2. 实验 51 底座压制交叉复验

- 动机:
  - 当前主线吸收链是实验 34 -> 49 -> 51 -> 70；近期很多语义更干净的 readout / propagation 结构都只带来 `0.001-0.002` 量级甚至更小的收益
  - 为验证是否存在“某个历史主线底座吸收后，导致后续语义干净结构难以发挥”，把两个在实验 51 底座上失败的结构放回更早底座做交叉复验
- 口径:
  - 训练口径仍为 ASSIST09 ordered + single propagation graph + `300 epoch` + `lr=1e-3` + `concept_dim=64`
  - 实验 49 底座保留 `high_concept_logit_adapter / pairwise_history_interaction_adapter / gs_difficulty_adapter`
  - 实验 51 底座在实验 49 基础上再打开 `interpretable_readout_expert_adapter`
  - 实验 70 的 `student_conditioned_ukc_readout_residual` 本轮不纳入矩阵，先隔离实验 51 这个嫌疑点
- `q-conditioned local mastery readout` 扩大复验:
  - 分支: `exp/q-conditioned-local-mastery-readout`
  - 输出: `results/base_suppression/q_local_mastery_matrix/`
  - 实验 34 底座 `seed=2024`: `AUC +0.000233`, `ACC +0.001808`, `RMSE -0.000747`, `Brier -0.000640`, `ECE +0.000123`
  - 实验 49 底座三 seed 差值:
    - `seed=2024`: `AUC +0.000942`, `ACC +0.000019`, `RMSE -0.000394`, `Brier -0.000337`, `ECE -0.000725`
    - `seed=2025`: `AUC +0.002195`, `ACC -0.000247`, `RMSE -0.000350`, `Brier -0.000300`, `ECE +0.001913`
    - `seed=2026`: `AUC +0.002430`, `ACC -0.000058`, `RMSE -0.000715`, `Brier -0.000612`, `ECE -0.000914`
    - 均值: `AUC +0.001856`, `ACC -0.000095`, `RMSE -0.000486`, `Brier -0.000416`, `ECE +0.000091`
  - 实验 51 底座三 seed 差值:
    - `seed=2024`: `AUC -0.002073`, `ACC -0.001998`, `RMSE +0.000278`, `Brier +0.000238`, `ECE -0.003823`
    - `seed=2025`: `AUC -0.001573`, `ACC -0.000438`, `RMSE +0.000115`, `Brier +0.000099`, `ECE -0.001287`
    - `seed=2026`: `AUC -0.000920`, `ACC -0.001503`, `RMSE +0.000182`, `Brier +0.000156`, `ECE -0.003454`
    - 均值: `AUC -0.001522`, `ACC -0.001313`, `RMSE +0.000192`, `Brier +0.000164`, `ECE -0.002855`
- `difficulty-weighted behavior propagation` 扩大复验:
  - 分支: `exp/difficulty-weighted-propagation`
  - 输出: `results/base_suppression/difficulty_weighted_matrix/`
  - 实验 34 底座 `seed=2024`: `AUC +0.000081`, `ACC +0.001731`, `RMSE +0.000034`, `Brier +0.000029`, `ECE -0.000443`
  - 实验 49 底座三 seed 均值: `AUC +0.000212`, `ACC +0.000881`, `RMSE -0.000620`, `Brier -0.000530`, `ECE -0.003367`
  - 实验 51 底座三 seed 均值: `AUC -0.000306`, `ACC -0.000926`, `RMSE +0.000609`, `Brier +0.000523`, `ECE +0.003129`
- 判断:
  - `q-conditioned local mastery` 已经形成强三 seed 反转: 在实验 49 底座上 `AUC` 三 seed 全正且均值达到 `+0.001856`，但在实验 51 底座上 `AUC` 三 seed 全负且均值为 `-0.001522`
  - `difficulty-weighted behavior propagation` 不是强 AUC 路线，但也呈现底座敏感: 在实验 49 底座上均值改善 `ACC/RMSE/Brier/ECE`，在实验 51 底座上均值则 `AUC/ACC/RMSE/Brier/ECE` 全部转坏
  - 因此“实验 51 full-trigger readout expert residual 可能压制部分后续语义干净结构”的判断，从单 seed 嫌疑升级为当前应默认纳入实验设计的风险
  - 当前证据不支持把实验 49 判为更强正式主线；实验 51 自身三 seed 仍有稳定 `AUC` 正向
  - 后续若新结构在最新主线上轻微负向但语义足够干净，应优先比较 `实验 49 底座` 与 `实验 51/70 底座` 的边际收益；只有在旧底座正向、新底座负向时，再考虑重设主线吸收顺序、隔离实验 51 residual、或做 distillation / residual isolation

### 诊断 3. 更早底座广覆盖扫描

- 动机:
  - 诊断 2 已证明实验 51 会压制部分后续结构；进一步验证是否还有更靠前的“错误底座”，或者是否存在只在更早底座才成立的结构
  - 本轮优先做 `seed=2024` 粗扫；只有出现强信号的格子才扩 seed
- 底座定义:
  - `B33-like`: 当前代码内置 `cognitive_difficulty_adapter`，不打开 `high_concept / gs_difficulty / pairwise / expert`
  - `B34`: `B33-like + high_concept_logit_adapter + gs_difficulty_adapter`
  - `B49`: `B34 + pairwise_history_interaction_adapter`
  - `B51`: `B49 + interpretable_readout_expert_adapter`
- `q-conditioned local mastery` 更早底座补点:
  - `B33-like`, `seed=2024`: `AUC +0.000655`, `ACC +0.000057`, `RMSE -0.000413`, `Brier -0.000356`, `ECE +0.000082`
  - 结合诊断 2: `B33/B34` 只是小正，`B49` 才放大成三 seed `AUC +0.001856`，`B51` 又反转成三 seed `AUC -0.001522`
  - 判断: 目前最清楚的链条不是“越早越好”，而是 `B49` 的 history carrier 给 local mastery 提供可发挥条件，实验 51 expert 又压住这条 readout 结构
- `difficulty-weighted behavior propagation` 更早底座补点:
  - `B33-like`, `seed=2024`: `AUC -0.000760`, `ACC +0.000228`, `RMSE +0.000276`, `Brier +0.000238`, `ECE -0.001231`
  - 结合诊断 2: `B49` 均值改善 `ACC/RMSE/Brier/ECE`，`B51` 均值全面回撤
  - 判断: 它也不是越早越好；主要说明实验 49 后 propagation-side scaling 至少不伤校准，而实验 51 后副作用放大
- `single-graph multi-hop propagation` 粗扫:
  - 分支: `exp/multi-hop-propagation`
  - `B33-like`: `AUC -0.000772`, `ACC +0.000628`, `RMSE +0.000416`, `Brier +0.000359`, `ECE +0.000310`
  - `B34`: `AUC +0.000241`, `ACC -0.000191`, `RMSE -0.000110`, `Brier -0.000094`, `ECE +0.000289`
  - `B49`: `AUC +0.000214`, `ACC -0.000285`, `RMSE -0.000279`, `Brier -0.000239`, `ECE -0.002063`
  - `B51`: `AUC +0.000187`, `ACC -0.002284`, `RMSE +0.000803`, `Brier +0.000687`, `ECE +0.004522`
  - 判断: 它在 `B34/B49/B51` 都只有极小 AUC 正向；实验 51 后副作用明显变大，但早底座也没有足够强的 clean gain，不扩 seed
- `qrepr-score residual` 粗扫:
  - 分支: `exp/qrepr-score-residual`
  - `B33-like`: `AUC +0.000296`, `ACC +0.001218`, `RMSE -0.000504`, `Brier -0.000434`, `ECE -0.000857`
  - `B34`: `AUC -0.000151`, `ACC +0.002036`, `RMSE -0.000985`, `Brier -0.000845`, `ECE -0.004865`
  - 判断: 这是误差/校准型小修补，不是 AUC 推进路线；不构成底座压制主证据
- `concept-conditioned propagation residual` 粗扫与扩 seed:
  - 分支: `exp/concept-conditioned-prop`
  - `B34`, `seed=2024`: `AUC +0.000109`, `ACC -0.000361`, `RMSE +0.000271`, `Brier +0.000233`, `ECE +0.002161`
  - `B33-like`, `seed=2024`: `AUC +0.001349`, `ACC +0.001998`, `RMSE -0.000924`, `Brier -0.000793`, `ECE -0.001454`
  - `B33-like` 扩三 seed 后均值: `AUC -0.000521`, `ACC -0.000070`, `RMSE -0.000410`, `Brier -0.000352`, `ECE -0.002821`
  - 判断: 单 seed 早底座强正没有复现；它仍是校准/误差折中，不是稳定可回退底座
- 显式历史统计 residual 与 local evidence:
  - 分支: `exp/local-concept-evidence-head`
  - `history-concept-stats`, `B33-like`: `AUC -0.001419`, `ACC +0.001123`, `RMSE -0.000712`, `Brier -0.000612`, `ECE -0.003306`
  - `history-concept-stats`, `B34`: `AUC -0.000658`, `ACC +0.002474`, `RMSE -0.000410`, `Brier -0.000352`, `ECE +0.000528`
  - `local-concept-evidence-head`, `B33-like`: 正式长训在 valid 阶段触发 BCE 输入越界的 CUDA device-side assert；不继续扫
  - 判断: 原始显式统计 residual 仍是 ACC/误差/校准折中，pairwise history carrier 的成功不应被简化为“统计 residual 越早越好”
- `student-pairwise-ranking-loss weight=0.02` 粗扫:
  - 分支: `exp/student-pairwise-ranking-loss`
  - `B49`: `AUC -0.001024`, `ACC -0.000096`, `RMSE +0.000249`, `Brier +0.000214`, `ECE -0.001026`
  - `B51`: `AUC -0.000780`, `ACC -0.000552`, `RMSE +0.000272`, `Brier +0.000233`, `ECE +0.000818`
  - 判断: ranking loss 早底座也不成立，不属于被实验 51 压住的结构
- clean readout routing 粗扫:
  - 分支: `exp/clean-readout-routing`
  - `B34 dense`: `AUC -0.001514`, `ACC -0.003502`, `RMSE +0.001018`, `Brier +0.000876`, `ECE +0.001853`
  - `B34 topk=2`: `AUC +0.000834`, `ACC -0.001827`, `RMSE +0.000578`, `Brier +0.000497`, `ECE +0.004667`
  - `B49 dense`: `AUC -0.001157`, `ACC -0.002836`, `RMSE +0.001590`, `Brier +0.001365`, `ECE +0.003502`
  - `B49 topk=2`: `AUC +0.000798`, `ACC -0.002455`, `RMSE +0.000201`, `Brier +0.000173`, `ECE +0.001169`
  - 判断: 更干净 routing 在早底座也不是 clean win；实验 51 的问题不是简单靠 `seen/unseen` gate 或 top-k routing 就能修复
- 总结:
  - 更早底座扫描没有推翻诊断 2；目前唯一强且稳定的底座反转仍是 `q-conditioned local mastery`: `B49` 三 seed 明显正向，`B51` 三 seed 明显负向
  - 其它结构大多属于三类: 早底座也不成立、只改善误差/校准而不推 AUC、或在实验 51 后副作用放大但早底座收益太小
  - 后续若要继续验证“错误底座”，优先不再盲目扩所有旧路线，而应围绕实验 51 做更直接的隔离实验: 降低/冻结/蒸馏/正交化 readout expert residual，或测试“B49 + 新 readout + 不吸收实验 51”的主线替代链

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
- 实验 67 进一步说明: 动态归因虽然比静态分摊语义更强，但直接替换 exercise-to-concept 主聚合容易削弱行为证据并伤害 `none_seen` 校准
- 实验 68 进一步排除了几个自然 rescue: 保持 message scale、只打高 concept-count、只学 incorrect blame 都没有恢复主线收益
- 这些方向的常见模式是 `AUC` 有时略正，但 `ACC/RMSE/Brier/ECE` 更容易回撤
- 因此 propagation 侧与目标层 tweak 的优先级应继续下调；若再回到这些方向，前提应是已有更明确的局部 slice 假设，或已有更强的结构正向底座

## 默认下一步

通用协作、运行与分支规则沿用 [docs/session_bootstrap.md](./session_bootstrap.md)；这里仅补充历史台账导出的默认优先级:

1. 仍从当前 `master` 主线出发；新假设优先单改一个结构因素，单次成立后再补 `2-3` 个 seed。
2. 当前处于单因素边际收益放缓的平台期；实验 70 已把 `none_seen` 学生条件化信号转成 overall 正收益，但多知识点题仍不是 clean win。
3. 允许少量测试“已各自成立”的正交组合，但默认只测最强的 `1-2` 组候选，不做组合爆炸；实验 70 + 实验 61 的直接组合已在实验 71 单 seed 验证为不 clean，不默认扩 seed。
4. 允许探索更大一级、真正改变表示瓶颈的模块；若单次结果表现为“overall 未过门槛但目标 slice 有明显改善”，可额外允许一次很小的 rescue sweep；普通 sidecar / residual 默认不进入这类 sweep。
5. 若继续沿实验 51/70 的 readout 底座推进，默认保留实验 51 full-trigger expert 与实验 70 `none_seen` sidecar；但诊断 2 已提示实验 51 可能压制部分后续 clean structure 的边际表现，因此新结构若在最新主线上表现为轻微负向、但语义足够干净，可优先追加一次实验 49 底座单 seed 交叉复验，再决定是否淘汰。
6. 对已经系统复访但未形成 clean overall gain 的 readout / propagation / ranking-loss 路线，默认不再高优先级继续；只有在出现明确新假设、且机制上明显区别于已失败版本时，才考虑重开。
7. 若用户明确要继续训练协议优化，当前优先候选是实验 37 与实验 61 两条支线；否则默认优先继续模型结构改动。
8. 判断是否值得继续时，默认主看 `AUC/ACC`，并优先寻找至少 `1e-3` 量级的改善；`RMSE/Brier/ECE` 与分桶校准默认只用于判断副作用。
