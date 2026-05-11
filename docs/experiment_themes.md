# Experiment Themes And Diagnostics

This file holds cross-experiment diagnostics and theme-level conclusions. Use it when designing a new line of experiments, after checking the current snapshot and experiment index.

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
