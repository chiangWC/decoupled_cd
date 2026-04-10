# Model Improvement Archive

这份文档只保留“实验台账”用途: 记录做过哪些改动、结果如何、哪些方向已经被证伪或暂缓。

它不是正式规范，也不是新会话默认必读文档。只有在设计实验、核对历史结论、避免重复试错时再读。

当前 workflow、正式基线、已定主线结论、下一步默认规则，统一以 [docs/session_bootstrap.md](/home/jameschiang/work/decoupled_cd/docs/session_bootstrap.md)、[docs/workflow.md](/home/jameschiang/work/decoupled_cd/docs/workflow.md) 和 [docs/handoff.md](/home/jameschiang/work/decoupled_cd/docs/handoff.md) 为准。

## 如何使用这份档案

- 需要快速进入当前项目状态时，先读 [docs/session_bootstrap.md](/home/jameschiang/work/decoupled_cd/docs/session_bootstrap.md)，再按需读 [docs/workflow.md](/home/jameschiang/work/decoupled_cd/docs/workflow.md) 和 [docs/handoff.md](/home/jameschiang/work/decoupled_cd/docs/handoff.md)。
- 需要确认某条路线是否已经试过、为什么没继续、是否值得复访时，再回到这份文档。
- 下文按“用途”而不是按时间顺序组织。
- 每条实验都保留原实验编号，方便和旧讨论记录对照。
- 文中若写“旧主线”，默认指实验 12；当前正式主线默认指实验 21。
- 除非条目里明确写了“已在实验 21 当前主线上复验”，否则基于实验 12 的备选分支不能直接视为对当前正式主线的结论。

## A. 基线演化

这部分回答的是: 当前主线是怎么一步步形成的。

### A1. 从早期图与题目表示到 transition graph

#### 实验 1. 原始 Q 共现图基线

改动:

- `N(k)` 由 Q 矩阵裸共现图构造。
- `q_e` 使用题目所需知识点 embedding 的简单平均。

实验结论:

- `assist_09` 上 `20 epoch` 全量训练可稳定运行。
- 指标略高于随机，但提升有限。

结论:

- 它证明了最小闭环可跑通，但不适合作为长期主线。

#### 实验 2. 稀疏归一化共现图 + 可学习 `q_e`

改动:

- `N(k)` 改为归一化共现 + top-k 稀疏图。
- `q_e` 改为可学习加权汇聚 + MLP。

实验结论:

- 训练稳定。
- 没有形成稳定泛化收益。
- `test_auc` 只有微小变化，`val_auc` 反而更差。

结论:

- 这轮改动不能作为主基线。

#### 实验 3. 论文式转移图

改动:

- 基于有序版 ASSIST09 数据。
- 使用论文风格的 `C / T` 关系构建方法。
- 导出:
  - `prerequisite_graph`
  - `similarity_graph`
  - `propagation_graph`

实验结论:

- 先修边数: `1161`
- 相似边数: `2492`
- 在 `assist_09` 上的早期全量训练里，效果明显优于前两版图。

结论:

- 这版图是后续主线的真正起点。

### A2. 合理超参数与训练充分性

#### 实验 6. 超参数扫描

改动:

- 扫描范围:
  - `learning_rate in {1e-3, 3e-4, 1e-4}`
  - `concept_dim in {16, 32, 64}`
- `concept_dim = 128` 在当前 full-batch GPU 路径下稳定 OOM。

实验结论:

- 最优组合是 `learning_rate = 1e-3`, `concept_dim = 64`。
- 当时最好文件:
  - `results/hparam_sweeps/assist09_transition_lr_1e-3_dim_64.json`

结论:

- 之前一些“结构改动无效”的判断，部分受不合理超参数影响。
- 后续结构比较必须固定在更合理的超参数上进行。

#### 实验 7. `g / s` 公平对比

改动:

- 在统一超参数下比较:
  - 学生常数 `g / s`
  - 条件化 `g_{u,e} / s_{u,e}`

实验结论:

- `conditional g/s` 明显优于 `constant g/s`。

结论:

- `conditional g/s` 是成立的正向结构改动，应保留在主线上。

#### 实验 8. 长训与训练策略

改动:

- 保持:
  - ordered ASSIST09
  - transition graph
  - `learning_rate = 1e-3`
  - `concept_dim = 64`
  - `conditional g/s`
- 新增:
  - best checkpoint 保存
  - `ReduceLROnPlateau` 调度框架
- 依次完成:
  - `100 epoch`
  - `200 epoch`
  - `300 epoch`

实验结论:

- `100 epoch`: `test_auc = 0.666546`
- `200 epoch`: `test_auc = 0.704830`
- `300 epoch`:
  - `best_val_auc = 0.716939`
  - `best_epoch = 243`
  - `test_auc = 0.709411`
  - best checkpoint:
    - `results/assist_09_current_worktree_300ep_best.pt`

结论:

- 当前模型之前远未训满。
- 结构比较默认应看 `300 epoch` 量级，而不是 `20 epoch`。

#### 实验 9. 多 seed 复现

改动:

- 在当时的正式单图基线上加入显式 `seed` 控制。
- 跑 `seed in {2024, 2025, 2026}` 的 `300 epoch` 对比。

实验结论:

- 三组 `test_auc` 分别为:
  - `0.709478`
  - `0.707565`
  - `0.706154`
- 均值约 `0.7077`。
- 波动很小。

结论:

- 长训后的单图基线是稳定的，不是单次偶然值。

### A3. 当前主线形成

#### 实验 11. `TKC/UKC` 结构传播参数解耦

改动:

- 在单图 `propagation_graph` 基线上，将原先共享的概念结构传播变换拆成:
  - `tkc_concept_to_concept`
  - `ukc_concept_to_concept`

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.744711`
  - `test_auc = 0.738442`
- `seed=2025`:
  - `best_val_auc = 0.742114`
  - `test_auc = 0.737970`
- `seed=2026`:
  - `best_val_auc = 0.743833`
  - `test_auc = 0.737942`
- `test_auc` 均值约 `0.7381`。
- 这组结果建立在“`valid/test` 复用 `train` 行为历史输入”的修复后评估口径上。

结论:

- 这是当前最可靠的正向结构改动之一。

#### 实验 12. `TKC` 正误双通道行为消息

改动:

- 保留:
  - 单图 `propagation_graph`
  - `conditional g/s`
  - `TKC/UKC` 结构传播参数独立
- 将 `TKC` 行为消息从“只看正确题”改成:
  - 正确题通道
  - 错误题通道
  - 行为侧 gated fusion
- 同时把按知识点聚合的实现改成更省显存的索引式写法，避免双通道在 full-batch 下 OOM。

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.748738`
  - `test_auc = 0.744609`
- `seed=2025`:
  - `best_val_auc = 0.749699`
  - `test_auc = 0.745276`
- `seed=2026`:
  - `best_val_auc = 0.754099`
  - `test_auc = 0.747414`
- `test_auc` 均值约 `0.7458`。
- 相比上一版评估修复后基线均值 `0.7381`，提升约 `+0.0077`。

结论:

- “错题信号被丢掉”是当前实现里的真实问题。
- `TKC` 行为消息应显式保留错误作答证据。
- 这一步和实验 11 一起构成了旧主线收敛到稳定形态的关键节点。

#### 实验 21. `TKC/UKC` 学生自适应融合 gate

改动:

- 保留:
  - 单图 `propagation_graph`
  - `conditional g/s`
  - `TKC/UKC` 结构传播参数独立
  - `TKC` 正误双通道行为消息
- 将原先全局固定的:
  - `alpha * tkc_mean + beta * ukc_mean`
  替换为学生级自适应融合。
- 融合 gate 输入为:
  - `coverage`
  - `tkc_mean`
  - `ukc_mean`
- 最终形式为:
  - `w_u * tkc_mean + (1 - w_u) * ukc_mean`

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.754493`
  - `best_epoch = 278`
  - `test_auc = 0.749272`
  - 文件:
    - `results/exp_adaptive_tkc_ukc_gate/assist_09_tkc_dual_channel_seed2024_300ep.json`
- `seed=2025`:
  - `best_val_auc = 0.757129`
  - `best_epoch = 300`
  - `test_auc = 0.751710`
  - 文件:
    - `results/exp_adaptive_tkc_ukc_gate/assist_09_tkc_dual_channel_seed2025_300ep.json`
- `seed=2026`:
  - `best_val_auc = 0.756685`
  - `best_epoch = 288`
  - `test_auc = 0.749739`
  - 文件:
    - `results/exp_adaptive_tkc_ukc_gate/assist_09_tkc_dual_channel_seed2026_300ep.json`
- 三个 seed 的 `test_auc` 均值约 `0.7502`。
- 相比实验 12 的旧主线均值 `0.7458`，提升约 `+0.0044`。

结论:

- 全局固定 `alpha/beta` 的融合方式过于粗糙，学生级自适应 gate 能更好利用覆盖率差异。
- 这次提升不是单 seed 偶然值，而是三 seed 一致提升。
- 这条线已经足够取代实验 12，成为当前正式主线。

## B. 已明确不作为主线的路线

这部分回答的是: 哪些方向已经明显退化，或者至少不值得默认继续。

### B1. 早期 `TKC` 融合层微调路线

#### 实验 4. `TKC` 标量融合 + `student + exercise` 偏置 `g/s`

改动:

- `TKC` 的行为消息和邻接消息加入独立可学习标量权重。
- `g / s` 从学生常数改为 `student + exercise` 偏置。

实验结论:

- 训练稳定。
- 早期全量对比没有带来提升。
- `test_auc` 下降到随机附近。

结论:

- 简单标量融合不够。
- 这条早期路线不应回到主线。

#### 实验 5. `TKC` gated fusion

改动:

- 保留论文式转移图基线。
- 将 `TKC` 分支的行为消息和邻接消息改为 gated fusion。
- `g / s` 回退到学生级常数，避免和融合策略混在一起。

实验结论:

- 训练稳定。
- 早期全量对比仍未超过转移图基线。
- `test_auc` 仍在 `0.503` 左右。

结论:

- 当时 `TKC` 路径的瓶颈不只是融合形式。
- 不能把“换 gated fusion”本身当成有效方向。

### B2. `dual graph` 路线

#### 实验 10. 双图分开传播

改动:

- 保留单图 `propagation_graph`。
- 新增可选 `graph_mode = dual`。
- 在 `dual` 模式下分别读取:
  - `prerequisite_graph`
  - `similarity_graph`
- 在传播层分别做两路邻接传播，再门控融合。

实验结论:

- `300 epoch` 全量对比明显退化。
- `best_val_auc = 0.508805`
- `test_auc = 0.501716`
- `best_epoch = 3`

结论:

- `dual graph` 接口可以保留，但不应作为正式主线。

### B3. `q_e` 残差增强路线

#### 实验 13. `q_e` 轻量 residual 融合

改动:

- 保留:
  - 单图 `propagation_graph`
  - `conditional g/s`
  - `TKC/UKC` 结构传播参数独立
  - `TKC` 正误双通道行为消息
- 在 `q_repr` 里给 `Q` pooling 表示和题目独立 embedding 增加轻量 gated residual 融合。

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.744008`
  - `test_auc = 0.737416`
- 对比当时主线:
  - `best_val_auc = 0.748738`
  - `test_auc = 0.744609`

结论:

- 这条 `q_e` residual 方案没有带来收益。
- 当前不建议再直接把题目独立 embedding 以残差形式叠回 `q_repr`。

### B4. 局部硬汇聚路线

#### 实验 14. 题目条件局部汇聚: `TKC/UKC` 同时局部 mean

改动:

- 保留:
  - 单图 `propagation_graph`
  - `conditional g/s`
  - `TKC/UKC` 结构传播参数独立
  - `TKC` 正误双通道行为消息
- 将认知预测从全局 `student_state` 改成题目条件局部汇聚。
- 对目标题 `q_e` 相关概念分别做:
  - `TKC` 局部 mean
  - `UKC` 局部 mean
- 再用 `alpha * local_tkc + beta * local_ukc` 进入 `P_cog`。

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.737040`
  - `test_auc = 0.731763`
  - 文件:
    - `results/assist_09_item_conditioned_seed2024_300ep.json`
- `seed=2025`:
  - `best_val_auc = 0.736479`
  - `test_auc = 0.731650`
  - 文件:
    - `results/assist_09_item_conditioned_seed2025_300ep.json`
- 两个 seed 的 `test_auc` 均值约 `0.7317`。

结论:

- “题目条件下再聚合”不是错误方向，但“局部硬 mean”明显伤害效果。

#### 实验 15. 题目条件局部汇聚: `TKC` 局部 mean + `UKC` 全局 mean

改动:

- 延续题目条件预测思路。
- 只让 `TKC` 对目标题相关概念做局部 mean。
- `UKC` 回退为原来的全局支持项 mean。

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.735676`
  - `test_auc = 0.731123`
  - 文件:
    - `results/assist_09_tkc_local_ukc_global_seed2024_300ep.json`

结论:

- 把 `UKC` 保持为全局支持项并没有救回这条路线。
- 局部硬汇聚本身就是主要问题之一。

### B5. 直接叠加两种 `UKC` 抑噪策略

#### 实验 20. `local UKC neighbor` + `UKC-only coverage gate`

改动:

- 以实验 17 和实验 19 为基础。
- 在认知分支中同时使用:
  - 实验 17 的题目一跳邻域 `local UKC`
  - 实验 19 的 `UKC-only coverage gate`
- 保持 `TKC` 全局汇聚、`guess/slip` 全局学生向量不变。

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.752087`
  - `best_epoch = 299`
  - `test_auc = 0.746921`
- `seed=2025`:
  - `best_val_auc = 0.744208`
  - `best_epoch = 291`
  - `test_auc = 0.739302`
- `seed=2026`:
  - `best_val_auc = 0.749998`
  - `best_epoch = 300`
  - `test_auc = 0.744460`
- 三个 seed 的 `test_auc` 均值约 `0.7436`。

结论:

- 这条融合没有形成 `1 + 1 > 1` 的效果。
- 它低于实验 19 单独版本，也低于当前正式主线。
- 当前不建议继续沿“直接叠加 local `UKC` 约束 + coverage gate”这条路线推进。

## C. 相对实验 12 旧主线的备选方向

这部分回答的是: 哪些路线在实验 12 的旧主线口径下还有信号，但目前还不该直接视为对实验 21 当前正式主线的结论。

### C1. 在实验 12 旧主线上接近、但尚未在实验 21 上系统复验

这几条路线的原始对比对象都是实验 12，而不是实验 21。它们可以作为“值得在当前主线上复验”的候选，但不能直接称为“接近当前主线”。

#### 实验 16. 题目条件局部汇聚: `TKC` item-aware attention + `UKC` 全局 mean

改动:

- 保留:
  - `TKC` 只在目标题相关概念内做局部选择
  - `UKC` 作为全局支持项
- 将 `TKC` 的局部硬 mean 改成由 `q_repr` 条件化的 item-aware attention。

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.743627`
  - `test_auc = 0.739887`
  - 文件:
    - `results/assist_09_tkc_attention_ukc_global_seed2024_300ep.json`
- `seed=2025`:
  - `best_val_auc = 0.741752`
  - `test_auc = 0.736367`
  - 文件:
    - `results/assist_09_tkc_attention_ukc_global_seed2025_300ep.json`
- 两个 seed 的 `test_auc` 均值约 `0.7381`。

结论:

- 相比局部硬 mean，这条 attention 版本更合理，也明显救回了性能。
- 但这里的“正式主线”比较对象仍是实验 12 时代的主线口径，不应直接外推到实验 21。
- 当前更适合把它视为“值得在实验 21 上补复验的候选路线”。

### C2. 局部化 `UKC` 的语义约束

#### 实验 17. coverage-aware `UKC` 融合: `UKC` 只在题目相关概念的一跳邻域内聚合

改动:

- 保留:
  - 单图 `propagation_graph`
  - `conditional g/s`
  - `TKC/UKC` 结构传播参数独立
  - `TKC` 正误双通道行为消息
  - `TKC` 全局已测概念汇聚
- 只修改认知分支中的 `UKC` 融合方式。
- 对每道题的 `q_e`:
  - 在 `propagation_graph` 上取题目概念集合的一跳邻域
  - 用该局部概念范围筛选 `UKC`
  - 只对“未测试且位于题目局部邻域内”的概念做 mean

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.747750`
  - `best_epoch = 299`
  - `test_auc = 0.742428`
  - 文件:
    - `results/assist_09_local_ukc_neighbor_seed2024_300ep.json`
- `seed=2025`:
  - `best_val_auc = 0.744854`
  - `best_epoch = 300`
  - `test_auc = 0.741246`
  - 文件:
    - `results/assist_09_local_ukc_neighbor_seed2025_300ep.json`
- 两个 seed 的 `test_auc` 均值约 `0.7418`。

结论:

- 这条线明显优于前面的 item-conditioned mean / attention 变体。
- 它说明 `UKC` 的主要问题更像是全局平均引入了与当前题无关的噪声。
- 但这里仍是实验 12 口径下的结论，不能直接视为对实验 21 的判断。

### C3. coverage-aware 全局融合

#### 实验 18. coverage-aware global fusion: 用 coverage 同时调节全局 `TKC/UKC`

改动:

- 保留:
  - 单图 `propagation_graph`
  - `conditional g/s`
  - `TKC/UKC` 结构传播参数独立
  - `TKC` 正误双通道行为消息
  - `TKC` 和 `UKC` 原来的全局概念汇聚
- 在认知分支新增一个由学生 coverage 条件化的标量 gate。
- coverage 定义为:
  - `tested_concepts / (tested_concepts + untested_concepts)`
- 用该 gate 同时调节全局 `TKC` 与全局 `UKC` 的融合强度。

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.752721`
  - `best_epoch = 300`
  - `test_auc = 0.747068`
  - 文件:
    - `results/assist_09_coverage_gate_seed2024_300ep.json`
- `seed=2025`:
  - `best_val_auc = 0.746704`
  - `best_epoch = 300`
  - `test_auc = 0.740811`
  - 文件:
    - `results/assist_09_coverage_gate_seed2025_300ep.json`
- `seed=2026`:
  - `best_val_auc = 0.752357`
  - `best_epoch = 300`
  - `test_auc = 0.746431`
  - 文件:
    - `results/assist_09_coverage_gate_seed2026_300ep.json`
- 三个 seed 的 `test_auc` 均值约 `0.7448`。

结论:

- 这条线说明“coverage-aware 融合”本身是有信号的。
- 但这里的比较对象仍是实验 12；由于它与实验 21 的自适应融合机制语义重叠较多，当前更适合视为历史参考，而不是默认优先复访方向。

#### 实验 19. coverage-aware global fusion follow-up: 只对全局 `UKC` 做 coverage gate

改动:

- 延续 coverage-aware 全局融合思路。
- 保持全局 `TKC` 项不变。
- 只用 coverage gate 去缩放全局 `UKC`。
- 形式为:
  - `alpha * global_tkc + gate(coverage) * beta * global_ukc`

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.754431`
  - `best_epoch = 299`
  - `test_auc = 0.748475`
  - 文件:
    - `results/assist_09_coverage_beta_gate_seed2024_300ep.json`
- `seed=2025`:
  - `best_val_auc = 0.748439`
  - `best_epoch = 300`
  - `test_auc = 0.741629`
  - 文件:
    - `results/assist_09_coverage_beta_gate_seed2025_300ep.json`
- `seed=2026`:
  - `best_val_auc = 0.751716`
  - `best_epoch = 300`
  - `test_auc = 0.745962`
  - 文件:
    - `results/assist_09_coverage_beta_gate_seed2026_300ep.json`
- 三个 seed 的 `test_auc` 均值约 `0.7454`。

结论:

- 只对 `UKC` 做 coverage gate 比“同时调 `TKC/UKC`”更稳。
- 它在旧主线口径下几乎追平，但相对实验 21 的新主线仍有明显差距。
- 因此它是目前 coverage-aware 路线里最强的一条，但仍应视为次优备选。

### C4. 已在实验 21 当前主线上复验，但未胜出

#### 实验 22. 在实验 21 当前主线上复验 `local UKC neighbor`

改动:

- 以实验 21 的当前正式主线为底座:
  - 单图 `propagation_graph`
  - `conditional g/s`
  - `TKC/UKC` 结构传播参数独立
  - `TKC` 正误双通道行为消息
  - 学生级自适应 `TKC/UKC` 融合 gate
- 在此基础上，只把 `UKC` 的全局 mean 改成题目相关概念一跳邻域内的 `local UKC neighbor`。
- 这相当于把实验 17 的核心想法重新挂到实验 21 上做增量复验。

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.752178`
  - `best_epoch = 293`
  - `test_auc = 0.747523`
  - 文件:
    - `results/exp_adaptive_local_ukc/assist_09_adaptive_local_ukc_seed2024_300ep.json`
- `seed=2025`:
  - `best_val_auc = 0.755889`
  - `best_epoch = 300`
  - `test_auc = 0.750068`
  - 文件:
    - `results/exp_adaptive_local_ukc/assist_09_adaptive_local_ukc_seed2025_300ep.json`
- `seed=2026`:
  - `best_val_auc = 0.754637`
  - `best_epoch = 300`
  - `test_auc = 0.747724`
  - 文件:
    - `results/exp_adaptive_local_ukc/assist_09_adaptive_local_ukc_seed2026_300ep.json`
- 三个 seed 的 `test_auc` 均值约 `0.7484`。
- 相比实验 21 当前正式主线均值 `0.7502`，下降约 `-0.0018`。

结论:

- 实验 17 中“`UKC` 全局平均会引入题目无关噪声”的观察，在实验 12 口径下是有信号的。
- 但把这条思路直接叠到实验 21 当前主线上后，三 seed 结果稳定略低于当前正式主线。
- 因此这条线现在不应再放在“接近当前主线”的候选里，而应归类为“已在实验 21 上复验但未胜出”。

### C5. 实验 21 当前主线上的新结果与其 follow-up

#### 实验 23. 修正 `TKC` 行为项的全局二次缩小

改动:

- 以实验 21 的当前正式主线为底座:
  - 单图 `propagation_graph`
  - `conditional g/s`
  - `TKC/UKC` 结构传播参数独立
  - `TKC` 正误双通道行为消息
  - 学生级自适应 `TKC/UKC` 融合 gate
- 检查 `_build_exercise_component` 后确认当前实现会先按学生全历史对 `weighted_exercises` 做一次全局归一化，再在概念维度上再除一次 `concept_weights`。
- 这会让学生历史越长，`TKC` 行为证据越容易被系统性压小，与 Step 2 中“`TKC` 保留行为信号”的语义不一致。
- 将该聚合改为概念内加权平均:
  - 不再先按学生全历史做全局归一化
  - 直接在每个概念内按该概念实际命中的行为权重做平均
- 同时补了最小回归测试，防止“无关历史变长会压小当前概念行为项”的问题回归。

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.763858`
  - `best_epoch = 175`
  - `test_auc = 0.760568`
  - 文件:
    - `results/exp_tkc_exercise_aggregation/assist_09_tkc_dual_channel_seed2024_300ep.json`
- `seed=2025`:
  - `best_val_auc = 0.764494`
  - `best_epoch = 180`
  - `test_auc = 0.759346`
  - 文件:
    - `results/exp_tkc_exercise_aggregation/assist_09_tkc_dual_channel_seed2025_300ep.json`
- `seed=2026`:
  - `best_val_auc = 0.764509`
  - `best_epoch = 185`
  - `test_auc = 0.759156`
  - 文件:
    - `results/exp_tkc_exercise_aggregation/assist_09_tkc_dual_channel_seed2026_300ep.json`
- 三个 seed 的 `test_auc` 均值约 `0.7597`。
- 相比实验 21 当前正式主线均值 `0.7502`，提升约 `+0.0095`。

结论:

- 这不是“实现风格差异”，而是当前 `TKC` 行为聚合里的真实缩放偏差。
- 修掉这一步后，三 seed 提升幅度明显且一致。
- 这条改动已经吸收到当前 `master`，并取代实验 21 之前的正式主线口径。

#### 实验 24. 在实验 23 基础上做多知识点题按知识点数分摊

改动:

- 以实验 23 为底座，继续只改一个结构因素。
- 对多知识点题先做 `q_matrix` 行归一化，使一道题的总行为权重按知识点数分摊到各概念。
- 分子和分母同时使用归一化后的 `q_matrix`:
  - 概念分子只累加该题分配给当前概念的份额
  - 概念分母也使用同样的份额
- 同时补了回归测试，验证多知识点题的行为权重确实会被分摊，而不是整题对每个概念都完整贡献一次。

实验结论:

- `seed=2024`:
  - `best_val_auc = 0.764223`
  - `best_epoch = 181`
  - `test_auc = 0.760619`
  - 文件:
    - `results/exp_tkc_exercise_aggregation_qnorm/assist_09_tkc_dual_channel_seed2024_300ep.json`
- `seed=2025`:
  - `best_val_auc = 0.764555`
  - `best_epoch = 191`
  - `test_auc = 0.759291`
  - 文件:
    - `results/exp_tkc_exercise_aggregation_qnorm/assist_09_tkc_dual_channel_seed2025_300ep.json`
- `seed=2026`:
  - `best_val_auc = 0.764564`
  - `best_epoch = 184`
  - `test_auc = 0.759446`
  - 文件:
    - `results/exp_tkc_exercise_aggregation_qnorm/assist_09_tkc_dual_channel_seed2026_300ep.json`
- 三个 seed 的 `test_auc` 均值约 `0.7598`。
- 相比实验 23 均值 `0.7597`，仅提升约 `+0.0001`。

结论:

- 这一步在语义上更干净，也更符合“多知识点题总权重守恒”的直觉。
- 但在当前主线口径下，它相对实验 23 的增量几乎可以视为持平，暂时没有足够证据把它当成必须合入的关键改动。
- 当前更可靠的判断是: 实验 23 的收益主要来自“修掉 `TKC` 行为项的全局二次缩小”，而不是多知识点题分摊本身。

## D. 快速索引

这部分只用于快速查重，不替代上面的详细条目。

### 已经形成当前主线的关键实验

- 实验 3: transition graph 成为后续图结构起点。
- 实验 6: 锁定 `learning_rate = 1e-3`, `concept_dim = 64`。
- 实验 7: `conditional g/s` 成立。
- 实验 8: `300 epoch` 训练充分性被确认。
- 实验 9: 长训单图基线的多 seed 稳定性被确认。
- 实验 11: `TKC/UKC` 结构传播参数解耦成立。
- 实验 12: `TKC` 正误双通道成立。
- 实验 21: `TKC/UKC` 学生自适应融合 gate 成立，并已取代旧主线。
- 实验 23: 修正 `TKC` 行为项全局二次缩小后，三 seed 均值约 `0.7597`，并已吸收到当前主线。

### 已明确不建议默认继续的路线

- 实验 4: `TKC` 标量融合无效。
- 实验 5: 早期 `TKC` gated fusion 无效。
- 实验 10: `dual graph` 明显退化。
- 实验 13: `q_e` residual 融合无收益。
- 实验 14: `TKC/UKC` 同时局部硬 mean 明显退化。
- 实验 15: `TKC` 局部硬 mean + `UKC` 全局 mean 仍无改善。
- 实验 20: 直接叠加 `local UKC neighbor` 和 `UKC-only coverage gate` 低于实验 19 单独版本。

### 相对实验 12 旧主线接近、可留作后续参考的路线

- 实验 16: item-aware attention 说明“题目条件选择性”有信号，但尚未在实验 21 上系统复验。
- 实验 17: local neighbor `UKC` 说明 `UKC` 全局噪声是关键问题之一，但该观察来自实验 12 口径。
- 实验 18: coverage-aware 全局融合有信号，但与实验 21 的融合机制存在语义重叠，当前优先级不高。
- 实验 19: coverage-aware `UKC` gate 在旧主线下几乎追平，但相对实验 21 仍是次优备选。

### 已在实验 21 当前主线上复验的路线

- 实验 22: 将实验 17 的 `local UKC neighbor` 叠到实验 21 后，三 seed 均值约 `0.7484`，低于当前正式主线 `0.7502`，当前不建议继续推进。
- 实验 23: 修正 `TKC` 行为项全局二次缩小后，三 seed 均值约 `0.7597`，显著高于实验 21 当前正式主线 `0.7502`。
- 实验 24: 在实验 23 基础上做多知识点题权重分摊后，三 seed 均值约 `0.7598`，相对实验 23 几乎持平。
