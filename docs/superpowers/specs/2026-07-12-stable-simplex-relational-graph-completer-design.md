# 稳定 Simplex 与证据关系图补全设计

日期：2026-07-12

远端实验仓库：`/home/xph/jwc/research/decoupled_cd_codex_worktrees/complete_model`

目标分支：`codex/remote-complete-model-20260710`

## 1. 当前证据与继续原因

统一 A0 与低秩 A1 已在冻结的 MOOCRadar、ASSIST17、XES3G5M 上完成两套 validation。A1 只在 ASSIST17 改善 AUC；MOO/XES 的 ordinary DOA 上升，但 weighted DOA 与 overall/zero AUC 回退。这说明自由 student/concept 因子能够改善局部次序，却没有从 curriculum-driven 观测模式中识别 missing-cell mastery。

现有 v3 还有一个独立数值缺陷：float32 三元 softmax 在极端 logits 下可令 `cognitive_weight` 下溢为精确零，使 `guess+slip==1`、最终概率对 cognitive probability 的导数为零。该问题不推翻已登记的 A1 负例，但 v3 不能直接作为下一模块搜索的可信基线。

本轮先建立 version-4 数值稳定基线 A0v4，再只替换 missing-mastery completer 为 A2。冻结 cohort、数据/基线审计、seed、划分、数值 recipe 与 test-closed 约束保持不变。

## 2. 三种机制比较

### 2.1 证据关系图补全（推荐）

把 train-only 学生–概念证据表示为带关系和可靠性权重的二部图，通过共享消息传递产生学生/概念状态，再解码未测 mastery。该机制直接利用“一个学生在其他概念上的证据”和“其他学生在该概念上的证据”，避免 A1 的自由 pairwise 因子。它不依赖外部概念图，因此 XES 也有有效传播路径。

依据是 Graph Convolutional Matrix Completion：将矩阵补全表示为二部图上的 link prediction，并用关系消息传递聚合观测边。

### 2.2 MNAR/曝光感知补全（保留为后续候选）

联合建模 curriculum exposure 与 mastery，最直接对应“未测不是随机缺失”的诊断，也具有较强论文叙事。但 exposure 与 mastery 的可识别性较弱，需要额外 exposure likelihood，首次候选同时改变的机制较多。

### 2.3 Side-information Inductive Matrix Completion（不作为首选）

只用学生覆盖统计和概念/Q 特征做 inductive bilinear completion，工程简单、参数较少，但与已失败的低秩模型过于接近，预计不足以填补 A17/XES 的绝对 AUC 缺口。

因此首个新候选只实现 2.1。2.2 只有在 A2 产生新的 validation 失效诊断后才可注册。

## 3. Version-4 稳定行为通道

固定最小 cognitive mass：

```text
delta = 2^-20
raw = softmax(z_guess, z_slip, z_cognitive)
cognitive_weight = delta + (1-delta) * raw_cognitive
behavior_mass = 1 - cognitive_weight
guess_ratio = sigmoid(z_guess - z_slip)
guess = behavior_mass * guess_ratio
slip = behavior_mass - guess
p = guess + cognitive_weight * p_cognitive
```

该参数化不设置 `0.3` 等行为上限；guess 或 slip 仍可接近 1。`delta` 只是固定的机器精度安全下界，并保证在 float32 极端 logits 下：

```text
guess >= 0
slip >= 0
guess + slip < 1
dp / dp_cognitive = cognitive_weight >= 2^-20
```

architecture manifest 更新为 version `4`，behavior module 名称固定为 `conditional-simplex-floor-2m20`。旧 v3 checkpoint/manifest 必须严格拒绝；A0v4 与 A2 使用相同 version-4 behavior module。该修复不追溯改写 v3 A0/A1 结果。

## 4. A0v4 数值基线

A0v4 保留 evidence-parameter mastery estimator、global concept prior completer、NeuralCDM monotone decoder，仅替换为稳定行为通道。使用已冻结 recipe：

- MOOCRadar：r2；
- ASSIST17：r1；
- XES3G5M：r0。

每个数据集运行 standard/holdout validation 一次，seed `42`、split seed `2024`。A0v4 形成新的同 fingerprint baseline；报告相对 v3 A0 的差值，但不把 v3 结果伪装成 v4 proof。任何 NaN、非正导数、空 mastery、顺序错误或 test 访问都使本轮全局停止。

## 5. A2 Evidence-Relation Graph Completer

### 5.1 输入与输出

输入仅包含：

- train-only `student_concept_evidence`；
- `observed_mask`；
- attempts-derived reliability；
- 学生和概念 ID/domain sizes。

输出：

- 完整 `[num_students, num_concepts]` missing-mastery probability；
- 可审计的学生/概念图状态和 reconstruction mask。

输出仍只在 `~observed_mask` 上进入唯一 mastery：

```text
mastery = observed_mask * m_obs + (1-observed_mask) * m_graph
```

### 5.2 关系二部图

每个 observed student–concept cell 形成一条边。边由平滑正确率区分为两个关系通道，并由 reliability 连续加权：

```text
positive relation: target >= 0.5
negative relation: target < 0.5
edge_weight = min(attempts, 20) / 20
```

学生初始特征为 train-only coverage/accuracy 汇总；概念初始特征为 train-only coverage/accuracy 与 Q-derived item-frequency 汇总。ID 只用于节点索引，不允许未经消息传递的 ID embedding 旁路直接进入 decoder。

两层 relation-specific sparse message passing 交替执行 concept→student 与 student→concept 聚合。每层使用共享线性变换、degree normalization、ReLU 和 layer normalization；不使用 dense `[S,K,K]` 张量、旧 A1 factor residual、mixture gate 或第二 mastery。

### 5.3 图解码器

图状态通过共享双线性 decoder 输出所有目标 cell：

```text
m_graph[s,k] = sigmoid(h_student[s]^T W h_concept[k]
                       + b_student_summary[s]
                       + b_concept_summary[k])
```

decoder 参数随数据集 concept/student 数量无关；图 hidden dimension 是允许跨数据集改变的数值超参数，但首轮固定继承各数据集 A0v4 concept dimension。

### 5.4 防止观测边泄漏

若 completion loss 重建一条仍存在于消息图中的观测边，模块可学习复制输入。训练时以 `seed=42` 的确定性哈希在每个 epoch 选择 `20%` observed cells 作为 reconstruction targets，并从该 epoch 的消息图移除这些边。第三项 loss 仍是 observed-cell completion BCE，但只作用于被移除的 target edges。

推理时使用全部 train-only observed edges 构图，再预测真正 missing cells。遮蔽比例、关系阈值 `0.5`、reliability cap `20` 和两层传播是固定模块语义，不允许按数据集切换。

## 6. 损失、架构身份与禁止项

A0v4 仍使用 final response BCE 与 observed mastery evidence loss。A2 使用同两项，并加入固定的 masked graph reconstruction loss。所有数据集损失组成一致，只有非零权重可作为数值超参数。

A2 architecture manifest 固定为：

```text
mastery_estimator = evidence-parameter
completion = evidence-relational-graph
cognitive_decoder = neuralcdm-monotonic
behavior_model = conditional-simplex-floor-2m20
mastery_output = student-concept
version = 4
```

禁止：旧 prior/lowrank residual、learned completer mixture、数据集专属分支、第二响应头、scalar loss 冒充模块、validation/test 构图、真实 test 选择、静默 batch 调整。

## 7. 验证门与停止规则

A2 相对 A0v4 必须满足原五项门：

- 每个 primary 数据集 standard overall AUC 不下降；
- 每个 primary 数据集 holdout overall AUC 不下降；
- 至少 `2/3` 数据集 zero AUC 严格提升；
- 至少一个 zero delta `>=0.001`；
- cohort 与 version-4 candidate fingerprint 完全一致。

DOA 仍只作软排序。A2 通过后才从 Task 5 audit 动态重建外部门；三个数据集必须同时胜过 strongest same-protocol external 的 zero AUC，并守住 standard/holdout overall。未通过则 test 继续关闭，登记图补全失效机制，并立即进入下一轮 MNAR/曝光感知模块设计，而不是停止总目标。

## 8. 测试与远端执行

数值修复聚焦测试：极端 logits `±10^4`、float32/float64、`guess+slip<1`、导数下界、旧 manifest/checkpoint 拒绝、A0 模块不变。

图补全聚焦测试：

- train-only edge construction；
- 正/负关系与 reliability；
- deterministic edge masking 不泄漏 target；
- sparse message passing shape/gradient；
- no-ID bypass；
- hard missing-only assembly；
- non-contiguous student minibatch indexing；
- 跨数据集同 fingerprint。

每个 architecture fingerprint 只运行一次 1-epoch GPU smoke。真实 validation 使用现有可信 controller、不可覆盖 attempt、proof replay、不同 GPU 并行与同卡 flock。数据、预测、checkpoint、mastery 和日志仍只写 campaign root；里程碑提交生成 bundle，不 push。

## 9. 文献映射

- Graph Convolutional Matrix Completion：二部 interaction graph、relation-specific message passing、link decoder。
- Modeling User Exposure in Recommendation：missingness/exposure 不是随机；保留为 A2 失败后的下一机制。
- Inductive Matrix Completion：side information 约束 latent completion；作为较低风险但表达力较弱的备选。
