# Bottleneck-Aware Monotone Requirement Surface：预注册

## 决策问题

本轮检验一个完整 Diagnosis 候选：**Bottleneck-Aware Monotone
Requirement Surface**。

> 给定同一份由学生历史构造的逐概念状态，多概念题的作答需求是否包含
> “最弱目标概念 × 目标概念平均水平”的不可加性交互，并且这种交互是否能
> 稳定胜过朴素聚合、同容量加性模型和标准正权 NCD 对照？

候选不预先声称 non-compensatory。HLL 只保证单调性；只有学习曲面确实
偏离加性、固定平均水平时仍对最弱概念敏感，且性能门与消融门同时通过，
才可讨论 bottleneck/non-compensatory 行为。DINA/GDINA 已有非补偿诊断，
NCDM 正权网络也能表达部分交互，本轮不提出“首次非补偿诊断”主张。

初始预注册提交前只做 train/Q 结构计数和公开实现审阅。没有候选代码、
训练、预测或 valid/test 读取。train-only 筛选通过仅激活端到端实现，不自动成为
论文模块；失败后不调阈值、不补 residual/gate/loss、不换名重跑。

### 正式执行前修订：非加性判据的尺度

在任何正式训练或 prediction 生成之前，经独立只读审查发现：若直接对
probability surface 计算 mixed difference，即使 Capacity Control 在 logit
尺度严格可加，外层 sigmoid 也可能产生非零二阶差分，从而造成假阳性。
因此固定将下文曲面判据中的 `F` 定义为去除公共 `item_offset` 和仅用于
严格单调数值保证的 `epsilon*g` 后，机制自身 `learned_surface` 的 logit，
即 `F=logit(learned_surface)`；固定网格、可实现域、`0.001`
阈值和所有性能门均不变。落盘工件同时保存 probability 与 surface logit，
正式 barrier 从落盘后的 logit 列重新计算判定。该修订只修复判据尺度，
没有查看候选训练、预测或任何 audit/valid/test 结果。

### 正式揭盲前工程修订：外部数据集锚与 checkpoint 重放

至少一次正式 `predict-dataset` train-only 运行已完成，其中包含冻结配方
下四个 head 的训练、checkpoint 写入及 audit prediction/surface 生成；
但未运行 `seal/evaluate`，未加载 audit-query labels，也未人工读取
prediction、surface 或 audit metrics。只查看了运行状态、文件清单/大小
及 outcome-free hash；产物随后作废删除。其后独立对抗审计发现：数据集
barrier 尚未写入外部锚时，联合重签 checkpoint、prediction、manifest 与
surface 可形成内部自洽伪造；原 `2e-6` replay 容差也可能吞掉改变排序的
微小预测变化。

因此正式流程增加下述逐数据集外部锚、CPU checkpoint 精确重放、surface
逐点重放与严格 outcome-free schema。模型、数据、训练配方、性能门和
noncollapse 门均未改变。

## 框架边界

若通过，模块接在 Claude v2 的
`tkc_states + ukc_states -> mastery[student, concept]` 后，读取 mastery、
权威 Q 和所有变体共享的题目 nuisance，产生唯一作答概率。它整块替换
target-aware、pooled NCF、hybrid、guess/slip 与第二预测头，旧 decoder
不得旁路。只有启用 mastery auxiliary BCE 的 Claude v2 mastery 可作为
冻结 State；后来 two-stage 中未受响应监督的 mastery 不能直接复用。

机会筛选先使用完全可审计的 support-only readiness，避免 State 差异冒充
Diagnosis 收益。它不继承旧性能，也不主张外部胜局。

## 来源与采用边界

曲面参数化参考 Yanagisawa 等人的
[Hierarchical Lattice Layer for Partially Monotone Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2022/hash/47ed62021460f2e9bba7be3e74260090-Abstract-Conference.html)
（NeurIPS 2022）。官方 [IBM/pmlayer](https://github.com/IBM/pmlayer)
固定为 `v1.0.1@89bcceb966b5c2feb506be3da413ed2549a71db1`，
Apache-2.0；wheel SHA-256 为
`02f596b6ad0c8a1be929bff33e0cc2a03909c7087f3751c4053b56fcb58b9153`。

正式代码只按论文公式独立实现所需 1D/2D 全单调 lattice，不复制作者
源码、模型或权重。固定 wheel 仅可作 CPU 小 fixture 数值 oracle。独立
实现须支持 registered buffers、dtype/device、mask 和网格单调性测试。
借鉴点是硬单调曲面参数化；拟议贡献边界是把学生局部状态映射为
`(weakest readiness, aggregate readiness)` 后进行多概念题诊断。

正式训练前已用上述固定 tag 的官方 CPU 实现做 257 点随机数值 oracle：
在相同 raw gates 下，1D `[16]` 最大绝对误差 `5.9604645e-08`，2D
`[4,4]` 最大绝对误差 `1.1920929e-07`；组合输出 SHA-256 分别为
`caff47a07b0ff7c5835580021b9a92344139e2212cf08d7a6258eb32c525c1fa`
和 `e171ab0fbd3da0ea4e012c2927a1dca16c3ec19666230ffa71291212003b18ab`。
仓库测试另冻结一组小型官方输出 fixture；此审计没有读取候选数据标签。

## 冻结的 train-only 协议

### 数据与划分

只读取近期 KnoField holdout 的 `train.csv` 和 `Q_matrix.csv`：

| 数据集 | train SHA-256 | Q SHA-256 |
|---|---|---|
| ASSIST17 | `e3c281f01d2fedaafa289c6d950c6ae64c48e2d2d3b2f6a69bd28e88a3ef236b` | `23a59ec57c3b454d2d3fece3760aa91ec65a297e357766265466591b5bb0f9e3` |
| MOOCRadar | `a60037cffe8378f63e38a0b96dc79bc0ec8396abc2af620a513a11f1d5882f89` | `b2526274cf733d0170028037b727c682eb81366804dc0f72703699fb90aaf8af` |

model seed=42，split seed=2024。ID 先 canonicalize；Q 使用 Q-matrix 对
同题所有行的 concept union，interaction `cpt_seq` 不得覆盖它。
`stable_fraction` 将参数以 `\x1f` 拼接后取 SHA-256 前 8 bytes 的
big-endian 整数除以 `2**64`。

```text
stable_fraction(2024, "requirement_surface", "student", student) < 0.8
```

小于 0.8 为 optimizer student，否则为 audit student，二者严格不交。
每组学生内部以 `(student,item)` 为原子：

```text
stable_fraction(2024, "requirement_surface", "group", student, item) < 0.7
```

小于 0.7 为 support，否则为 query。重复作答整体同侧。学生至少有
10 个 support groups 和 3 个 query groups。

### State 与公共题目项

学生–概念 readiness 只由该学生 support 的不同题组构造：

```text
readiness = (correct + 1) / (attempts + 2)
```

多概念 support response 固定复制给 Q 中每个概念；这是共同 State 定义，
不是待测机制。query 的完整 `(student,item)` 组不在 support。Eligible
要求 `2 <= |Q| <= 4`，且每个目标概念在 support 中至少出现 3 个题组。

公共题目 offset 只由 optimizer-support 构造：

```text
global_ease = (optimizer_support_correct + 1)
              / (optimizer_support_attempts + 2)
item_ease = (item_correct + 10 * global_ease) / (item_attempts + 10)
item_offset = logit(clamp(item_ease, 1e-4, 1 - 1e-4))
```

未见 item 回退 global ease。offset 对变体逐行相同。除明确列出的 strong
NCD control 外，变体看不到 item/concept/exact-Q ID、Q-combination
embedding 或 target-conditioned retrieval；Q 只选 readiness 和 cardinality。

协议、slice 和 outcome-free row ID 在打开 audit-query label 前冻结。row
ID 由源 train 行号、student、item 构造，禁止使用含 label 的现有
`source_row_id`。翻转 audit-query label 不得改变协议、输入、slice、
batch 或 prediction hash。

## Stage 0 结构可行性

冻结 namespace 后：

| 数据集 | Eligible rows / 学生 / 正负 | Q2 rows / 学生 / 正负 | Q3 | Q4 |
|---|---|---|---:|---:|
| ASSIST17 | 2,769 / 247 / 1,176–1,593 | 2,735 / 246 / 1,158–1,577 | 34 | 0 |
| MOOCRadar | 1,037 / 206 / 773–264 | 786 / 205 / 588–198 | 236 | 15 |

主推断固定 Q2；Q3/Q4 只描述，不能救 Q2。集中度为：

| 数据集 | items / 最大占比 / HHI | exact-Q pairs / 最大占比 / HHI |
|---|---|---|
| ASSIST17 | 564 / 0.77% / .002782 | 249 / 3.80% / .011540 |
| MOOCRadar | 167 / 4.83% / .012402 | 79 / 6.62% / .026892 |

两处均超过 500 rows、100 students、20 items、10 Q-pairs，且有正负标签。
另做移除 top-5 item、移除 top-5 Q-pair 和 Q-pair clustered bootstrap
敏感性分析；student-cluster bootstrap 仍是主判。

## 模块输入与严格单调性

Q2 readiness `r1,r2` 定义 `b=min(r1,r2)`、`g=mean(r1,r2)`。
Q2 的 `(b,g)` 与排序后两项一一对应。Q3/Q4 使用 masked min/mean，
不提供 cardinality embedding。padding 对 min 置 1，mean 用 valid count；
Q=0 报错。concept permutation 和 padding 数量不得改变输出。

HLL 依论文用合法顶点上下界与 multilinear interpolation，逐坐标硬非递减。
三条 lattice 路径共同使用固定 `epsilon=1e-3`：

```text
surface = (1 - epsilon) * learned_surface + epsilon * g
prob = sigmoid(item_offset + logit(clamp(surface, 1e-6, 1 - 1e-6)))
```

它只是数值单调保证，不算贡献；另报告 `epsilon=0` forward sensitivity，
不能用于选择。Q1 不参与筛选；端到端三变体 Q1 必须 bitwise 共用 unary。

## 变体与强对照

| 变体 | 冻结数据流 |
|---|---|
| **Full** | 2D HLL `[4,4]` 读取 `(b,g)`；16 个 active lattice 参数，可表达联合曲面。 |
| **Pooled Direct** | 1D HLL `[16]` 只读 `g`；16 参数。它是 pooled calibrator，不冒充标准 NCD。 |
| **Capacity Additive** | 1D HLL `[7]+[7]` 分别读取 `b,g`，以 `bias + w*logit(h_b) + (1-w)*logit(h_g)` 组合，`w=sigmoid(raw_w)`；14+2=16 参数，没有联合曲面。 |
| **Standard NCD Direct** | 完整 Q-masked readiness vector输入仓库既定 softplus-positive `K -> 256 -> 128 -> 1` NCD interaction 和 sigmoid activations，共享 item offset；可读 concept coordinates，容量不匹配，防止玩具对照。 |

Full/Capacity/Pooled 活跃参数均恰 16。报告 active/total 参数、首步
finite/nonzero gradient 覆盖和 FLOPs。Full/Capacity 初始化为同一加性
函数，固定 fixture 初始 prediction 最大差小于 `1e-6`。公共 offset、
row order、batch plan 和输入 hash 完全一致。

Standard NCD 是强性能保险，不属于同容量消融。保守对比为：

```text
delta_auc = AUC(Full)
            - max(AUC(Pooled), AUC(Capacity), AUC(Standard NCD))
```

bootstrap 每次重采样重新取三项差值最小值；Brier 与三个 control 最小值
比较。不得按指标删除或拼接对手。

## 固定训练与揭盲

四个 head 只在 optimizer query 中满足同一 eligible 定义的行上训练；
其 State 只来自 optimizer support，audit State 只来自 audit support。
统一使用：

- seed 42，float32；
- response BCE，无 auxiliary objective；
- full-batch Adam，learning rate `5e-3`，weight decay 0；
- 固定 2,000 steps，第 2,000 步为唯一 checkpoint；
- 无 scheduler、early stopping、checkpoint window 或调参。

每个 head 训练完成后先落盘唯一 checkpoint，再从该 checkpoint 在 CPU
重新加载并生成权威 prediction；CSV 读回后按 float32 bit pattern 与
checkpoint replay 精确一致，评价只消费 replay frame。Full surface 同样从
落盘 checkpoint 在 CPU 重放，固定点顺序及全部数值列必须逐点精确一致；
gate 只消费重放绑定后的 summary。Prediction schema 固定为六个
outcome-free metadata 字段加 `prob`，拒绝任何额外列。

每个数据集的四种 prediction 齐全后，`predict-dataset` 将
`dataset + barrier SHA + commit + architecture fingerprint` 打印到仓库及
output root 之外的受控调用日志。`seal` 必须由调用者显式传回 A17/MOO
各自首次输出的 SHA，不得从当前文件推导；先核对两个外部锚，再校验冻结
数据、recipe、manifest、checkpoint、prediction 和曲面工件。之后写入全局
barrier、打印 SHA 并退出。全局 SHA 也必须先记录在实验目录之外；后续
`evaluate` 必须显式传回且不能重写 barrier，之后才可揭盲。
runner 不接受 valid/test path；正式执行验证 clean HEAD、origin 同 SHA 和
`decoupled_cd` 环境。训练前保存 batch、row、input、offset、初始化、source
和 architecture hashes。过程只记录初始/末步 loss 与逐参数梯度健康，不显示
audit metric；OOM/非有限值明确失败，不改配方。

## 输出与问题实证

预先固定描述图：按 `b,g` 的 `[0,.2,.4,.6,.8,1]` 网格画 audit Q2
每格样本量和经验正确率，以及 Full、Capacity 和差值曲面；只展示至少
30 行的格子。它观察固定平均水平时 weakest readiness 是否仍有条件信息，
不能代替 held-out gate。

主表报告 Q2 AUC、ACC、RMSE、Brier、ECE；描述表报告 eligible pooled、
Q3/Q4、去 top item/pair。曲面另报告固定 `g` 改变 `b` 的
counterfactual gap、相对最佳加性投影的 interaction departure、以及是否
塌缩为加性。叙事诊断不能救性能失败。

### 曲面非塌缩的固定数值判据

在读取任何 prediction 前，将 `item_offset=0`，在步长 0.05 的固定
`b,g in {0.05,...,0.95}` 网格上枚举所有相邻 2x2 cells；只有四个角均
满足 Q2 可实现域 `b <= g <= (1+b)/2` 的 cell 才参与。对每个 cell 计算

```text
F(b,g) = logit(learned_surface(b,g; before epsilon blend and item_offset))
interaction = F(b0,g0) - F(b0,g1) - F(b1,g0) + F(b1,g1)
```

`max(abs(interaction)) >= 0.001` 才记为 noncollapsed。该算法和阈值对
两数据集相同；它只判断 Full 是否学到不可加性交互，不衡量效果好坏，
也不能替代相对 Capacity/Standard NCD 的性能门。

## Stage 1 通过门

Full 相对三 control 的保守 envelope 必须同时满足：

1. A17、MOO Q2 `delta_auc >= 0.005`；
2. 至少一处 `delta_auc >= 0.010`；
3. 至少一处 2,000 次 student-cluster paired-bootstrap 95% CI 下界大于 0；
4. 两处 Brier 相对最强 control 回归不超过 0.001；
5. eligible pooled delta 两处非负；
6. 去 top-5 item、去 top-5 Q-pair 后，两处 Q2 delta 仍为正；
7. Full 曲面不在数值容差内塌缩为 Capacity Additive。

CI 只表示当前学生聚类重采样下的效应稳定性，不表示训练 seed 不确定性，
也不宣称多轮搜索后的 family-wise significance；bootstrap 不属于多 seed。

任一失败即拒绝并删除候选代码，只保留预注册、结果、哈希和 git 历史；
不读 valid/test、不进入 v2、不调参。

## 通过后的端到端顺序

通过后另行预注册：先在 A17/MOO holdout validation 固定并共享 Claude v2
State；所有变体使用相同 State/Q/题目项/训练配方，只替换 Diagnosis；
holdout 门通过后才跑 standard；达到至少 3 个 validation 普通外部胜局后
冻结并进行 validation-driven test confirmation；在 Full 实际胜出集补齐
controls。

最终仍服从 `docs/research_goal.md`：同一架构至少 3 个普通外部胜局；
本模块在至少两个 Full-winning 数据集相对较强合理 control
`delta_T >= 0.005`，至少一处 `>= 0.010`、至少一处 student-cluster
CI 下界大于 0，且其他 S/H/T 不实质回归。Stage 1 通过只获得端到端集成
资格，不是合格论文模块；只有完成上述端到端三胜及 Full-winning 数据集
上的模块资格门后，它才成为第一个合格论文模块。终局仍需第二个独立且
通过干净消融的模块。
