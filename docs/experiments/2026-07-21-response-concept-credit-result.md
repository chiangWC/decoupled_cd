# Response-to-Concept Credit Routing：验证结果与拒绝记录

## 冻结结论

Response-to-Concept Credit Routing and State Completion 未通过预注册的
Stage 1 激活门，因此本实现被正式拒绝，不能称为论文模块，也不能进入
standard、test 或外部胜局确认阶段。

本结论来自唯一正式目录：

    results/goal_two_module/response_credit_stage1_f9ad892/stage1/

实验只运行 MOOCRadar_chold_v2 和 NIPS34_chold_v2 的 holdout
validation。Full、Direct 和 Capacity Control 均使用模型 seed 42、划分
seed 2024、20 epochs、batch size 128、AdamW、learning rate 1e-3、
weight decay 1e-4、无 scheduler、无 early stopping，并只评估 epoch 20。
没有超参数搜索、结果后调参、checkpoint 选择或多 seed 实验。学生聚类
bootstrap 是同一组冻结预测上的不确定性分析，不是模型多 seed。

由于 Stage 1 失败，预注册协议禁止启动 standard 的六个 Stage 2 任务；
真实 test 文件、test manifest 和 test prediction 均未读取。各正式
manifest 和 evaluation 均记录 test_files_opened=false。

后续不得通过调参、加入辅助目标、残差、gate、adapter、第二预测头或换名
重跑来挽救该候选。若继续探索，必须提出职责或数据流实质不同的新机制，
并重新预注册其对照和门槛。

## 比较对象与指标

本次只检验一个候选框架方框：

- Full：响应条件化、Q 约束的 concept credit routing，随后生成完整
  student-concept state；
- Direct：把一条多概念作答完整复制给其 Q 集中的每个概念，再使用与
  Full 相同的 state completion 和 Diagnosis；
- Capacity Control：保留与 Full 相同的迭代骨架、参数量和 completion，
  但路由只能使用静态 item/Q/concept 信息，不能使用学生实际作答或
  response-conditioned state。

固定的 Q-conditioned Diagnosis 只消费 framework_state，不作为候选
贡献。所有变体仅使用 response BCE，没有模块专属目标。

每一行的“较强 control”按该 scope 的 AUC 在 Direct 与 Capacity
Control 中取较大者。Delta 定义为：

    AUC(Full) - max(AUC(Direct), AUC(Capacity Control))

overall 是 holdout validation 全部查询行；C 是预注册的广义
credit-sensitive slice；C_strict 是因 MOO 样本不足而在实现前降级为
纯描述性诊断的原严格 slice；T 在 MOOCRadar 为 exact-zero，在 NIPS34
为 low-coverage。C_strict 不参与 checkpoint、控制选择或任何 gate。

## MOOCRadar holdout validation

| Scope | Full AUC | Direct AUC | Capacity AUC | 较强 control | Delta |
|---|---:|---:|---:|:---:|---:|
| overall | 0.889900 | 0.887634 | 0.888118 | Capacity | +0.001782 |
| C | 0.802942 | 0.800578 | 0.803330 | Capacity | -0.000388 |
| C_strict | 0.700501 | 0.691729 | 0.691520 | Direct | +0.008772 |
| T | 0.910263 | 0.906500 | 0.906647 | Capacity | +0.003616 |

C 包含 1,581 行、295 名学生；T 包含 3,271 行、399 名学生。正式机制
指标 C 上，Full 略低于 Capacity Control，因此没有证明
response-conditioned routing 优于静态 item-concept routing。

C_strict 的 +0.008772 仅来自 156 行、35 名学生。它是值得保留的描述性
信号，但该 slice 在任何预测产生前已经因低于 500 行、100 名学生的
可行性下限而失去 gating 资格。它不能覆盖正式 C 的 -0.000388，也不能
用来激活模块或启动 standard/test。

## NIPS34 holdout validation

| Scope | Full AUC | Direct AUC | Capacity AUC | 较强 control | Delta |
|---|---:|---:|---:|:---:|---:|
| overall | 0.727614 | 0.726894 | 0.727062 | Capacity | +0.000552 |
| C | 0.729375 | 0.728722 | 0.728877 | Capacity | +0.000498 |
| C_strict | 0.722558 | 0.721405 | 0.721159 | Direct | +0.001153 |
| T | 0.690810 | 0.689449 | 0.691964 | Capacity | -0.001155 |

C 包含 28,051 行、1,013 名学生；C_strict 包含 8,717 行、507 名学生；
T 包含 1,415 行、169 名学生。Full 在正式 C 上仅比较强 control 高
0.000498，远低于 0.002 的门槛；在 T 上又比 Capacity Control 低
0.001155，略微越过允许回归 0.001 的边界。

## 配对 bootstrap

C 的不确定性使用 2,000 次确定性的 student-clustered paired bootstrap。
每个 replicate 同时计算 Full-minus-Direct 与 Full-minus-Capacity，并
以二者较小值作为 joint two-control contrast。

| Dataset | 有效 / 请求 replicates | Joint C delta 95% CI | CI 下界 > 0 |
|---|---:|---:|:---:|
| MOOCRadar | 2000 / 2000 | [-0.005339, +0.004389] | 否 |
| NIPS34 | 2000 / 2000 | [-0.000394, +0.001104] | 否 |

两个区间均跨过零，因此无法排除 Full 相对至少一个冻结 control 没有真实
收益。bootstrap 不改变模型、训练或 seed，也不能替代跨数据集效应门槛。

## 预注册 Stage 1 gate

| 条件 | 结果 | 正式证据 |
|---|:---:|---|
| 两个数据集均有 delta C >= 0.002 | **失败** | MOO -0.000388；NIPS +0.000498 |
| 至少一个数据集有 delta C >= 0.003 | **失败** | 两者均未达到 |
| 至少一个 joint C 95% CI 下界 > 0 | **失败** | 两个区间均跨零 |
| 两个数据集均有 delta T >= -0.001 | **失败** | NIPS -0.001155 |
| 至少一个数据集有 delta T >= +0.001 | 通过 | MOO +0.003616 |
| 两个 holdout overall delta 均 >= -0.001 | 通过 | MOO +0.001782；NIPS +0.000552 |
| overall/C/T Brier 回归均 <= 0.0002 | 通过 | 六项 delta 均为负，Full 的 Brier 更低 |
| 每个数据集至少 1,800 个有效 bootstrap | 通过 | 两者均为 2,000 |

aggregate 决策为 stage1_passed=false。这不是边缘性人工判断：主要
机制指标 C 的跨数据集幅度、联合置信区间和 T 非回归三类必要条件同时
未满足。即使 MOO 的 T 与 C_strict 方向较好，也不能推翻联合 gate。

## 失败的 8e97d35 尝试仅作工程记录

较早目录：

    results/goal_two_module/response_credit_stage1_8e97d35/stage1/

来自 commit 8e97d35a8d2f4e31882d27bf67591535f4a1c9a3。调度器在启动
MOOCRadar/full 前执行 live origin 校验时，git ls-remote 返回 exit
status 128，目录因此标记 FAILED。该不完整尝试虽然已经生成部分 NIPS34
预测，但没有形成完整的两数据集 barrier 和 aggregate decision；其中
任何预测或潜在指标均不进入本结论，也不能与正式结果拼接。

commit f9ad892 只为 live-origin 查询加入有限重试并减少同一 dispatch
cycle 的冗余网络校验；它没有修改模型机制、训练配方或 gate。正式六任务
从新的 f9ad892 目录完整运行。因此，8e97d35 是一次网络/调度工程故障
记录，不是第二个 seed、调参轮次、模型失败证据或可引用实验结果。

## Provenance 与不可变 artifacts

正式运行代码 commit：

    f9ad892ca2c02ad8f91fae4bfd33776ef95bec06

architecture topology SHA-256：

    52c057fe83894ca88f57b14a8bb4f16d2d41946857944680c32c6202fd00c434

以下路径均相对于正式 Stage 1 目录。

| Artifact | SHA-256 |
|---|---|
| aggregate/activation_decision.json | 7fad670c28cbedbcfdd8e34b7eda7aae78b86c636e634028205c45221083e6ed |
| prediction-barrier/prediction_barrier.json | 4085edc940b542e561765ceb59990f27eaff9c45945307ba9033b136f15dd082 |
| evaluate/MOOCRadar/evaluation.json | f49e91bd4c00f61e840c711e896c78eac6c4b708f30a4a9bf64be1dc54d8fdfd |
| evaluate/MOOCRadar/joint_c_student_bootstrap.npz | c2cb4ccb9158cfbabfad0080dc53f1a31e627c22991a72a472e4ea8526d3239c |
| evaluate/MOOCRadar/aligned_predictions_with_labels.csv | 73bac02761f3099f0578c224df2a8fe212088c560c55438455ea77c89bbcc53a |
| evaluate/NIPS34/evaluation.json | 255741ae54966e088ee52a68b9d60842ad6f2657680005cf51d1b1e3a8ff2fd8 |
| evaluate/NIPS34/joint_c_student_bootstrap.npz | 230b6d5b5b447acfc0f0e27f2a73fb6d84886cb6d7241d9f19f558ceb5c5a33a |
| evaluate/NIPS34/aligned_predictions_with_labels.csv | 93b65e973aad466b955bd5160b276174f2792148d2883f9c874f3ee28c2ce451 |
| predict/MOOCRadar/full/prediction_manifest.json | d2e24142b9a28d2db8f87c7dae078a8cee94215129222663cacba35e3065d5bb |
| predict/MOOCRadar/direct/prediction_manifest.json | 5420ffc972b88574f33ba7a28b3069eca9224fe617bfaf83d7f4316add9c401b |
| predict/MOOCRadar/capacity/prediction_manifest.json | 3324a2e9d18ea15447b79a9de3b7ed2ca0e77b6aa45b43905198941c552f5c83 |
| predict/NIPS34/full/prediction_manifest.json | 592c4a4e4cf7494816be593a2b4841c524aefb048d1392e6268c3ab9e0942197 |
| predict/NIPS34/direct/prediction_manifest.json | 788a706cfedf6d5f43081dd98e212da72ec9dd6e5140f3af3ca6a4b878fe27c0 |
| predict/NIPS34/capacity/prediction_manifest.json | 58de478100a213bfb0b835ff86517c4953618d15063cbd5e1db0bba262ba2982 |

各 prediction manifest 继续保存对应 checkpoint、逐行 prediction、
student batch plan、初始化、数据、Q、mapping 和行序哈希；evaluation
JSON 保存完整 ACC、RMSE、Brier、ECE、slice ID、逐行对齐及 bootstrap
provenance。生成结果位于 results/，不提交仓库。

机器可读 source of truth 是 aggregate/activation_decision.json；本文档
只是其人类可读负结果记录，不得引用为 test、standard、外部模型比较或
论文模块成立的证据。
