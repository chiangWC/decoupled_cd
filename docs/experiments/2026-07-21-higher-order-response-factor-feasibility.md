# Higher-Order Response Factor：Stage 0 可行性结论

## 结论

Higher-Order Response Factor 候选不激活，不进入模型实现或训练。ASSIST17
满足预注册的最小样本要求，但 MOOCRadar 同时低于 500 行和 100 名学生；
因此没有两个数据集共同支持该机制的首屏可行性。

本结论是只读、train-only 的 Stage 0 结构审计，不是模型结果。后续不得在
看到计数后把结构因子从 |Q| >= 3 放宽到 |Q| = 2，也不得用一般 multi-Q
交互替代 higher-order factor 后沿用同一候选名称。若要研究二阶关系，必须
把它作为不同假设重新定义，而不是挽救本候选。

## 冻结审计口径

审计使用 split seed 2024，只读取当前数据版本的 train interactions 和
Q-matrix：

1. 从 single-Q train interaction 构造 train-only pseudo-UKC 目标；
2. 对每个目标删除该学生所有包含目标 concept 的 support，使目标 concept
   在剩余学生历史中不可见；
3. structural factor 必须是 union-Q 大小至少为 3 的 train item；
4. 目标必须能通过该 higher-order factor 连接到至少 2 个仍被学生观察到的
   bridge concepts；
5. 每个数据集至少需要 500 个合格 pseudo-UKC rows、100 名学生，且目标
   response 的正负类别均存在。

split seed 只固定 train 内的伪目标/支持构造。审计没有打开 valid 或 test
interaction、label、prediction、manifest 或 checkpoint，也没有拟合模型或
根据结果调整阈值。

## 只读计数

| Dataset | 合格 rows | 学生 | 正例 | 负例 | 500 rows | 100 students | Stage 0 |
|---|---:|---:|---:|---:|:---:|:---:|:---:|
| ASSIST17 | 2,517 | 245 | 1,113 | 1,404 | 通过 | 通过 | 通过 |
| MOOCRadar | 184 | 74 | 125 | 59 | 失败 | 失败 | **失败** |

MOOCRadar 虽然同时包含正负标签，但仅有 184 行、74 名学生，不能支撑后续
Full/Direct/Capacity 的稳定切片比较或学生聚类区间估计。候选的双数据集
可行性条件因此失败；ASSIST17 单独通过不能激活实现。

## 文献来源与采用边界

机制动机来自 Zhang 等人的
[Factor Graph Neural Networks（JMLR 2023）](https://jmlr.org/papers/v24/21-0434.html)：
用 variable-to-factor 与 factor-to-variable 消息传递表达普通 pairwise
图难以直接表示的 higher-order relation。对本项目的拟议映射是：概念为
variable，多概念作答为 structural factor，factor 的响应信息通过已观察
bridge concepts 向 pseudo-UKC 状态传递。

作者提供了
[MIT-licensed reference repository](https://github.com/zzhang1987/Factor-Graph-Neural-Network)。
本轮只借鉴论文层面的 higher-order factor/message-passing 机制，没有下载、
复制、改写或采用该仓库的任何源代码、权重、配置或完整模型。由于 Stage 0
已失败，仓库内也没有新增该候选的模型或训练代码。

## 决策与 provenance

审计对应代码快照为
cce040ef310f592241dcbbfdc7f0422c1f75e264。最终决定是：

- 不实现 Higher-Order Response Factor；
- 不运行 smoke、validation、standard 或 test；
- 不调参，不放宽 |Q| >= 3 与至少 2 个 observed bridges 的定义；
- 不把 |Q| = 2 或一般 multi-Q 版本换名重跑；
- 将本计数保留为该文献方向在当前首屏数据集上的可行性负证据。

本文档仅记录 Stage 0 数据可行性，不证明 FGNN 机制本身无效，也不构成模型
性能、模块消融或外部胜局证据。
