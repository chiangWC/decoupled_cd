# 项目目标
实现一个 decoupled cognitive diagnosis model

说明:

- 本文件保留为高层模型语义说明，不追踪所有实现细节。
- 新会话默认入口是 [docs/session_bootstrap.md](docs/session_bootstrap.md)。
- 当前正式 workflow、主线结论和运行约定以 [docs/session_bootstrap.md](docs/session_bootstrap.md)、[docs/workflow.md](docs/workflow.md) 和 [docs/handoff.md](docs/handoff.md) 为准。

# Step 1：认知空间解耦

## 背景问题

传统认知诊断模型（如 NCDM、GCDM）只建模 TKC，忽略 UKC，导致认知空间不完整。

- `TKC`: 有行为监督（对/错）
- `UKC`: 无行为信号，仅依赖知识结构

## 符号定义

- 学生集合: `U`
- 习题集合: `E`
- 知识点集合: `K`
- 学生 `u` 做过的题: `E_u`
- Q 矩阵: `Q_{e,k} in {0,1}`
- 已测试知识点: `TKC_u`
- 未测试知识点: `UKC_u`

## 1.1 知识空间划分

`TKC_u` 与 `UKC_u` 满足:

- `TKC_u ∩ UKC_u = ∅`
- `TKC_u ∪ UKC_u = K`

构建 Q 矩阵:

- `Q in {0,1}^{|E| x |K|}`

## 1.2 行为空间（做题集合）

`E_u = { e in E | u 对 e 有作答记录 }`

## 1.3 已测试知识点

`TKC_u = ⋃_{e in E_u} { k | Q_{e,k} = 1 }`

## 1.4 未测试知识点

`UKC_u = K \\ TKC_u`

# Step 2：异构图信息传播

目标：`TKC` 与 `UKC` 分离建模，避免行为空间污染。

## 2.1 TKC 更新

`TKC` 接收三类信息:

- 习题信息
- 答题行为（对/错）
- 邻接知识点

记学生 `u` 在习题 `e` 上的答题结果为 `r_{u,e}`，习题表示为 `h_e`，知识点 `k` 的邻居集合为 `N(k)`，则:

`h_k^{TKC} = AGG({ r_{u,e} * h_e | e in N(k) }) + AGG({ h_{k'} | k' in N(k) })`

其中:

- `r_{u,e}`: 答题结果，取值 `0/1`
- `h_e`: 习题表示

## 2.2 UKC 更新

`UKC` 只依赖知识结构:

`h_k^{UKC} = AGG({ h_{k'} | k' in N(k) })`

不使用:

- 习题信息
- 行为信号

## 2.3 学生认知状态融合

学生最终认知状态定义为:

`h_u = w_u * mean({ h_k^{TKC} | k in TKC_u }) + (1 - w_u) * mean({ h_k^{UKC} | k in UKC_u })`

其中:

- `w_u in (0, 1)`: 学生级自适应融合权重
- `w_u` 由学生覆盖率及 `TKC / UKC` 聚合状态共同决定

设计动机:

- 不同学生的知识覆盖率差异较大，`TKC / UKC` 的融合不应由全局固定标量决定。
- 覆盖率较低的学生可以更依赖 `UKC` 推断，覆盖率较高的学生可以更依赖 `TKC`。

# Step 3：答题概率建模

## 3.1 纯认知答题概率

`P^{cog}_{u,e} = sigma(f_match(h_u, q_e) - b_e)`

其中:

- `h_u`: 学生认知状态
- `q_e`: 习题知识需求向量
- `f_match`: 学生认知状态与题目需求之间的可学习匹配函数
- `b_e`: 习题难度

## 3.2 加入非认知因素

`P_{u,e} = (1 - s_u) * P^{cog}_{u,e} + g_u * (1 - P^{cog}_{u,e})`

其中:

- `g_u`: 猜测概率
- `s_u`: 滑错概率

# Step 4：反向优化

损失函数采用二元交叉熵:

`L = - [ y_{u,e} log P_{u,e} + (1 - y_{u,e}) log (1 - P_{u,e}) ]`

优化参数包括:

- `TKC / UKC` 消息传递权重
- 学生级 `TKC / UKC` 融合 gate 参数
- 非认知参数 `g_u, s_u`
- 习题难度 `b_e`

# 当前优先级

- 完善工程复用
- 保持 Step 1-4 全流程可运行
- 在此基础上逐步优化模型效果

# 参考项目
../ConceptSkillCDM
仅参考其数据处理、训练框架、评估代码
不要复用其核心模型结构

# 本项目原则
- Python + PyTorch
- 先做最小可运行版本
- 先复用工程层，再重写模型核心
