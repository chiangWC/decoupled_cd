# Corrected Support-Adaptive 双图门

## 背景与历史兼容

r22 的 `--v2-dual-graph-adaptive` 使用归一化概念图的行和作为“密度”输入。
概念图含自环且按行归一化后，非空 TKC 集合上的该特征恒为
`log1p(1)=log(2)`，因此不能表达学生之间的图支持差异。旧参数、线性层形状和
checkpoint 加载路径仍完整保留，但该开关只标记为 legacy，不再解释为有效的
密度门。本文不修改或重新归因任何 r22 历史结果。

## 新机制

新开关为：

```text
--v2-dual-graph-support-adaptive
```

机制从归一化图的非零拓扑创建副本并移除自环。图方向沿用传播约定
`A[target, source]`。对每名学生，先计算从任一 TKC 一跳可达的概念，再统计其中
属于该学生 UKC 的比例：

```text
support = reachable_ukc_count / ukc_count
```

没有 UKC 时 support 定义为 1；只有自环的图上，有 UKC 的学生 support 为 0。
边权大小不参与 support，只使用边是否非零。

新路线保留 Mo1 原始 `Linear(2D, 1)` 门，不扩展输入维度：

```text
gate_logit = base_gate([decoupled_state, response_state])
gate_logit += support_weight * support
```

`support_weight` 是零初始化标量。系数为 0 时，共享参数初始化和预测与同 seed 的
Mo1 严格一致；关闭新开关时该参数不进入 state dict，旧 checkpoint 继续严格加载。

## 组合与实验约束

- 新 support 开关可与 Tr2（`--v2-consistency-weight`）组合。
- Tr2 只在 `student_recompute_minibatch` 实现；其他训练模式会被参数校验拒绝，
  防止配置被静默忽略。
- 新开关与 legacy `--v2-dual-graph-adaptive`、`--v2-router` 互斥。
- xes3g5m 的概念图若只有自环，support 路径预期退化为 Mo1；这属于结构诊断，
  不是性能结论。
- 当前提交只完成正确性实现与 CPU 回归测试，尚未运行正式 GPU 实验，也未引入
  mastery separation。

示例：

```bash
python scripts/train.py \
  --model v2 \
  --v2-dual-graph-support-adaptive \
  --v2-consistency-weight 0.5 \
  --training-mode student_recompute_minibatch \
  --student-batch-size 64 \
  --seed 42
```
