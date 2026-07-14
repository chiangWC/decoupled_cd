# 两模块候选：Evidence Posterior + Concept Query Completion

本文件只登记当前可证伪候选，不冻结论文叙事。

## Full 数据流

1. **Induced Evidence Posterior**
   - 输入：train-only 学生作答集合、题目表示、Q 派生概念表示和题目统计。
   - 输出：固定长度 evidence memory、学生后验及不确定性。
   - 机制来源：Partial VAE 的部分观察后验与 Set Transformer 的 inducing-point set attention；仅借鉴机制，代码独立实现。
   - 朴素对照：全局边际统计和 masked-mean 经同容量 MLP 生成同形状 memory。
2. **Attentive Concept Query Completion**
   - 输入：概念 query、Module 1 memory/后验、局部 train-only concept evidence。
   - 输出：完整 student-concept framework state、mastery 与 reliability。
   - 机制来源：Attentive Neural Process 的 target-to-context cross-attention；代码独立实现。
   - 朴素对照：参数匹配的 latent-concept MLP，不做 cross-attention。
3. 固定 Diagnosis：Q-conditioned pooled NCF，只消费 framework state，不作为贡献。

Full 与全部消融始终实例化四条路径；模式开关只替换数据流。因此 state dict、初始化顺序、总参数量和 architecture fingerprint 一致。两个有效路径及其对照的独占参数差均不超过 10%。

## 变体

| 变体 | Evidence Posterior | Concept Query |
|---|---|---|
| Full | induced posterior | cross-attention |
| w/o Module 1 | summary control | cross-attention |
| w/o Module 2 | induced posterior | latent control |
| Double control | summary control | latent control |

首屏只使用 response BCE，不启用 KL、masked reconstruction 或一致性目标，避免将训练目标收益混入结构消融。

## 晋级条件

Full 先在 MOOCRadar 与 XES3G5M validation 上相对各自更强合理对照形成 Pareto 点。最终仅在 Full 胜出数据集判断模块资格；每个模块至少两个胜出数据集 ΔT >= 0.005、至少一个 ΔT >= 0.01、至少一个 student-clustered paired-bootstrap 95% CI 下界大于 0，且胜出数据集其他 S/H/T 轴无明显回归。
