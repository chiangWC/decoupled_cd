# TKC/UKC 两模块替换候选

上一候选的 Full 在 MOOCRadar 和 XES3G5M validation 均胜出，但两个模块相对干净对照的 target 增益不足：MOO 为 +0.002264/+0.000337，XES 为 +0.000652/+0.000178。因此它只登记为性能候选，两个模块均 rejected。

新候选仍只有两个论文方框：

1. **Relational TKC Evidence**
   - 输入 train-only 学生–题目 correct/incorrect 关系、Q 和题目表示。
   - 按 Q edge 将正负关系消息聚合为 observed concept state。
   - 对照为同容量 raw concept statistics MLP。
2. **Personalized UKC Completion**
   - 输入 Module 1 的 TKC states、observed mask 和 concept queries。
   - 每个 UKC query 直接注意该学生全部 observed TKC states，再硬组装 TKC/UKC。
   - 对照为同容量 global TKC summary–concept MLP，不做 query-specific attention。

固定 Diagnosis 只消费完整 framework state。四个变体实例化完全相同的参数树；模式只替换数据流。训练统一使用 20% context-target masking，禁止目标 response 进入自身状态。
