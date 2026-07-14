# 两模块结果反推候选：状态补全与目标条件诊断

## 已拒绝的上游机制

前面三类 Evidence 机制均按原始 validation 结果拒绝：

- Neural-Process Evidence Posterior：相对控制的 target 增益为 MOO +0.002264、XES +0.000652。
- Relational TKC Evidence：相对控制的 target 增益为 MOO +0.001931、XES +0.001041。
- 当前 Calibrated Evidence Representation：相对 raw summary control 的 target 增益为 MOO +0.000144、XES +0.001026。

这说明题目身份、难度校准和复杂 evidence encoder 在当前协议下没有产生足够的独立收益。Evidence Representation 保留为普通输入编码，不作为论文贡献，也不再围绕它调整消融。

## 已通过首屏的 Module 1: Personalized TKC/UKC State Completion

输入普通 student evidence、concept nodes、train-only population concept prior 和 observed concept evidence，输出下游唯一消费的完整 framework state：

- Full 通过 student–concept 乘性交互一次生成全部 TKC/UKC 状态。
- Control 对 observed concept 直接投影原始统计，对 UKC 使用学生无关的静态 concept prior。
- 两条 decoder 容量完全相同，Diagnosis 不存在其他学生特异旁路。

Holdout validation 首屏：

| 数据集 | Full H | Full T | Control H | Control T | ΔT |
|---|---:|---:|---:|---:|---:|
| MOOCRadar | 0.925962 | 0.935233 | 0.854393 | 0.760626 | +0.174607 |
| XES3G5M | 0.786172 | 0.783953 | 0.720483 | 0.708437 | +0.075516 |

Full 在两者均保持 validation strict win。该模块已经远超幅度门槛，待补 paired student bootstrap、第三个胜出数据集及 standard 轴后正式确认。

## 待验证的 Module 2: Target-Conditioned Diagnosis

输入完整 framework state、Q 和题目表示，输出作答概率：

- Full 使用目标状态与题目需求的乘积/差异交互，以及由目标状态条件化的有界 guess/slip。
- Control 是标准单调诊断：将 Q 加权 mastery 作为唯一学生变量，以正 discrimination、题目偏置/难度和仅题目条件化的 bounded guess/slip 生成概率。
- Full/control 的 cognitive、guess、slip 网络容量相差不超过 10%；两者共享完整状态和其他所有训练条件。

新变体加入后，所有模式仍共享同一 state dict、初始化哈希和 architecture fingerprint。因为新增对照模块会推进全局 RNG，Full 与 diagnosis control 必须一起重新训练，不能把旧 Full checkpoint 与新 control 拼接比较。

最终门槛不变：同一架构至少三胜；两个模块各自在至少两个 Full 胜出数据集 target 提升不低于 0.005，其中一个不低于 0.01，并至少一个 student-clustered paired-bootstrap CI 下界大于 0。
