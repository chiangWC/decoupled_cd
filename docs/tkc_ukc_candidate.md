# Goal 两模块候选与拒绝记录

## 重新激活候选：Calibrated Evidence Representation

前面三类 Evidence 机制在 MOO/XES 上均未达到门槛：

- Neural-Process Evidence Posterior：相对控制的 target 增益为 MOO +0.002264、XES +0.000652。
- Relational TKC Evidence：相对控制的 target 增益为 MOO +0.001931、XES +0.001041。
- 当前 Calibrated Evidence Representation：相对 raw summary control 的 target 增益为 MOO +0.000144、XES +0.001026。

扩池后，同一个 Calibrated Evidence 机制在 Junyi 的 target 提升 +0.011985 且 Full 胜外部线；EdNet target 提升 +0.013519，但 Full 尚未胜外部线。ASSIST17 正在复核。因此该模块不再按 MOO/XES 两个数据集提前判废；是否晋级仍只看 Full 实际胜出数据集。

## 待强对照复核：Personalized TKC/UKC State Completion

输入普通 student evidence、concept nodes、train-only population concept prior 和 observed concept evidence，输出下游唯一消费的完整 framework state：

- Full 通过 student–concept 乘性交互一次生成全部 TKC/UKC 状态。
- Control 对 observed concept 直接投影原始统计，对 UKC 使用学生无关的静态 concept prior。
- Strong Control 同样消费 student evidence、concept 和 population prior，仅移除显式乘性交互。
- 两条 decoder 容量完全相同，Diagnosis 不存在其他学生特异旁路。

Holdout validation 首屏：

| 数据集 | Full H | Full T | Control H | Control T | ΔT |
|---|---:|---:|---:|---:|---:|
| MOOCRadar | 0.925962 | 0.935233 | 0.854393 | 0.760626 | +0.174607 |
| XES3G5M | 0.786172 | 0.783953 | 0.720483 | 0.708437 | +0.075516 |

Full 在两者均保持 validation strict win，但上表只相对弱 direct-prior control。只有相对新增 Strong Control 仍达到门槛后，该模块才算通过。

## 已拒绝：Target-Conditioned Diagnosis

输入完整 framework state、Q 和题目表示，输出作答概率：

- Full 使用目标状态与题目需求的乘积/差异交互，以及由目标状态条件化的有界 guess/slip。
- Control 是标准单调诊断：将 Q 加权 mastery 作为唯一学生变量，以正 discrimination、题目偏置/难度和仅题目条件化的 bounded guess/slip 生成概率。
- Full/control 的 cognitive、guess、slip 网络容量相差不超过 10%；两者共享完整状态和其他所有训练条件。

MOO target 只提升 +0.000783；XES target 反而下降 0.001070。该机制明确拒绝，不作为论文模块。

最终门槛不变：同一架构至少三胜；两个模块各自在至少两个 Full 胜出数据集 target 提升不低于 0.005，其中一个不低于 0.01，并至少一个 student-clustered paired-bootstrap CI 下界大于 0。
