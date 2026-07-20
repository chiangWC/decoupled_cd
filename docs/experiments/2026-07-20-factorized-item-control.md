# Factorized Item Control 生死实验

## 判别问题

现有 `Exercise-Specific Requirement Query` 的 Full 输入为目标 Q 加权
concept view 与目标题 `exercise_node`，并用一个 `2d -> d` 联合投影在同一
隐藏单元中先组合两种信息、再经过 ReLU 和 LayerNorm。

此前的 `q_only_control` 与 `concept_prototype_control` 都不读取目标题 ID，
因此 Full 相对它们的收益无法区分：

1. 目标题信息本身的收益；
2. joint nonlinear Q-item composition 的收益。

新 `factorized_item_control` 不删除输入信息：

- Q view 与 Full 完全相同；
- exact target exercise ID 及对应 `exercise_node` 与 Full 完全相同；
- 下游公共的独立 item difficulty 保持不变；
- 下游公共的 `q_state * q_repr` item-conditioned
  interaction/discrimination-like 通道保持不变；
- 当前架构没有显式标量 discrimination 参数，因此不作该声称。

唯一替换是：

```text
Full:
  LN(ReLU(W_joint [q_view; item_view] + b))

Factorized control:
  LN((ReLU(W_q q_view + b_q) + ReLU(W_i item_view)) / sqrt(2))
```

`W_q` 有 bias、`W_i` 无 bias，故两路的参数量均为 `2d² + 3d`，精确相等。
control 仍可为每道题学习不同 item factor，只移除 joint hidden unit。
判别结论必须限制为：

- Full 稳定胜出：支持 joint nonlinear composition 优于同信息的
  factorized additive encoding；
- Full 不胜出：原 Requirement 消融主要由删除 target item 信息造成；
- 即使 Full 胜出，也不能仅凭该实验自动声称存在新的 cognitive
  requirement 语义。

## 强 2x2 定义

History 主对照统一使用 `calibrated_summary_control`。它保留学生正确率、
做题难度校准、历史置信度和 coverage，只移除 attempted-item semantic
pool。`raw_summary_control` 与 `identity_raw_control` 只可作为机制拆解附表。

| 变体 | Evidence mode | Requirement mode |
|---|---|---|
| Full | `calibrated_history` | `exercise_specific` |
| w/o History | `calibrated_summary_control` | `exercise_specific` |
| w/o Requirement | `calibrated_history` | `factorized_item_control` |
| w/o both | `calibrated_summary_control` | `factorized_item_control` |

每个数据集的 standard/holdout 使用同一训练配方。固定 `seed=42`，只看
validation；runner 用 `valid.csv` 作为 train harness 的 test placeholder，
因此不会打开真实 test 文件，也不会产生 test metric。

## Validation artifact 盘点

以下旧 artifact 的 `evaluation_stage` 均为 `validation` 且
`test_metrics=null`。新增 control 构造时恢复 RNG，因此这些 Full 或
dormant-control artifact 的旧活跃参数初始化和预测仍可作为 2x2 已完成格；
使用前仍须核对数据哈希与预测哈希。

| 数据集 | Split | Full | 强 w/o History |
|---|---|---|---|
| ASSIST17 | S | `final_standard_v9/a17.json` | 缺失 |
| ASSIST17 | H | `target_screen_v9/a17_full.json` | `evidence_final_v9/a17_partial.json` |
| MOOCRadar | S | `final_standard_v9/moo.json` | 缺失 |
| MOOCRadar | H | `final_holdout_v9/moo.json` | 缺失 |
| XES3G5M | S | `final_standard_v9/xes.json` | 缺失 |
| XES3G5M | H | `target_screen_v9/xes_full.json` | 缺失 |
| Junyi | S | `final_standard_v9/junyi.json` | 缺失 |
| Junyi | H | `evidence_final_v9/junyi_full.json` | `evidence_final_v9/junyi_partial.json` |

旧 `target_screen_v9/*_{direct,capacity}` 删除目标题身份，不是新的强
w/o Requirement，不能复用。正式文档中 Junyi 的 `Full - raw_summary`
约 `+0.0116`；相对本实验采用的强 `calibrated_summary_control` 仅约
`+0.00178`，后者才是 History 贡献的有效判据。

## 最少缺失任务

总矩阵为 `4 datasets x 2 splits x 4 variants = 32` 格。已存在 8 个 Full
和 2 个强 w/o History，共可复用 10 格；最多新增 22 个 validation 训练任务：

| 数据集 | S 缺失 | H 缺失 | 合计 |
|---|---|---|---:|
| ASSIST17 | w/o H、w/o R、w/o both | w/o R、w/o both | 5 |
| MOOCRadar | w/o H、w/o R、w/o both | w/o H、w/o R、w/o both | 6 |
| XES3G5M | w/o H、w/o R、w/o both | w/o H、w/o R、w/o both | 6 |
| Junyi | w/o H、w/o R、w/o both | w/o R、w/o both | 5 |

不直接启动 22 个任务，而分两阶段：

1. `requirement_gate`：只跑四数据集 S/H 的 8 个 w/o Requirement。
2. `full_factorial`：仅在 gate 通过后补 6 个 w/o History 与 8 个
   w/o both，共 14 个任务。

Requirement gate 预注册为：相对 `factorized_item_control`，Full 在至少
两个当前胜出数据集的 T 提升 `>=0.002`，其中一个 `>=0.003`；至少
一个数据集的 student-clustered paired-bootstrap 95% CI 下界大于 0；且每个
胜出数据集的 Full 任一 S/H/T 相对 control 回归不超过 `0.001`。第一阶段
不通过即停止，不补全 2x2，也不保留 Requirement 为论文模块。

阶段一计划：

```bash
bash scripts/run_factorized_requirement_factorial.sh \
  --data-root /home/xph/jwc/research/knofield_data \
  --legacy-result-root /home/xph/jwc/research/decoupled_cd_codex/results/goal_two_module \
  --stage requirement_gate \
  --devices cuda:0,cuda:2,cuda:3 \
  --max-parallel 3
```

默认只打印计划；正式运行还需追加
`--expected-commit <sha> --execute`。阶段二额外要求
`--stage full_factorial --gate-approved`。执行模式会拒绝 dirty worktree、
HEAD 与预期实现 commit 不一致、已存在的目标 artifact；并把 branch/HEAD
写入 identity log。启动任务前，runner 用
`configs/factorized_requirement_artifacts.json` 锁定并核对旧 Full/History
summary、checkpoint、逐行预测、train/valid/Q 哈希与完整配方；不读取 test。
当前轮只运行 smoke，不执行 8 个正式 gate 任务。

## 代码验收

必须同时满足：

- same-Q/different-item 在 control 中仍可区分；
- 固定 item vector 改变 Q 时 control 输出改变；
- Full 与 control 活跃 Requirement 参数量精确相等；
- 构造 dormant control 不推进 RNG，旧 Full 关键参数初始化哈希不变；
- 默认 Full architecture fingerprint 仍为 `099906acdba8c3b4`，其合成
  预测张量和 SHA-256 与冻结实现逐位相同；
- Full 只对 joint projection 产生梯度，control 只对 factorized branches
  产生梯度；
- 两路都继续消费公共 item difficulty 与 item-conditioned interaction；
- 旧 Full checkpoint 只允许缺失 dormant factorized 参数；若 summary 声明
  active `factorized_item_control`，缺失这些参数必须报错；
- summary 记录 `target_requirement_mode`、基础 architecture fingerprint、
  ablation variant fingerprint 与完整 initialization hash。
