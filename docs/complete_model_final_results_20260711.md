# 完整模型路线最终结果与协议审计（2026-07-11）

## 1. 可发表结论

本轮完整模型的正式结论是：**coverage-only 成功 3/3**，数据集为 ASSIST17、MOOCRadar、XES3G5M。这里的成功定义是预注册的严格未覆盖知识预测门，不包含 DOA。

| 数据集 | 配置 | overall AUC | exact-zero AUC | gate | 判定 |
|---|---|---:|---:|---|---|
| ASSIST17 | v2 base；full batch；dim 64；300 epochs | `0.7863331202` | `0.7839686719` | zero `>0.7808065` | PASS |
| MOOCRadar | hybrid+monotonic+UKC；student batch 64；dim 64；30 epochs | `0.9293294694` | `0.9461244636` | zero `>0.9454`；overall 允许相对 `0.9300` 回退至 `-0.002` | PASS |
| XES3G5M holdout | v2 base-long；full batch；dim 64；3000 epochs | `0.7864038697` | `0.7854583863` | zero `>0.7845`；overall `>=0.7828765165` | PASS |

MOOCRadar 的 overall 距 `0.9300` 为 `-0.0006705`，处于 `-0.002` 容忍区间；它不是 overall 超越 `0.9300` 的结果。XES holdout 的 zero 余量为 `+0.0009583863`，原 overall 门余量为 `+0.0035273532`。

不得把这个 3/3 扩展成“完整模型 AUC+DOA 3/3”。XES 固定 base recipe 没有 per-concept mastery 输出，DOA 为 missing，joint AUC+DOA 的正式状态是 **NOT PROVEN / 不可判定**。

## 2. XES 双划分固定单配置复现

XES 使用同一固定 recipe 在 standard 与 holdout 分别训练：v2 base、concept dim 64、full batch、learning rate `0.001`、epochs `3000`、patience `50`、seed `42`。训练命令的 `--test-interactions` 在 validation 阶段指向各自 `valid.csv`；选择前没有读取、哈希或评估真实 test。

### Validation 与冻结

| split | best epoch | validation overall AUC | fresh ORCDF λ=0 门 | delta |
|---|---:|---:|---:|---:|
| standard | 482 | `0.7865296518` | `0.7863955904` | `+0.0001340614` |
| holdout | 256 | `0.7821793088` | `0.7816204531` | `+0.0005588557` |

standard/holdout validation exact-zero AUC 分别为 `0.7753021505` 和 `0.7807820399`。两套 coverage 都只有 exact-zero 与 full 桶，因此 low coverage 与 exact-zero 相同。

冻结后得到两个独立 frozen ID：

- standard：`cf6c12d31b99c63ee3bdda1c88219479b4bd0e7f685dc1264383b205f68e1818`
- holdout：`3b4f8ef50d5a35d6980baf3004c167318d8ebd49a28e896b9492775c88a1cb9e`

### Test-once 指标

| split | scope | n | AUC | ACC | RMSE |
|---|---|---:|---:|---:|---:|
| standard | overall | 20,714 | `0.7925201779` | `0.8331563194` | `0.3467559511` |
| standard | exact-zero / low | 9,063 | `0.7943522478` | `0.8342712126` | `0.3437676174` |
| standard | full | 11,651 | `0.7911340259` | `0.8322890739` | `0.3490628058` |
| holdout | overall | 41,758 | `0.7864038697` | `0.8361751042` | `0.3484068128` |
| holdout | exact-zero / low | 30,632 | `0.7854583863` | `0.8362170279` | `0.3484876030` |
| holdout | full | 11,126 | `0.7891098907` | `0.8360596800` | `0.3481842853` |

strict two-split overall AUC 以 fresh ORCDF λ=0 为对照：

| gate | observed | baseline | delta | 判定 |
|---|---:|---:|---:|---|
| standard overall | `0.7925201779` | `0.7925144228` | `+0.0000057551` | PASS |
| holdout overall | `0.7864038697` | `0.7848765165` | `+0.0015273532` | PASS |

standard 的增量只有约 `5.76e-6`；结论是按原始精度不回退，不能表述成有实质性安全余量。

### DOA 边界

validation DOA runner 正常完成，但记录明确为：`model exposes no per-concept mastery`。纯 base recipe 不构建 `mastery_head`，forward 返回 `mastery=None`，所以 `doa.json` 的 rows 为空、CSV 为空表。

这表示 DOA **missing / 不可计算**，而不是数值 0。coverage-only test-once 没有生成 test DOA；因此 joint AUC+DOA 既不能判 PASS，也不能声称 DOA 增强成功。

## 3. Test-once 审计

完整模型统一协议固定为 seed 42；ledger 位于 `/home/xph/jwc/research/decoupled_cd_codex/.git/codex-test-ledger/complete-v1/`。最终共有 4 个唯一 claim，分别对应 ASSIST17、MOOCRadar、XES standard 与 XES holdout。

XES 两个 frozen ID 的 test-once root 都严格只有：

- `attempt-001=dry_run`
- `attempt-002=completed/0`

没有 failed、running、第三个 attempt 或重复 claim。两个 actual outer runner 的 dataset 输入都只有 `frozen.json`，`test.csv` 输入数为 0；外层没有预先读取或哈希 test。inner protocol 先以排他方式落盘并 fsync claim，成功后才复制和读取真实 test。两个 evaluation record 与 claim 中声明的输出哈希已逐项重算一致。

XES standard/holdout test snapshot SHA-256 分别为：

- standard：`55d45ad44d0c278c41a210ed5e5737289b8ea713e2665e5fee175a6edac083fb`
- holdout：`a60f0a0d0701d34a46057beeaa6424e6c44bcfb44c21df07220b6818ee1e49d2`

XES test 在历史 complete 探索和插件路线中已经出现。本轮没有利用旧 test 数值调参，只执行固定单配置的 validation、freeze 和正式 test-once；所以应称“预注册正式复现”，不得称“全局盲测”或“历史首次打开 test”。

actual 启动前曾出现一次远端 shell 引号 syntax error。错误发生在 runner 创建 attempt 前；当时两个 `attempt-002`、两个新 claim 均不存在，也没有 test 读取或哈希。之后以同一 dry-run argv、仅去掉 `--dry-run` 运行唯一 actual。该错误属于包装器操作披露，不是模型失败，不计作 test attempt 或重试。

## 4. NIPS34 validation-only 负面证据

NIPS34 在 standard/holdout 上固定比较 core、Tr2、Mo3、support、support+Tr2 五个候选，所有训练输入与 evaluation 都只使用 validation。50 份 status 中真实 `/test.csv` 路径出现 0 次，complete ledger 没有新增 NIPS claim。

| candidate | Δ std overall | Δ std low | Δ hold overall | Δ hold low | Δ ordinary DOA | Δ weighted DOA |
|---|---:|---:|---:|---:|---:|---:|
| Tr2 | +0.0009535938 | 0 | +0.0008660760 | -0.0010765966 | -0.0019719249 | -0.0033629222 |
| Mo3 | -0.0121562254 | -0.1948051948 | -0.0113509932 | -0.0207551447 | -0.0049755657 | +0.0016334806 |
| support | +0.0030978662 | -0.1558441558 | +0.0020556377 | -0.0016273602 | -0.2045724805 | -0.2236898113 |
| support+Tr2 | +0.0044724810 | -0.0129870130 | +0.0025920409 | +0.0000474883 | -0.1734080419 | -0.1814230313 |

support 系列的 overall AUC 虽有提高，但 DOA 大幅崩塌，且 standard low slice 只有 18 条交互。没有任何 variant 通过“四项 AUC 与 weighted DOA 不下降、ordinary DOA 严格提升”的 hard gate；最终状态为 `NO_SAFE / needs_more_validation`。没有 selection、freeze、test claim 或 test evaluation。

## 5. 与插件 joint campaign 严格分账

插件 joint standard/holdout campaign 的 validation 结果为 `shared=3/3`，只表示三个配方达到冻结门；最终 test 完整硬门为 **0/3**：

- ASSIST17：AUC 与 DOA delta 均为正，但 ordinary DOA `0.7088566682` 未严格超过 `0.708857`。
- XES3G5M：standard/holdout AUC 均提高，但 ordinary/weighted DOA 均下降，且绝对门失败。
- ASSIST09：四项相对门均通过，但 ordinary DOA `0.6688742557` 未严格超过 `0.670806`。

这一插件 0/3 与完整模型 coverage-only 3/3 是不同模型、不同预注册问题和不同 ledger 的结果。不得用 validation `shared=3/3` 替代插件 test 0/3，也不得用完整模型 3/3 回填插件失败。

## 6. 资产与复现边界

- ASSIST17/MOO complete：`/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/formal-aaai-20260710/complete/`
- XES complete：`/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/formal-aaai-complete-joint-20260711/xes3g5m/`
- NIPS validation：`/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/formal-aaai-complete-joint-20260711/nips34/`
- 插件 joint final：`/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/formal-aaai-joint-splits-20260710/`

所有实验资产、claim ledger、日志、checkpoint 和数据都留在远端生成目录，不进入 Git。正式代码 worktree 为 `/home/xph/jwc/research/decoupled_cd_codex_worktrees/complete_model`；`/home/xph/jwc/research/decoupled_cd_v2` 仅作历史结果参考，不是代码来源。

## 7. 最终状态

`DONE`：完整模型 coverage-only 3/3；XES strict two-split AUC PASS；完整模型 joint AUC+DOA NOT PROVEN（DOA missing）；NIPS validation `NO_SAFE` 且 test=0；插件 joint final hard gate 0/3。
