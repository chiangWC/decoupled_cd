# AAAI 双路线实验最终进度（更新于 2026-07-11）

## 当前结论

seed 42 的两条正式路线已经结束，但必须按各自预注册问题分别计数：

- **完整模型 coverage-only 路线为 3/3**：ASSIST17、MOOCRadar、XES3G5M 均通过各自预注册的未覆盖知识预测门。XES 是固定单配置的正式复现，并同时通过 strict standard/holdout 两个 overall AUC 非回归门。
- **插件 joint standard/holdout 路线最终 test hard gate 为 0/3**：三个数据集虽在 validation 均获得 `decision=shared`，但 final test 的完整硬门均失败。validation `shared=3/3` 不能写成 test 成功。
- **完整模型的 joint AUC+DOA 结论不可判定**：XES 固定 v2 base 不暴露 per-concept mastery，DOA 为 missing，不是 0，也不是成功。
- **NIPS34 完整模型五候选 validation 为 `NO_SAFE`**：没有冻结配置，真实 test 的读取、哈希、claim 与 evaluation 均为 0。

因此论文表述应把“完整模型严格未覆盖知识预测 3/3”和“插件 joint hard gate 0/3”严格分开，不能相加、替代或混计。完整结果与协议边界见 `docs/complete_model_final_results_20260711.md`。

统一设置为 `train_seed=42`、`doa_seed=42`、`split_seed=2024`、`min_responses=3`。配置只由 validation 决定；冻结后每个 frozen ID 只执行一次 test actual。

## 完整模型路线：coverage-only 3/3

| 数据集 | 冻结配置 | test overall AUC | test zero AUC | 预注册目标 | 结论 |
|---|---|---:|---:|---:|---|
| ASSIST17 | v2 base, full batch, dim 64, 300 epochs | 0.7863331202 | **0.7839686719** | zero `>0.7808065` | PASS |
| MOOCRadar | hybrid+mono+UKC, student minibatch 64, dim 64, 30 epochs | 0.9293294694 | **0.9461244636** | zero `>0.9454` | PASS |
| XES3G5M holdout | v2 base-long, full batch, dim 64, 3000 epochs | 0.7864038697 | **0.7854583863** | zero `>0.7845` 且 overall `>=0.7828765165` | PASS |

MOOCRadar overall 比 `0.9300` 低 `0.0006705`，仍在允许的 `-0.002` 容忍区间内；其胜出依据是 zero AUC 严格超过 `0.9454`，不得扩写为 overall 超过 `0.9300`。

XES standard 的 test overall/zero AUC 为 `0.7925201779 / 0.7943522478`，holdout 为 `0.7864038697 / 0.7854583863`。相对 fresh ORCDF λ=0，strict overall AUC 的 standard 增量只有 `+0.0000057551`，holdout 增量为 `+0.0015273532`；按原始精度两门均通过，但 standard 的安全余量极小。

XES 的 base recipe 返回 `mastery=None`，validation DOA 为 missing；coverage-only test-once 也没有生成 test DOA。因此这里只能声称 coverage-only 3/3 和 strict two-split AUC PASS，不能声称 joint AUC+DOA 成功。

## 插件 joint standard/holdout：test hard gate 0/3

| 数据集 | standard AUC Δ | holdout AUC Δ | ordinary DOA Δ | weighted DOA Δ | 最终结果 |
|---|---:|---:|---:|---:|---|
| ASSIST17 | +0.0009150899 | +0.0011988625 | +0.0142295764 | +0.0074211223 | FAIL；ordinary DOA `0.7088566682` 未满足 `>0.708857` |
| XES3G5M | +0.0029974701 | +0.0032796101 | -0.0024450673 | -0.0077172576 | FAIL；两项 DOA delta 与绝对门均失败 |
| ASSIST09 | +0.0037351197 | +0.0030561481 | +0.0036118059 | +0.0035461488 | FAIL；ordinary DOA `0.6688742557` 未满足 `>0.670806` |

三个共享配方在 validation 的最小 AUC 安全余量分别为 ASSIST17 `0.0010801356`、XES3G5M `0.0021348435`、ASSIST09 `0.0029302412`，因此都获准进入 test-once；这只说明冻结条件通过。最终完整硬门要求两项 AUC 与 weighted DOA 不下降、ordinary DOA 严格提升且满足数据集绝对 DOA 门，结果仍为 0/3。

早期单划分插件“3/3”只保留为历史实验口径，不代表本轮 joint campaign 的最终结果，也不得与完整模型 3/3 混算。

## NIPS34 validation-only 负面结果

NIPS34 对 core、Tr2、Mo3、support、support+Tr2 五个预注册候选只运行 standard/holdout validation。support 系列虽然提高 overall AUC，但低覆盖和 DOA 出现严重回退：support 的 ordinary/weighted DOA 相对 core 分别为 `-0.2045724805 / -0.2236898113`，support+Tr2 分别为 `-0.1734080419 / -0.1814230313`。五个候选均未通过联合 hard gate，结论为 `NO_SAFE`；没有 selection、freeze 或 test claim，真实 test 保持关闭。

## Test-once、历史 test 与异常披露

完整模型统一 ledger 最终有 4 个唯一 claim：ASSIST17、MOOCRadar、XES standard、XES holdout。XES 两个新 test-once root 均只有 `attempt-001=dry_run` 和 `attempt-002=completed/0`；outer runner 只哈希对应 `frozen.json`，没有读取或哈希 test，真实 test 由 inner protocol 在排他 claim 成功后读取。

XES test 在更早的 complete 探索和插件路线中已经可见。因此本轮 XES 的含义是“忽略历史 test 数值、固定单一 base-long 配置、先按 validation 冻结、再执行正式 test-once 的预注册复现”，不能称为全局盲测。

XES actual 启动前曾有一次远端 shell 引号 syntax error；错误发生在 runner 启动前。复核确认当时没有 `attempt-002`、没有 claim、没有 actual、没有 test 读取或哈希。随后使用同一 dry-run argv、仅移除 `--dry-run` 执行唯一 actual。该操作错误需要披露，但不计为模型失败或 test 重试。

## 已确认的实现与解释边界

- r18 的 `plugin-decouple` 历史 optimizer 漏掉 `ukc_gate`；完全相同的 base/aux 结果是 no-op，不作为机制证据。
- r22 的归一化图行和恒为 `log(2)`，不能称作密度；corrected support 使用去自环后的 UKC 可达比例，并在系数为 0 时数值等价于 Mo1。
- NIPS 真实 validation 证明 corrected-support 不是可直接冻结的改进：overall 提升并未抵消低覆盖与 DOA 崩塌。
- 历史 SVGCD 阶段间残留梯度与 OneCycleLR step 问题已在正式 ASSIST09 重跑中修正；旧指标没有混入 joint final。
- vendor ORCDF/SVGCD 快照缺少许可证，只限远端内部实验，不得 push、公开或再分发。

## 正式产物

- 早期 complete ASSIST17/MOO：`/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/formal-aaai-20260710/complete/`
- XES complete 与 NIPS validation：`/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/formal-aaai-complete-joint-20260711/`
- 插件 joint final：`/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/formal-aaai-joint-splits-20260710/`

这些实验产物与 ledger 均不进入 Git；Git 只保存实现和可审查的结果文档。
