# Unified Mastery 失败登记（A1，2026-07-12）

## 冻结结论

A1 相对 A0 的候选门失败，`test` 保持关闭。本登记只诊断 Task 8 注册的低秩 missing-mastery completer；没有启动数值调参、外部门、第二个替换模块或真实测试。

## 三数据集冻结证据

| Dataset | A1 std/hold/zero AUC | Δ vs A0 std/hold/zero | Δ vs audited external std/hold/zero | ordinary/weighted DOA | Recipe/proof |
|---|---|---|---|---|---|
| MOOCRadar | 0.9237381615694075 / 0.9207740355531081 / 0.9319816652749389 | -0.0003049148639236998 / -0.0014195154951669453 / -0.0006994812862632926 | -0.0059036502704337135 / -0.004168092697416692 / -0.0012955591178283044 | 0.5247008544427902 / 0.6647645004684439 | r2 / `01dd5e79...` |
| ASSIST17 | 0.7772810133205648 / 0.7771050187131403 / 0.7790516387954539 | 0.0022809629178004442 / 0.0003449112774648322 / 0.0017363756363880656 | -0.007139198095423072 / -0.007012527345769515 / -0.002996664331784693 | 0.679439145816845 / 0.6774372281073252 | r1 / `9395ce45...` |
| XES3G5M | 0.7744117095426273 / 0.7690837575240221 / 0.7658099393382194 | -0.004287354613667804 / -0.0011226916175651747 / -0.0034803057119332514 | -0.015043131434353985 / -0.011913832937965774 / -0.011004759754278837 | 0.49826839826839825 / 0.7358490566037735 | r0 / `00affc12...` |

外部差值只从 Task 5 audit SHA `071b5df25d8641fdc249a9a56175961025b3deeba36a4a471640563ef5d0b181` 的 `strongest_comparators` 动态读取，用于失败诊断，没有运行或伪造 `final-external-gate.json`。MOO 三项 comparator 均为 SVGCD；ASSIST17 和 XES 三项均为 ORCDF。六个 validation 均为 `attempt-001/exit 0`；GPU UUID/peak 与完整 raw proof SHA 见 A1 主报告。

## F-A1-001

`failure_id` -> `F-A1-001`

`failed dataset/split` -> `MOOCRadar/standard overall AUC`、`MOOCRadar/holdout overall AUC`、`MOOCRadar/holdout exact-zero AUC`

`exact double-precision delta` -> `-0.0003049148639236998`、`-0.0014195154951669453`、`-0.0006994812862632926`

`responsible module` -> A1 低秩 missing-mastery completer（rank `32`，observed-cell completion loss weight `0.1`）；其余 estimator、monotonic decoder、conditional-simplex behavior module 与 A0 相同。

`mechanism hypothesis` -> 仅用已测单元重构目标训练的自由 student/concept 因子，在 MOOCRadar 的高维 `concept_dim=256` 条件下可改善普通 DOA，却没有把可迁移结构约束到未测单元；新增自由度轻微扰动了整体排序并降低 exact-zero 泛化。

`literature retrieval question` -> 在认知诊断或稀疏二元矩阵补全中，哪些带结构先验、置信度校准或 inductive cold-start 约束的 mastery completer，能在不降低 observed/overall AUC 的前提下改善完全未观测 student-concept 单元？

`variable mapping` -> student=`student_factors/student_bias`；concept=`concept_factors/concept_bias`；观测指示=`mastery_observed_mask`；监督统计=`attempts/correct`；目标=`(correct+1)/(attempts+2)`；未测输出=`completion_predictions[~mastery_observed_mask]`；整体保护量=`standard_overall_auc, holdout_overall_auc`；冷启动量=`zero_auc`。

`replacement module input/output` -> 输入仍为 train-only student-concept evidence、observed mask 与可审计 concept structure；输出仍为 `[num_students, num_concepts]` 的 mastery 概率，只在 missing mask 上进入唯一 mastery。候选替换应是一个完整的结构约束/inductive completer，不改 decoder 或行为模块。

`acceptance gate` -> 同一冻结 cohort、seed `42`、split seed `2024`、双 validation 协议下：三个数据集 standard/holdout overall AUC 均不回退；至少 2/3 数据集 zero AUC 严格提升；至少一项 zero delta `>=0.001`；随后才允许执行 Task 5 strongest-comparator 外部门。

## F-A1-002

`failure_id` -> `F-A1-002`

`failed dataset/split` -> `XES3G5M/standard overall AUC`、`XES3G5M/holdout overall AUC`、`XES3G5M/holdout exact-zero AUC`

`exact double-precision delta` -> `-0.004287354613667804`、`-0.0011226916175651747`、`-0.0034803057119332514`

`responsible module` -> A1 低秩 missing-mastery completer（rank `32`，observed-cell completion loss weight `0.1`），不是 XES 的未注册 r1 fallback；本次严格继承 proof-ranked XES r0。

`mechanism hypothesis` -> XES 上 ordinary DOA 增加 `0.04242424242424242`，但 weighted DOA 降低 `0.009433962264150941`，表明自由低秩补全可能改善少数概念的次序一致性，同时在交互权重较大的概念和 overall 排序上产生系统性偏移；observed-cell 重构并不足以标识 missing-cell mastery。

`literature retrieval question` -> 哪些 weighted-risk-aware、graph-regularized 或 hierarchical Bayesian mastery completion 机制能把 observed-cell 证据外推到完全未测概念，并显式保护高频概念的加权排序与 overall AUC？

`variable mapping` -> 学生覆盖度=`attempt count per student-concept`；概念权重=`weighted DOA support/interaction frequency`；低秩表示=`student_factors @ concept_factors.T + biases`；训练目标=`masked_completion_loss on observed cells`；缺失预测=`hard missing-only assembly`；失败观测=`overall/zero/weighted-DOA deltas`。

`replacement module input/output` -> 输入保持 train-only evidence、mask、student/concept identity 和预注册结构先验；输出保持完整 `[S,K]` mastery 概率及可审计置信度，不允许读取 validation target 或真实 test。整体替换 completer，接口不扩散到 estimator/decoder/behavior modules。

`acceptance gate` -> 与 F-A1-001 相同的全局候选门；任何单个 standard 或 holdout overall 回退都失败。通过相对 A0 门后，才可从 Task 5 audited `strongest_comparators` 动态重建外部门，禁止手抄阈值。

## VOID-A2-r5-001（protocol-invalid diagnostic，不是失败登记）

`failure_id` -> `VOID-A2-r5-001`；本条从有效失败 registry 撤销，仅保留 immutable 诊断 provenance。

`architecture fingerprint` -> `270b7a8d456110f3c23b91df14b05d970569b46efb3fee57d974a79d9c91faa4`

`campaign / proof` -> `unified-ergc-r5-20260712`；A2 replay proof `b04958941f632ab41aa665f36eecf6058df590e33e3bae1b8e7c4ab29b400de7`；relative gate proof `300deb2cdd6e5d0d5313b0ef3fc781dc7f4048177cb892a5bcf8c1ec0aac084e`。

`all exact deltas (A2 - A0v4)` -> ASSIST17 standard/holdout/zero/ordinary DOA/weighted DOA = `+0.0014313163188552913 / -0.001791471355351626 / -0.001219137427258432 / -0.004044384733056128 / -0.001563468596224582`；MOOCRadar = `+0.0009221217654461489 / +0.0005682702409960383 / -0.000002208094605493649 / +0.0064278145627227334 / -0.00021292905203984525`；XES3G5M = `+0.0003531456287206858 / +0.003821104390673624 / +0.003943417179277153 / -0.03030303030303033 / -0.009433962264150941`。

`non-authoritative failure vector` -> `standard_nonregression=true`；`holdout_nonregression=false`；`overall_nonregression=false`；`zero_wins=1`；`zero_delta_at_least_0.001=true`。因 protocol defect，旧 `passed=false` 不是注册 A2 verdict。

`protocol defect` -> 三数据集没有显式传 `--unified-graph-hidden-dim`，实际均为默认 `32`，违反冻结 recipe `64/256/64`；旧 smoke 也没有可验证的 masked-edge、hard-assembly 与 graph-gradient diagnostic schema。故不能把 delta 归因于注册 A2 module。

`invalid-run observation only` -> hidden dim `32` 的非注册变体得到上述 diagnostic delta；不得据此提出注册 A2 的机制结论。

`next independent mechanism` -> **未登记**。原 MNAR / exposure-aware mastery completion 建议撤回，必须先用正确 hidden dim 与强制 smoke diagnostics 完成有效 A2 relative gate。

`test state` -> `test 未打开`；external gate 未运行并保持关闭；没有 retry、batch 改动、外部 gate、push 或 reference repo 修改。

`acceptance gate` -> 继续使用相同冻结 cohort、seed `42`、split seed `2024` 和五项相对门：三数据集 standard/holdout overall 均不回退，zero AUC 至少 2/3 严格提升且至少一项 delta `>=0.001`；通过后才能重建 external gate，随后才可能打开 test。

## F-A2-R6-001（有效 A2，Evidence-Relation Graph Completer）

`failure_id` -> `F-A2-R6-001`

`architecture fingerprint` -> `270b7a8d456110f3c23b91df14b05d970569b46efb3fee57d974a79d9c91faa4`

`campaign / proof` -> `unified-ergc-r6-20260712`；A2 replay proof `6c359b6b60ec7dd5e2cda089866c744814b41eeaeef7be99b2325efdbe58356e`；relative gate proof `ced71f0458e1477afddee7d94dd16187ca3d65672de581c170abfcd9a036e441`。

`all exact deltas (A2 - A0v4)` -> ASSIST17 standard/holdout/zero/ordinary DOA/weighted DOA = `+0.0017449611570987678 / -0.000001826372766688955 / -0.0008467777400537058 / -0.00464286452904028 / -0.004158277880485173`；MOOCRadar = `-0.0004243465040141281 / +0.00012095297817782402 / -0.0004240349482135253 / +0.04884646857106317 / -0.00012775743122395156`；XES3G5M = `-0.00187074496124906 / +0.0021598078518532127 / +0.0005681385058069477 / +0.030303030303030276 / -0.009433962264150941`。

`failure vector` -> `standard_nonregression=false`；`holdout_nonregression=false`；`overall_nonregression=false`；`zero_wins=1`；`zero_delta_at_least_0.001=false`；最终 `passed=false`。

`protocol validity` -> graph hidden dim 严格绑定冻结 recipe `64/256/64`；completion/evidence weights `1.0/0.1`；唯一 smoke masked edge `1`、hard-assembly error `0.0`、reconstruction loss `0.851538896560669`、grad present/finite `23/23`、nonzero parameter count `2`、aggregate norm `0.7495275139808655`。r6 A0 对 r4/r5 的 15 项 delta 全为 `0`。

`mechanism hypothesis` -> 正确容量的关系图 completer 能明显提高 MOO ordinary DOA 与 XES ordinary DOA，但 MOO/XES standard overall 回退、ASSIST17 holdout 微退，且 exact-zero 仅 XES 小幅提升。train-only observed-edge reconstruction 没有显式建模 concept exposure / attempt selection，missingness 的 MNAR 偏差仍可改变 overall 排序。

`next independent mechanism` -> **MNAR / exposure-aware mastery completion**。显式联合建模 exposure propensity 与 mastery completion，不给 A2 添加 residual 小补丁或 dataset-specific 数值。

`test state` -> `test 未打开`；external gate 未运行并保持关闭；没有 retry、batch/recipe 改动、push 或 reference repo 修改。

`acceptance gate` -> 同一冻结 cohort、seed `42`、split seed `2024`：三数据集 standard/holdout overall 均不回退；zero AUC 至少 2/3 严格提升且至少一项 delta `>=0.001`；通过后才允许重建 external gate。
