# A0v4 稳定图基线冻结验证（2026-07-12）

## 结论

A0v4 已在 `unified-ergc-r4-20260712` 冻结。唯一 smoke 与 MOOCRadar、ASSIST17、XES3G5M 三个 standard/holdout validation 均使用 seed 42、split seed 2024 和冻结 recipe；未运行 A2，未访问 test split。架构 fingerprint 为 `ec525cdef915be4d0bbfa7f82c23b5fe86be4d6cb81fbc795752cb09aeb209eb`，cohort SHA-256 为 `6342dc8a5f73a4e03a1645780597b625c1480ba7a6513668b6766089cdd5b8a5`。

## 路由与输入 provenance

- 冻结代码 commit：`32714c54237dee430ca7a99e896f79e9b32d4faf`（作者 `chiangWC <215551297+chiangWC@users.noreply.github.com>`）。
- implementation code SHA-256：`5bb8a12dc0bf4969d61313b672a08dc201bf08a4d65cff4d7365a06bfc84058f`。
- 权威数据根：`/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-mastery-20260712/controllers/a0-controller/data`。21 个 standard/holdout train、valid、Q、holdout-assignment 文件的 size/SHA 继承自已验证 v3 A0 proof，并在每次 issuance 前重新校验。
- r4 正式 preflight：3 数据集、6 split、21 文件、18 条 train/coverage/DOA argv；`test.csv` 引用数为 0。preflight log SHA-256：`03aed53a18273d6c6059d5136edab0f6d445265219e51a45a057f804569a3ed1`；CPU 双 split rehearsal manifest SHA-256：`70bec25b48064ce0faf1aa8faeff47c44545271c4832cfd8378d2ba48788cfc9`，source/aux-manifest 收集数为 6/1。
- ledger SHA-256：`3b17720f97f972b56f0c86c3e1711a5d1b676d0f2d8d82a48cede401bb23d953`；replay 文件 SHA-256：`d01059685dd0e2461a1e1885d1f415686ce4e7bf546e5703a761053a3633ee31`；freeze 文件 SHA-256：`4fef4ae01ec7eebfff4a8dd1c7f37fdb03d556a2a42e079ff5fc659a05a5aff3`。

## Smoke

`attempt-001` 在 `GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0` 完成 1 epoch：final loss `0.7542915940284729`，mastery shape `[3, 3]`，参数量 `133756`，峰值显存 `0.019381046295166016 GiB`。proof SHA-256 为 `2374678d0cc3a57d5116c15dbdf489b11b3847ef372f8cfaf1e23dfa800c17d2`，runner artifact SHA-256 为 `20b7327c035eef8f24d4a97a8f00046731ada29d6ab7a434bbd46c959c5cede8`。

## 冻结 validation 指标

| 数据集 | recipe | standard AUC | holdout AUC | zero AUC | ordinary DOA | weighted DOA | GPU UUID | 峰值显存 GiB | proof SHA-256 | aux manifest SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| ASSIST17 | 1 | 0.7749852655918901 | 0.7768349796132571 | 0.7773802473210246 | 0.6789363563914350 | 0.6781723326402870 | `GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0` | 0.3141498565673828 | `f9e58b3acb8df375e400ac4100b7c89a0418196e17bcf4d2ceba6f65619ddd95` | `044a53622a99572807aa8e0dd2645b245f403da25403cc93cb439900aa98c505` |
| MOOCRadar | 2 | 0.9240612356300701 | 0.9221811388430907 | 0.9330495482965493 | 0.4935233100757395 | 0.6650838940465037 | `GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0` | 0.44170427322387695 | `e64525050b02874f538fe3f7a2a758b1adac692be5fe057999fe48129599cff1` | `07c827449f2fb8b290b163c71cbaa5eb0689ebe4202aa8c8d707f568d8d05a24` |
| XES3G5M | 0 | 0.7786313636312655 | 0.7701273834795197 | 0.7692254512185255 | 0.45584415584415583 | 0.7452830188679245 | `GPU-314fdf43-7a94-4e29-20d0-fbd496632b1b` | 0.1502375602722168 | `a71ba4891f35467f20f44b186b72f69a16a49c84167f0649413c65cf72e7ee56` | `096351c8246c4807356c58e8f7533f871b86b60b160185db0730835c699e0272` |

replay proof SHA-256 为 `1bb6b39bd7454eda63b8b00b1d6652e0ff416c023ff18098c59b00a0e19e2fc8`；frozen proof SHA-256 为 `d6e3646be2e9a70f2f999ea5872c665464b3bf0bee206476632b27f08b5e8271`。

## 相对 v3 A0 的历史诊断

v3 A0 仅作历史对照，不属于 A0v4 proof。差值定义为 `A0v4 - v3 A0`。

| 数据集 | standard AUC 差值 | holdout AUC 差值 |
|---|---:|---:|
| ASSIST17 | -0.0000147848108742 | +0.0000748721775816 |
| MOOCRadar | +0.0000181591967389 | -0.0000124122051843 |
| XES3G5M | -0.0000677005250296 | -0.0000790656620676 |

v3 对照来源为 `unified-mastery-20260712/controllers/a0-primary-validation-rows.json`，对应 proof SHA 分别为 MOOCRadar `3cd57fe3477ab7b1ea2dc88b20b366dc42fc927966204e62df1fbbfed531d319`、ASSIST17 `d56bbb415d52f7182ebbb57616fb03c32ca72d079fee5ae91b077638ba418095`、XES3G5M `f18639f0058a7f7adfa5870f252f7ef6ee2ac8c35dd7c2d4a8647609519ca0c0`。

## 失败 campaign 审计

- r1 `unified-ergc-20260712`：smoke 在训练前因 obsolete `--unified-completion-rank 32` 被当前 train CLI 拒绝；失败 nonce `9a06c6931d859107b1aec1402ddb66a0afa7e01e6f0f1de6b2a3bf8438bcbc4c`，ledger SHA-256 `e8b1302f95e0ab7333364d53b9eeff4916b073f82166c5b087cdb473a1ff9f76`。
- r2 `unified-ergc-r2-20260712`：smoke 成功，三个 validation 在训练前因 controller 未绑定权威 data root 而失败；失败 counters 2–4 保持只读。
- r3 `unified-ergc-r3-20260712`：smoke 成功，三个 validation 完成训练后因闭集 aux 目录校验拒绝合法嵌套 `logs/` 而失败；失败 counters 2–4 保持只读，ledger SHA-256 `c96d9e28870f3faee4fac59c541c4d788996218ca453a2bbbb3ef2b1f53fdd5e`。

修复后，r4 使用由 trusted runner 最后写入的单一 `aux/artifact-manifest.json`，controller 以 no-follow/openat 独立递归枚举并重算所有 aux 文件；遗漏、修改或 symlink 均会拒绝。r1–r3 均未删除、重置或重分类。

## 测试与 test-closed 证明

提交前完整 CPU suite 共 354 tests；唯一失败是已知 SIGTERM 时序测试，随后隔离复跑 1/1 通过。`compileall` 与 `git diff --check` 通过。r4 campaign 只含 smoke 与 validation counters 1–4，以及 replay/freeze decisions；对 attempts/decisions 搜索 `test.csv`、`stable-test`、test split 无匹配，未执行 `run-test-once`。
