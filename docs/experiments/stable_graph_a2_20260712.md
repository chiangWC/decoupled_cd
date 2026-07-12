# A2 关系图补全冻结验证（2026-07-12）

## 结论

A2 r5 是 **protocol-invalid diagnostic**，不是已注册 A2 的冻结 relative gate。三个 A2 validation 实际使用 train CLI 默认 graph hidden dim `32`，违反冻结 recipe `64/256/64`；r5 smoke 也没有生成可验证的 masked-edge、hard-assembly 与 graph-gradient diagnostic schema。因此 r5 `relative-gate.json` 不具权威性，不能判定注册 A2。`test` 与 external gate 仍关闭；MNAR / exposure-aware mastery completion 的下一机制登记撤回，等待有效重跑。

本次只读诊断 campaign 为 `/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-ergc-r5-20260712`，seed `42`、split seed `2024`、cohort SHA-256 `6342dc8a5f73a4e03a1645780597b625c1480ba7a6513668b6766089cdd5b8a5`。A2 fingerprint 为 `270b7a8d456110f3c23b91df14b05d970569b46efb3fee57d974a79d9c91faa4`。

## 签发前修复与 r5 独立复现

首次 r4 preflight 在任何 A2 issuance 前发现 stable runner 把注册的 A2 completion loss weight `1.0` 错当作 evidence weight `0.1` 的倍率，生成了 `0.1`。TDD 修复把 stable-v4 的绝对 completion weight 与旧 v3 A1 的 evidence-scaled 语义分离；旧 A1 `0.1` 行为保持不变。基础设施提交为 `7aaf9a3551f8f3264051ee5852a7909e5edc7442`（`fix: separate stable completion loss weight`）。聚焦 controller/runner/train CLI 共 79 tests 通过，compileall 与 `git diff --check` 通过；按 standing instruction 未重跑完整 CPU suite。

r4 ledger 绑定旧 implementation tree，因此没有 reset 或 reseal r4，而是新建独立 r5。r1–r4 全部原样只读保留。r5 两个正式 preflight 均 exit 0、各核验 21 个冻结文件且 `test.csv` 引用为 0：A0v4 的 train argv 为 evidence/completion `0.1/0.0`；A2 正式 smoke 加六个 validation train argv 为 `0.1/1.0`。preflight 日志 SHA-256 分别为 `9d75045a275549355948bf2769bff07506bc98be2b867b3c864c00f2fc3c1ea1` 和 `2aa20bb203bdf3743cfc8dd4f04ecbd680a944a215ef1d3f01e998c73fdd504c`。

r5 A0v4 仅运行一次 smoke 和三个冻结 validation，随后 replay/freeze。与 r4 frozen A0v4 比较的 3 数据集 × 5 指标共 15 个 delta **全部精确为 0**，通过预注册的确定性容差 `|delta| <= 1e-12`。r5 A0 replay/freeze 文件 SHA-256 为 `01b8f53a27b6acb384f737ce9607fe898c5874c01dde8211bd887d7f7142f36b` / `7c31bf1a28e5363ed00c9cd8002a9b68d661b2897ab7be14c0fa5719477a7645`。r4 ledger/replay/freeze SHA-256 仍分别为 `3b17720f97f972b56f0c86c3e1711a5d1b676d0f2d8d82a48cede401bb23d953`、`d01059685dd0e2461a1e1885d1f415686ce4e7bf546e5703a761053a3633ee31`、`4fef4ae01ec7eebfff4a8dd1c7f37fdb03d556a2a42e079ff5fc659a05a5aff3`。

## 唯一 A2 smoke

A2 smoke 为 r5 `attempt-005`，只运行一次：GPU `GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0`，final loss `2.5158112049102783`，mastery shape `[3,3]`，parameter count `143520`，peak GPU memory `0.01954507827758789 GiB`。旧 runner没有记录 masked-edge、hard-assembly error、逐参数 gradient present/finite count 或 aggregate gradient norm；不能声称 runtime diagnostics 已通过。无 OOM 只是一项资源事实。

## 冻结 A2 validation 指标

三个任务继承相同 seed、split、epoch、optimizer budget 与 student batch，completion loss weight 为 `1.0`，但 graph hidden dim 错误地统一为默认 `32`，没有继承冻结 concept dim `64/256/64`。任务虽首次 exit 0，以下指标只能作为 protocol-invalid diagnostic。

| Dataset | standard AUC | holdout AUC | zero AUC | ordinary DOA | weighted DOA | GPU / peak GiB | proof SHA-256 |
|---|---:|---:|---:|---:|---:|---|---|
| ASSIST17 | 0.7764165819107454 | 0.7750435082579055 | 0.7761611098937662 | 0.6748919716583789 | 0.6766088640440624 | `GPU-314fdf43-7a94-4e29-20d0-fbd496632b1b` / 0.3175182342529297 | `6e5800ec6281541bfe57584fe290f96e058356f0e9d493566ec683adeaa09bdb` |
| MOOCRadar | 0.9249833573955163 | 0.9227494090840868 | 0.9330473402019438 | 0.49995112463846225 | 0.6648709649944639 | `GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0` / 0.4546394348144531 | `bc838c5bdf8b395d7e2e8feea3810a90066e8c34d7bcdd365df384780177c6e1` |
| XES3G5M | 0.7789845092599862 | 0.7739484878701933 | 0.7731688683978026 | 0.4255411255411255 | 0.7358490566037735 | `GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0` / 0.15712976455688477 | `b0398555394ff6e7428695b8d281c19330a740ec2e92fb6dfa24cd253cf5f5ec` |

A2 replay proof SHA-256 为 `b04958941f632ab41aa665f36eecf6058df590e33e3bae1b8e7c4ab29b400de7`；replay 文件 SHA-256 为 `b640bc6f87e7dd5217c967283f949478120f27b117924024c8de2e54eda6ca13`。

## Protocol-invalid A2 − A0v4 诊断 delta

| Dataset | Δ standard | Δ holdout | Δ zero | Δ ordinary DOA | Δ weighted DOA |
|---|---:|---:|---:|---:|---:|
| ASSIST17 | +0.0014313163188552913 | -0.001791471355351626 | -0.001219137427258432 | -0.004044384733056128 | -0.001563468596224582 |
| MOOCRadar | +0.0009221217654461489 | +0.0005682702409960383 | -0.000002208094605493649 | +0.0064278145627227334 | -0.00021292905203984525 |
| XES3G5M | +0.0003531456287206858 | +0.003821104390673624 | +0.003943417179277153 | -0.03030303030303033 | -0.009433962264150941 |

五项 failure vector：

- `standard_nonregression=true`；
- `holdout_nonregression=false`；
- `overall_nonregression=false`；
- `zero_wins=1`，因此 2/3 zero wins 门为 `false`；
- `zero_delta_at_least_0.001=true`。

旧 r5 文件写出 `passed=false`，但该结果不是有效五项门 verdict。relative gate proof SHA-256 为 `300deb2cdd6e5d0d5313b0ef3fc781dc7f4048177cb892a5bcf8c1ec0aac084e`；`relative-gate.json` 文件 SHA-256 为 `530ca021f2b0096dea07e1ce83d9ce09b7f01d3e77592a263b671d8ac5a6cd48`，两者仅作为 immutable invalid-run evidence。

## 协议边界

relative gate **未被有效执行**；external gate remains closed。没有运行 `external-gate`、`run-test-once`、reference repo 或 push。`test` 未打开。没有对 A2 做 residual patch。下一机制暂不登记；先在新独立 campaign 以正确 hidden dim 和强制 runtime diagnostics 重跑注册 A2。
