# A2 关系图补全冻结验证（2026-07-12）

## 结论

A2 相对 A0v4 的冻结 relative gate **FAIL**；`test` 未打开，external gate 也保持关闭。A2 没有进行数值调参、dataset-specific 修改、retry、batch 改动或残差补丁。下一独立机制登记为 **MNAR / exposure-aware mastery completion**。

本次权威 campaign 为 `/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-ergc-r5-20260712`，seed `42`、split seed `2024`、cohort SHA-256 `6342dc8a5f73a4e03a1645780597b625c1480ba7a6513668b6766089cdd5b8a5`。A2 fingerprint 为 `270b7a8d456110f3c23b91df14b05d970569b46efb3fee57d974a79d9c91faa4`。

## 签发前修复与 r5 独立复现

首次 r4 preflight 在任何 A2 issuance 前发现 stable runner 把注册的 A2 completion loss weight `1.0` 错当作 evidence weight `0.1` 的倍率，生成了 `0.1`。TDD 修复把 stable-v4 的绝对 completion weight 与旧 v3 A1 的 evidence-scaled 语义分离；旧 A1 `0.1` 行为保持不变。基础设施提交为 `7aaf9a3551f8f3264051ee5852a7909e5edc7442`（`fix: separate stable completion loss weight`）。聚焦 controller/runner/train CLI 共 79 tests 通过，compileall 与 `git diff --check` 通过；按 standing instruction 未重跑完整 CPU suite。

r4 ledger 绑定旧 implementation tree，因此没有 reset 或 reseal r4，而是新建独立 r5。r1–r4 全部原样只读保留。r5 两个正式 preflight 均 exit 0、各核验 21 个冻结文件且 `test.csv` 引用为 0：A0v4 的 train argv 为 evidence/completion `0.1/0.0`；A2 正式 smoke 加六个 validation train argv 为 `0.1/1.0`。preflight 日志 SHA-256 分别为 `9d75045a275549355948bf2769bff07506bc98be2b867b3c864c00f2fc3c1ea1` 和 `2aa20bb203bdf3743cfc8dd4f04ecbd680a944a215ef1d3f01e998c73fdd504c`。

r5 A0v4 仅运行一次 smoke 和三个冻结 validation，随后 replay/freeze。与 r4 frozen A0v4 比较的 3 数据集 × 5 指标共 15 个 delta **全部精确为 0**，通过预注册的确定性容差 `|delta| <= 1e-12`。r5 A0 replay/freeze 文件 SHA-256 为 `01b8f53a27b6acb384f737ce9607fe898c5874c01dde8211bd887d7f7142f36b` / `7c31bf1a28e5363ed00c9cd8002a9b68d661b2897ab7be14c0fa5719477a7645`。r4 ledger/replay/freeze SHA-256 仍分别为 `3b17720f97f972b56f0c86c3e1711a5d1b676d0f2d8d82a48cede401bb23d953`、`d01059685dd0e2461a1e1885d1f415686ce4e7bf546e5703a761053a3633ee31`、`4fef4ae01ec7eebfff4a8dd1c7f37fdb03d556a2a42e079ff5fc659a05a5aff3`。

## 唯一 A2 smoke

A2 smoke 为 r5 `attempt-005`，只运行一次：GPU `GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0`，final loss `2.5158112049102783`，mastery shape `[3,3]`，parameter count `143520`，peak GPU memory `0.01954507827758789 GiB`。runner 内部 masked-edge、有限 graph gradient 与 observed mastery hard assembly 检查均未报错；无 OOM。

## 冻结 A2 validation 指标

三个任务使用 A0v4 的相同 recipe、seed、split seed、training mode、epoch、optimizer budget、student batch 与 concept dim；唯一模块变化是 A2 evidence-relational-graph completer，completion loss weight 固定 `1.0`。任务在三张不同物理 GPU 上并行首次完成，均 exit 0。

| Dataset | standard AUC | holdout AUC | zero AUC | ordinary DOA | weighted DOA | GPU / peak GiB | proof SHA-256 |
|---|---:|---:|---:|---:|---:|---|---|
| ASSIST17 | 0.7764165819107454 | 0.7750435082579055 | 0.7761611098937662 | 0.6748919716583789 | 0.6766088640440624 | `GPU-314fdf43-7a94-4e29-20d0-fbd496632b1b` / 0.3175182342529297 | `6e5800ec6281541bfe57584fe290f96e058356f0e9d493566ec683adeaa09bdb` |
| MOOCRadar | 0.9249833573955163 | 0.9227494090840868 | 0.9330473402019438 | 0.49995112463846225 | 0.6648709649944639 | `GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0` / 0.4546394348144531 | `bc838c5bdf8b395d7e2e8feea3810a90066e8c34d7bcdd365df384780177c6e1` |
| XES3G5M | 0.7789845092599862 | 0.7739484878701933 | 0.7731688683978026 | 0.4255411255411255 | 0.7358490566037735 | `GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0` / 0.15712976455688477 | `b0398555394ff6e7428695b8d281c19330a740ec2e92fb6dfa24cd253cf5f5ec` |

A2 replay proof SHA-256 为 `b04958941f632ab41aa665f36eecf6058df590e33e3bae1b8e7c4ab29b400de7`；replay 文件 SHA-256 为 `b640bc6f87e7dd5217c967283f949478120f27b117924024c8de2e54eda6ca13`。

## A2 − A0v4 精确 delta 与五项门

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

最终 `passed=false`。relative gate proof SHA-256 为 `300deb2cdd6e5d0d5313b0ef3fc781dc7f4048177cb892a5bcf8c1ec0aac084e`；`relative-gate.json` 文件 SHA-256 为 `530ca021f2b0096dea07e1ce83d9ce09b7f01d3e77592a263b671d8ac5a6cd48`。失败主因是 ASSIST17 holdout overall 回退以及 zero AUC 仅 XES3G5M 严格提升，不能用外部较弱基线掩盖同架构回退。

## 协议边界

relative gate FAIL；external gate remains closed。没有运行 `external-gate`、`run-test-once`、reference repo、push 或第二次 attempt。`test` 未打开。A2 不再做 residual patch；下一独立机制是 **MNAR / exposure-aware mastery completion**。
