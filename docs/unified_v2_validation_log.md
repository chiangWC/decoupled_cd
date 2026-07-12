# Unified V2 validation-only module matrix

Date: 2026-07-11

Seed: 42

Artifact root: `/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-v2-20260711`

## Protocol boundary

This ledger is validation-only. Every training invocation passed the matching
`valid.csv` as both the CLI validation input and the CLI evaluation input. No
Task 8 campaign attempt received, opened, or hashed a real test CSV. Outer
attempt input hashes cover only train, valid, Q-matrix, and (for holdout DOA)
the holdout-assignment CSV.

The B0 outer attempts predate the cohort freeze. They therefore have no outer
immutable binding to either the later cohort artifact or an architecture
manifest. Their inner summaries do verify seed 42, a common B0 fingerprint,
finite metrics, nonempty mastery, and validation-only routing, but those inner
checks make B0 a structural reference only; they do not retroactively turn the
attempts into immutable cohort-bound candidate evidence.

The exact-zero eligible pool is ASSIST09, ASSIST17, MOOCRadar, and XES3G5M.
NIPS34 is asset-ready but has 0 standard and 0 holdout exact-zero validation
rows, so it is not counted or frozen. Junyi and EdNet-ICDM remain provisional.

The frozen cohort is **structural/audit-complete only**, not baseline-safe and
not a successful final cohort. Its selection rule was: exact-zero eligibility,
both B0 validation splits complete, finite loss and metrics, nonempty mastery,
seed 42, and one B0 architecture fingerprint. This rule does not assert an
absolute performance floor. The B0 zero AUC values fail the known absolute
targets for ASSIST17 (`0.533314 < 0.7808065`), MOOCRadar
(`0.661149 < 0.9454`), and XES3G5M (`0.618938 < 0.7845`); ASSIST09 has no
fresh frozen external baseline in this campaign.

- Cohort artifact: `<artifact-root>/cohort/cohort.json`
- Cohort SHA-256: `77ba446b4cd1e67cb25a5c6e754c6ffb78788fba1519b3ec65d269703313444f`
- B0 fingerprint: `2323e72b72333efcc74fc51ee315c9c41f741757a8b142886f83b7298fed0c45`
- M2 fingerprint: `18681d8cf6e52f2a8055c87d7194034ef70d708bf0d96b7fd55e2d78ebaadb6c`
- M2+M3 fingerprint: `3b41caad78a64db7c96e9a6ba43b7397fa82f21b366e3b0909c8010f9fd0e461`

Metric rows use standard overall AUC, holdout overall AUC, holdout exact-zero
AUC, and standard ordinary/weighted DOA, matching the global selector schema.

## Validation metrics

| Candidate | Dataset | Standard overall AUC | Holdout overall AUC | Zero AUC | Ordinary DOA | Weighted DOA | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| B0 | ASSIST09 | 0.5791774813 | 0.5636198806 | 0.5285085930 | 0.6107170488 | 0.6667775590 | structural reference only |
| B0 | ASSIST17 | 0.5661479857 | 0.5493992376 | 0.5333137376 | 0.6503196268 | 0.6812306966 | structural reference only |
| B0 | MOOCRadar | 0.7303817255 | 0.7204413174 | 0.6611493025 | 0.5959125246 | 0.6705242313 | structural reference only |
| B0 | XES3G5M | 0.6237583746 | 0.6193148526 | 0.6189378030 | 0.5057359307 | 0.6792452830 | structural reference only |
| M2 graph+mask | ASSIST09 | 0.5782989044 | 0.5654950642 | 0.5354634948 | 0.6162482790 | 0.6688933837 | global FAIL |
| M2 graph+mask | ASSIST17 | 0.5790737943 | 0.5709013063 | 0.5731749771 | 0.6615838856 | 0.6808686302 | global FAIL |
| M2 graph+mask | MOOCRadar | 0.7321349112 | 0.7246998670 | 0.6472872885 | 0.5946449721 | 0.6702474236 | global FAIL |
| M2 graph+mask | XES3G5M | 0.6238292218 | 0.6122507385 | 0.5996447360 | 0.5057359307 | 0.6792452830 | global FAIL |
| M2+M3 graph+coverage | ASSIST09 | 0.5779440396 | 0.5672360288 | 0.5395943682 | 0.6219328595 | 0.6668707085 | global FAIL |
| M2+M3 graph+coverage | ASSIST17 | 0.5783688058 | 0.5675990123 | 0.5675493235 | 0.6873696459 | 0.6804133043 | global FAIL |
| M2+M3 graph+coverage | MOOCRadar | 0.7501073899 | 0.7407644988 | 0.6669306866 | 0.6098533288 | 0.6660740141 | global FAIL |
| M2+M3 graph+coverage | XES3G5M | 0.6309429876 | 0.6172572317 | 0.6051007203 | 0.5057359307 | 0.6792452830 | global FAIL |

## Global gate decisions

M2 was compared with B0 only after all eight M2 split attempts completed.
Decision artifact: `<artifact-root>/m2/decision/candidate-decision.json`.
This Task 8 comparison is retained as exploratory diagnosis. It predates the
executable authorization path described below and is not a registered Task 9
iteration.

| Gate | M2 result |
|---|---|
| Standard overall AUC non-regression on every dataset | FAIL: ASSIST09 `-0.0008785769` |
| Holdout overall AUC non-regression on every dataset | FAIL: XES3G5M `-0.0070641142` |
| Weighted DOA non-regression on every dataset | FAIL: ASSIST17 `-0.0003620664`, MOOCRadar `-0.0002768078` |
| Zero AUC strict improvement on at least 3/4 | FAIL: 2/4 (ASSIST09, ASSIST17) |
| At least one zero AUC delta >= 0.001 | PASS: ASSIST09, ASSIST17 |
| Ordinary DOA strict improvement on at least 3/4 | FAIL: 2/4 (ASSIST09, ASSIST17) |

M2+M3 was compared with M2 only after all eight M2+M3 split attempts
completed. This was an exploratory/falsification run outside the
preregistered stop rule: no executable authorization token was issued or
consumed before its attempts. It must not be cited as evidence that the stop
controller authorized a registered continuation.
Decision artifact: `<artifact-root>/m2-m3/decision/candidate-decision.json`.

| Gate | M2+M3 result |
|---|---|
| Standard overall AUC non-regression on every dataset | FAIL: ASSIST09 `-0.0003548648`, ASSIST17 `-0.0007049885` |
| Holdout overall AUC non-regression on every dataset | FAIL: ASSIST17 `-0.0033022940` |
| Weighted DOA non-regression on every dataset | FAIL: ASSIST09 `-0.0020226753`, ASSIST17 `-0.0004553259`, MOOCRadar `-0.0041734094` |
| Zero AUC strict improvement on at least 3/4 | PASS: ASSIST09, MOOCRadar, XES3G5M |
| At least one zero AUC delta >= 0.001 | PASS: ASSIST09, MOOCRadar, XES3G5M |
| Ordinary DOA strict improvement on at least 3/4 | PASS: ASSIST09, ASSIST17, MOOCRadar |

Both candidates fail globally. No numerical tuning was launched because no
module set passed its hard gate.

### Executable stop controller for Task 9

Task 9 begins a new registered iteration, independent of the exploratory Task
8 failures. `controller-init` must first create an exclusive controller state
directory bound to the exact route HEAD, canonical cohort and candidate
manifest/fingerprint, fixed dataset order, and a controller-owned copy/hash of
the complete baseline rows. The first `authorize` call starts from zero
successes and four remaining datasets but issues one-time capabilities only
for the first dataset's standard/holdout pair. There is no override option.

Capabilities are random bearer references backed by exact records in the
controller's `issued/` registry; they have no caller-computable
self-authentication hash. At the first line of `run-split`, the controller
holds its flock, verifies the exact route/counter/dataset/split/nonce registry
record, and atomically moves that split capability to `consumed/` before child
output, GPU, or data work. Fabricated, stale, mismatched, and reused
capabilities fail. An unauthorized outer attempt may leave a failed shell, but
cannot train/read child data or become progress proof.

The next `authorize` derives progress only from the prior pair's consumed
records and their fixed outer attempt directories. Both `status.json` files
must close over the exact route commit, seed, cohort/manifest immutable inputs,
dataset path/size/hashes, and validation-summary output path/size/hash. Each
summary must bind the registered controller/capability and remain valid-only.
A dataset counts as a success only if standard and holdout overall AUC and
weighted DOA do not regress while zero AUC and ordinary DOA both strictly
improve. If `successes + remaining < 3`, issuance fails closed. Final global
success also requires at least one zero-AUC delta of `0.001` or greater.
Task 9 has not been initialized or run by this ledger update.

### Cumulative diagnosis against B0

Although the historical incremental M2+M3 comparison uses M2 as its reference,
the cumulative B0 comparison isolates the replacement target for Task 9.
ASSIST09, ASSIST17, and MOOCRadar all improve holdout zero AUC and standard
ordinary DOA relative to B0. However, ASSIST09 standard overall AUC falls by
`-0.0012334417`; ASSIST17 weighted DOA falls by `-0.0008173924`; and
MOOCRadar weighted DOA falls more clearly by `-0.0044502172`. XES3G5M
holdout zero AUC falls by `-0.0138370827` and holdout overall AUC falls by
`-0.0020576209`. The absolute zero AUC values also remain far below the
historical external targets, so these validation gains do not establish a
safe or competitive final architecture.

## GPU and attempt ledger

Every real job is `<artifact-root>/<candidate>/<dataset>/<split>/attempt-001`.
All 24 real attempts completed with exit code 0, finite loss, nonempty mastery,
and no OOM-triggered retry or batch change.

These Task 8 outer attempts were created before the executable authorization
contract. In particular, their absence of authorization/manifest bindings is
why M2 and M2+M3 remain exploratory evidence despite the valid inner metrics.

| Candidate | Dataset | Standard GPU / peak GiB | Holdout GPU / peak GiB |
|---|---|---:|---:|
| B0 | ASSIST09 | GPU2 / 1.1156 | GPU2 / 1.0968 |
| B0 | ASSIST17 | GPU2 / 0.7759 | GPU2 / 0.7563 |
| B0 | MOOCRadar | GPU0 / 0.3203 | GPU3 / 0.3003 |
| B0 | XES3G5M | GPU0 / 0.1258 | GPU3 / 0.1258 |
| M2 | ASSIST09 | GPU2 / 1.4772 | GPU2 / 1.4584 |
| M2 | ASSIST17 | GPU3 / 0.9671 | GPU3 / 0.9475 |
| M2 | MOOCRadar | GPU0 / 0.4689 | GPU0 / 0.4638 |
| M2 | XES3G5M | GPU0 / 0.1621 | GPU0 / 0.1617 |
| M2+M3 | ASSIST09 | GPU2 / 1.7914 | GPU2 / 1.7881 |
| M2+M3 | ASSIST17 | GPU2 / 1.0957 | GPU2 / 1.0760 |
| M2+M3 | MOOCRadar | GPU2 / 0.5028 | GPU2 / 0.4804 |
| M2+M3 | XES3G5M | GPU2 / 0.1637 | GPU2 / 0.1632 |

The runner audits
`index,memory.used,memory.total,utilization.gpu` before each real job, allows
only GPUs below half memory, and holds `/tmp/unified-v2-gpu-<index>.lock`.
When all eight M2+M3 attempts were launched in a short burst, they all
preselected then-idle GPU2 before the first job allocated memory. The flock
correctly serialized them and no attempt was interrupted or restarted, but
parallel throughput was lost. Future Task 9 attempts should be started in
small GPU-availability batches so each audit observes the preceding allocation.

## Synthetic smoke

Successful smoke artifact: `<artifact-root>/smoke/attempt-004/smoke-summary.json`.
It contains CPU and GPU one-epoch records for B0, M2, and M2+M3. All six have
nonempty `[3,3]` mastery, finite loss, the expected fingerprint, and GPU jobs
record positive peak memory (`0.016878` to `0.016890` GiB).

Immutable negative smoke attempts were retained:

- attempt-001: outer argv accidentally tried to execute `--` (exit 127).
- attempt-002: direct script entrypoint lacked project-root import setup (exit 1); fixed with TDD.
- attempt-003: synthetic validation targets duplicated history interactions and were rejected by the visibility guard (exit 1); fixed with a disjoint-target TDD regression.

## Numerical recipes

- ASSIST09 and ASSIST17: full batch, concept dim 64, 300 epochs, learning rate
  `0.001`, weight decay `0`, patience 5.
- MOOCRadar: student-recompute minibatch 64, concept dim 64, 30 epochs,
  learning rate `0.001`, weight decay `0`, patience 5.
- XES3G5M: student-recompute minibatch 64, concept dim 64, 30 epochs,
  learning rate `0.001`, weight decay `0`, patience 5. This M2-safe recipe was
  fixed before attempts; it was not an OOM-triggered replacement.
- Unified mastery loss weight: nonzero `0.1` for all candidates.

Architecture manifests and fingerprints were identical across datasets;
numerical recipes did not change architecture fingerprints.

## Task 9：NeuralCDM M4 注册验证（2026-07-12）

本轮在干净路由提交
`e1d370717d6872d2bf32aafe7d84c165ac983616` 上执行，固定 cohort SHA-256 为
`77ba446b4cd1e67cb25a5c6e754c6ffb78788fba1519b3ec65d269703313444f`，
artifact root 为
`/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-v2-neuralcdm-20260712`。
B0、M2、M2+M3 的新 fingerprint 分别为 `83084810da46...cd239`、
`756fe4732d58...4888`、`7b59fafb5d27...3191`。所有真实实验仅由各自的
controller `authorize` 后调用 controller-owned `run-pair`；没有手工
`run-split`、调参、测试集读取、OOM 重试或 batch 修改。

括号内为相对同构旧 M4 行的精确 delta。指标口径依次为 standard overall
AUC、holdout overall AUC、holdout zero AUC、standard ordinary DOA、standard
weighted DOA。

| 架构 | 数据集 | Standard overall AUC | Holdout overall AUC | Zero AUC | Ordinary DOA | Weighted DOA | Joint gate |
|---|---|---:|---:|---:|---:|---:|---|
| B0 | ASSIST09 | 0.6949973902 (+0.1158199089) | 0.6533762320 (+0.0897563513) | 0.6272741467 (+0.0987655538) | 0.6286784893 (+0.0179614404) | 0.6753073934 (+0.0085298345) | PASS |
| M2 | ASSIST09 | 0.6936857244 (+0.1153868200) | 0.6532055897 (+0.0877105255) | 0.6413193019 (+0.1058558071) | 0.6261427773 (+0.0098944982) | 0.6753207005 (+0.0064273168) | PASS |
| M2+M3 | ASSIST09 | 0.6884736734 (+0.1105296338) | 0.6548264160 (+0.0875903871) | 0.6205597548 (+0.0809653866) | 0.6213017065 (-0.0006311529) | 0.6750146378 (+0.0081439293) | FAIL |
| B0 | ASSIST17 | 0.6980089267 (+0.1318609410) | 0.6268888753 (+0.0774896377) | 0.5697472419 (+0.0364335044) | 0.6511864275 (+0.0008668006) | 0.6807698849 (-0.0004608118) | FAIL |
| M2 | ASSIST17 | 0.6219238593 (+0.0428500650) | 0.6610505093 (+0.0901492030) | 0.6579489025 (+0.0847739254) | 0.6886857636 (+0.0271018780) | 0.6801993560 (-0.0006692743) | FAIL |
| M2+M3 | ASSIST17 | 0.6185213781 (+0.0401525724) | 0.5818714611 (+0.0142724489) | 0.5445743919 (-0.0229749316) | 0.6902001755 (+0.0028305296) | 0.6762879415 (-0.0041253628) | FAIL |
| B0 | MOOCRadar | 0.8182326299 (+0.0878509043) | 0.8115020177 (+0.0910607003) | 0.7522291658 (+0.0910798633) | 0.5942040982 (-0.0017084263) | 0.6576846095 (-0.0128396218) | FAIL |
| M2 | MOOCRadar | 0.8151590143 (+0.0830241032) | 0.8117347245 (+0.0870348575) | 0.7583983667 (+0.1111110782) | 0.5922534129 (-0.0023915592) | 0.6532556852 (-0.0169917384) | FAIL |

GPU 记录中的 peak 为训练摘要的 PyTorch allocated peak；每个对应 outer
`status.json` 均为 `completed`、exit code 0，并另有进程树采样峰值。

| 架构 | 数据集 | Standard GPU / peak GiB / outer MiB | Holdout GPU / peak GiB / outer MiB |
|---|---|---:|---:|
| B0 | ASSIST09 | GPU3 / 2.3524584770 / 3650 | GPU3 / 2.2860584259 / 3558 |
| M2 | ASSIST09 | GPU2 / 2.7135171890 / 4140 | GPU0 / 2.6470146179 / 4054 |
| M2+M3 | ASSIST09 | GPU0 / 2.9388108253 / 4408 | GPU0 / 2.8728456497 / 4332 |
| B0 | ASSIST17 | GPU2 / 2.5146756172 / 4182 | GPU2 / 2.4285235405 / 4058 |
| M2 | ASSIST17 | GPU0 / 2.7063961029 / 4542 | GPU0 / 2.6197180748 / 4418 |
| M2+M3 | ASSIST17 | GPU0 / 2.8349676132 / 4688 | GPU2 / 2.7481746674 / 4568 |
| B0 | MOOCRadar | GPU3 / 0.4798274040 / 2248 | GPU3 / 0.4413785934 / 2226 |
| M2 | MOOCRadar | GPU2 / 0.6296601295 / 10064 | GPU2 / 0.5895938873 / 10064 |

所有 split 均为各自 immutable root 下的 `attempt-001`。由于外部 GPU2 上
同时存在其他进程，M2 的 MOOCRadar outer 进程树采样峰值 10064 MiB 明显高于
PyTorch 自身的 0.63 GiB；因此模型峰值以训练摘要为准，outer 值保留为设备级
审计证据，不能归因于本模型。

### Controller 终态与结论

- B0：cursor 3，successes 1，issuance counter 3，`blocked=true`；
  ASSIST09 PASS，ASSIST17/MOOCRadar FAIL。
- M2：cursor 3，successes 1，issuance counter 3，`blocked=true`；
  ASSIST09 PASS，ASSIST17/MOOCRadar FAIL。
- M2+M3：cursor 2，successes 0，issuance counter 2，`blocked=true`；
  ASSIST09/ASSIST17 均 FAIL。

三个 controller 均因 `successes + remaining < 3` 按注册 stop rule 永久停止。
没有签发 XES3G5M capability，也没有 XES3G5M attempt。新 M4 在所有已运行
数据集上提高 standard overall AUC，但 ASSIST17 和 MOOCRadar 的 weighted DOA
发生回退，直接触发预注册反证条件。因此 Task 9 为负结果、global gate FAIL；
停止继续修改 M4，后续如切换整套 M1 必须另立任务与注册决策。

辅助 work JSON 中，B0 各 holdout 的 `holdout_doa_spearman` 因常数/不可定义
为 `NaN`，ASSIST09 standard 的若干空 coverage slice AUC 也为 `NaN`；这些值
未进入冻结 summary、proof 或上述 gate 指标。必需指标、loss 与 mastery 均有限。
