# A2 关系图补全有效冻结验证 r6（2026-07-12）

## 结论

`unified-ergc-r6-20260712` 是修复 graph hidden dim 与强制 smoke runtime diagnostics 后的首个协议有效 A2 campaign。A0v4 对 r4/r5 的 15 项指标全部精确复现；唯一 A2 smoke diagnostics 全部通过；三个冻结 A2 validation 首次完成。权威 relative gate **FAIL**，因此 external gate 和 test 均保持关闭。下一独立机制登记为 **MNAR / exposure-aware mastery completion**。

seed `42`、split seed `2024`、cohort SHA-256 `6342dc8a5f73a4e03a1645780597b625c1480ba7a6513668b6766089cdd5b8a5`；A2 fingerprint `270b7a8d456110f3c23b91df14b05d970569b46efb3fee57d974a79d9c91faa4`。

## 协议修复与 preflight

提交 `c7fc2b81bd58389ecb88f76d0a549db850fadb86` 将 A2 validation 的 graph hidden dim 显式绑定到冻结 recipe concept dim，并增加 omission/mismatch preflight 拒绝；A2 smoke 使用显式 `--a2-smoke-diagnostics`，train-side diagnostic 只做 deterministic forward 与 `autograd.grad`，不做 optimizer step，也不改 checkpoint。runner 与 controller 都要求 exact diagnostic schema，并有 missing/tamper tests。

focused controller/runner/train tests 83 个通过，compileall 与 `git diff --check` 通过。preflight A2 日志 SHA-256 `51dcf0a61a81fe8a4ebb576d0b928eafeeb1b32fd3e6cb4972767516d26496ce`：smoke hidden `4` 和 diagnostic flag 各一次；ASSIST17/XES 双 split hidden `64` 共四次；MOO 双 split hidden `256` 两次；正式 completion `1.0` 七次；21 个冻结文件，`test.csv` 引用 0。

## A0v4 独立复现

r6 仅运行一次 A0 smoke、三个 A0 validation、replay/freeze。相对 r4 和 r5 的 ASSIST17/MOOCRadar/XES3G5M × standard/holdout/zero/ordinary DOA/weighted DOA 共 15 个 delta 全部精确为 `0`。A0 replay/freeze 文件 SHA-256 为 `6bb152b39cb1deed0e3ca014df5d09f65f902ae6c0dfed550524cee6a282aa2e` / `b90ace883b2bbb4a7ec4071dcc778e2e92982fccb8b1999a12cde98adff51aa2`。

## 唯一 A2 smoke

r6 `attempt-005` 在 `GPU-8b057858-863a-bb19-1102-4c7ee3d1afe0` 首次完成，final loss `1.6073514223098755`、mastery `[3,3]`、peak `0.019424915313720703 GiB`。diagnostics：

- masked edge count `1`；
- observed hard-assembly max absolute error `0.0`；
- reconstruction loss `0.851538896560669`；
- graph completer trainable parameter tensors `23`，gradient present `23`、finite `23`、nonzero parameter count `2`；
- aggregate gradient norm `0.7495275139808655`。

所有 preregistered runtime invariants 通过，无 OOM；diagnostic 未 optimizer-step 或写 checkpoint。

## 有效 A2 validation 指标与 delta

三个任务使用 completion weight `1.0`、evidence weight `0.1`，graph hidden dim 分别为 ASSIST17 `64`、MOOCRadar `256`、XES3G5M `64`，并在三张不同物理 GPU 上首次 exit 0。

| Dataset | standard | holdout | zero | ordinary DOA | weighted DOA | Δ standard | Δ holdout | Δ zero | Δ ordinary | Δ weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ASSIST17 | 0.7767302267489888 | 0.7768331532404904 | 0.7765334695809709 | 0.6742934918623947 | 0.6740140547598018 | +0.0017449611570987678 | -0.000001826372766688955 | -0.0008467777400537058 | -0.00464286452904028 | -0.004158277880485173 |
| MOOCRadar | 0.923636889126056 | 0.9223020918212685 | 0.9326255133483358 | 0.5423697786468027 | 0.6649561366152797 | -0.0004243465040141281 | +0.00012095297817782402 | -0.0004240349482135253 | +0.04884646857106317 | -0.00012775743122395156 |
| XES3G5M | 0.7767606186700164 | 0.7722871913313729 | 0.7697935897243324 | 0.4861471861471861 | 0.7358490566037735 | -0.00187074496124906 | +0.0021598078518532127 | +0.0005681385058069477 | +0.030303030303030276 | -0.009433962264150941 |

A2 replay proof SHA-256 `6c359b6b60ec7dd5e2cda089866c744814b41eeaeef7be99b2325efdbe58356e`；replay 文件 SHA-256 `429dc503f84ebac62d44e94db379374447c9249d0f13279254a4f3140d9ba690`。

## 权威 relative gate

Failure vector：

- `standard_nonregression=false`（MOO、XES 回退）；
- `holdout_nonregression=false`（ASSIST17 回退）；
- `overall_nonregression=false`；
- `zero_wins=1`，2/3 门失败；
- `zero_delta_at_least_0.001=false`，最大正增益仅 XES `+0.0005681385058069477`。

最终 `passed=false`。relative gate proof SHA-256 `ced71f0458e1477afddee7d94dd16187ca3d65672de581c170abfcd9a036e441`；decision file SHA-256 `a1a63502440d2de3feabc8912692a9e2865b4fa18fb9e8ac20d7b549c0f715fc`。

## 协议边界

没有 retry、reset、batch/recipe 改动、dataset-specific 数值、A2 residual patch、external gate、test、push 或 reference repo 修改。r1–r5 全部只读保留；r5 A2 仍只作为 protocol-invalid evidence。有效 r6 gate FAIL 后，`test` 未打开，external gate remains closed。下一独立机制是 **MNAR / exposure-aware mastery completion**。

## 最终软件验证与审计边界

完整 CPU suite 运行 `359` 项，唯一失败为已知 timing-sensitive `test_sigterm_to_runner_forwards_to_nested_child_process_group`；按记录政策立即隔离重跑该精确用例，`1/1` 通过。完整日志 `audit/final-verification/full-unittest-discover.log` SHA-256 为 `e2e9f193746f6bb313108ea1fc20ed355ebe9ff77546e75bb28e4c72ab1e2fae`；隔离日志 `audit/final-verification/isolated-sigterm-rerun.log` SHA-256 为 `5a860203e4d129be58e7b3e196ee5d19787208995a231c73c4dd5b9299744883`。退出码与 `SHA256SUMS` 同目录归档。`compileall`、`git diff --check`、controller status 均通过；attempt/ledger/decision 受保护文件的聚合 SHA-256 保持 `a9c4e6eddd7da40953eb070d76fca2ebf7f5b1f1f42515eb81a0573cb5ffffbb`。

controller summary 数值交叉校验留给下一独立 campaign，因为任何 controller/summary 实现改动都会改变 r6 implementation identity 并使现有权威链失效。当前 r6 的 preflight、冻结 runner argv 与 checkpoint/summary provenance 已绑定 graph hidden dim `64/256/64`、completion/evidence weights `1.0/0.1`；本次关闭只增加审计日志和文档，没有修改代码、ledger、decision 或 test-closed 状态。
