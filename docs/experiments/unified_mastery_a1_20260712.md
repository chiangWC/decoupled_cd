# Unified Mastery A1 冻结验证报告（2026-07-12）

## 结论

A1 已在冻结 cohort 的三个 primary 数据集上完成 standard 与 student-concept holdout 两套 validation，共六个且仅六个 `attempt-001`。相对 A0 候选门失败：standard overall 非回退、holdout overall 非回退、zero AUC 至少 2/3 提升三项未通过。因此本 campaign 在 failure diagnosis 分支停止；未运行数值调参、final external gate、test authorization 或真实 test。

## 冻结身份与注册设置

- cohort canonical SHA-256：`6342dc8a5f73a4e03a1645780597b625c1480ba7a6513668b6766089cdd5b8a5`
- comparator audit canonical SHA-256：`071b5df25d8641fdc249a9a56175961025b3deeba36a4a471640563ef5d0b181`
- A0 fingerprint：`ef5d6e82e32d095248a362ab01b6a768bbf47785fe34196dee5b6ddd0ece2131`
- A1 fingerprint：`5a3f25c93b51131017c16bd886f723b8c7f31499a6320b89d5b7ea7127e5b687`
- seed：`42`；split seed：`2024`；DOA seed：`42`
- A1 固定 completion rank：`32`；evidence loss weight 与 completion loss weight 均为 `0.1`
- proof-ranked 继承：MOOCRadar r2、ASSIST17 r1、XES3G5M r0；没有使用 last-attempt XES r1。
- route commit：`561faf830069a3dbbad5e6e88c330113e69c739b`

数值 recipe：MOO r2=`dim256/30 epochs/lr0.001/wd0/patience5/student batch64`；ASSIST17 r1=`dim64/80/lr0.002/wd0/patience5/student batch128`；XES r0=`dim64/30/lr0.001/wd0/patience5/student batch64`。三者均为 `student_recompute_minibatch`。

## Validation 指标与精确差值

| Dataset | A1 standard AUC | Δ vs A0 | A1 holdout AUC | Δ vs A0 | A1 zero AUC | Δ vs A0 |
|---|---:|---:|---:|---:|---:|---:|
| MOOCRadar | 0.9237381615694075 | -0.0003049148639236998 | 0.9207740355531081 | -0.0014195154951669453 | 0.9319816652749389 | -0.0006994812862632926 |
| ASSIST17 | 0.7772810133205648 | 0.0022809629178004442 | 0.7771050187131403 | 0.0003449112774648322 | 0.7790516387954539 | 0.0017363756363880656 |
| XES3G5M | 0.7744117095426273 | -0.004287354613667804 | 0.7690837575240221 | -0.0011226916175651747 | 0.7658099393382194 | -0.0034803057119332514 |

| Dataset | A1 ordinary DOA | Δ vs A0 | A1 weighted DOA | Δ vs A0 | Parameters |
|---|---:|---:|---:|---:|---:|
| MOOCRadar | 0.5247008544427902 | 0.031192949129376313 | 0.6647645004684439 | -0.00031939357805976787 | 3485511 |
| ASSIST17 | 0.679439145816845 | 0.00041043017341058086 | 0.6774372281073252 | -0.0007680196613033541 | 1062890 |
| XES3G5M | 0.49826839826839825 | 0.04242424242424242 | 0.7358490566037735 | -0.009433962264150941 | 1444837 |

DOA 仅作为内部软排序证据，不覆盖 AUC hard gate。

## Audited strongest external 诊断差距

相对 A0 hard gate 已失败，所以没有执行 final external gate。下表仅从 Task 5 已审计 `strongest_comparators` 动态读取相同 validation protocol 的绝对值并计算诊断差距；没有手抄阈值到控制逻辑。

| Dataset | Strongest model | External std/hold/zero AUC | A1 - external std/hold/zero |
|---|---|---|---|
| MOOCRadar | SVGCD / SVGCD / SVGCD | 0.9296418118398412 / 0.9249421282505248 / 0.9332772243927672 | -0.0059036502704337135 / -0.004168092697416692 / -0.0012955591178283044 |
| ASSIST17 | ORCDF / ORCDF / ORCDF | 0.7844202114159878 / 0.7841175460589098 / 0.7820483031272386 | -0.007139198095423072 / -0.007012527345769515 / -0.002996664331784693 |
| XES3G5M | ORCDF / ORCDF / ORCDF | 0.7894548409769813 / 0.7809975904619879 / 0.7768146990924982 | -0.015043131434353985 / -0.011913832937965774 / -0.011004759754278837 |

精确 source identity：MOO standard=`audit/comparator-jobs/SVGCD/MOOCRadar/standard/attempt-001/baseline-rows.json`，MOO holdout/zero=`audit/comparator-jobs/SVGCD/MOOCRadar/holdout/attempt-001/baseline-rows.json`；ASSIST17 standard/holdout/zero 对应 `audit/comparator-jobs/ORCDF/ASSIST17/{standard|holdout}/attempt-001/baseline-rows.json`；XES 对应 `audit/comparator-jobs/ORCDF/XES3G5M/{standard|holdout}/attempt-001/baseline-rows.json`。路径均位于 campaign root。

## 候选门

| Gate | Result | Evidence |
|---|---|---|
| same frozen cohort | PASS | 三行均为 canonical cohort `6342dc8a...` |
| standard overall non-regression | FAIL | MOOCRadar、XES3G5M 回退 |
| holdout overall non-regression | FAIL | MOOCRadar、XES3G5M 回退 |
| zero AUC improved on at least 2/3 | FAIL | 仅 ASSIST17 提升，required=`2` |
| at least one zero delta >= 0.001 | PASS | ASSIST17=`0.0017363756363880656` |

软排序量：mean zero delta=`-0.0008144704539361595`；worst zero delta=`-0.0034803057119332514`；mean weighted-DOA delta=`-0.003507125167838021`；negative parameter count=`-5993238.0`。候选总参数量为 `5993238`。

## 证明链与 GPU 记录

- A1 validation rows SHA-256：`635fe863f625eb7cc6d77012b4af43e040a39fae3cffdc3237fdc7c7f0121447`
- A0 primary rows SHA-256：`4f579d7445eb8635f5e68babb6a826dd6ae5355ee4d47f4d70137ea86d67205b`
- decision SHA-256：`2c29c5190f5c01327c16497e0ae0bfef7bd7a36b08d67fe790a901fcc6e4218f`
- final controller state SHA-256：`74370ed3adc81dab8f3aee6df9f06d42454473d3192551577ab42b66590403c2`
- raw proofs：MOO=`01dd5e79cb17c25e614619b0f0b14a82fb2a7a1bd3d6fbf5ce538f423086e599`；ASSIST17=`9395ce45384f51a94969130745849d1b1be97ae5682ed53d49f4c0f10e8c7d7a`；XES=`00affc1240e3c6f831c59ad075c98953f5c8adb288e401a07e0fe4b205a4f7f0`。

| Dataset/split | Physical GPU UUID | Outer peak used memory MiB | Train peak allocated GiB |
|---|---|---:|---:|
| MOO/standard | GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0 | 1504 | 0.4430990219116211 |
| MOO/holdout | GPU-314fdf43-7a94-4e29-20d0-fbd496632b1b | 1402 | 0.4059443473815918 |
| ASSIST17/standard | GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0 | 1308 | 0.3155522346496582 |
| ASSIST17/holdout | GPU-314fdf43-7a94-4e29-20d0-fbd496632b1b | 1280 | 0.3071908950805664 |
| XES/standard | GPU-314fdf43-7a94-4e29-20d0-fbd496632b1b | 724 | 0.1519327163696289 |
| XES/holdout | GPU-56971197-483f-7ec4-a71a-4e092c2fa6b0 | 734 | 0.14238643646240234 |

所有 outer status 为 `completed/exit 0`；每个 pair 使用不同物理 GPU lock 并发，无 OOM、NaN、batch change、overwrite 或 GPU attempt retry。

## 工程验证与前置诊断

- bridge RED：A1 parser 拒绝；候选 controller 在首个整体回退后过早停止。GREEN 后 focused controller/exploration `86` tests OK。
- schema-2 cohort compatibility RED/GREEN 后 focused controller/exploration `87` tests OK。
- A1 token namespace RED/GREEN 后 exploration `7` tests OK。
- `compileall` 与 `git diff --check` 在每个 pre-attempt commit 前通过；真实尝试前 worktree clean。
- full suite 曾运行两次：第一次暴露并修复 subprocess-only MKL 环境污染；第二次 `292` tests 中仅 remote campaign SIGTERM timing test 一项偶发失败，该测试随即 isolated `1/1` 通过。按执行协调要求未再重复 full suite。
- 两次高层 orchestration 失败都发生在 capability/attempt/GPU 之前：第一次是 schema-2 loader 缺口，第二次是 A0 token 文件名冲突。两者均有 TDD 修复，没有形成训练 retry。

相关 commits：`f78ade402e542b29f7e86a7e73705e0e097ec1d2`、`cc21f625aa698cacb69dd6b4c016ce3e33d1bcfc`、`561faf830069a3dbbad5e6e88c330113e69c739b`。

## Freeze-or-diagnose

失败责任限定在本计划唯一替换的 low-rank completer。详细机制假设、检索问题、变量映射、替换接口和验收门见 `unified_mastery_failure_registry.md`。本 Task 9 到此停止，不在本计划内启动第二模块搜索。

## Test 状态

`TEST CLOSED`。没有运行 `external-gate`、`authorize-test` 或 `run-test`，也没有读取真实 test label/metric。只有未来经批准的新 brainstorming/plan 完成单模块替换，并依次通过相对 A0 candidate gate 与从 Task 5 audited strongest registry 动态重建的 external gate，才可重新讨论 test authorization。

## Review-fix：anchored A1 replay（2026-07-12）

Blocking provenance finding 已修复于
`93877cf9c0e22f57f8aece630b63d072420379a5`（`fix: anchor A1 candidate replay`）。
原 `finalize_candidate` 只读取 mutable `state.json` 和 raw proof metrics；现在 A0/A1
共用 Task 7 anchored replay core，并对 A1 强制执行：

- `proof_sha256_by_counter` 的 key 必须精确等于 `1..issuance_counter`；
- 三个 counter 必须按冻结 cohort 顺序一一对应 MOO、ASSIST17、XES；
- 从 consumed capability 重放 outer status、注册命令、输入 snapshots、split summary、
  DOA、architecture/cohort、recipe、seed/split seed 与 parameter count；
- 重新计算 standard/holdout/zero/ordinary-DOA/weighted-DOA delta，并与 raw proof
  语义及 controller-registered raw SHA 比较；
- candidate rows 只从重放后的 split proofs 组装，不再信任 proof 内 metric 副本。

新增 tamper tests 证明 coordinated state+proof hash 修改和缺失 proof counter 均失败。
Schema-2 cohort 也不再从同一 cohort payload 自派生 expectations；新 campaign 在 init
时从 independently verified A0 proof registry、dataset audit 和 comparator audit 计算并写入
controller state，completed legacy A1 verification 则显式传入同一独立重放结果。CLI 描述已改为
A0/A1 shared validation。

在 clean committed verifier 上运行 `verify-existing`，没有覆盖既有 A1 rows 或 decision；
它逐字节重建并确认：

- existing rows SHA-256：
  `635fe863f625eb7cc6d77012b4af43e040a39fae3cffdc3237fdc7c7f0121447`；
- existing decision SHA-256：
  `2c29c5190f5c01327c16497e0ae0bfef7bd7a36b08d67fe790a901fcc6e4218f`；
- verifier route commit：`93877cf9c0e22f57f8aece630b63d072420379a5`；
- candidate training route commit：`561faf830069a3dbbad5e6e88c330113e69c739b`；
- verification canonical SHA：
  `088c3165dab400d77f6e868c344cec11185a0d7fe3850203d82909337664706a`；
- verification file SHA-256：
  `31048af248217c26611b62c0802a09de2271cbce37c7b8f40d1bbed39aa86fe5`；
- verification artifact：`$ROOT/decisions/a1-existing-verification.json`；
- `test_opened=false`。

Review-fix focused exploration/controller/campaign verification：`104` tests，OK；
`compileall` 和 `git diff --check` exit 0。没有运行 GPU、full suite、external gate 或 test。
此前未经授权的两次 pre-attempt full-suite 运行及其 MKL/timing 结果仍按上文原样保留，
没有改写为 suite-green claim。
