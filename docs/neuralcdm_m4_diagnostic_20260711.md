# NeuralCDM M4 diagnostic and registered hypothesis

Date: 2026-07-11

## Registered diagnosis

Task 8 froze the structural/audit cohort ASSIST09, ASSIST17, MOOCRadar, and
XES3G5M under cohort SHA
`77ba446b4cd1e67cb25a5c6e754c6ffb78788fba1519b3ec65d269703313444f`.
M2+M3 improved holdout zero AUC and standard ordinary DOA on three datasets,
but absolute AUC remained well below historical strong baselines. The
registered M4 hypothesis is that the cognitive state contains useful signal
while the old global-concept, Q-weighted linear IRT interaction cannot express
item-by-concept difficulty or nonlinear required-concept interactions.

## Literature mechanism

The primary source is Fei Wang et al., “Neural Cognitive Diagnosis for
Intelligent Education Systems,” AAAI 2020
([official article](https://ojs.aaai.org/index.php/AAAI/article/view/6080),
DOI `10.1609/aaai.v34i04.6080`). The paper motivates learning complex exercise
interactions with multiple neural layers while enforcing monotonicity for
factor interpretability, and specializes required concepts through the
Q-matrix. The extended framework reference is
[NeuralCD: A General Framework for Cognitive Diagnosis](https://ieeexplore.ieee.org/document/9865139/),
IEEE TKDE 2023.

The repository adaptation is deliberately narrower than either general
framework. It keeps the existing student-concept mastery state, learns dense
item-by-concept difficulty, applies positive item discrimination and the exact
Q mask, then uses one positive-weight `K -> 512 -> 256 -> 1` interaction MLP.
The full mastery tensor returned for DOA is the source of the selected mastery
rows used by that sole prediction path.

## Frozen implementation choices

- `beta[e,k]` is sigmoid-constrained to `[0,1]` for every dataset.
- discrimination is softplus-constrained and initialized near one.
- effective MLP weights are softplus-constrained; biases are unconstrained.
- initialization is centered in effective-weight space to avoid immediate
  512/256-layer saturation on sparse Q inputs.
- the architecture uses dense `E x K` beta and fixed 512/256 widths globally.
- the manifest identifies decoder `neuralcdm-monotonic`, version 2, and a new
  fingerprint; old manifests and decoder checkpoints fail closed.

This iteration excludes guessing/slip prediction transforms, exercise bias,
legacy or residual readouts, adapters, a second head, dataset-specific
switches, tuning, and automatic reserve candidates.

## Falsification rule and evidence boundary

The future registered validation must compare the new M4 against the matching
old-M4 candidate on the frozen four-dataset standard/holdout matrix at seed 42.
If standard overall AUC does not improve on at least three of four datasets, or
if any AUC improvement necessarily costs weighted DOA, the M4 hypothesis is
recorded as negative and M4 modification stops. Switching to a whole M1
replacement requires a separate decision.

The CPU structural, gradient, monotonicity, and checkpoint tests in the
implementation phase establish only mechanism and auditability. They are not
evidence that validation AUC or DOA improved. No controller initialization,
GPU run, real validation, real test-data read, or experiment is part of this
commit.

## 注册验证结论（2026-07-12）

Task 9 后续实验严格使用三个独立 trusted controller、固定 seed 42、固定 Task 8
数值 recipe、controller-owned validation snapshot 与 immutable outer attempt。
详细逐项指标、delta、GPU 和 proof 见 `docs/unified_v2_validation_log.md`。

结果否定了“仅替换 M4 即可同时恢复 AUC 且不损失 DOA”的注册假设。B0 与 M2
只在 ASSIST09 通过 joint gate；两者在 ASSIST17 的 weighted DOA 分别回退
`-0.0004608118`、`-0.0006692743`，在 MOOCRadar 又分别回退
`-0.0128396218`、`-0.0169917384`。M2+M3 在 ASSIST09 的 ordinary DOA 回退
`-0.0006311529`，ASSIST17 的 weighted DOA 回退 `-0.0041253628` 且 zero AUC
回退 `-0.0229749316`。

controller 的可达性 stop gate 因而使 M2+M3 在 cursor 2 停止，使 B0/M2 在
cursor 3 停止；三者均为 `blocked=true`。XES3G5M 未获授权、未运行，不能补齐
或外推为四数据集成功。虽然所有已运行组合的 standard overall AUC delta 均为
正数，但“任一 AUC 改善以 weighted DOA 为代价”已经满足预注册反证条件。

最终判定：NeuralCDM 整体 M4 替换在结构与可训练性测试上成立，但本次冻结 cohort
上的 global gate FAIL，Task 9 记录为负结果并停止 M4 修改。若要验证 whole-M1
replacement，必须在新的任务中重新注册，不得从本轮 controller 续跑或调参。
