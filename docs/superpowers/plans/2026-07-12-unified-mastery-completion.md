# Unified Mastery Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建并验证一个跨数据集 architecture fingerprint 完全一致的认知诊断模型，以唯一学生–概念 mastery 同时守住 standard/holdout overall AUC，并在至少三个冻结数据集的 zero AUC 上超过最强同协议外部模型。

**Architecture:** 模型固定为四个可画入框架图的职责模块：训练证据初始化的已测 mastery 估计器、只填补未测单元的 mastery 补全器、以唯一 mastery 为输入的 NeuralCDM 单调解码器、满足 `guess+slip<1` 的条件行为模块。A0 使用 concept prior 补全完成数值对齐与 primary cohort 冻结；A1 只把补全器整体替换为低秩矩阵补全，其他模块、损失语义和验证协议保持不变。

**Tech Stack:** Python 3.11/3.12、PyTorch、NumPy、scikit-learn、标准库 `unittest`、Conda 环境 `decoupled_cd`、Git worktree、NVIDIA CUDA、JSON/SHA-256 实验账本。

## Global Constraints

- 只修改远端 `/home/xph/jwc/research/decoupled_cd_codex_worktrees/complete_model` 的 `codex/remote-complete-model-20260710` 分支；`/home/xph/jwc/research/decoupled_cd_v2` 始终只读，不 push。
- Git 身份固定为 `chiangWC <215551297+chiangWC@users.noreply.github.com>`；每个里程碑提交后生成 Git bundle。
- seed 只使用 `42`，已有数据划分保持 `split_seed=2024`；不得用 test 选择架构、数据集、epoch 或超参数。
- 最终入选数据集使用同一 architecture fingerprint；数据集间只允许改变维度、batch size、学习率、weight decay、epoch、patience 和固定损失项的非零权重。
- 最终预测只使用一份非空 `[students, concepts]` mastery；不得添加 legacy/hybrid 旁路、residual、adapter、mixture-of-experts、数据集专属开关或第二预测头。
- `mastery = observed_mask*m_obs + (1-observed_mask)*m_miss` 必须硬组装；最终概率必须满足 `d p / d p_cognitive = 1-guess-slip > 0`。
- 探索池固定为 ASSIST09、ASSIST17、NIPS34、MOOCRadar、XES3G5M；A1 搜索前按预注册规则冻结三个 primary 数据集，之后不得替换失败成员。
- 模块候选硬门：三个 primary 的 standard overall AUC 与 holdout overall AUC 均不下降，至少 `2/3` 的 zero AUC 严格提升，且至少一个 zero AUC 提升 `>=0.001`。
- DOA 始终计算和保存，但只作近似 AUC 候选的软排序，不作淘汰门；最终若不具竞争力可不进入论文主表。
- 每次整架构变化只做六项聚焦 correctness 测试和一次 1-epoch GPU smoke；完整测试只在最终架构冻结和打开 test 前运行。
- GPU 启动前动态检查；优先空闲卡，显存占用低于物理显存一半的卡也可用。不同物理 GPU 可并行，同一 GPU 必须以 `/tmp/unified-mastery-gpu-N.lock` 加 `flock`。
- 生成资产统一写入 `/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-mastery-20260712/`，attempt 目录不可覆盖；数据、日志、checkpoint、预测和 mastery 不提交 Git。

---

## File Map

- `models/unified_v2_spec.py`：version-3 架构 manifest、fingerprint 与旧 checkpoint 边界。
- `models/unified_v2_components.py`：已测 mastery、A0/A1 补全器、单调认知解码器、simplex 行为模型。
- `models/unified_decoupled_cdm.py`：四模块唯一前向路径与硬 mastery 组装。
- `models/decoupled_cdm.py`：向公共输出 dataclass 增加可选的证据/补全张量。
- `trainers/engine.py`：已测证据损失和 A1 masked completion 损失。
- `scripts/train.py`：version-3 CLI、训练证据初始化、summary/checkpoint 元数据。
- `scripts/unified_baseline_audit.py`：外部结果协议与 provenance 审计、同协议最强值选择。
- `scripts/unified_dataset_audit.py`：zero 样本量及正负标签计数。
- `scripts/unified_cohort.py`：A0 排序、eligibility 与不可变 primary cohort。
- `scripts/unified_campaign.py`：A0/A1 validation 门、软 DOA 排序和最终外部胜出门。
- `scripts/run_unified_validation.py`：A0/A1 命令构造、GPU 选择与并发互斥。
- `scripts/unified_validation_controller.py`：新 campaign state 版本、clean route 与 test-once。
- `tests/test_unified_*.py`：六类架构 correctness、审计、cohort、campaign 和 controller 回归。

### Task 1: Version-3 Architecture Identity and Checkpoint Boundary

**Files:**
- Modify: `models/unified_v2_spec.py`
- Modify: `scripts/train.py`
- Modify: `scripts/run_unified_validation.py`
- Test: `tests/test_unified_v2_spec.py`
- Test: `tests/test_unified_validation.py`

**Interfaces:**
- Produces: `UnifiedArchitectureSpec(completion: Literal["prior", "lowrank"], version: int = 3)`.
- Produces: canonical `manifest() -> dict[str, str | int]` and `fingerprint() -> str`; numeric rank and training hyperparameters are deliberately absent.
- Rejects: every version-1/version-2 manifest and any checkpoint whose stored fingerprint differs from the requested architecture.

- [ ] **Step 1: Write the RED manifest tests**

Add these exact assertions to `tests/test_unified_v2_spec.py`:

```python
def test_a0_and_a1_have_canonical_distinct_version_3_fingerprints(self):
    a0 = UnifiedArchitectureSpec(completion="prior")
    a1 = UnifiedArchitectureSpec(completion="lowrank")
    self.assertEqual(a0.manifest()["modules"], "m1-prior-m3-m4")
    self.assertEqual(a1.manifest()["modules"], "m1-lowrank-m3-m4")
    self.assertEqual(a0.manifest()["version"], 3)
    self.assertNotEqual(a0.fingerprint(), a1.fingerprint())

def test_version_2_manifest_is_rejected(self):
    old = {
        "inference": "prior", "composer": "mask",
        "decoder": "neuralcdm-monotonic",
        "mastery_output": "student-concept", "version": 2,
        "modules": "m1-m4-neuralcdm",
    }
    with self.assertRaisesRegex(ValueError, "version 3"):
        UnifiedArchitectureSpec.from_manifest(old)
```

- [ ] **Step 2: Run the RED tests**

Run remotely:

```bash
cd /home/xph/jwc/research/decoupled_cd_codex_worktrees/complete_model
conda run -n decoupled_cd python -m unittest \
  tests.test_unified_v2_spec tests.test_unified_validation -v
```

Expected: failures because `completion` and version 3 are not implemented and the runner still emits version-2 flags.

- [ ] **Step 3: Replace the canonical architecture schema**

Use this dataclass shape in `models/unified_v2_spec.py`:

```python
@dataclass(frozen=True)
class UnifiedArchitectureSpec:
    mastery_estimator: Literal["evidence-parameter"] = "evidence-parameter"
    completion: Literal["prior", "lowrank"] = "prior"
    cognitive_decoder: Literal["neuralcdm-monotonic"] = "neuralcdm-monotonic"
    behavior_model: Literal["conditional-simplex"] = "conditional-simplex"
    mastery_output: Literal["student-concept"] = "student-concept"
    version: int = 3

    def __post_init__(self) -> None:
        expected = {
            "mastery_estimator": (self.mastery_estimator, "evidence-parameter"),
            "cognitive_decoder": (self.cognitive_decoder, "neuralcdm-monotonic"),
            "behavior_model": (self.behavior_model, "conditional-simplex"),
            "mastery_output": (self.mastery_output, "student-concept"),
        }
        for name, (actual, required) in expected.items():
            if type(actual) is not str or actual != required:
                raise ValueError(f"{name} must be {required!r}")
        if type(self.completion) is not str or self.completion not in {"prior", "lowrank"}:
            raise ValueError("completion must be 'prior' or 'lowrank'")
        if type(self.version) is not int or self.version != 3:
            raise ValueError("version must be integer 3")

    def manifest(self) -> dict[str, str | int]:
        payload = asdict(self)
        payload["modules"] = {
            "prior": "m1-prior-m3-m4",
            "lowrank": "m1-lowrank-m3-m4",
        }[self.completion]
        return payload
```

Update `from_manifest()` to require exactly the six dataclass fields plus `modules`; reconstruct the class and compare `manifest` byte-for-byte before checking the SHA-256 fingerprint. Update train/checkpoint loading and runner summary checks to call this version-3 parser; do not add a compatibility conversion.

- [ ] **Step 4: Run identity and runner-summary tests**

Run:

```bash
conda run -n decoupled_cd python -m unittest \
  tests.test_unified_v2_spec tests.test_unified_validation -v
```

Expected: tests pass; generated A0/A1 summaries contain canonical version-3 manifests, and old version-2 summaries are rejected.

- [ ] **Step 5: Commit the architecture boundary**

```bash
git add models/unified_v2_spec.py scripts/train.py \
  scripts/run_unified_validation.py tests/test_unified_v2_spec.py \
  tests/test_unified_validation.py
git commit -m "feat: register unified mastery architecture v3"
```

### Task 2: Observed Mastery Estimator and Hard A0 Completion

**Files:**
- Modify: `models/unified_v2_components.py`
- Test: `tests/test_unified_v2_components.py`

**Interfaces:**
- Produces: `ObservedMasteryState(mastery, reliability, observed_mask)` with tensors `[S,K]`.
- Produces: `ObservedMasteryEstimator(num_students, num_concepts, initial_logits, evidence_cap=20.0)`.
- Produces: `GlobalConceptPriorCompleter(num_concepts, initial_prior=None)` returning `[S,K]` missing predictions.
- Produces: `assemble_mastery(observed, missing, observed_mask) -> Tensor` with exact hard masking.

- [ ] **Step 1: Write six focused component tests**

Add tests covering the registered invariants; the core assertions must be:

```python
def test_observed_estimator_uses_train_only_initial_logits(self):
    evidence = torch.tensor([[[4., 3.], [0., 0.]], [[2., 0.], [5., 4.]]])
    initial = smoothed_evidence_logits(evidence, alpha=1.0)
    module = ObservedMasteryEstimator(2, 2, initial_logits=initial)
    state = module(evidence)
    self.assertTrue(torch.equal(state.observed_mask, evidence[..., 0] > 0))
    torch.testing.assert_close(state.mastery, initial.sigmoid())

def test_hard_assembly_has_no_learned_blending(self):
    observed = torch.tensor([[.2, .8]])
    missing = torch.tensor([[.9, .1]])
    mask = torch.tensor([[True, False]])
    torch.testing.assert_close(
        assemble_mastery(observed, missing, mask),
        torch.tensor([[.2, .1]]),
    )
```

The remaining four tests must assert: zero-attempt cells are marked missing; reliability equals `attempts.clamp(max=20)/20`; the prior completer broadcasts one learnable concept vector to every student; gradients from an observed output never reach the missing completer.

- [ ] **Step 2: Run RED component tests**

```bash
conda run -n decoupled_cd python -m unittest \
  tests.test_unified_v2_components -v
```

Expected: import errors for the new dataclasses, helper, estimator and completer.

- [ ] **Step 3: Implement train-evidence initialization and hard assembly**

Add these public definitions to `models/unified_v2_components.py`:

```python
@dataclass(frozen=True)
class ObservedMasteryState:
    mastery: torch.Tensor
    reliability: torch.Tensor
    observed_mask: torch.Tensor

def smoothed_evidence_logits(evidence: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    if evidence.ndim != 3 or evidence.shape[-1] != 2:
        raise ValueError("evidence must have shape [students, concepts, 2]")
    attempts, correct = evidence.unbind(dim=-1)
    rate = (correct + alpha) / (attempts + 2.0 * alpha)
    return torch.logit(rate.clamp(1e-6, 1.0 - 1e-6))

class ObservedMasteryEstimator(nn.Module):
    def __init__(self, num_students: int, num_concepts: int,
                 initial_logits: torch.Tensor | None = None,
                 evidence_cap: float = 20.0) -> None:
        super().__init__()
        if evidence_cap <= 0.0:
            raise ValueError("evidence_cap must be positive")
        shape = (num_students, num_concepts)
        values = torch.zeros(shape) if initial_logits is None else initial_logits.detach().clone()
        if tuple(values.shape) != shape:
            raise ValueError(f"initial_logits must have shape {shape}")
        self.logits = nn.Parameter(values)
        self.evidence_cap = float(evidence_cap)

    def forward(self, evidence: torch.Tensor,
                student_ids: torch.Tensor | None = None) -> ObservedMasteryState:
        selected = evidence if student_ids is None else evidence[student_ids]
        logits = self.logits if student_ids is None else self.logits[student_ids]
        attempts = selected[..., 0]
        return ObservedMasteryState(
            mastery=logits.sigmoid(),
            reliability=attempts.clamp(max=self.evidence_cap) / self.evidence_cap,
            observed_mask=attempts > 0,
        )

class GlobalConceptPriorCompleter(nn.Module):
    def __init__(self, num_concepts: int, initial_prior: torch.Tensor | None = None) -> None:
        super().__init__()
        prior = torch.zeros(num_concepts) if initial_prior is None else torch.logit(
            initial_prior.detach().clone().clamp(1e-6, 1.0 - 1e-6)
        )
        self.logits = nn.Parameter(prior)

    def forward(self, num_students: int) -> torch.Tensor:
        return self.logits.sigmoid().unsqueeze(0).expand(num_students, -1)

def assemble_mastery(observed: torch.Tensor, missing: torch.Tensor,
                     observed_mask: torch.Tensor) -> torch.Tensor:
    if observed.shape != missing.shape or observed.shape != observed_mask.shape:
        raise ValueError("observed, missing, and observed_mask shapes must match")
    return torch.where(observed_mask, observed, missing)
```

- [ ] **Step 4: Run the focused tests**

```bash
conda run -n decoupled_cd python -m unittest \
  tests.test_unified_v2_components -v
```

Expected: all component tests pass, including the six new architecture invariants.

- [ ] **Step 5: Commit M1 and A0 completion**

```bash
git add models/unified_v2_components.py tests/test_unified_v2_components.py
git commit -m "feat: estimate and assemble unified mastery"
```

### Task 3: Sole Monotone Decoder and Conditional Simplex Behavior Module

**Files:**
- Modify: `models/unified_v2_components.py`
- Modify: `models/decoupled_cdm.py`
- Test: `tests/test_unified_v2_components.py`

**Interfaces:**
- Changes: `MonotonicDiagnosisDecoder.forward(mastery, q_matrix, target_student_ids, target_exercise_ids) -> tuple[cognitive_probs, difficulty]`; decoder no longer owns a `mastery_head`.
- Produces: `BehaviorState(probs, guess_probs, slip_probs, cognitive_weight)`.
- Produces: `ConditionalSimplexBehaviorModel(num_students, num_exercises, dim)`.
- Extends: `DecoupledForwardOutput` with optional `observed_mastery`, `completion_predictions`, `mastery_observed_mask`, `cognitive_weight` fields defaulting to `None`.

- [ ] **Step 1: Write RED monotonicity and simplex tests**

```python
def test_decoder_consumes_the_returned_unique_mastery(self):
    decoder = MonotonicDiagnosisDecoder(num_exercises=2, num_concepts=3, dim=8)
    mastery = torch.tensor([[.2, .4, .6]], requires_grad=True)
    q = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
    cognitive, _ = decoder(mastery, q, torch.tensor([0]), torch.tensor([0]))
    cognitive.backward()
    self.assertGreaterEqual(float(mastery.grad[0, 0]), -1e-8)
    self.assertEqual(float(mastery.grad[0, 1]), 0.0)
    self.assertGreaterEqual(float(mastery.grad[0, 2]), -1e-8)

def test_behavior_simplex_preserves_positive_cognitive_derivative(self):
    behavior = ConditionalSimplexBehaviorModel(2, 3, dim=4)
    cognitive = torch.tensor([.2, .8], requires_grad=True)
    state = behavior(cognitive, torch.tensor([0, 1]), torch.tensor([1, 2]))
    torch.testing.assert_close(
        state.guess_probs + state.slip_probs + state.cognitive_weight,
        torch.ones(2), atol=1e-6, rtol=0,
    )
    state.probs.sum().backward()
    self.assertTrue(torch.all(cognitive.grad > 0))
```

Also assert there is no parameter name containing `mastery_head`, `guess+slip<1` for boundary logits, and non-Q mastery perturbations leave cognitive probability unchanged.

- [ ] **Step 2: Run RED tests**

```bash
conda run -n decoupled_cd python -m unittest \
  tests.test_unified_v2_components -v
```

Expected: failures from the old decoder signature, internal mastery head, and missing behavior class.

- [ ] **Step 3: Implement the single behavior formula**

Add the behavior module and change the decoder to index the supplied mastery:

```python
@dataclass(frozen=True)
class BehaviorState:
    probs: torch.Tensor
    guess_probs: torch.Tensor
    slip_probs: torch.Tensor
    cognitive_weight: torch.Tensor

class ConditionalSimplexBehaviorModel(nn.Module):
    def __init__(self, num_students: int, num_exercises: int, dim: int) -> None:
        super().__init__()
        self.student_embedding = nn.Embedding(num_students, dim)
        self.exercise_embedding = nn.Embedding(num_exercises, dim)
        self.out = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, 3)
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.constant_(self.out[-1].bias, -2.0)
        with torch.no_grad():
            self.out[-1].bias[2] = 2.0

    def forward(self, cognitive_probs: torch.Tensor,
                student_ids: torch.Tensor,
                exercise_ids: torch.Tensor) -> BehaviorState:
        features = torch.cat((self.student_embedding(student_ids),
                              self.exercise_embedding(exercise_ids)), dim=-1)
        guess, slip, weight = self.out(features).softmax(dim=-1).unbind(dim=-1)
        probs = (1.0 - slip) * cognitive_probs + guess * (1.0 - cognitive_probs)
        return BehaviorState(probs, guess, slip, weight)
```

The decoder must calculate NeuralCDM interactions only from `mastery[target_student_ids]`; delete its internal mastery projection. Add the four optional fields to `DecoupledForwardOutput` after existing defaulted fields so legacy constructors remain valid.

- [ ] **Step 4: Verify mathematical invariants**

```bash
conda run -n decoupled_cd python -m unittest \
  tests.test_unified_v2_components -v
conda run -n decoupled_cd python -m compileall models
```

Expected: tests pass; compileall exits 0.

- [ ] **Step 5: Commit the decoder and behavior modules**

```bash
git add models/unified_v2_components.py models/decoupled_cdm.py \
  tests/test_unified_v2_components.py
git commit -m "feat: add monotone cognitive behavior path"
```

### Task 4: Integrate A0 and Its Two Registered Losses

**Files:**
- Modify: `models/unified_decoupled_cdm.py`
- Modify: `trainers/engine.py`
- Modify: `scripts/train.py`
- Test: `tests/test_unified_v2_training.py`

**Interfaces:**
- `UnifiedDecoupledCDM(..., architecture, initial_mastery_logits, completion_rank=32)` owns exactly estimator, completer, decoder and behavior modules.
- `observed_mastery_evidence_loss(output, evidence, student_ids=None) -> Tensor` supervises observed cells only.
- `train_model(..., unified_evidence_loss_weight: float, unified_completion_loss_weight: float)` rejects A0 completion weight other than zero.

- [ ] **Step 1: Write RED integration tests**

Add tests that construct A0 from synthetic train evidence and assert:

```python
output = model(
    q_matrix=q, concept_graph=graph,
    student_exercise_mask=mask, response_matrix=responses,
    student_tkc_mask=tkc, student_ukc_mask=ukc,
    student_concept_evidence=evidence,
    target_student_ids=torch.tensor([0, 1]),
    target_exercise_ids=torch.tensor([0, 1]),
)
self.assertEqual(tuple(output.mastery.shape), (2, 3))
self.assertIs(output.mastery, output.mastery)
self.assertTrue(torch.equal(output.mastery_observed_mask, evidence[..., 0] > 0))
self.assertTrue(torch.all(output.guess_probs + output.slip_probs < 1.0))
```

Add a loss test proving changing missing-cell targets leaves `observed_mastery_evidence_loss` unchanged, and a CLI test proving `--unified-completion prior --unified-completion-loss-weight 0` is accepted while a positive completion weight is rejected.

- [ ] **Step 2: Run RED training tests**

```bash
conda run -n decoupled_cd python -m unittest \
  tests.test_unified_v2_training -v
```

Expected: failures because the old model discards evidence and the engine exposes only the old mastery loss.

- [ ] **Step 3: Wire the only allowed forward path**

Implement the central A0 path in `UnifiedDecoupledCDM.forward`:

```python
if student_concept_evidence is None:
    raise ValueError("unified v3 requires train-only student_concept_evidence")
student_ids, local_target_ids = self._select_students(
    target_student_ids, use_student_subset
)
observed = self.mastery_estimator(student_concept_evidence, student_ids)
missing = self.completer(observed.mastery.shape[0])
mastery = assemble_mastery(observed.mastery, missing, observed.observed_mask)
cognitive_probs, difficulty = self.decoder(
    mastery, q_matrix, local_target_ids, target_exercise_ids
)
behavior = self.behavior_model(
    cognitive_probs, target_student_ids, target_exercise_ids
)
```

Return that exact `mastery`, `observed.mastery`, `missing`, mask, cognitive probability, behavior probability and simplex parameters. `tkc_states/ukc_states` may be compatibility views `mastery.unsqueeze(-1)*mask.unsqueeze(-1)`; they must not participate in prediction.

Populate the legacy-shaped public fields only as transparent views of the same mastery path:

```python
concept_basis = torch.eye(q_matrix.shape[1], device=mastery.device, dtype=mastery.dtype)
return DecoupledForwardOutput(
    student_state=mastery,
    tkc_states=mastery.unsqueeze(-1) * observed.observed_mask.unsqueeze(-1),
    ukc_states=mastery.unsqueeze(-1) * (~observed.observed_mask).unsqueeze(-1),
    tkc_weight=observed.reliability,
    concept_embeddings=concept_basis,
    exercise_embeddings=q_matrix.to(dtype=mastery.dtype),
    cognitive_probs=cognitive_probs,
    probs=behavior.probs,
    guess_probs=behavior.guess_probs,
    slip_probs=behavior.slip_probs,
    difficulty=difficulty,
    mastery=mastery,
    source_weights=None,
    observed_mastery=observed.mastery,
    completion_predictions=missing,
    mastery_observed_mask=observed.observed_mask,
    cognitive_weight=behavior.cognitive_weight,
)
```

Tests must assert these compatibility views equal their documented mastery/mask transforms; no loss or prediction code may consume them.

- [ ] **Step 4: Implement evidence initialization and loss validation**

In `scripts/train.py`, derive `initial_mastery_logits = smoothed_evidence_logits(train_bundle.student_concept_evidence_tensor)` before model construction; never read validation/test evidence. Replace the old CLI with:

```python
parser.add_argument("--unified-completion", choices=("prior", "lowrank"), default="prior")
parser.add_argument("--unified-completion-rank", type=int, default=32)
parser.add_argument("--unified-evidence-loss-weight", type=float, default=1.0)
parser.add_argument("--unified-completion-loss-weight", type=float, default=0.0)
```

The evidence loss target is `(correct+1)/(attempts+2)` and its mask is `attempts>0`. Validate all weights finite/nonnegative; require evidence weight positive for both A0/A1, completion weight exactly zero for `prior`, and positive for `lowrank`. Save all four values plus manifest/fingerprint in summary and checkpoint.

Replace the old helper with this exact observed-cell loss:

```python
def observed_mastery_evidence_loss(
    output: DecoupledForwardOutput,
    evidence: torch.Tensor,
    student_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if output.observed_mastery is None or output.mastery_observed_mask is None:
        raise ValueError("observed mastery output is required")
    selected = evidence if student_ids is None else evidence[student_ids]
    attempts, correct = selected.unbind(dim=-1)
    target = (correct + 1.0) / (attempts + 2.0)
    mask = output.mastery_observed_mask
    if not bool(mask.any()):
        raise ValueError("at least one observed mastery cell is required")
    return F.binary_cross_entropy(output.observed_mastery[mask], target[mask])
```

- [ ] **Step 5: Run focused A0 tests and commit**

```bash
conda run -n decoupled_cd python -m unittest \
  tests.test_unified_v2_components tests.test_unified_v2_spec \
  tests.test_unified_v2_training -v
git diff --check
git add models/unified_decoupled_cdm.py trainers/engine.py scripts/train.py \
  tests/test_unified_v2_training.py
git commit -m "feat: train unified A0 through one mastery path"
```

Expected: focused tests pass and diff check exits 0.

### Task 5: Audit Datasets and Strong External Baselines

**Files:**
- Create: `scripts/unified_baseline_audit.py`
- Modify: `scripts/unified_dataset_audit.py`
- Modify: `scripts/unified_cohort.py`
- Test: `tests/test_unified_baseline_audit.py`
- Test: `tests/test_unified_cohort.py`

**Interfaces:**
- `audit_baseline_rows(rows, dataset_audits) -> dict` returns accepted/rejected rows with exact reasons and strongest same-protocol comparator per dataset/split/metric.
- Dataset audit emits `zero_count`, `zero_positive_count`, `zero_negative_count`, data hash and Q hash.
- `freeze_primary_cohort(a0_rows, comparator_rows, audit_rows) -> dict` writes exactly three immutable dataset IDs and a canonical cohort SHA-256.

- [ ] **Step 1: Write RED protocol and eligibility tests**

Use synthetic JSON records to prove rejection on each mismatch:

```python
row = {
    "dataset_id": "assist17", "model": "ORCDF", "seed": 42,
    "split_seed": 2024, "split": "holdout", "metric": "zero_auc",
    "value": 0.78, "data_sha256": "a" * 64, "q_sha256": "b" * 64,
    "prediction_sha256": "c" * 64, "prediction_order_sha256": "d" * 64,
    "config_sha256": "e" * 64, "checkpoint_sha256": "f" * 64,
    "source_path": "/abs/internal/result.json",
}
self.assertEqual(audit_baseline_rows([row], audits)["accepted_count"], 1)
for field in ("seed", "split_seed", "data_sha256", "q_sha256", "prediction_order_sha256"):
    broken = dict(row)
    broken[field] = 7 if field.endswith("seed") else "0" * 64
    self.assertEqual(audit_baseline_rows([broken], audits)["accepted_count"], 0)
```

Also test that two accepted same-protocol rows choose the larger AUC; a dataset with `zero_count<1000`, `zero_positive_count<100`, or `zero_negative_count<100` is ineligible; NIPS34 remains partial-only if its zero slice is absent.

- [ ] **Step 2: Run RED audit tests**

```bash
conda run -n decoupled_cd python -m unittest \
  tests.test_unified_baseline_audit tests.test_unified_cohort -v
```

Expected: missing module/import failures and missing label-count fields.

- [ ] **Step 3: Implement canonical baseline records**

Define the required fields as an exact tuple and reject any non-finite metric or hash mismatch:

```python
REQUIRED_BASELINE_FIELDS = (
    "dataset_id", "model", "seed", "split_seed", "split", "metric", "value",
    "data_sha256", "q_sha256", "prediction_sha256",
    "prediction_order_sha256", "config_sha256", "checkpoint_sha256", "source_path",
)
SAME_PROTOCOL = {"seed": 42, "split_seed": 2024}
```

Read-only source inventory must cover:

```text
/home/xph/jwc/research/local_data/pyedmine_cd_baselines/job_outputs/
/home/xph/jwc/research/local_data/svgcd_baselines/job_outputs/
/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/formal-aaai-20260710/
/home/xph/jwc/research/decoupled_cd_v2/results/r14/ext_a17.csv
/home/xph/jwc/research/decoupled_cd_v2/results/r14/ext_moo.csv
/home/xph/jwc/research/decoupled_cd_v2/results/r14/ext_xes.csv
```

Reference-repo rows lacking exact provenance must be rejected from hard gates, not repaired by assumption. The generated audit JSON records every rejection reason and source-file SHA-256.

- [ ] **Step 4: Implement zero label counts and immutable cohort ranking**

Rank eligible A0 rows by this exact key, descending:

```python
rank_key = (
    a0_zero_auc - strongest_external_zero_auc,
    min(a0_standard_auc - strongest_external_standard_auc,
        a0_holdout_auc - strongest_external_holdout_auc),
    zero_count,
    -failed_attempt_count,
)
```

Require exactly three selected datasets and write with exclusive creation (`Path.open("x")`). Canonicalize with sorted JSON keys before hashing. A later load must recompute the hash and reject any changed dataset list, A0 fingerprint, comparator hash or audit hash.

- [ ] **Step 5: Run audit tests and commit**

```bash
conda run -n decoupled_cd python -m unittest \
  tests.test_unified_baseline_audit tests.test_unified_cohort -v
git diff --check
git add scripts/unified_baseline_audit.py scripts/unified_dataset_audit.py \
  scripts/unified_cohort.py tests/test_unified_baseline_audit.py \
  tests/test_unified_cohort.py
git commit -m "feat: audit unified external comparison cohort"
```

### Task 6: Validation Gates, Fresh Controller State, and GPU Parallelism

**Files:**
- Modify: `scripts/unified_campaign.py`
- Modify: `scripts/run_unified_validation.py`
- Modify: `scripts/unified_validation_controller.py`
- Test: `tests/test_unified_campaign.py`
- Test: `tests/test_unified_validation.py`
- Test: `tests/test_unified_validation_controller.py`

**Interfaces:**
- `evaluate_candidate(baseline_rows, candidate_rows)` gates only standard/holdout overall and zero AUC; DOA appears only in ranking.
- Runner recognizes only `a0` and `a1`, uses disjoint attempt roots, and returns a proof only after AUC and DOA artifacts are complete.
- Controller state uses a new schema/campaign ID and refuses prior Task-9 state.

- [ ] **Step 1: Write RED gate and concurrency tests**

Use three rows where all AUC gates pass but DOA falls; assert candidate passes and DOA remains reported:

```python
decision = evaluate_candidate(baseline_rows, candidate_rows)
self.assertTrue(decision["pass"])
self.assertNotIn("weighted_doa_non_regression", decision["gates"])
self.assertAlmostEqual(decision["ranking"]["mean_weighted_doa_delta"], -0.02)
```

Add tests that: A0/A1 commands contain version-3 completion flags; old architecture names fail; a GPU at 49% memory is eligible while 50% is not; two different GPU locks can be held simultaneously; the same GPU lock cannot; proof publication waits for both split metrics and DOA; Task-9 controller schema is rejected.

- [ ] **Step 2: Run RED campaign/controller tests**

```bash
conda run -n decoupled_cd python -m unittest \
  tests.test_unified_campaign tests.test_unified_validation \
  tests.test_unified_validation_controller -v
```

Expected: failures from old DOA hard gates, B0/M2 names and reusable old state.

- [ ] **Step 3: Replace gates and architecture commands**

Keep these five hard gates only:

```python
gates = {
    "standard_overall_auc_non_regression": all(d["standard_overall_auc"] >= 0 for d in deltas.values()),
    "holdout_overall_auc_non_regression": all(d["holdout_overall_auc"] >= 0 for d in deltas.values()),
    "zero_auc_improved_two_thirds": sum(d["zero_auc"] > 0 for d in deltas.values()) >= required,
    "zero_auc_delta_at_least_0.001": any(d["zero_auc"] >= 0.001 for d in deltas.values()),
    "same_frozen_cohort": baseline_cohort == candidate_cohort,
}
```

Ranking order is mean zero delta, worst zero delta, mean weighted DOA delta, then negative parameter count. Runner mappings are exactly:

```python
ARCHITECTURES = {
    "a0": ("prior", 0.0),
    "a1": ("lowrank", 1.0),
}
```

Register this ordered numerical recipe table before training; later entries for a dataset are launched only if the earlier entry does not meet the external overall-AUC guard or if fewer than three eligible cohort candidates remain:

```python
RECIPES = {
    "ASSIST09": (
        NumericalRecipe("full_batch", 64, 300, 1e-3, 0.0, 5, None),
    ),
    "ASSIST17": (
        NumericalRecipe("student_recompute_minibatch", 64, 40, 1e-3, 0.0, 5, 64),
        NumericalRecipe("student_recompute_minibatch", 64, 80, 2e-3, 0.0, 5, 128),
    ),
    "NIPS34": (
        NumericalRecipe("full_batch", 64, 300, 1e-3, 0.0, 5, None),
    ),
    "MOOCRadar": (
        NumericalRecipe("student_recompute_minibatch", 64, 30, 1e-3, 0.0, 5, 64),
        NumericalRecipe("student_recompute_minibatch", 128, 30, 1e-3, 0.0, 5, 64),
        NumericalRecipe("student_recompute_minibatch", 256, 30, 1e-3, 0.0, 5, 64),
    ),
    "XES3G5M": (
        NumericalRecipe("student_recompute_minibatch", 64, 30, 1e-3, 0.0, 5, 64),
        NumericalRecipe("full_batch", 64, 3000, 1e-3, 0.0, 50, None),
    ),
}
```

Define `NumericalRecipe` fields in that displayed order: training mode, concept dimension, epochs, learning rate, weight decay, patience and optional student batch size. XES long training is the last fallback and is never launched once three eligible A0 datasets already meet cohort requirements. A1 initially inherits the exact selected A0 recipe per frozen dataset, including optimizer budget and loss weights.

Every command includes seed 42, split seed 2024, architecture manifest/fingerprint, cohort hash and exclusive attempt directory.

- [ ] **Step 4: Version state and implement per-device locks**

Set `CAMPAIGN_ID="unified-mastery-20260712"` and controller `schema_version=3`. GPU eligibility is `used_memory==0 or used_memory*2<total_memory`; sort by used bytes, then index. Acquire `fcntl.flock(..., LOCK_EX|LOCK_NB)` on `/tmp/unified-mastery-gpu-{physical_index}.lock`; skip locked devices and fail without changing batch size if none qualify. Preserve existing trusted absolute Git checks, clean route checks, immutable issuance and test-once enforcement.

- [ ] **Step 5: Run focused tests and commit**

```bash
conda run -n decoupled_cd python -m unittest \
  tests.test_unified_campaign tests.test_unified_validation \
  tests.test_unified_validation_controller -v
git diff --check
git add scripts/unified_campaign.py scripts/run_unified_validation.py \
  scripts/unified_validation_controller.py tests/test_unified_campaign.py \
  tests/test_unified_validation.py tests/test_unified_validation_controller.py
git commit -m "feat: govern unified mastery validation campaign"
```

### Task 7: A0 GPU Smoke, Baseline Audit, Numerical Alignment, and Cohort Freeze

**Files:**
- Create: `docs/experiments/unified_mastery_a0_20260712.md`
- Generated outside Git: campaign `audit/`, `smoke/a0/`, `a0/`, `cohort.json`

**Interfaces:**
- Consumes: committed clean A0 code and version-3 controller.
- Produces: one GPU smoke proof, audited comparator registry, validation-only A0 rows for every eligible exploration dataset, and immutable three-dataset cohort.

- [ ] **Step 1: Verify route, identity, data hashes, and GPU availability**

```bash
cd /home/xph/jwc/research/decoupled_cd_codex_worktrees/complete_model
git config user.name chiangWC
git config user.email 215551297+chiangWC@users.noreply.github.com
test -z "$(git status --porcelain=v1 --untracked-files=all)"
git rev-parse --abbrev-ref HEAD | grep -Fx codex/remote-complete-model-20260710
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total \
  --format=csv,noheader,nounits
```

Expected: clean target branch; at least one unlocked GPU is empty or below half memory. If route/data hashes change or environment import fails, stop the whole campaign.

- [ ] **Step 2: Run the only A0 architecture smoke**

```bash
ROOT=/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-mastery-20260712
conda run -n decoupled_cd python scripts/run_unified_validation.py smoke \
  --architecture a0 --seed 42 --epochs 1 \
  --output-root "$ROOT/smoke/a0"
```

Expected: one completed immutable attempt with nonempty `[S,K]` mastery, finite BCE, `guess+slip<1`, correct version-3 fingerprint, GPU UUID and peak memory.

- [ ] **Step 3: Build the audited strongest-comparator registry**

```bash
conda run -n decoupled_cd python scripts/unified_dataset_audit.py \
  --output "$ROOT/audit/datasets.json"
conda run -n decoupled_cd python scripts/unified_baseline_audit.py \
  --dataset-audit "$ROOT/audit/datasets.json" \
  --output "$ROOT/audit/baselines.json"
```

Expected: each accepted comparator has matching data/Q/order hashes and seed 42; rejected historical rows remain listed. Rerun only missing or rejected required ORCDF/SVGCD/KaNCD rows through their existing internal adapters, and append them only after the same audit accepts them.

- [ ] **Step 4: Run A0 validation-only numerical alignment**

Use controller-issued commands only. For each eligible dataset, register the historical strong recipe plus r20–r23 numeric alternatives; keep architecture `a0` unchanged. At most one attempt for each predeclared numeric recipe; no test split:

```bash
conda run -n decoupled_cd python scripts/unified_validation_controller.py init \
  --campaign-id unified-mastery-20260712 --architecture a0 \
  --dataset-audit "$ROOT/audit/datasets.json" \
  --baseline-audit "$ROOT/audit/baselines.json" \
  --state "$ROOT/controllers/a0.json"
conda run -n decoupled_cd python scripts/unified_validation_controller.py run-validation \
  --state "$ROOT/controllers/a0.json" --parallel-gpus
```

Expected: standard and holdout validation proofs for eligible members of the five-dataset pool; NIPS34 is recorded ineligible if exact zero remains unavailable. Numeric recipes may run concurrently only on different physical GPUs.

- [ ] **Step 5: Freeze primary cohort and record the A0 milestone**

```bash
conda run -n decoupled_cd python scripts/unified_cohort.py freeze \
  --a0-metrics "$ROOT/controllers/a0-validation-rows.json" \
  --baseline-audit "$ROOT/audit/baselines.json" \
  --dataset-audit "$ROOT/audit/datasets.json" \
  --output "$ROOT/cohort.json"
conda run -n decoupled_cd python scripts/unified_cohort.py verify \
  --cohort "$ROOT/cohort.json"
```

Write `docs/experiments/unified_mastery_a0_20260712.md` with exact double-precision metrics, strongest comparator/model/source, gaps, excluded datasets/reasons, recipe hashes, fingerprint, cohort hash and all failed attempts. Commit and bundle:

```bash
git add docs/experiments/unified_mastery_a0_20260712.md
git commit -m "docs: freeze unified mastery A0 cohort"
git bundle create "$ROOT/bundles/a0-cohort-$(git rev-parse --short=12 HEAD).bundle" HEAD
sha256sum "$ROOT"/bundles/a0-cohort-*.bundle
```

### Task 8: Replace Only the Missing-Mastery Completer with Low Rank A1

**Files:**
- Modify: `models/unified_v2_components.py`
- Modify: `models/unified_decoupled_cdm.py`
- Modify: `trainers/engine.py`
- Test: `tests/test_unified_v2_components.py`
- Test: `tests/test_unified_v2_training.py`

**Interfaces:**
- Produces: `LowRankMasteryCompleter(num_students, num_concepts, rank)` returning `[S,K]` probabilities.
- Produces: `masked_completion_loss(output, evidence, student_ids=None) -> Tensor` over observed cells only.
- A1 differs from A0 only in `architecture.completion`, the completer parameters, and the registered third loss.

- [ ] **Step 1: Write RED A1 tests**

```python
def test_low_rank_completer_has_student_concept_factorization(self):
    module = LowRankMasteryCompleter(3, 4, rank=2)
    result = module()
    self.assertEqual(tuple(result.shape), (3, 4))
    self.assertEqual(tuple(module.student_factors.shape), (3, 2))
    self.assertEqual(tuple(module.concept_factors.shape), (4, 2))
    self.assertTrue(torch.all((result > 0) & (result < 1)))

def test_completion_loss_uses_observed_cells_but_prediction_uses_it_only_when_missing(self):
    loss = masked_completion_loss(output, evidence)
    loss.backward()
    self.assertIsNotNone(model.completer.student_factors.grad)
    observed_prediction = model.completer()[output.mastery_observed_mask]
    self.assertFalse(torch.equal(observed_prediction, output.mastery[output.mastery_observed_mask]))
```

Also prove rank does not change A1 fingerprint, A0 has no low-rank parameters, `completion_predictions` covers all cells for loss computation, and the hard assembly selects A1 values only where the train mask is false.

- [ ] **Step 2: Run RED A1 tests**

```bash
conda run -n decoupled_cd python -m unittest \
  tests.test_unified_v2_components tests.test_unified_v2_training -v
```

Expected: missing low-rank completer and completion loss failures.

- [ ] **Step 3: Implement low-rank completion**

```python
class LowRankMasteryCompleter(nn.Module):
    def __init__(self, num_students: int, num_concepts: int, rank: int) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.student_factors = nn.Parameter(torch.empty(num_students, rank))
        self.concept_factors = nn.Parameter(torch.empty(num_concepts, rank))
        self.student_bias = nn.Parameter(torch.zeros(num_students, 1))
        self.concept_bias = nn.Parameter(torch.zeros(1, num_concepts))
        nn.init.normal_(self.student_factors, std=rank ** -0.5)
        nn.init.normal_(self.concept_factors, std=rank ** -0.5)

    def forward(self, student_ids: torch.Tensor | None = None) -> torch.Tensor:
        users = self.student_factors if student_ids is None else self.student_factors[student_ids]
        bias = self.student_bias if student_ids is None else self.student_bias[student_ids]
        return (users @ self.concept_factors.T + bias + self.concept_bias).sigmoid()
```

Construct it only for `completion="lowrank"`; preserve all A0 modules and initialization. The masked completion target and mask are identical to the evidence loss; the prediction assembly still uses completion values only on missing cells.

Dispatch the two completers without changing the hard assembly:

```python
if self.architecture.completion == "prior":
    missing = self.completer(observed.mastery.shape[0])
else:
    missing = self.completer(student_ids)
mastery = assemble_mastery(observed.mastery, missing, observed.observed_mask)
```

Add the third registered loss:

```python
def masked_completion_loss(
    output: DecoupledForwardOutput,
    evidence: torch.Tensor,
    student_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    if output.completion_predictions is None or output.mastery_observed_mask is None:
        raise ValueError("completion predictions are required")
    selected = evidence if student_ids is None else evidence[student_ids]
    attempts, correct = selected.unbind(dim=-1)
    target = (correct + 1.0) / (attempts + 2.0)
    mask = output.mastery_observed_mask
    if not bool(mask.any()):
        raise ValueError("at least one observed completion target is required")
    return F.binary_cross_entropy(output.completion_predictions[mask], target[mask])
```

- [ ] **Step 4: Verify A1 and run its single GPU smoke**

```bash
conda run -n decoupled_cd python -m unittest \
  tests.test_unified_v2_components tests.test_unified_v2_spec \
  tests.test_unified_v2_training -v
git diff --check
git add models/unified_v2_components.py models/unified_decoupled_cdm.py \
  trainers/engine.py tests/test_unified_v2_components.py \
  tests/test_unified_v2_training.py
git commit -m "feat: complete missing mastery with low rank factors"
ROOT=/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-mastery-20260712
conda run -n decoupled_cd python scripts/run_unified_validation.py smoke \
  --architecture a1 --seed 42 --epochs 1 \
  --output-root "$ROOT/smoke/a1"
```

Expected: focused tests pass; one A1 GPU smoke completes with finite losses and all six architecture invariants.

- [ ] **Step 5: Bundle the A1 implementation**

```bash
git bundle create "$ROOT/bundles/a1-implementation-$(git rev-parse --short=12 HEAD).bundle" HEAD
sha256sum "$ROOT"/bundles/a1-implementation-*.bundle
```

### Task 9: Frozen-Cohort A1 Validation, Freeze-or-Diagnose, and Test Opening

**Files:**
- Create: `docs/experiments/unified_mastery_a1_20260712.md`
- Create only on failure: `docs/experiments/unified_mastery_failure_registry.md`
- Generated outside Git: campaign `a1/`, `decisions/`, and optional `test-confirmation/`

**Interfaces:**
- Consumes: immutable `cohort.json`, A0 validation rows, audited strongest external comparator registry.
- Produces: candidate decision and either a frozen final architecture or one fully registered failure diagnosis.

- [ ] **Step 1: Run A1 on the complete frozen cohort**

```bash
ROOT=/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/unified-mastery-20260712
conda run -n decoupled_cd python scripts/unified_validation_controller.py init \
  --campaign-id unified-mastery-20260712 --architecture a1 \
  --cohort "$ROOT/cohort.json" --baseline-audit "$ROOT/audit/baselines.json" \
  --state "$ROOT/controllers/a1.json"
conda run -n decoupled_cd python scripts/unified_validation_controller.py run-validation \
  --state "$ROOT/controllers/a1.json" --parallel-gpus
```

Expected: all three primary datasets finish both validation protocols before decision publication; no dataset replacement or test access.

- [ ] **Step 2: Evaluate the registered candidate gate**

```bash
conda run -n decoupled_cd python scripts/unified_campaign.py \
  --baseline-metrics "$ROOT/controllers/a0-primary-validation-rows.json" \
  --candidate-metrics "$ROOT/controllers/a1-validation-rows.json" \
  --output "$ROOT/decisions/a1-vs-a0.json"
```

Expected: JSON reports all metric deltas, AUC gates, DOA soft ranking, parameter count, both fingerprints and identical cohort hash.

- [ ] **Step 3: Follow the predeclared branch without changing test data**

If A1 fails any hard gate, create `docs/experiments/unified_mastery_failure_registry.md` with this filled schema using values from the decision JSON:

```text
failure_id -> failed dataset/split -> exact double-precision delta -> responsible module
-> mechanism hypothesis -> literature retrieval question -> variable mapping
-> replacement module input/output -> acceptance gate
```

Stop implementation after committing that diagnosis. Return to the approved brainstorming protocol: review at most three mechanisms, replace one whole module, and do not start a second replacement in this plan.

If A1 passes, allow only predeclared numeric tuning over rank, dimension, learning rate, weight decay, batch size, epoch, patience and existing nonzero loss weights. Select by mean zero AUC, worst zero AUC, weighted DOA, then parameter count; rerun the selected validation configuration once through the controller and freeze its hashes.

- [ ] **Step 4: Require external wins before opening test**

Run the final external gate against the strongest audited same-protocol rows. Open test only if all three primary datasets have `zero_auc > strongest_external_zero_auc` and both overall AUCs meet their external comparator:

```bash
conda run -n decoupled_cd python scripts/unified_campaign.py external-gate \
  --candidate-metrics "$ROOT/controllers/frozen-validation-rows.json" \
  --baseline-audit "$ROOT/audit/baselines.json" \
  --cohort "$ROOT/cohort.json" \
  --output "$ROOT/decisions/final-external-gate.json"
conda run -n decoupled_cd python -m unittest discover -s tests -v
conda run -n decoupled_cd python -m compileall configs data models trainers utils scripts tests
git diff --check
```

Expected before test authorization: external gate passes three of three, full suite passes, compileall and diff check exit 0, worktree is clean. If the gate fails, test remains closed and the failure branch in Step 3 applies.

- [ ] **Step 5: Run confirmation test once and close the campaign**

After controller authorization only:

```bash
conda run -n decoupled_cd python scripts/unified_validation_controller.py authorize-test \
  --state "$ROOT/controllers/final.json" \
  --external-gate "$ROOT/decisions/final-external-gate.json"
conda run -n decoupled_cd python scripts/unified_validation_controller.py run-test \
  --state "$ROOT/controllers/final.json" --parallel-gpus
```

Write the Chinese final report with separate standard, holdout and zero AUC tables; exact comparator identity; DOA as internal/optional evidence; same fingerprint proof; excluded datasets; failed candidates; GPU UUID/peak memory; seed, hashes, commits and test-once caveat. Then:

```bash
git add docs/experiments/unified_mastery_a1_20260712.md
test ! -f docs/experiments/unified_mastery_failure_registry.md || \
  git add docs/experiments/unified_mastery_failure_registry.md
git commit -m "docs: close unified mastery completion campaign"
git bundle create "$ROOT/bundles/final-$(git rev-parse --short=12 HEAD).bundle" HEAD
sha256sum "$ROOT"/bundles/final-*.bundle
git status --short
```

Expected: final status is empty, bundle checksum is recorded, and no push has occurred. If the failure registry file was not created, omit it from `git add` rather than creating an empty file.

## Execution Checkpoints

1. Tasks 1–6 form the implementation/control checkpoint; no real training starts before focused tests pass and the worktree is clean.
2. Task 7 freezes the primary cohort from A0 validation; this is the last point where exploration-pool membership may affect selection.
3. Task 8 is the only registered architecture change in this plan.
4. Task 9 either freezes a winning A1 or stops with a mechanism diagnosis; it never silently broadens the module search.
5. Any dirty tree, data/vendor hash change, environment import failure or controller provenance failure stops the campaign globally. OOM, NaN, label-order mismatch or missing artifact fails only that attempt and never authorizes an unregistered batch-size change.
