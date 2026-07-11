# Unified V2 Same-Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现并验证一套在至少三个数据集上使用完全相同 M1–M4 计算图、同时输出预测与逐知识点 mastery 的 Unified V2。

**Architecture:** 新建 `UnifiedDecoupledCDM`，不继续扩张历史 `DecoupledCDMV2`。M1 编码已测知识，M2 整体生成未测知识状态，M3 构成唯一 mastery state map，M4 以单调结构输出答对概率；B0、M2、M2+M3 组合仅用于全局模块识别。

**Tech Stack:** Python 3.12、PyTorch、pandas、NumPy、标准库 `unittest`、现有 campaign/test-once 工具、CUDA。

## Global Constraints

- 所有真实实验固定 `seed=42`、`doa_seed=42`、`min_responses=3`、`split_seed=2024`。
- 主表至少三个数据集；所有主表 checkpoint 的 architecture fingerprint 必须相同。
- 数据集间只允许改变 concept dimension、batch size、learning rate、weight decay、epochs、patience 和非零数值型 loss weight。
- M1/M4 在所有候选中必选；M2/M3 只能在全局候选中整体加入或删除，同一候选内所有数据集的模块开关必须一致。旧 V2 只用于历史 checkpoint，不迁移旧 residual、adapter、hybrid 或 router 路径。
- 模块选择只读 validation；冻结后才运行确认性 test。已见历史 test 不宣称全局盲测。
- 不修改 `/home/xph/jwc/research/decoupled_cd_v2`，不 push；远端提交身份固定为 `chiangWC <215551297+chiangWC@users.noreply.github.com>`。
- 数据、日志、checkpoint、预测、mastery 与 ledger 写入 `/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/`，不进入 Git。
- GPU 优先使用空闲卡；显存使用低于一半的卡可并发，同一卡由 `flock` 保护；OOM 不得静默改变 batch size。

---

### Task 1: Architecture specification and fingerprint

**Files:**
- Create: `models/unified_v2_spec.py`
- Create: `tests/test_unified_v2_spec.py`
- Modify: `models/__init__.py`

**Interfaces:**
- Produces: `UnifiedArchitectureSpec.manifest() -> dict[str, str | int]` and `UnifiedArchitectureSpec.fingerprint() -> str`。
- Consumes: Python JSON/SHA-256 only; no model code dependency.

- [ ] **Step 1: Write the failing contract tests**

```python
import unittest

from models.unified_v2_spec import UnifiedArchitectureSpec


class UnifiedArchitectureSpecTests(unittest.TestCase):
    def test_numeric_training_hyperparameters_do_not_change_fingerprint(self):
        left = UnifiedArchitectureSpec(inference="graph", composer="coverage")
        right = UnifiedArchitectureSpec(inference="graph", composer="coverage")
        self.assertEqual(left.fingerprint(), right.fingerprint())

    def test_module_change_changes_fingerprint(self):
        base = UnifiedArchitectureSpec(inference="prior", composer="mask")
        graph = UnifiedArchitectureSpec(inference="graph", composer="mask")
        self.assertNotEqual(base.fingerprint(), graph.fingerprint())

    def test_invalid_combination_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "coverage composer requires graph inference"):
            UnifiedArchitectureSpec(inference="prior", composer="coverage")
```

- [ ] **Step 2: Run the test and verify the missing module failure**

Run: `python -m unittest tests.test_unified_v2_spec -v`

Expected: `ModuleNotFoundError: No module named 'models.unified_v2_spec'`.

- [ ] **Step 3: Implement the immutable specification**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal


@dataclass(frozen=True)
class UnifiedArchitectureSpec:
    inference: Literal["prior", "graph"] = "prior"
    composer: Literal["mask", "coverage"] = "mask"
    decoder: Literal["monotonic"] = "monotonic"
    mastery_output: Literal["student-concept"] = "student-concept"
    version: int = 1

    def __post_init__(self) -> None:
        if self.composer == "coverage" and self.inference != "graph":
            raise ValueError("coverage composer requires graph inference")

    def manifest(self) -> dict[str, str | int]:
        payload = asdict(self)
        payload["modules"] = {
            ("prior", "mask"): "m1-m4",
            ("graph", "mask"): "m1-m2-m4",
            ("graph", "coverage"): "m1-m2-m3-m4",
        }[(self.inference, self.composer)]
        return payload

    def fingerprint(self) -> str:
        payload = json.dumps(self.manifest(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

Export the class from `models/__init__.py`. The manifest must contain no dataset name, tensor size, concept dimension, optimizer, epoch or loss weight.

- [ ] **Step 4: Run contract tests**

Run: `python -m unittest tests.test_unified_v2_spec -v`

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add models/unified_v2_spec.py models/__init__.py tests/test_unified_v2_spec.py
git commit -m "feat: define unified architecture fingerprint"
```

### Task 2: M1 evidence encoder and M4 monotonic B0

**Files:**
- Create: `models/unified_v2_components.py`
- Create: `models/unified_decoupled_cdm.py`
- Create: `tests/test_unified_v2_components.py`
- Modify: `models/__init__.py`

**Interfaces:**
- Produces: `TestedKnowledgeEvidenceEncoder.forward(q_matrix, student_exercise_mask, response_matrix, student_tkc_mask) -> TestedKnowledgeState`.
- Produces: `MonotonicDiagnosisDecoder.forward(state_map, q_matrix, target_student_ids, target_exercise_ids) -> tuple[Tensor, Tensor, Tensor]` and deterministic test helper `decode_from_mastery(target_mastery, q_vectors, target_exercise_ids) -> Tensor`.
- Produces: `UnifiedDecoupledCDM.forward(q_matrix, concept_graph, student_exercise_mask, response_matrix, student_tkc_mask, student_ukc_mask, student_concept_evidence, target_student_ids, target_exercise_ids, use_student_subset=False) -> DecoupledForwardOutput` with non-null mastery.
- Consumes: `UnifiedArchitectureSpec` from Task 1 and the existing `DecoupledForwardOutput` protocol.

- [ ] **Step 1: Write failing M1/M4 tests**

```python
import inspect
import unittest

import torch

from models.unified_decoupled_cdm import UnifiedDecoupledCDM
from models.unified_v2_components import MonotonicDiagnosisDecoder, TestedKnowledgeEvidenceEncoder
from models.unified_v2_spec import UnifiedArchitectureSpec


class UnifiedComponentTests(unittest.TestCase):
    def test_b0_always_emits_student_concept_mastery(self):
        model = UnifiedDecoupledCDM(
            num_students=3,
            num_exercises=2,
            num_concepts=3,
            dim=4,
            architecture=UnifiedArchitectureSpec(inference="prior", composer="mask"),
        )
        q = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        history = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        tkc = (history @ q > 0).float()
        tensors = {
            "q_matrix": q,
            "concept_graph": torch.eye(3),
            "student_exercise_mask": history,
            "response_matrix": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            "student_tkc_mask": tkc,
            "student_ukc_mask": 1.0 - tkc,
            "student_concept_evidence": None,
            "target_student_ids": torch.tensor([0, 1, 2]),
            "target_exercise_ids": torch.tensor([0, 1, 1]),
        }
        output = model(**tensors)
        self.assertEqual(tuple(output.mastery.shape), (3, 3))
        self.assertTrue(torch.isfinite(output.mastery).all())

    def test_decoder_is_monotone_in_target_mastery(self):
        decoder = MonotonicDiagnosisDecoder(num_exercises=1, num_concepts=1, dim=4)
        low = decoder.decode_from_mastery(torch.tensor([[0.2]]), torch.tensor([[1.0]]), torch.tensor([0]))
        high = decoder.decode_from_mastery(torch.tensor([[0.8]]), torch.tensor([[1.0]]), torch.tensor([0]))
        self.assertGreaterEqual(float(high), float(low))

    def test_m1_changes_with_student_responses(self):
        encoder = TestedKnowledgeEvidenceEncoder(dim=4, evidence_cap=20.0)
        q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        history = torch.ones(1, 2)
        common = {
            "q_matrix": q,
            "student_exercise_mask": history,
            "student_tkc_mask": torch.ones(1, 2),
        }
        left = encoder(response_matrix=torch.zeros(1, 2), **common)
        right = encoder(response_matrix=torch.ones(1, 2), **common)
        self.assertFalse(torch.equal(left.tkc_states, right.tkc_states))
```

- [ ] **Step 2: Verify tests fail because classes do not exist**

Run: `python -m unittest tests.test_unified_v2_components -v`

Expected: import errors for the three new classes.

- [ ] **Step 3: Implement M1 as the only response-history encoder**

Implement `TestedKnowledgeState(tkc_states, direct_reliability)` and the following computation in `unified_v2_components.py`:

```python
correct = (student_exercise_mask * response_matrix) @ q_matrix
incorrect = (student_exercise_mask * (1.0 - response_matrix)) @ q_matrix
attempts = correct + incorrect
features = torch.stack([
    torch.log1p(correct),
    torch.log1p(incorrect),
    correct / attempts.clamp_min(1.0),
], dim=-1)
tkc_states = self.encoder(features) * student_tkc_mask.unsqueeze(-1)
direct_reliability = (torch.log1p(attempts) / math.log1p(self.evidence_cap)).clamp(0.0, 1.0)
return TestedKnowledgeState(tkc_states=tkc_states, direct_reliability=direct_reliability)
```

`self.encoder` is `Linear(3, dim) -> ReLU -> Linear(dim, dim)` and is shared across all students/concepts.

- [ ] **Step 4: Implement M4 as the only prediction decoder**

Use one mastery projection, positive discrimination, concept difficulty and exercise bias:

```python
mastery_logits = self.mastery_head(state_map).squeeze(-1)
mastery = torch.sigmoid(mastery_logits)
target_mastery = mastery[target_student_ids]
weights = q_matrix[target_exercise_ids]
weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
difficulty = self.concept_difficulty.weight.squeeze(-1)
margin = (target_mastery - difficulty.unsqueeze(0)) * weights
discrimination = F.softplus(self.exercise_discrimination(target_exercise_ids)).squeeze(-1)
logits = discrimination * margin.sum(dim=1) + self.exercise_bias(target_exercise_ids).squeeze(-1)
cognitive_probs = torch.sigmoid(logits)
return cognitive_probs, cognitive_probs, mastery
```

Do not instantiate the old NCF match, per-student guess/slip, hybrid head, target residual or router. `UnifiedDecoupledCDM` uses a learned concept-prior tensor to fill UKCs in B0 and returns the existing output dataclass.

- [ ] **Step 5: Run component and legacy regression tests**

Run: `python -m unittest tests.test_unified_v2_components tests.test_v2_support_adaptive -v`

Expected: new tests pass and historical V2 tests remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add models/unified_v2_components.py models/unified_decoupled_cdm.py models/__init__.py tests/test_unified_v2_components.py
git commit -m "feat: add unified evidence encoder and monotonic decoder"
```

### Task 3: M2 replacement UKC inference network

**Files:**
- Modify: `models/unified_v2_components.py`
- Modify: `models/unified_decoupled_cdm.py`
- Modify: `tests/test_unified_v2_components.py`

**Interfaces:**
- Produces: `UntestedKnowledgeInferenceNetwork.forward(tkc_states, tkc_mask, ukc_mask, concept_graph, direct_reliability) -> InferredKnowledgeState`.
- `InferredKnowledgeState` contains `ukc_states`, `inferred_reliability`, `reachable_mask`.

- [ ] **Step 1: Add failing personalization and no-residual tests**

```python
from models.unified_v2_components import UntestedKnowledgeInferenceNetwork


def test_m2_is_student_conditioned(self):
    module = UntestedKnowledgeInferenceNetwork(num_concepts=3, dim=4, layers=2)
    graph = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    tkc_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ukc_mask = 1.0 - tkc_mask
    states = torch.zeros(2, 3, 4)
    states[0, 0] = 1.0
    states[1, 0] = -1.0
    out = module(
        tkc_states=states,
        tkc_mask=tkc_mask,
        ukc_mask=ukc_mask,
        concept_graph=graph,
        direct_reliability=tkc_mask,
    )
    self.assertFalse(torch.equal(out.ukc_states[0, 1], out.ukc_states[1, 1]))

def test_m2_unreachable_ukc_uses_prior_not_static_broadcast(self):
    module = UntestedKnowledgeInferenceNetwork(num_concepts=3, dim=4, layers=1)
    graph = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    tkc_mask = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    ukc_mask = 1.0 - tkc_mask
    out = module(
        tkc_states=torch.randn(2, 3, 4),
        tkc_mask=tkc_mask,
        ukc_mask=ukc_mask,
        concept_graph=graph,
        direct_reliability=tkc_mask,
    )
    self.assertTrue(torch.equal(out.ukc_states[:, 2], module.concept_prior[2].expand(2, -1)))
```

- [ ] **Step 2: Run and observe missing M2**

Run: `python -m unittest tests.test_unified_v2_components.UnifiedComponentTests.test_m2_is_student_conditioned -v`

Expected: `ImportError` or missing attribute failure.

- [ ] **Step 3: Implement M2 as a primary graph diffusion path**

Normalize a self-loop-free topology, clamp observed TKC nodes after each propagation step, and use the concept prior only for unreachable UKCs:

```python
topology = concept_graph.ne(0).to(tkc_states.dtype)
topology = topology - torch.diag_embed(torch.diagonal(topology))
source = topology * direct_reliability.unsqueeze(1)
first_transition = source / source.sum(dim=-1, keepdim=True).clamp_min(1e-8)
later_transition = topology / topology.sum(dim=-1, keepdim=True).clamp_min(1e-8)
hidden = tkc_states
for index, layer in enumerate(self.layers):
    transition = first_transition if index == 0 else later_transition.unsqueeze(0)
    propagated = torch.einsum("skj,sjd->skd", transition, layer(hidden))
    hidden = torch.where(tkc_mask.bool().unsqueeze(-1), tkc_states, torch.tanh(propagated))
reachable = hidden.abs().sum(dim=-1).gt(0) & ukc_mask.bool()
prior = self.concept_prior.unsqueeze(0).expand_as(hidden)
ukc_states = torch.where(reachable.unsqueeze(-1), hidden, prior) * ukc_mask.unsqueeze(-1)
inferred_reliability = reachable.to(tkc_states.dtype)
```

Do not import or call `_personalized_ukc_residual`; M2 entirely replaces static UKC generation.

- [ ] **Step 4: Wire `inference="graph"` into the unified model**

For `inference="prior"`, fill UKCs from the learned prior. For `inference="graph"`, call M2. In both cases construct `state_map = tkc_states + ukc_states` before M4.

- [ ] **Step 5: Run tests**

Run: `python -m unittest tests.test_unified_v2_components -v`

Expected: mastery, monotonicity, personalization and disconnected fallback all pass.

- [ ] **Step 6: Commit**

```bash
git add models/unified_v2_components.py models/unified_decoupled_cdm.py tests/test_unified_v2_components.py
git commit -m "feat: replace static ukc path with unified inference network"
```

### Task 4: M3 coverage-aware state composer

**Files:**
- Modify: `models/unified_v2_components.py`
- Modify: `models/unified_decoupled_cdm.py`
- Modify: `tests/test_unified_v2_components.py`

**Interfaces:**
- Produces: `CoverageAwareStateComposer.forward(tkc_states, ukc_states, concept_prior, tkc_mask, direct_reliability, inferred_reliability) -> (state_map, source_weight)`.

- [ ] **Step 1: Add failing boundary tests**

```python
from models.unified_v2_components import CoverageAwareStateComposer


def test_m3_prefers_direct_state_at_full_reliability(self):
    composer = CoverageAwareStateComposer(dim=4)
    tkc = torch.tensor([[[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]]])
    ukc = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]])
    prior = torch.zeros(2, 4)
    tkc_mask = torch.tensor([[1.0, 0.0]])
    state, weight = composer(
        tkc_states=tkc,
        ukc_states=ukc,
        concept_prior=prior,
        tkc_mask=tkc_mask,
        direct_reliability=torch.tensor([[1.0, 0.0]]),
        inferred_reliability=torch.tensor([[0.0, 1.0]]),
    )
    self.assertTrue(torch.allclose(state[:, 0], tkc[:, 0], atol=1e-5))

def test_m3_uses_inference_for_zero_coverage_reachable_concept(self):
    composer = CoverageAwareStateComposer(dim=4)
    tkc = torch.zeros(1, 2, 4)
    ukc = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]])
    state, weight = composer(
        tkc_states=tkc,
        ukc_states=ukc,
        concept_prior=torch.zeros(2, 4),
        tkc_mask=torch.tensor([[1.0, 0.0]]),
        direct_reliability=torch.zeros(1, 2),
        inferred_reliability=torch.tensor([[0.0, 1.0]]),
    )
    self.assertTrue(torch.allclose(state[:, 1], ukc[:, 1], atol=1e-5))

def test_m3_does_not_depend_on_dataset_name(self):
    self.assertNotIn("dataset", inspect.signature(CoverageAwareStateComposer.forward).parameters)
```

- [ ] **Step 2: Verify the composer is absent**

Run: `python -m unittest tests.test_unified_v2_components.UnifiedComponentTests.test_m3_prefers_direct_state_at_full_reliability -v`

Expected: import or name failure.

- [ ] **Step 3: Implement one quality-aware composition module**

Use reliability-calibrated logits over direct, inferred and prior states:

```python
quality = torch.stack([direct_reliability, inferred_reliability, 1.0 - torch.maximum(direct_reliability, inferred_reliability)], dim=-1)
logits = self.quality_network(quality).view(*quality.shape[:-1], 3)
valid = torch.stack([tkc_mask, 1.0 - tkc_mask, torch.ones_like(tkc_mask)], dim=-1).bool()
weights = torch.softmax(logits.masked_fill(~valid, -1e9), dim=-1)
candidates = torch.stack([tkc_states, ukc_states, concept_prior.unsqueeze(0).expand_as(tkc_states)], dim=-2)
state_map = (weights.unsqueeze(-1) * candidates).sum(dim=-2)
return state_map, weights
```

Use `Linear(3, 3)` for `quality_network`; initialize its weight to `28 * I` and bias to `-14`. Full direct reliability or full inferred reliability then selects its corresponding state within `1e-5`; this is an initialization rule, not a parallel legacy path.

- [ ] **Step 4: Wire `composer="coverage"` and expose diagnostics**

Return `source_weight` through a new optional `source_weights` field on `DecoupledForwardOutput`; keep `mastery` mandatory. `composer="mask"` uses deterministic TKC/UKC assembly and exists only for module attribution.

- [ ] **Step 5: Run tests**

Run: `python -m unittest tests.test_unified_v2_components -v`

Expected: all component tests pass without legacy V2 changes.

- [ ] **Step 6: Commit**

```bash
git add models/unified_v2_components.py models/unified_decoupled_cdm.py models/decoupled_cdm.py tests/test_unified_v2_components.py
git commit -m "feat: add coverage-aware mastery state composer"
```

### Task 5: Training, checkpoint loading, DOA and CLI integration

**Files:**
- Modify: `scripts/train.py`
- Modify: `trainers/engine.py`
- Modify: `scripts/analyze_prediction_slices.py`
- Modify: `scripts/evaluate_doa.py`
- Create: `tests/test_unified_v2_training.py`

**Interfaces:**
- CLI: `--model unified_v2 --unified-inference {prior,graph} --unified-composer {mask,coverage}`.
- Summary fields: `architecture_manifest`, `architecture_fingerprint`, `unified_mastery_loss_weight`.
- Loader: reconstructs only from the summary manifest and numerical dimensions.

- [ ] **Step 1: Add failing CLI and loader tests**

Test that coverage+prior is rejected, fingerprint is written, mastery loss weight must be positive, and a saved unified checkpoint reloads with identical predictions.

```python
self.assertRaisesRegex(ValueError, "coverage composer requires graph inference", validate_model_args, args)
self.assertRegex(summary["architecture_fingerprint"], r"^[0-9a-f]{64}$")
self.assertGreater(summary["unified_mastery_loss_weight"], 0.0)
self.assertTrue(torch.equal(before.probs, after.probs))
```

- [ ] **Step 2: Run and confirm `unified_v2` is unknown**

Run: `python -m unittest tests.test_unified_v2_training -v`

Expected: parser/model construction failures.

- [ ] **Step 3: Add the unified CLI and model builder**

Add the enum arguments, create `UnifiedArchitectureSpec`, instantiate `UnifiedDecoupledCDM`, and record its manifest/fingerprint. Reject every `V1_ONLY_FLAG_ATTRS` and `V2_ONLY_FLAG_ATTRS` flag when `args.model == "unified_v2"`.

- [ ] **Step 4: Use one mastery loss in both training modes**

Add `unified_mastery_bce_weight` to full-batch and student-minibatch epoch functions. Directly supervise the same `output.mastery` on observed student–concept cells using smoothed empirical correctness; do not create a second prediction head:

```python
attempts = tensors["student_concept_evidence"][..., 0]
correct = tensors["student_concept_evidence"][..., 1]
target = (correct + 1.0) / (attempts + 2.0)
observed = attempts >= 3.0
mastery_loss = F.binary_cross_entropy(output.mastery[observed], target[observed])
loss = loss + unified_mastery_bce_weight * mastery_loss
```

For student minibatches, index `attempts`, `correct` and `observed` by the same sorted unique student IDs used to produce `output.mastery`. Require a weight greater than zero for unified models.

- [ ] **Step 5: Extend loaders and evaluators**

In `analyze_prediction_slices.load_model`, dispatch `model_variant == "unified_v2"` from `architecture_manifest`. In `evaluate_doa.extract_mastery`, assert unified mastery is non-null rather than printing `[skip]`.

- [ ] **Step 6: Run focused and full suites**

Run: `python -m unittest tests.test_unified_v2_spec tests.test_unified_v2_components tests.test_unified_v2_training tests.test_complete_model_protocol tests.test_doa_external tests.test_remote_campaign tests.test_v2_support_adaptive -v`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add scripts/train.py trainers/engine.py scripts/analyze_prediction_slices.py scripts/evaluate_doa.py tests/test_unified_v2_training.py
git commit -m "feat: integrate unified model training and evaluation"
```

### Task 6: Dataset eligibility and immutable primary cohort

**Files:**
- Create: `scripts/unified_dataset_audit.py`
- Create: `scripts/unified_cohort.py`
- Create: `tests/test_unified_cohort.py`

**Interfaces:**
- Produces: dataset audit JSON with standard/holdout paths, Q hash, zero-coverage counts and eligibility.
- Produces: immutable `cohort.json` with at least three dataset IDs and its canonical SHA-256.

- [ ] **Step 1: Write failing cohort tests**

Cover five ready datasets, reject Junyi/EdNet until Q+holdout exist, reject cohorts smaller than three, and reject replacement after freeze.

- [ ] **Step 2: Run and verify missing scripts**

Run: `python -m unittest tests.test_unified_cohort -v`

Expected: import failures.

- [ ] **Step 3: Implement train-only eligibility audit**

For each split pair require `train.csv`, `valid.csv`, `test.csv`, `Q_matrix.csv`, holdout `split_summary.json`, identical ID domains and nonzero exact-zero validation rows. The audit may hash test files but must not read labels or metrics during cohort selection; store `test_content_hash_only=true`.

- [ ] **Step 4: Implement exclusive cohort freeze**

Write `cohort.json` with `os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)`, fsync the file and parent directory, and include audit hashes plus B0 validation references. Reject any later dataset list mismatch.

- [ ] **Step 5: Run tests and audit the remote pool**

Run: `python -m unittest tests.test_unified_cohort -v`

Then run: `python -m scripts.unified_dataset_audit --root /home/xph/jwc/research/knofield_data --output /tmp/unified-dataset-audit.json`

Expected: ASSIST09, ASSIST17, NIPS34, MOOCRadar and XES3G5M eligible; Junyi and EdNet-ICDM provisional.

- [ ] **Step 6: Commit**

```bash
git add scripts/unified_dataset_audit.py scripts/unified_cohort.py tests/test_unified_cohort.py
git commit -m "feat: audit datasets and freeze unified cohort"
```

### Task 7: Global module-selection campaign

**Files:**
- Create: `scripts/unified_campaign.py`
- Create: `tests/test_unified_campaign.py`
- Modify: `scripts/run_remote_campaign.py`

**Interfaces:**
- Consumes: cohort hash, architecture fingerprint, per-dataset validation metric JSON.
- Produces: `candidate-decision.json` containing every AUC/DOA delta and the global hard-gate decision.

- [ ] **Step 1: Write failing gate tests**

Test `ceil(2N/3)`, all-dataset overall/weighted-DOA non-regression, one zero-AUC delta at least `0.001`, and rejection of mixed fingerprints.

- [ ] **Step 2: Run and verify missing selector**

Run: `python -m unittest tests.test_unified_campaign -v`

Expected: import failures.

- [ ] **Step 3: Implement `evaluate_candidate()`**

The pure function accepts baseline/candidate rows, verifies identical dataset sets and fingerprints, computes raw double-precision deltas, and returns `pass=False` with named failed gates. It must not accept test paths or test metrics.

- [ ] **Step 4: Bind fingerprint and cohort to campaign attempts**

Add `--architecture-manifest` and `--cohort` to `run_remote_campaign.py`; hash both as immutable inputs and reject commands whose summary output fingerprint differs.

- [ ] **Step 5: Run suites**

Run: `python -m unittest tests.test_unified_campaign tests.test_remote_campaign -v`

Expected: all tests pass, including dirty-tree and immutable-attempt regressions.

- [ ] **Step 6: Commit**

```bash
git add scripts/unified_campaign.py scripts/run_remote_campaign.py tests/test_unified_campaign.py tests/test_remote_campaign.py
git commit -m "feat: enforce global unified module selection"
```

### Task 8: Smoke runs and validation-only module matrix

**Files:**
- Create: `scripts/run_unified_validation.py`
- Create: `docs/unified_v2_validation_log.md`

**Interfaces:**
- Runs B0, M2 and M2+M3 on every frozen cohort member for both standard/holdout validation.
- Never receives a real test path; the training command maps its evaluation input to `valid.csv`.

- [ ] **Step 1: Run one-epoch synthetic CPU and GPU smoke**

Run B0, graph+mask and graph+coverage with seed 42. Assert each summary has nonempty mastery, finite loss, the expected fingerprint and peak GPU memory.

- [ ] **Step 2: Audit GPU availability before each real job**

Run: `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits`

Select idle GPUs first; permit a used GPU only when `memory.used / memory.total < 0.5`. Store the selection in the attempt metadata and use one `flock` lock per physical GPU.

- [ ] **Step 3: Train Unified-V2-B0 on all five eligible datasets**

Use the existing per-dataset numerical recipes as starting points, but identical architecture `inference=prior, composer=mask`. Run standard and holdout validation, coverage slices and DOA. Freeze a primary cohort of at least three datasets using only B0 validation and audit criteria.

- [ ] **Step 4: Run the global module matrix on the frozen cohort**

Run `graph+mask`, then `graph+coverage`; each candidate must cover every cohort member and both splits before `evaluate_candidate()` is called. Do not launch the next candidate if the current candidate makes `successes + remaining < 3`.

- [ ] **Step 5: Tune numerical hyperparameters only after a module set passes**

For each cohort dataset permit concept dimension, batch/student-batch, learning rate, weight decay, epochs, patience and nonzero mastery loss weight. Keep manifest/fingerprint fixed. Use seed 42 only.

- [ ] **Step 6: Record validation decisions**

Append a table to `docs/unified_v2_validation_log.md` with candidate fingerprint, cohort hash, standard/holdout overall and zero AUC, ordinary/weighted DOA, failed gates, GPU attempts and decision. Commit negative outcomes as well.

- [ ] **Step 7: Commit the runner and validation ledger**

```bash
git add scripts/run_unified_validation.py docs/unified_v2_validation_log.md
git commit -m "exp: record unified module validation matrix"
```

### Task 9: Literature-driven whole-module replacement loop

**Files:**
- Create: `docs/unified_v2_module_diagnostics.md`
- Modify only one of: `models/unified_v2_components.py` M1, M2, M3 or M4 implementation per iteration.
- Modify: `tests/test_unified_v2_components.py`

**Interfaces:**
- Consumes: a named validation hard-gate failure from Task 8.
- Produces: one replacement module with the same public input/output contract.

- [ ] **Step 1: Write the failure diagnosis before searching**

Record the failing metrics, affected datasets/splits and a falsifiable mechanism hypothesis. Example schema: `failure_id`, `module`, `observed_delta`, `search_question`, `acceptance_gate`.

- [ ] **Step 2: Search primary papers using the diagnosed mechanism**

Search cognitive diagnosis first, then recommendation, incomplete multi-view learning, graph semi-supervision, MNAR/exposure, PU/weak supervision, noisy-label learning, matrix completion or other matching fields. Record paper title, venue, URL and variable mapping; the existing seed list is not a whitelist.

- [ ] **Step 3: Select at most three mechanisms for code-level review**

Reject candidates that are only a scalar loss, residual, adapter, dataset-specific switch or a second parallel prediction head. Select one whole-module replacement whose interface matches the current M1/M2/M3/M4 contract.

- [ ] **Step 4: Add a failing mechanism-specific test and replace one module**

The test must encode the diagnosed failure, such as disconnected graph fallback, non-random missingness calibration or noisy evidence robustness. Replace the implementation rather than stacking it with the previous module.

- [ ] **Step 5: Re-run the entire frozen cohort validation gate**

Use the same cohort, seed and numerical budget. Continue searching only when a new validation diagnosis differs from the previous failure. Stop adding structure immediately once the global success conditions pass.

- [ ] **Step 6: Commit each literature iteration separately**

```bash
git add docs/unified_v2_module_diagnostics.md models/unified_v2_components.py tests/test_unified_v2_components.py
git commit -m "exp: replace unified module from validation diagnosis"
```

### Task 10: Freeze, confirmatory test, report and bundle

**Files:**
- Modify: `scripts/complete_model_protocol.py`
- Modify: `tests/test_complete_model_protocol.py`
- Create: `docs/unified_v2_final_results.md`

**Interfaces:**
- Frozen selection binds cohort hash, architecture fingerprint, numerical config, checkpoint hash and validation record.
- Test evaluation rejects mixed fingerprints, duplicate claims and absent mastery/DOA outputs.

- [ ] **Step 1: Add failing protocol tests**

Test that freeze rejects fewer than three datasets, mismatched fingerprints, zero mastery loss weight and missing DOA configuration; test duplicate evaluation remains rejected.

- [ ] **Step 2: Extend freeze and evaluation records**

Add `cohort_sha256`, `architecture_manifest`, `architecture_fingerprint` and `mastery_required=true` to frozen records. At evaluation completion require overall/coverage JSON, prediction hash, mastery hash and DOA hash.

- [ ] **Step 3: Run protocol and full tests**

Run: `python -m unittest discover -s tests -v`

Expected: all tests pass.

- [ ] **Step 4: Freeze the single final architecture**

Verify every selected dataset has the same fingerprint and passed standard/holdout validation gates. Freeze per-dataset numerical configurations without changing the manifest.

- [ ] **Step 5: Run one confirmatory test per frozen dataset**

Run overall, exact-zero/low/full coverage, ACC/RMSE, ordinary and weighted DOA. Do not alter architecture or cohort based on these results. Clearly disclose that historical test values existed before this unified campaign.

- [ ] **Step 6: Write the Chinese final report**

`docs/unified_v2_final_results.md` must separate: unified main results, module ablations, excluded datasets, literature replacements, negative results, standard vs concept-holdout splits, DOA, protocol limitations and current-vs-historical test visibility.

- [ ] **Step 7: Verify, commit and bundle**

```bash
python -m unittest discover -s tests -q
git diff --check
git add scripts/complete_model_protocol.py tests/test_complete_model_protocol.py docs/unified_v2_final_results.md
git commit -m "docs: finalize unified v2 campaign"
git status --porcelain
BUNDLE=/home/xph/jwc/research/local_data/decoupled_cd_codex_routes/bundles/unified-v2-final-$(git rev-parse --short HEAD).bundle
git bundle create "$BUNDLE" codex/remote-complete-model-20260710
git bundle verify "$BUNDLE"
sha256sum "$BUNDLE"
```

Expected: tests pass, `git diff --check` is silent, worktree is clean after the commit, bundle verification succeeds and SHA-256 is recorded in the final report.
