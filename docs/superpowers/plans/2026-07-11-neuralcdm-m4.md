# NeuralCDM M4 Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the complete Unified V2 linear M4 decoder with one globally identical NeuralCDM monotonic interaction network while preserving public model I/O and rejecting old manifests/checkpoints.

**Architecture:** Dense sigmoid-constrained `E x K` item-concept difficulty and positive item discrimination form `Q * (mastery - beta) * discrimination`. Three softplus-weight linear layers (`K -> 512 -> 256 -> 1`) with sigmoid activations are the only prediction path; the same mastery tensor is returned for DOA.

**Tech Stack:** Python 3.12, PyTorch, `unittest`, dataclass manifests.

## Global Constraints

- Work only in the remote `complete_model` worktree on the approved design commit derived from `6a6544e`.
- Use `chiangWC <215551297+chiangWC@users.noreply.github.com>`; do not push.
- Do not touch `decoupled_cd_v2`, initialize a controller, use GPU, run a real experiment, or read a real test label/path.
- Use seed 42 for deterministic initialization/gradient tests.
- Dense `E x K` difficulty and fixed widths 512/256 apply to every dataset.
- No legacy branch, residual, adapter, guessing/slip prediction transform, exercise-bias path, or second head.
- Observe every required RED before production edits.
- Commit implementation, tests, and diagnostic note together, separately from any future experimental ledger.

---

### Task 1: Add the complete failing decoder contract

**Files:**
- Modify: `tests/test_unified_v2_components.py`
- Modify: `tests/test_unified_v2_spec.py`
- Modify: `tests/test_unified_v2_training.py`

**Interfaces:**
- Consumes: existing decoder/model forward APIs and manifest parser.
- Produces: behavioral contract for `item_concept_difficulty`, `exercise_discrimination`, and `interaction_layers` without production changes.

- [ ] **Step 1: Add positive-weight fixture helpers**

```python
def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))

def set_effective_weight(layer: nn.Module, value: float) -> None:
    with torch.no_grad():
        layer.raw_weight.fill_(inverse_softplus(value))
```

Import `math`, `unittest.mock`, and `torch.nn as nn`.

- [ ] **Step 2: Add random/boundary monotonicity, Jacobian, and non-Q tests**

Under `torch.manual_seed(42)`, use Q rows `[1,1,0,0]` and `[0,1,0,1]`.
For zero/one boundaries and random ordered mastery pairs, increase each required
coordinate independently and require `p_high >= p_low - 1e-8`. Use
`torch.autograd.grad` once per output and require every required-coordinate
Jacobian `>= -1e-8`. Changing only a Q-zero coordinate from zero to one must
leave prediction bitwise equal.

- [ ] **Step 3: Add item heterogeneity and nonlinear-logit tests**

For two items sharing `Q=[1,1]`, set beta logits to `logit(0.2)` and
`logit(0.8)`, equalize discrimination, and require different predictions.

For one multi-concept item, fill raw weights/biases with `-20`, then configure
one path through all three layers with effective weights `4` and centered
biases. Evaluate mastery corners and require:

```python
mixed = logit(p11) - logit(p10) - logit(p01) + logit(p00)
self.assertGreater(abs(float(mixed)), 1e-2)
```

The old weighted-linear IRT logit has exactly zero mixed difference.

- [ ] **Step 4: Add initialization/gradient audit**

With seed 42, three sparse Q rows over eight concepts, and interior mastery,
require all probabilities in `(0.05,0.95)`. Backpropagate `probs.sum()` and
require finite nonzero gradients on required mastery and beta entries, selected
discrimination rows, and every interaction layer's raw weight and bias.
Recreate two decoders after resetting seed 42 and require identical state dicts
and nonzero first-layer effective-weight standard deviation.

- [ ] **Step 5: Add sole mastery/prediction-path test**

Wrap `decoder.decode_from_mastery` while running a small model forward. Require
nonempty `[students,concepts]` mastery, captured selected mastery equal to
`output.mastery[target_student_ids]`, and identical storage for `probs` and
`cognitive_probs`. Inspect decoder parameter/source names and reject
`concept_difficulty`, `exercise_bias`, `guess`, `slip`, `legacy`, or `residual`.

- [ ] **Step 6: Add old metadata/state rejection tests**

Freeze this old manifest and require `from_manifest` to reject it:

```python
old = {"inference": "graph", "composer": "mask", "decoder": "monotonic",
       "mastery_output": "student-concept", "version": 1,
       "modules": "m1-m2-m4"}
```

Construct an exact old decoder state dict containing `mastery_head.*`,
`concept_difficulty.weight`, `exercise_discrimination.weight`, and
`exercise_bias.weight`; require strict loading to raise missing/unexpected keys.
Add expectations for decoder `neuralcdm-monotonic`, version 2, and new module
labels to manifest/training tests.

- [ ] **Step 7: Run and record RED**

```bash
/home/xph/anaconda3/bin/conda run --no-capture-output -n decoupled_cd \
  python -m unittest tests.test_unified_v2_components \
  tests.test_unified_v2_spec tests.test_unified_v2_training -v
```

Expected: behavioral failures for absent dense beta/layers, zero mixed-logit
interaction, accepted v1 metadata/state, and missing initialization contract.
Fix setup errors only; do not edit production.

---

### Task 2: Implement the NeuralCDM decoder

**Files:**
- Modify: `models/unified_v2_components.py`
- Modify: `models/unified_decoupled_cdm.py`

**Interfaces:**
- Consumes: mastery `[S,K]`, Q `[B,K]`, exercise IDs `[B]`.
- Produces: unchanged tuple `(cognitive_probs[B], probs[B], mastery[S,K])` and standard PyTorch state for dense beta/discrimination/positive layers.

- [ ] **Step 1: Add inverse-softplus initialization and `PositiveLinear`**

```python
def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, *, effective_weight,
                 bias, perturbation=0.01):
        super().__init__()
        center = _inverse_softplus(effective_weight)
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.raw_weight, mean=center, std=perturbation)
        self.bias = nn.Parameter(torch.full((out_features,), bias))

    @property
    def effective_weight(self):
        return F.softplus(self.raw_weight)

    def forward(self, inputs):
        return F.linear(inputs, self.effective_weight, self.bias)
```

Global PyTorch seed controls perturbations; seed-reset tests prove determinism.

- [ ] **Step 2: Replace every legacy decoder parameter/computation**

```python
self.item_concept_difficulty = nn.Embedding(num_exercises, num_concepts)
self.exercise_discrimination = nn.Embedding(num_exercises, 1)
self.interaction_layers = nn.ModuleList([
    PositiveLinear(num_concepts, 512, effective_weight=0.5, bias=0.0),
    PositiveLinear(512, 256, effective_weight=1/512, bias=-0.5),
    PositiveLinear(256, 1, effective_weight=1/256, bias=-0.5),
])
```

Zero beta logits and initialize discrimination so softplus plus floor is one.
`decode_from_mastery` must use only:

```python
beta = torch.sigmoid(self.item_concept_difficulty(target_exercise_ids))
disc = F.softplus(self.exercise_discrimination(target_exercise_ids)) + floor
hidden = q_vectors * (target_mastery - beta) * disc
hidden = torch.sigmoid(self.interaction_layers[0](hidden))
hidden = torch.sigmoid(self.interaction_layers[1](hidden))
return torch.sigmoid(self.interaction_layers[2](hidden).squeeze(-1))
```

Keep mastery computation/tuple unchanged; add no alternate probability call.

- [ ] **Step 3: Replace public scalar difficulty derivation**

In `UnifiedDecoupledCDM.forward` compute Q-normalized selected beta:

```python
beta = torch.sigmoid(self.decoder.item_concept_difficulty(target_exercise_ids))
difficulty = (q_weights * beta).sum(dim=1)
```

Do not change other public output fields.

- [ ] **Step 4: Run component tests to GREEN**

Run the Task 1 command. Expected: decoder tests pass; only manifest fixtures
awaiting Task 3 may remain RED.

---

### Task 3: Bump canonical architecture identity

**Files:**
- Modify: `models/unified_v2_spec.py`
- Modify: `tests/test_unified_v2_spec.py`
- Modify: `tests/test_unified_v2_training.py`
- Modify valid manifest fixtures found in `tests/test_remote_campaign.py`, `tests/test_unified_validation.py`, and other tests.

**Interfaces:**
- Consumes: canonical manifest/fingerprint validation.
- Produces: only decoder `neuralcdm-monotonic`, version 2, NeuralCDM module labels; v1 fails before construction.

- [ ] **Step 1: Change dataclass literals and exact validation**

```python
decoder: Literal["neuralcdm-monotonic"] = "neuralcdm-monotonic"
version: int = 2
```

Map modules to `m1-m4-neuralcdm`, `m1-m2-m4-neuralcdm`, and
`m1-m2-m3-m4-neuralcdm`. Error messages state exact accepted decoder/version.

- [ ] **Step 2: Update only canonical new-manifest fixtures**

Use `rg '"decoder": "monotonic"|"version": 1|m1-m4' tests`. Update valid
fixtures; preserve explicit old-manifest rejection fixtures. Wrong-version and
wrong-decoder tamper cases become version 1 and decoder `monotonic`.

- [ ] **Step 3: Run metadata/loader/campaign/controller tests**

```bash
/home/xph/anaconda3/bin/conda run --no-capture-output -n decoupled_cd \
  python -m unittest tests.test_unified_v2_spec \
  tests.test_unified_v2_training tests.test_unified_validation_controller \
  tests.test_unified_validation tests.test_remote_campaign -v
```

Expected: all pass; valid fingerprints are new and v1 metadata is rejected
before model construction.

---

### Task 4: Add literature diagnostic and verify implementation

**Files:**
- Create: `docs/neuralcdm_m4_diagnostic_20260711.md`

**Interfaces:**
- Consumes: approved design and implementation.
- Produces: primary-source diagnostic note and one implementation commit, with no experimental claim/ledger.

- [ ] **Step 1: Write diagnostic note**

Include Task 8 diagnosis/cohort SHA; AAAI article URL and DOI
`10.1609/aaai.v34i04.6080`; secondary NeuralCD link; exact dense-beta,
positive-discrimination, Q-mask, 512/256 sole-path adaptation; exclusions; and
the registered falsification rule. State explicitly that CPU structural tests
are not evidence of validation improvement.

- [ ] **Step 2: Run focused CPU tests**

```bash
/home/xph/anaconda3/bin/conda run --no-capture-output -n decoupled_cd \
  python -m unittest tests.test_unified_v2_components \
  tests.test_unified_v2_spec tests.test_unified_v2_training \
  tests.test_unified_validation tests.test_unified_validation_controller -v
```

Expected: `OK`, CPU only.

- [ ] **Step 3: Run full CPU discovery and static checks**

```bash
/home/xph/anaconda3/bin/conda run --no-capture-output -n decoupled_cd \
  python -m unittest discover -s tests -v
/home/xph/anaconda3/bin/conda run -n decoupled_cd \
  python -m compileall configs data models trainers utils scripts tests
git diff --check
```

Expected: all exit zero. Do not run synthetic GPU smoke or real validation.

- [ ] **Step 4: Audit the diff**

Require one probability path, no legacy M4 names in decoder/model, dense beta,
fixed 512/256 positive layers, new canonical manifest, no unrelated/generated
assets.

- [ ] **Step 5: Commit implementation/tests/note**

```bash
git add models/unified_v2_components.py models/unified_decoupled_cdm.py \
  models/unified_v2_spec.py tests docs/neuralcdm_m4_diagnostic_20260711.md
git commit -m "feat: replace M4 with NeuralCDM interaction"
```

Record design, plan, implementation SHAs and exact counts in
`/home/jameschiang/work/decoupled_cd_codex/.superpowers/sdd/task-9-report.md`.
Do not push.
