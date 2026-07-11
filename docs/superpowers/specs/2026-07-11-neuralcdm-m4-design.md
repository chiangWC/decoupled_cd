# NeuralCDM Whole-Module M4 Replacement Design

Date: 2026-07-11

Base commit: `6a6544e6ae56f34270deea6ce5ab6a34a77e9b72`

## Scope and registered hypothesis

Task 9 replaces the complete Unified V2 M4 interaction module. Task 8 found
useful diagnostic movement but weak absolute AUC, and registered the hypothesis
that the current Q-weighted linear IRT readout cannot express item-by-concept
difficulty or nonlinear required-concept interactions. This implementation
tests that hypothesis without changing M1, M2, M3, data, loss recipes, or adding
another prediction path.

This phase is implementation-only. It does not initialize a controller, use a
GPU, read real validation/test data, or run a real experiment.

## Decoder architecture

`MonotonicDiagnosisDecoder` keeps its public constructor, `forward`, and
`decode_from_mastery` signatures, but all legacy M4 parameters and arithmetic
are removed.

The replacement owns:

- `item_concept_difficulty: Embedding(E, K)`, transformed by sigmoid to
  `beta[e,k]` in `[0,1]`;
- `exercise_discrimination: Embedding(E, 1)`, transformed by softplus plus a
  small positive floor;
- three monotonic linear layers with dimensions `K -> 512 -> 256 -> 1`.

For each target student/item pair it computes exactly:

```text
x = Q_e * (mastery_s - beta_e) * discrimination_e
h1 = sigmoid(positive_linear_1(x))
h2 = sigmoid(positive_linear_2(h1))
p  = sigmoid(positive_linear_3(h2))
```

Each effective linear weight is `softplus(raw_weight)` and therefore strictly
positive throughout optimization. Biases are unconstrained. Q masking occurs
before the interaction network, so an unrequired mastery coordinate is exactly
zero at the network input and cannot affect the result.

There is no exercise bias branch, global concept difficulty, weighted linear
IRT sum, guessing/slip transform, residual, adapter, legacy path, or second
head. Dense `E x K` difficulty is used for every dataset; no dataset-dependent
representation or architecture switch is permitted.

## Stable initialization

Naively initializing raw weights near zero would produce effective softplus
weights near `0.693`, saturating the 512/256 layers. Initialization is defined
in effective-weight space and mapped back with inverse softplus:

- layer 1 effective weights target `0.5` with a small raw-space perturbation;
  sparse Q masking keeps its summed input order-one while perturbations avoid
  identical hidden units;
- layer 2 effective weights target `1/512`, with bias
  `-(1/512) * 512 * 0.5 = -0.5`;
- layer 3 effective weights target `1/256`, with bias
  `-(1/256) * 256 * 0.5 = -0.5`;
- item-concept difficulty logits start at zero (`beta=0.5`);
- discrimination is initialized so softplus plus its floor is approximately
  `1`;
- the first-layer bias starts at zero.

A CPU regression uses representative sparse Q vectors and requires probabilities
strictly away from 0/1 plus finite, nonzero gradients for required mastery,
required beta entries, discrimination, and all three positive layers.

## Mastery and public outputs

The existing mastery head still produces one nonempty `[students, concepts]`
tensor. The sole prediction path indexes that tensor and passes the selected
rows to `decode_from_mastery`; the full same tensor is returned as
`DecoupledForwardOutput.mastery` for DOA. `cognitive_probs` and `probs` are the
same probability tensor. Guess/slip outputs remain zeros only to preserve the
existing public result schema; they do not participate in prediction.

The public scalar `difficulty` output becomes the Q-normalized average of the
selected item's sigmoid-constrained item-concept difficulty. All other model
inputs and outputs remain unchanged.

## Manifest and checkpoint boundary

`UnifiedArchitectureSpec` changes its only accepted decoder to
`neuralcdm-monotonic` and its only accepted version to integer `2`. Canonical
module labels become `m1-m4-neuralcdm`, `m1-m2-m4-neuralcdm`, and
`m1-m2-m3-m4-neuralcdm`. These fields create new architecture fingerprints.

Version-1/`monotonic` manifests fail before model construction. Old M4 state
dictionaries fail strict loading because the legacy concept-difficulty,
exercise-bias, and linear-readout keys are absent from the new module while the
new dense beta and interaction-layer keys are missing from the checkpoint.
Partial loading is not introduced.

## TDD contract

Production changes follow observed RED tests for:

1. coordinate-wise random and boundary monotonicity on every required Q entry;
2. required-coordinate autograd Jacobians at least `-1e-8`;
3. exact invariance to non-Q mastery changes;
4. different predictions for items sharing Q but having different beta rows;
5. nonzero mixed finite difference in logits for a constructed multi-concept
   item, which the old weighted-linear IRT logit cannot express;
6. one nonempty mastery tensor feeding the only prediction path and DOA output;
7. strict rejection of old manifests and old decoder checkpoints;
8. stable, nonsaturated initialization with finite nonzero gradients through
   every required component;
9. absence of legacy M4 parameters or source-level prediction branches.

Focused CPU tests run before full `unittest` discovery. The implementation and
literature diagnostic note are committed together, separately from this design
and from any future experimental ledger.

## Literature diagnostic note

The implementation note will cite the primary AAAI 2020 NeuralCD paper and its
official article URL, distinguish the paper's monotonic neural interaction
mechanism from this repository's exact adaptation, and record the falsifiable
M4 hypothesis. It will not claim validation improvement before the separately
authorized experiment phase.
