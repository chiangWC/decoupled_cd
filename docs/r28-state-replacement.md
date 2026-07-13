# r28 State Replacement Campaign

## Research boundary

r28 always contains one candidate State Completion component followed by a fixed
Q-conditioned pooled NCF diagnosis. The candidate is replaced as one block; it
is not stacked with earlier candidates. ORCDF/SVGCD remain external baselines,
and no external CD architecture or author implementation is copied into r28.

Every candidate emits the unique student-specific `framework_state` consumed by
diagnosis, a mastery projection, reliability, and module diagnostics. There is
no free student-ID embedding, legacy state path, hybrid residual, or second
prediction head.

## Validation protocol

- Seed is fixed to 42; other seeds are rejected by the trainer.
- Selection uses validation rows only. Test confirmation requires a frozen
  checkpoint and a separate explicit evaluation stage.
- The first screen is MOOCRadar and XES3G5M, standard plus `*_chold_v2`.
- Full is compared with both `direct_prior` and a parameter-matched capacity
  control; the stronger control is used independently for S/H/T.
- Non-winning datasets remain in the live registry but do not veto module
  attribution on the architecture's winning datasets.
- Paired bootstrap resamples students, not seeds.

## Registered iterations

### f01 Evidence-Relational State Completion — rejected

Formula-level mapping from graph convolutional matrix completion; no author
code was used. It produced significant T gains over both controls, but zero
external wins. MOO Full margins were `−0.022983/−0.018541/−0.003378`; XES Full
margins were `−0.041503/−0.038951/−0.016468` for S/H/T.

### f02 Exposure-aware completion — not activated

The preregistered MNAR audit gave richer-vs-coverage mask-AUROC gains of only
`+0.005169` on MOO and `+0.000421` on XES, below the `0.02/0.03` activation
threshold. Exposure modeling was therefore not implemented.

### f03 Partial VAE State Completion — rejected

The implementation follows the paper's variable-size observation set,
permutation-invariant aggregation, amortized Gaussian student posterior and
concept-conditioned complete-state decoder. No author code was used. It has
225,501 parameters on XES versus 226,078 for its deterministic capacity control.

Clean T gains and student-clustered 95% CIs were:

| Dataset | Delta T | 95% CI | Full S/H/T external margins |
|---|---:|---:|---:|
| MOOCRadar | +0.006737 | [+0.002568, +0.010747] | +0.000301 / +0.000156 / −0.001294 |
| XES3G5M | +0.018981 | [+0.014135, +0.024225] | −0.002062 / −0.001083 / +0.000372 |

The component effect is real, but neither dataset satisfies the exact ordinary
win rule, so it remains a rejected performance anchor rather than a paper module.

### f04 Difficulty-Calibrated Response-Set Completion — rejected

The failure hypothesis was that concept-aggregated evidence discards item
difficulty and success/failure structure needed for overall AUC and exact-zero
transfer. The replacement consumes train-only exercise-response pairs, global
exercise evidence and Q-derived features. It encodes item-difficulty-calibrated
success/failure tokens as a permutation-invariant set and decodes a complete
concept state. Its capacity control uses identical blocks after collapsing the
exercise-response pairs to student marginals.

The mechanism was informed by Deep-IRT's separation of ability and item
difficulty and PFA's distinct success/failure evidence, but the state-completion
architecture is implemented independently and does not use either source model.

The item-conditioned mechanism was worse than its identical-parameter marginal
control on every axis: MOO S/H/T deltas were
`−0.025939/−0.024798/−0.023703`; XES deltas were
`−0.036185/−0.031768/−0.029318`. It is rejected without repair.

The mechanism-free marginal control itself produced three strict validation
wins: ASSIST17 `0.801341/0.799410/0.793590`, MOO
`0.930663/0.926869/0.935790`, and XES `0.792147/0.787284/0.785301`. It is
retained only as a stronger performance anchor; it is not relabeled as a
contribution.

### f05 Product-of-Experts Ability Completion — rejected

VIBO reports that an item-conditioned Gaussian product-of-experts posterior is
more robust to missing responses than averaging observed factors. r28 maps each
train-only exercise response and Q-derived item representation to a Gaussian
ability expert; missing items contribute only the unit prior. The posterior is
decoded into the complete concept state. The first structural screen uses only
response BCE, so a variational auxiliary objective cannot manufacture the
module effect. No VIBO source code is used.

PoE retained strict validation wins on both screen datasets, but it did not
improve the exact-parameter marginal anchor. Its MOO S/H/T deltas were
`+0.000395/+0.000037/+0.000146`, and its XES deltas were
`−0.000003/−0.000468/−0.000846`. Both paired target CIs crossed zero, so the
mechanism is rejected unchanged.

### f06 Hierarchical Bayesian State Completion — rejected

The new failure question is why probabilistic item experts and item-conditioned
sets fail to improve a strong student-marginal anchor. The replacement treats
global student ability and population concept difficulty as empirical-Bayes
priors, updates each student-concept state with its train-only local evidence,
and decodes the resulting continuous posterior and uncertainty into the full
state. The design is informed by ReliCD's concept-level uncertainty and the
Bayesian partial-mastery CD literature; it is independently implemented and
does not adopt either source model.

Its capacity control has the exact same layers, initialization and parameter
count, but follows the already successful marginal-summary/static-concept-prior
data flow. Both variants use response BCE only.

The full posterior replacement degraded every axis. MOO S/H/T deltas were
`−0.017493/−0.015621/−0.018723`; XES deltas were
`−0.024722/−0.019271/−0.020133`. Both paired target CIs were strictly negative,
so the candidate is rejected without changing prior strength or architecture.

### f07 Cohort-Conditioned State Completion — rejected

The next failure question is whether an individual posterior is too weak for a
concept with no personal observations. The new module softly locates each
student in fixed train-history ability levels, estimates a non-parametric
concept response profile for every level from peer evidence, and uses that
profile as the missing-concept prior. Local evidence updates observed concepts.

This is a new state-completion mapping informed by HCD's use of student ability
hierarchies; no HCD layer or source code is copied. The capacity control uses
the exact same trainable blocks and parameter count but collapses all ability
levels to a single population concept profile. Both use response BCE only.

The cohort path also degraded all axes. MOO S/H/T deltas were
`−0.017145/−0.016023/−0.018768`; XES deltas were
`−0.016036/−0.016192/−0.017600`, with both target CIs strictly negative. It is
rejected without altering the level count or bandwidth.

### f08 Bipolar Evidence-Prototype Completion — rejected

The repeated failure is now localized to concept-posterior injection rather
than lack of individual or peer evidence. This replacement leaves missing
concept priors graph-free and instead reconstructs the student's global state
from two fixed sufficient-statistic prototypes: exercises answered correctly
and exercises answered incorrectly. It then decodes that unique student state
and the Q-derived concept prior into the complete framework state.

The response-polarity separation follows the original README_spec semantics
and PFA's distinction between successes and failures, but no Claude propagation
code or external model implementation is reused. The capacity control has the
same blocks and parameters and duplicates the response-agnostic exercise
centroid into both channels.

The component retained strict wins on both screen datasets and improved MOO S
by `+0.001705`, but it did not improve the declared target metric. MOO/XES T
deltas were `−0.000210/−0.000476`, and both CIs crossed zero. It is retained as
a performance result but rejected as a paper module.

## Commands

```bash
conda activate decoupled_cd

python -m unittest discover -s tests -p 'test_r28*.py' -v

python scripts/run_r28_campaign.py \
  --dataset assist_09 --dataset nips34 \
  --state-completer difficulty_capacity \
  --completion-objective none \
  --data-root "$KNOFIELD_DATA_ROOT" \
  --gpus 0,1,3 \
  --output-root results/r28/anchor_marginal_pool_expansion
```

The scheduler treats occupied GPUs as eligible when the registered peak still
fits. Observed full-validation peaks are reserved explicitly. An OOM fails the
task and never changes batch size or model configuration silently.

## Current boundary after f08

The marginal performance anchor has strict validation wins on ASSIST17, MOO and
XES. Pool expansion did not add a fourth win: ASSIST09 is behind the external
S/H/T lines by `0.012498/0.013389/0.010135`; NIPS wins T by `0.001493` but misses
S/H by `0.007259/0.004241`.

There are two active framework boxes (one replaceable completion slot and the
fixed diagnosis), but zero qualified paper modules. The latest literature
review leaves no eligible unimplemented mechanism under the current input and
topology rules: item-response ranking is a training objective, knowledge-sensed
CD relies on the prohibited student-ID/low-rank product, and the remaining
semantic cold-start methods require common content/course modalities not
present in the five-dataset harness or introduce prohibited multi-path gates.
Continuing therefore requires new semantic inputs or explicit authority to
reopen one excluded complete relation/query mechanism; the three-win anchor is
not relabeled as a contribution.

## Promotion rules

The candidate first needs a T-AUC gain of at least 0.002 on one screen dataset,
with no S/H/T regression beyond 0.001 on the other, and both ordinary wins.
Final qualification requires T gains of at least 0.002 on two winning datasets,
at least 0.003 on one, a positive clustered-bootstrap CI lower bound, three
ordinary external wins, and at least two strict wins.
