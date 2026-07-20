# Selected Story Experiments v13

## Scope

This report reduces the previous list of seven analyses to three paper-facing
experiment families.  All results are validation-only.  The model seed remains
42.  The three history-mask seeds are perturbation repeats of a frozen
checkpoint, not multi-seed model training.

## What a trained History control means

The three paths are trained independently with the same split, data order,
optimizer recipe, seed, downstream State Completion and Requirement Query, and
the same common-parameter initialization hash.

| Path | Attempted-exercise semantic pool | Difficulty residual | Raw accuracy / confidence / coverage |
|---|:---:|:---:|:---:|
| Full (`calibrated_history`) | yes | yes | yes |
| CalibratedSummary | no | yes | yes |
| RawSummary | no | no | yes |

All paths use an encoder with the same input/output dimensions and active
parameter count.  Therefore:

- Full versus RawSummary is the whole History-module ablation.
- Full versus CalibratedSummary tests the attempted-exercise semantic pool
  while retaining calibrated aggregate statistics.
- CalibratedSummary versus RawSummary tests difficulty calibration in the
  summary-only path.

These are internal attribution controls, not external baselines.  They are
separately optimized rather than obtained by disabling an input in a trained
Full checkpoint.  The current three-path experiment is not a complete 2x2
factorial: it does not identify the difficulty-residual main effect while the
semantic pool is present.

## Completed MOO/XES control training

The newly trained controls use fingerprint `099906acdba8c3b4`, seed 42, 40
epochs, student batch 64, learning rate 0.001, context target fraction 0.2 and
the same dataset-specific common initialization hash as Full.

| Dataset | Full validation AUC | CalibratedSummary | RawSummary | Full - stronger control |
|---|---:|---:|---:|---:|
| MOOCRadar | 0.926027 | 0.925026 | 0.923839 | +0.001000 |
| XES3G5M | 0.786737 | 0.783690 | 0.782401 | +0.003047 |

On the fixed original exact-zero slice, the Full-minus-stronger-control gains
are +0.000646 on MOOCRadar and +0.003274 on XES3G5M.  XES is a strong positive
attribution result; MOO is positive but small.

## History Hiding Stress Curve with trained controls

At validation inference, nested masks hide 20%, 40%, 60% or 80% of each
student's train history.  The target rows, original zero/full membership and
exercise-population evidence remain fixed.  Full and both controls receive
identical masks; all 13 mask groups per dataset have matching hashes.

The table compares Full with the stronger trained control at the unmasked
endpoint and at 20% retained history.

| Dataset | Full at 100% | Control at 100% | Gap | Full at 20% | Control at 20% | Gap | Full/control drop | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ASSIST17 | 0.799584 | 0.783309 | +0.016274 | 0.789224 | 0.772154 | +0.017069 | 0.010360 / 0.011155 | main |
| MOOCRadar | 0.926027 | 0.925026 | +0.001000 | 0.914954 | 0.914910 | +0.000044 | 0.011073 / 0.010116 | appendix / negative boundary |
| XES3G5M | 0.786737 | 0.783690 | +0.003047 | 0.761307 | 0.756717 | +0.004590 | 0.025430 / 0.026973 | main |
| Junyi | 0.828647 | 0.826864 | +0.001783 | 0.808875 | 0.803840 | +0.005035 | 0.019773 / 0.023024 | supplementary |

Junyi does not need a post-hoc normalization to be usable: Full has the
highest absolute AUC at every observation budget.  The weaker RawSummary drops
less than Full (0.017119 versus 0.019773), so Junyi must not support a universal
"Full degrades slowest" claim.  This is consistent with a lower-performing,
less history-dependent control having less performance to lose.

XES is the cleanest stress result: Full is best at every budget, drops less
than both controls, and its gap over CalibratedSummary widens from +0.003047 to
+0.004590.  ASSIST17 shows the same qualitative pattern with a much larger
absolute module gap.  MOO remains best overall at every budget, but the gap to
CalibratedSummary contracts to +0.000044 and the exact-zero control is slightly
better under the most severe mask.  It should be reported as a boundary, not
used to advertise robustness.

## Three retained paper-facing experiment families

### 1. History-composition ambiguity and module response

**What was done.**  Train-only, leave-current-student-out item ease was used to
test whether attempted-exercise composition predicts future outcomes beyond
raw history accuracy, history length and concept coverage.  A second analysis
checked whether Full-minus-RawSummary predictions move with the signed
calibration residual.

**Presentation.**  A four-dataset forest plot of the partial item-ease
coefficient, with incremental R-squared as a right-side annotation.  Add a
small ASSIST17/Junyi quartile point-range inset for the prediction response.

**Claim supported.**  Equal raw history accuracy does not imply equal student
state when the attempted exercise sets differ, and the implemented History
module consumes the signed calibration information in the intended direction.

This is strong: all four winning datasets have ease-coefficient intervals
below zero, with incremental R-squared from +0.0180 to +0.0952.  The response
analysis is directional mechanism evidence, not proof that error reduction is
monotone in mismatch severity.

### 2. History Hiding with trained controls

**What was done.**  Progressively hide student-side training history from
frozen Full and independently trained capacity/interface controls under shared
nested masks.

**Presentation.**  Two main line-chart panels for ASSIST17 and XES3G5M, with
history retained on the x-axis and validation AUC on the y-axis.  Mask-seed
standard deviation is shown as a narrow error band.  Junyi and MOO go to the
appendix with their caveats.

**Claim supported.**  The History-module advantage is not confined to the
unperturbed validation history.  On A17 and XES it persists and widens as the
observation budget decreases.

This experiment does not establish new-student generalization, training-time
robustness or superiority to external models under artificial hiding.

### 3. Exact-Q requirement aliasing and benefit concentration

**What was done.**  Within repeated exact-Q groups, train-only item-aware and
Q-only priors were compared to test whether identical Q signatures alias
behaviorally different exercises.  Full-minus-capacity-control gains were then
stratified by train-only within-Q heterogeneity.

**Presentation.**  A three-dataset forest plot of item-aware Brier advantage,
plus a compact XES low/high-heterogeneity point plot.

**Claim supported.**  Q alone loses exercise-specific requirement information.
The Exercise-Aware Requirement Query is most useful in the clearest severe
aliasing case on XES.

This is strong problem evidence on ASSIST17, MOOCRadar and XES3G5M.  The
severity-to-module-gain link is clean only on XES; ASSIST17 supports general
module benefit but not concentration by this severity measure.

## Setup statistic and appendix-only analyses

Natural zero/partial/full concept-coverage prevalence remains an introductory
dataset statistic, not a fourth mechanism experiment.  It establishes that
student-local incomplete coverage also occurs in standard splits, while
disclosing strong dataset dependence and Junyi's one-item-one-concept protocol
artifact.

The matched-item/matched-student target-advantage experiment remains an
ASSIST17 case study because the MOO/XES intervals cross zero.  It should not be
used as a multi-dataset main claim.  Full MOO hiding curves, Junyi's RawSummary
slope caveat, exact-Q item-rate distributions and non-significant
severity-benefit interactions belong in the appendix.

## Artifacts

- New trained controls: `results/goal_two_module/history_controls_v13/`.
- New MOO/XES controlled stress curves:
  `results/goal_two_module/history_hiding_v13/`.
- Existing A17/Junyi controlled stress curves:
  `results/goal_two_module/history_hiding_v12/`.
- Mechanism audit: `results/goal_two_module/story_mechanisms_v11/`.
- Matched case study: `results/goal_two_module/matched_target_v10/`.
