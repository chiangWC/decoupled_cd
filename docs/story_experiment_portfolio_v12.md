# Story Experiment Portfolio v12

## Selection rule

Small experiments are retained only when they answer a named narrative
question with a fixed analysis. They do not replace the main S/H/T comparison
or clean module ablations. Validation predictions are used throughout; no new
test result is selected. Bootstrap and hiding-mask repetitions are perturbation
analyses, not model multi-seed training.

## 1. Natural coverage prevalence

**Question.** Does student-level incomplete target-concept coverage occur in
ordinary standard splits, rather than only in the constructed concept-holdout
protocol?

**Method.** For every standard validation interaction, construct the student's
train-seen concept set and classify the target Q concepts as zero, partial, or
full coverage. Separately count target student-concept pairs that occur only
once in the complete split pool.

**Result.** Incomplete coverage rates are 3.46% on ASSIST09, 7.54% on ASSIST17,
62.02% on MOOCRadar, 1.60% on NIPS34, and 43.82% on XES3G5M. Junyi is 100%
because its one-exercise-one-concept mapping combines with a student-exercise
group split; it must not be used as natural-prevalence evidence.

**Recommended presentation.** A stacked bar chart of zero/partial/full rates.
Show ASSIST09, ASSIST17, MOOCRadar, NIPS34, and XES3G5M; put Junyi in a hatched
protocol-note bar or omit it from the prevalence claim. A small marker can show
the singleton lower bound.

**What it supports.** Incomplete student-local concept coverage exists without
the explicit holdout construction and is not removed merely by k-fold
averaging. Prevalence is dataset-dependent.

**Do not claim.** It is not global unseen-concept cold start, and these split
rates do not estimate prevalence on every real platform.

## 2. History-composition difficulty audit

**Question.** Is raw history accuracy ambiguous because students attempt
exercise sets with different difficulty composition?

**Method.** Estimate train-only item ease while removing the current student's
responses. Regress item-normalized future validation outcomes on raw history
accuracy, log history length, concept coverage, and mean attempted-item ease.
Use a student bootstrap for the partial ease coefficient.

**Result.** The ease coefficient is negative with a 95% interval below zero on
all four winning datasets:

| Dataset | Partial ease coefficient | Incremental R2 |
|---|---:|---:|
| ASSIST17 | -0.0290 [-0.0379, -0.0207] | +0.0180 |
| MOOCRadar | -0.0766 [-0.0820, -0.0711] | +0.0952 |
| XES3G5M | -0.0255 [-0.0316, -0.0199] | +0.0220 |
| Junyi | -0.0582 [-0.0670, -0.0494] | +0.0468 |

**Recommended presentation.** A four-row forest plot for the coefficient,
with incremental R2 printed as a right-hand annotation. A one-sentence example
can explain that the same raw accuracy on easier histories predicts weaker
future performance.

**What it supports.** Exercise composition contributes information beyond raw
correctness, history length, and concept coverage.

**Do not claim.** Prior CD models ignore item difficulty. The narrower issue is
calibrating a history-derived student representation.

## 3. Calibration signal-to-prediction response

**Question.** Does the current History module actually use the signed
calibration signal in the expected direction?

**Method.** On fixed exact-zero validation rows, aggregate Full-minus-RawSummary
prediction shifts per student and compare them with the student's signed
history calibration residual.

**Result.** Spearman correlations are +0.175 on ASSIST17 and +0.300 on Junyi.
The highest-minus-lowest residual quartile prediction shifts are +0.0218
[+0.0105, +0.0325] and +0.0522 [+0.0483, +0.0560], respectively.

**Recommended presentation.** A two-panel quartile line or point-range plot:
calibration-residual quartile on the x-axis and mean Full-minus-control
prediction shift on the y-axis.

**What it supports.** The module output changes in the direction prescribed by
the proposed mechanism; the difficulty feature is not dead metadata.

**Do not claim.** Absolute-error benefit does not significantly increase with
absolute mismatch severity, so this is a mechanism-direction result rather
than a monotonic treatment-effect result.

## 4. History Hiding Stress Curve

**Question.** Does the model remain useful as the available student history is
progressively reduced, and do the trained History-module controls close the
gap?

**Method.** At validation inference only, hide nested 20%, 40%, 60%, and 80%
subsets of each student's train history. The same masks are used for every
model. Exercise-population evidence remains fixed, isolating student-side
history loss. Three mask seeds give perturbation variability. Target rows and
the original zero/full slice membership remain fixed. No checkpoint is
retrained.

**Overall AUC.** Columns give the percentage of history retained.

| Dataset/model | 100% | 80% | 60% | 40% | 20% |
|---|---:|---:|---:|---:|---:|
| ASSIST17 Full | 0.7996 | 0.7989 | 0.7981 | 0.7960 | 0.7892 |
| ASSIST17 CalibratedSummary | 0.7833 | 0.7823 | 0.7811 | 0.7787 | 0.7722 |
| ASSIST17 RawSummary | 0.7827 | 0.7816 | 0.7802 | 0.7776 | 0.7706 |
| Junyi Full | 0.8286 | 0.8269 | 0.8244 | 0.8199 | 0.8089 |
| Junyi CalibratedSummary | 0.8269 | 0.8247 | 0.8220 | 0.8173 | 0.8038 |
| Junyi RawSummary | 0.8170 | 0.8156 | 0.8134 | 0.8095 | 0.7999 |
| MOOCRadar Full | 0.9260 | 0.9251 | 0.9240 | 0.9222 | 0.9150 |
| XES3G5M Full | 0.7867 | 0.7850 | 0.7827 | 0.7768 | 0.7613 |

At 20% retained history, Full remains above both trained controls on ASSIST17
and Junyi. ASSIST17 Full drops 0.0104 versus 0.0112/0.0121 for the two controls.
Junyi Full drops 0.0198 versus 0.0230 for CalibratedSummary but 0.0171 for the
already weaker RawSummary. Thus Full is always best, but it is not universally
the least sensitive in normalized degradation.

For Full, the 80%-hiding overall drops are 0.0104 (ASSIST17), 0.0111
(MOOCRadar), 0.0254 (XES3G5M), and 0.0198 (Junyi). Fixed original zero/full
slices degrade similarly on ASSIST17 and MOO. On XES, they drop 0.0240 and
0.0294 respectively. No slice exhibits an isolated collapse.

**Recommended presentation.** Use the generated line plots with history
retained on the x-axis and validation AUC on the y-axis. A17 and Junyi should be
the main two panels because they contain trained controls; MOO and XES Full-only
curves belong in an appendix or robustness panel. Error bars show mask-seed
standard deviation.

**What it supports.** Full retains its absolute advantage as observation budget
shrinks. On A17, the gap over controls slightly widens. On Junyi, the Full gap
over RawSummary narrows but remains substantial.

**Do not claim.** This is not new-student evaluation, training-time robustness,
or an external-baseline comparison under hidden history. MOO/XES have no
trained History controls in this experiment, so their curves cannot attribute
robustness to the module.

## 5. Exact-Q requirement aliasing

**Question.** Do exercises with the same exact Q signature nevertheless have
different behavioral requirements?

**Method.** Within each exact-Q group containing at least two items with at
least 20 training responses, compare a Q-group frequency prior with a
train-only item-aware prior on validation rows. Bootstrap exact-Q groups.

**Result.** Median within-Q item-rate ranges are 0.259 on ASSIST17, 0.154 on
MOOCRadar, and 0.219 on XES3G5M. Item-aware over Q-only AUC gains are +0.0981,
+0.0254, and +0.0879; group-bootstrap Brier intervals are positive on all
three. Junyi has no repeated Q signatures and is not applicable.

**Recommended presentation.** A violin or box plot of within-Q item-rate ranges
plus a three-row forest plot of item-aware Brier advantage. The forest plot is
the cleaner main-paper visual; the distribution can go to the appendix.

**What it supports.** Q alone aliases behaviorally different exercises, giving
an empirical reason for an exercise-aware requirement representation.

**Do not claim.** Frequency priors are not the proposed neural module and do
not prove that every difference is a distinct cognitive skill requirement.

## 6. Requirement heterogeneity-to-gain link

**Question.** Is the Exercise-Aware Requirement Query especially useful where
same-Q aliasing is severe?

**Method.** Stratify exact-zero rows by train-only within-Q excess variance and
compare Full with its capacity control under fixed prediction rows.

**Result.** On XES, Full-minus-control AUC changes from -0.0014 in the lowest
heterogeneity quartile to +0.0226 in the highest. The high-minus-low Brier
advantage is +0.00861 with CI [+0.00184, +0.01575]. On ASSIST17, Full helps in
both strata, but the high-minus-low contrast crosses zero.

**Recommended presentation.** A two-dataset grouped point plot showing low and
high heterogeneity Full-minus-control AUC or Brier advantage. Emphasize XES as
the positive mechanism case and retain A17 as a non-concentrated effect.

**What it supports.** XES provides the cleanest chain from problem severity to
module benefit. A17 supports general benefit, not severity concentration.

## 7. Matched target-advantage case study

**Question.** Is our advantage on zero-coverage rows merely caused by easier
items or stronger students entering that slice?

**Method.** Compare our external-model loss advantage between exact-zero and
full-coverage rows after exact item matching or exact student matching. Use
cluster bootstrap over the matching unit.

**Result.** ASSIST17 has positive item-matched Brier DID +0.00394
[+0.00015, +0.00791] and student-matched DID +0.01554
[+0.00681, +0.02381]. MOO and XES intervals cross zero.

**Recommended presentation.** Keep the existing item/student forest plot as an
ASSIST17 case study. Do not present it as a three-dataset general result.

**What it supports.** The A17 target advantage is not explained solely by item
or student composition under the exact matching definitions.

## Suggested paper layout

### Main problem-evidence figure

1. Standard-split zero/partial/full coverage stacked bars.
2. History difficulty coefficient forest plot.
3. Exact-Q aliasing Brier forest plot.

Together these establish the setting and the two representation ambiguities
before introducing the model.

### Main mechanism figure

1. ASSIST17 and Junyi History Hiding curves with Full and two controls.
2. Calibration residual quartile versus prediction shift.
3. XES requirement gain in low/high exact-Q heterogeneity groups.

Together these show that the modules consume the intended information and that
their gains are not only aggregate table effects.

### Appendix

- MOO/XES Full-only History Hiding curves.
- Full coverage-prevalence table including ASSIST09/NIPS34 and the Junyi
  protocol caveat.
- ASSIST17 matched-item/student target case study.
- Exact-Q item-rate distributions and all negative/non-concentrated mechanism
  interactions.

## Naming implication

The History module contains both attempted-exercise set context and difficulty
calibration. The stress controls suggest the dominant source differs by
dataset: ASSIST17 benefits strongly from the attempted-exercise representation,
whereas Junyi shows a large CalibratedSummary-over-RawSummary gap. A broader
name such as **Exercise-Contextualized History Encoder** or
**Context-Calibrated History Encoder** is therefore safer than making
"difficulty" the entire module identity.

The Requirement module can retain **Exercise-Aware Requirement Query**. Its
same-Q problem evidence is strong on three datasets and its severity-gain link
is positive on XES.

## Artifacts

- Story mechanism audit: `docs/story_mechanism_audit_v11.md`.
- Story mechanism raw results: `results/goal_two_module/story_mechanisms_v11/`.
- History Hiding raw results and SVG curves:
  `results/goal_two_module/history_hiding_v12/`.
- History Hiding entry point: `scripts/evaluate_history_hiding_stress.py`.
