# Story Mechanism Audit v11

## Scope

This is a fixed-seed, train-only diagnostic audit. It does not tune the model or
read new test predictions. The analyses use standard validation rows for
mechanism prevalence and existing holdout validation predictions for module
links. All 95% intervals use 2,000 student- or exact-Q-cluster bootstrap
replicates; this is not multi-seed training.

## Natural student-level concept coverage

Coverage is computed against each student's standard-split training history.
`zero` means none of the concepts required by the target exercise occur in that
student's training history; `partial` means only a strict subset occurs.

| Dataset | Standard valid zero | partial | incomplete | any student-concept singleton |
|---|---:|---:|---:|---:|
| ASSIST09 | 2.86% | 0.60% | 3.46% | 2.60% |
| ASSIST17 | 2.54% | 5.00% | 7.54% | 5.34% |
| MOOCRadar | 25.91% | 36.10% | 62.02% | 57.54% |
| NIPS34 | 0.00% | 1.60% | 1.60% | 1.23% |
| XES3G5M | 43.82% | 0.00% | 43.82% | 38.67% |
| Junyi | 100.00% | 0.00% | 100.00% | 100.00% |

All validation students occur in train, while student-exercise pairs do not.
Target exercises and concepts are globally known at rates of at least 99.0% on
the six datasets. The audit therefore measures student-local missing coverage,
not global unseen concepts. Junyi is a protocol edge case: exercise-concept
mapping is one-to-one and the group split keeps each student-exercise pair in a
single split, so its 100% rate must not be cited as natural platform prevalence.

The singleton column is a lower-bound diagnostic for why k-fold evaluation does
not eliminate the issue: when a student-concept pair occurs only once in the
complete split pool, the fold containing that interaction cannot have another
training observation for the same pair.

## History difficulty calibration

Item ease is estimated from training responses with the current student's
responses removed, then beta-smoothed. At student level, future validation
outcomes are normalized by train-only target-item ease. The full regression
controls raw history accuracy, log history length, and concept coverage; the
reported coefficient is the standardized partial effect of mean history-item
ease. A negative coefficient means that among otherwise comparable histories,
the same raw accuracy on easier attempted items predicts worse future outcomes.

| Dataset | Eligible students | ease coefficient (95% CI) | incremental R2 (95% CI) |
|---|---:|---:|---:|
| ASSIST17 | 1,659 | -0.0290 [-0.0379, -0.0207] | +0.0180 [+0.0092, +0.0302] |
| MOOCRadar | 2,000 | -0.0766 [-0.0820, -0.0711] | +0.0952 [+0.0739, +0.1170] |
| XES3G5M | 2,000 | -0.0255 [-0.0316, -0.0199] | +0.0220 [+0.0134, +0.0334] |
| Junyi | 3,270 | -0.0582 [-0.0670, -0.0494] | +0.0468 [+0.0333, +0.0617] |

This consistently supports the existence of history-composition confounding.
It does not establish that cognitive diagnosis models generally ignore item
difficulty; it establishes a narrower need for calibration when constructing a
student representation from response history.

### Link to the current History module

Compared with the raw-history control, the Full model changes predictions in the
direction implied by the signed calibration residual:

| Dataset | Spearman(residual, Full-control shift) | top-minus-bottom residual prediction shift (95% CI) |
|---|---:|---:|
| ASSIST17 | +0.1751 | +0.0218 [+0.0105, +0.0325] |
| Junyi | +0.2998 | +0.0522 [+0.0483, +0.0560] |

However, the Full model's Brier gain is not significantly larger in the highest
absolute-mismatch quartile than in the lowest: ASSIST17 +0.00138
[-0.00560, +0.00822], Junyi +0.00254 [-0.00015, +0.00524]. Thus the audit
supports mechanism direction, but not a claim that module benefit monotonically
increases with mismatch severity. The existing clean T ablations remain the
primary module-effect evidence.

## Exact-Q requirement aliasing

Items are grouped by their exact Q signature. Eligible items have at least 20
training responses and eligible signatures contain at least two such items.
Item-aware and Q-only train-frequency priors are then compared on standard
validation rows. The bootstrap gives each exact-Q group equal weight.

| Dataset | Q groups / items | median within-Q rate range | item-aware AUC gain | Brier advantage (95% CI) |
|---|---:|---:|---:|---:|
| ASSIST17 | 225 / 1,885 | 0.259 | +0.0981 | +0.0142 [+0.0116, +0.0170] |
| MOOCRadar | 120 / 339 | 0.154 | +0.0254 | +0.0148 [+0.0102, +0.0196] |
| XES3G5M | 154 / 812 | 0.219 | +0.0879 | +0.0070 [+0.0050, +0.0092] |
| Junyi | N/A | N/A | N/A | N/A |

The share of groups with a smoothed within-Q item-rate range of at least 0.1 is
82.7%, 58.3%, and 82.5% respectively. Exact Q signatures therefore alias
behaviorally different items on three datasets. Junyi is not applicable because
its exercise-concept mapping is one-to-one.

### Link to the current Requirement Query

For XES3G5M, Full versus the capacity control changes from an AUC delta of
-0.0014 in the lowest-heterogeneity quartile to +0.0226 in the highest. Its
high-minus-low Brier advantage is +0.00861 with CI [+0.00184, +0.01575]. This is
the cleanest mechanism-severity link in the audit.

ASSIST17 has positive Full-versus-control AUC gains in both low (+0.0174) and
high (+0.0117) heterogeneity strata, but the high-minus-low Brier contrast is
not significant: +0.00119 [-0.02037, +0.02332]. Therefore exact-Q aliasing is
present and the module helps overall, but its gain is not concentrated in the
most heterogeneous A17 groups.

## Narrative decision

1. Student-level incomplete concept coverage is empirically present in standard
   splits, but prevalence is dataset- and protocol-dependent. Do not use Junyi
   to claim natural prevalence.
2. Difficulty-aware history calibration has consistent problem-existence
   evidence across all four winning datasets and direction-consistent module
   behavior on ASSIST17 and Junyi. Its severity-gain link remains weak.
3. Exercise-aware requirement modeling has strong problem-existence evidence on
   ASSIST17, MOOCRadar, and XES3G5M, plus a complete severity-gain link on XES.
4. The strongest current paper framing is two representation ambiguities under
   incomplete student history, not explicit TKC/UKC structural decoupling and
   not a claim that prior work ignores item difficulty.

Raw outputs are under `results/goal_two_module/story_mechanisms_v11/`; the
reproducible entry point is `scripts/analyze_story_mechanisms.py`.
