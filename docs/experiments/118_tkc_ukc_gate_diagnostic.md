# Experiment 118: TKC/UKC gate diagnostic

## Status

`gate_diagnostic_mixed_mechanism_evidence`

## Verdict

This diagnostic gives useful but mixed mechanism evidence.

The paper-safe claim is:

> Higher target coverage is associated with modestly higher TKC gate weight and
> effective TKC contribution, especially on the student-concept holdout split.

The stronger proposed claim is not supported as written:

> Student-global train-history coverage is not positively correlated with
> `tkc_weight`; in these seed2027 checkpoints it is negatively correlated.

This distinction matters because the learned gate input is student-global
coverage plus `tkc_mean` and `ukc_mean`, while `target_coverage(s,e)` is an
interaction-level diagnostic. Use the target-coverage bucket chart as a
mechanism-associated plot; do not claim that the gate is a monotone function of
student-global coverage.

## Implementation

Added diagnostic access to the propagation return:

```text
PropagationOutput.tkc_weight
DecoupledForwardOutput.tkc_weight
```

Added evaluator:

```text
scripts/evaluate_gate_diagnostic.py
```

The evaluator is evaluation-only. It loads checkpoint summaries, rebuilds the
train-history evaluation bundle, computes per-student `tkc_weight`, maps it to
target interactions, and reports:

- student-global-coverage bins vs mean `tkc_weight`
- target-coverage buckets vs mean `tkc_weight`
- effective TKC contribution share:

```text
w * ||tkc_mean|| / (w * ||tkc_mean|| + (1 - w) * ||ukc_mean||)
```

For dual-tower checkpoints, the report includes `primary`, `secondary`, and
`mean` tower rows.

## Artifacts

Remote output directory:

```text
results/gate_diagnostic_exp118/
```

Reports:

- `assist09_ordered_seed2027_gate_report.json`
- `assist09_ordered_seed2027_gate_report_bins.csv`
- `assist09_ordered_seed2027_gate_report_summary.csv`
- `assist09_holdout_seed2027_gate_report.json`
- `assist09_holdout_seed2027_gate_report_bins.csv`
- `assist09_holdout_seed2027_gate_report_summary.csv`

Remote verification:

```bash
python -m py_compile scripts/evaluate_gate_diagnostic.py models/hetero_propagation.py models/decoupled_cdm.py models/ensemble_cdm.py
python -m unittest tests.test_gate_diagnostic tests.test_hetero_propagation tests.test_decoupled_cdm
```

Passed remotely: 31 tests OK.

## Overall Correlations

Rows use seed2027 ASSIST09 checkpoints. For Exp110 dual-tower rows, `tower=mean`
averages primary and secondary tower contribution metrics.

| dataset | model | tower | mean tkc weight | mean effective TKC share | corr student coverage -> tkc weight | corr target coverage -> tkc weight | corr student coverage -> effective share | corr target coverage -> effective share |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ordered | Exp81 baseline | single | 0.599693 | 0.498930 | -0.571745 | +0.019394 | -0.706543 | +0.007841 |
| ordered | Exp110 full | mean | 0.599045 | 0.495037 | -0.448485 | +0.025011 | -0.749537 | +0.006301 |
| holdout | Exp81 baseline | single | 0.645786 | 0.546903 | -0.043877 | +0.146391 | -0.577011 | +0.091885 |
| holdout | Exp110 full | mean | 0.649005 | 0.553433 | -0.137986 | +0.153907 | -0.602554 | +0.147307 |

Read:

- The direct student-global coverage axis does not support the intended
  monotonic-positive gate story.
- Target coverage has a weak positive association with `tkc_weight`; the
  association is clearer on the holdout split than on the default ordered split.
- Effective TKC share follows the same target-coverage direction on holdout.

## Target Coverage Buckets

Ordered split:

| model | tower | target bucket | count | mean tkc weight | mean effective TKC share |
|---|---|---|---:|---:|---:|
| Exp81 baseline | single | zero | 1658 | 0.590716 | 0.494254 |
| Exp81 baseline | single | low | 15 | 0.554446 | 0.444983 |
| Exp81 baseline | single | partial | 315 | 0.598415 | 0.500451 |
| Exp81 baseline | single | full | 50561 | 0.600009 | 0.499089 |
| Exp110 full | mean | zero | 1658 | 0.588616 | 0.491558 |
| Exp110 full | mean | low | 15 | 0.548600 | 0.440834 |
| Exp110 full | mean | partial | 315 | 0.593093 | 0.496226 |
| Exp110 full | mean | full | 50561 | 0.599439 | 0.495159 |

Holdout split:

| model | tower | target bucket | count | mean tkc weight | mean effective TKC share |
|---|---|---|---:|---:|---:|
| Exp81 baseline | single | zero | 21021 | 0.641919 | 0.542223 |
| Exp81 baseline | single | low | 446 | 0.640852 | 0.530529 |
| Exp81 baseline | single | partial | 3640 | 0.636422 | 0.533244 |
| Exp81 baseline | single | full | 28645 | 0.649891 | 0.552328 |
| Exp110 full | mean | zero | 21021 | 0.638992 | 0.541293 |
| Exp110 full | mean | low | 446 | 0.635759 | 0.535919 |
| Exp110 full | mean | partial | 3640 | 0.622634 | 0.514864 |
| Exp110 full | mean | full | 28645 | 0.659910 | 0.567516 |

The holdout target-coverage buckets give the most useful figure: Exp110 full
shows full-coverage samples with higher mean gate weight and higher effective
TKC contribution than zero/low coverage samples.

## Interpretation

- Use this as a **mechanism-associated diagnostic**, not a proof that the gate
  is monotone in the explicit student-global coverage input.
- The cleanest plot is target coverage bucket on the x-axis and either mean
  `tkc_weight` or mean effective TKC share on the y-axis, using the holdout
  split.
- If a stronger gate-behavior claim is needed, the model would need a gate
  regularizer or architecture that explicitly encourages monotonicity with
  coverage. That would be a model-change experiment, not this diagnostic.
