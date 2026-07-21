# Option-contrast signal gate result

Formal run: `option_contrast_signal_v1`, commit `ac4060b`, model seed `42`,
pseudo-split seed `2024`. The estimator used only holdout-train interaction and
option sidecars; validation/test option fields were not opened.

The preregistered gate required EdNet and one of NIPS34/ENEM to pass, with
Full-minus-stronger-control target AUC at least `0.005` on both, at least
`0.010` on one, no pseudo-overall AUC regression worse than `0.001`, and no
target Brier regression worse than `0.0002`.

| dataset | pseudo-T rows | stronger control | ΔT AUC | Δoverall AUC | ΔT Brier | result |
|---|---:|---|---:|---:|---:|---|
| NIPS34 | 30 | — | — | — | — | insufficient target support |
| EdNet | 13,508 | Shuffle | +0.003154 | +0.001208 | −0.001201 | gate fail |
| ENEM | 209,221 | Shuffle | +0.010376 | +0.010376 | −0.003108 | deterministic pass |

NIPS34 remains in the audit because its option data admission passed, but the
train-only pseudo-T low-coverage scope has only 30 rows and 24 students; it is
not used to produce an unstable delta. EdNet and ENEM complete the frozen
cross-fit estimator. The aggregate gate therefore has only one deterministic
passing dataset and does not run bootstrap.

Conclusion: categorical wrong-option identity is a real predictive signal on
ENEM and a positive but sub-threshold signal on EdNet. The preregistered joint
gate fails, so no neural Categorical-Response State Completion module is
implemented or promoted. The result is retained as a positive diagnostic and
as evidence for the next literature-driven replacement search; thresholds and
controls are not relaxed post hoc.
