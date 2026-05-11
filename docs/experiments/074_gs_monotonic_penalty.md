# Experiment 74: `guess/slip` Monotonic Soft Penalty

## Summary

- Branch: `exp/gs-monotonic-penalty`
- Base: experiment 70 mainline
- Status: rejected
- Verdict: diagnostics are useful, but the soft penalty is not a mainline candidate.

## Change

No hard reparameterization was applied. Training adds an optional BCE-side regularizer:

```text
weight * mean(relu(guess + slip - 1)^2)
```

The branch also adds `scripts/analyze_guess_slip_diagnostics.py`, which reports `guess/slip` means, `guess+slip` quantiles, `ratio(guess+slip>1)`, and the ratio bucketed by `concept_count`, target concept coverage, and student history length.

## Results

`weight=1e-4`, relative to experiment 70 with the same seed:

| seed | AUC | ACC | RMSE | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | +0.000076 | +0.001846 | -0.000399 | -0.000342 | -0.000324 |
| 2025 | -0.000195 | -0.000895 | +0.000585 | +0.000500 | +0.002527 |
| 2026 | -0.000391 | +0.000742 | -0.000221 | -0.000189 | -0.001603 |
| mean | -0.000170 | +0.000565 | -0.000012 | -0.000010 | +0.000200 |

`weight=1e-3`, `seed=2024`, relative to experiment 70 with the same seed:

| AUC | ACC | RMSE | Brier | ECE |
| ---: | ---: | ---: | ---: | ---: |
| -0.000913 | -0.001237 | -0.000191 | -0.000164 | -0.004819 |

The `1e-3` setting improves calibration but hurts AUC/ACC, so it was not expanded.

## Diagnostics

`weight=1e-4`, best checkpoint:

| seed | guess.mean | slip.mean | ratio(guess+slip>1) | p99(guess+slip) |
| --- | ---: | ---: | ---: | ---: |
| 2024 | 0.166936 | 0.088017 | 0.000742 | 0.897360 |
| 2025 | 0.962968 | 0.991846 | 0.999772 | 1.999874 |
| 2026 | 0.926414 | 0.958467 | 0.999600 | 1.997649 |

The `seed=2024` checkpoint looked healthy, but `seed=2025/2026` collapsed into almost full semantic inversion. The `1e-4` regularizer is too weak to reliably constrain the independent sigmoid `guess/slip` branch.

## Decision

Do not merge this loss into the mainline. Keep the diagnostic script on the experiment branch as a useful inspection tool. Do not return directly to hard monotonic parameterization unless there is a new mechanism beyond the experiment 62-65 constrained `guess/slip` family, which already showed that hard constraints can remove semantic inversion without producing clean overall gains.

