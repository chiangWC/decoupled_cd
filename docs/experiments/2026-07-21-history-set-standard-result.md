# History Set Representation standard-only audit result

## Decision

The existing History Set Representation is rejected as a second paper module.
It does not beat the strongest capacity-matched control by `0.002` on any
dataset. No bootstrap or test confirmation is run because the deterministic
effect-size gate is impossible.

This result does not change the retained RCPK candidate or its three external
standard wins. The History box remains ordinary upstream model plumbing.

## Results

All values are standard-validation AUC. Full is the frozen
`calibrated_history` RCPK model. The strongest control is selected from
`identity_raw_control`, `calibrated_summary_control` and
`raw_summary_control`.

| Dataset | Full | Identity raw | Calibrated summary | Raw summary | Strongest control | Full - strongest |
|---|---:|---:|---:|---:|---:|---:|
| ASSIST09 | 0.795594 | **0.797070** | 0.796135 | 0.794980 | 0.797070 | -0.001477 |
| NIPS34 | **0.784829** | 0.774935 | 0.783776 | 0.777564 | 0.783776 | +0.001052 |
| XES3G5M | **0.789947** | 0.789637 | 0.788964 | 0.788174 | 0.789637 | +0.000310 |
| Junyi | **0.831026** | 0.829844 | 0.828965 | 0.822109 | 0.829844 | +0.001182 |

ASSIST09 directly violates the required positive direction. NIPS34, XES3G5M
and Junyi are positive but all remain below the predeclared `0.002` minimum.
Consequently the required two winning datasets and one `0.005` effect cannot
be reached.

## Interpretation

The raw-summary control is usually weaker, so removing both item identity and
difficulty calibration does hurt. The information-matched controls show why
that is insufficient for a module claim:

- ASSIST09 performs better when difficulty residualization is removed;
- NIPS34 retains nearly all Full performance with calibrated marginals and no
  attempted-item identity pool;
- XES3G5M and Junyi gains over identity-preserving controls are too small.

The weak raw control must not be used to manufacture a large ablation. RCPK's
separate direct access to response-conditioned path statistics also means the
global History state is not the only carrier of response information, making
the strong-control result the relevant one.

## Integrity

- Model seed 42; no multi-seed run.
- RCPK, real relation graphs, target requirement and Diagnosis were frozen.
- Full and controls share initialization hashes within every dataset.
- Every run used validation stage and has null test metrics.
- No hyperparameter, objective or control was changed after observing results.
