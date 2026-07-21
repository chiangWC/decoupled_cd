# RCPK standard test confirmation result

## Decision

The frozen standard-only RCPK model passes test confirmation. The same
dimension-64 architecture fingerprint (`f0151739cd1cbfaa`) beats the frozen
external standard-test line on ASSIST09, NIPS34, XES3G5M, and Junyi.

RCPK is now a qualified paper module for standard, relation-aware cognitive
diagnosis rather than merely a performance candidate. On ASSIST09, NIPS34, and
Junyi, Full both beats the external model and beats the validation-selected
complete control. XES3G5M is an additional architecture win, but its small
negative Full-control delta is reported and is not used as module evidence.

The rejected concept-holdout and TKC/UKC-completion claims remain rejected.
No H/T result is repurposed to support this standard-only conclusion.

## Frozen test comparison

| Dataset | Full AUC | Frozen control AUC | Module delta | External test AUC | External margin | Result |
|---|---:|---:|---:|---:|---:|---|
| ASSIST09 | **0.795227** | 0.759061 | +0.036166 | 0.778200 | +0.017027 | external + module win |
| NIPS34 | **0.790072** | 0.780317 | +0.009755 | 0.789300 | +0.000772 | external + module win |
| XES3G5M | 0.793637 | **0.793911** | -0.000273 | 0.792500 | +0.001137 | external/base win only |
| Junyi | **0.836877** | 0.833324 | +0.003553 | 0.824587 | +0.012290 | external + module win |

ASSIST09 and NIPS34 use Q-only controls. XES3G5M and Junyi use rewires 0 and
2, respectively, because those controls were selected by validation AUC before
test evaluation.

## Full-model secondary metrics

| Dataset | ACC | RMSE | Brier | ECE |
|---|---:|---:|---:|---:|
| ASSIST09 | 0.749396 | 0.415243 | 0.172427 | 0.059258 |
| NIPS34 | 0.719222 | 0.429708 | 0.184649 | 0.006760 |
| XES3G5M | 0.837984 | 0.346002 | 0.119717 | 0.010547 |
| Junyi | 0.776225 | 0.392861 | 0.154340 | 0.011957 |

## Paired test uncertainty

Student-clustered paired bootstrap uses 2,000 replicates and seed 2024. It
quantifies row-prediction uncertainty for the frozen seed-42 checkpoints and is
not multi-seed training.

| Dataset | Module delta | 95% CI | P(delta > 0) |
|---|---:|---:|---:|
| ASSIST09 | +0.036166 | [+0.032313, +0.039899] | 1.000 |
| NIPS34 | +0.009755 | [+0.008713, +0.010844] | 1.000 |
| Junyi | +0.003553 | [+0.002702, +0.004362] | 1.000 |

The pre-existing module gate is met: at least two externally won datasets have
module gains above `0.005`, one exceeds `0.01`, and all three reported paired
confidence intervals have positive lower bounds. XES3G5M's regression is only
`0.000273`, below the `0.001` safety tolerance.

Junyi also permits a paired Full-versus-external test comparison because its
ORCDF row predictions are retained. The external delta is `+0.012290`, with
student-clustered 95% CI `[+0.010964, +0.013741]` and
`P(delta > 0) = 1.000`.

## Integrity

- All checkpoints and controls were selected from validation results before
  standard test rows were opened.
- Test confirmation loaded existing checkpoints; it did not retrain or change
  a recipe.
- Full and control predictions have identical rows and train-only histories.
- Junyi retains row-level external test predictions. The legacy ASSIST09,
  NIPS34, and XES3G5M registry entries retain same-split external test AUCs but
  not their test prediction files, so their external margins are metric-level
  comparisons and do not receive paired external confidence intervals.
- Row-level predictions, metric reports, bootstrap reports, and SHA-256 hashes
  remain under `results/rcpk_standard_test/` on xph.
- Model seed is 42; no multi-seed experiment was run.

## Current framework interpretation

The framework has several functional boxes, but one claimed contribution:

1. ordinary train-history/state construction;
2. **Response-Conditioned Path Kernel**, the qualified contribution module;
3. ordinary Q/item-conditioned Diagnosis.

The upstream History Set mechanism was tested against three matched controls
and rejected as a second contribution. It remains implementation plumbing and
must not receive a contribution claim. Further structure should be added only
for a new, evidence-backed responsibility; a second module is not required to
make the current ablation complete.
