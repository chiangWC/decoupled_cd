# TKC/UKC Recovery Scope

The active research line starts from Claude-stage commit `4b7d75b` and the
problem definition in `README_spec.md`.

## Retained assets

- Reproducible Junyi and EdNet pool preparation.
- Dataset fingerprints, row identities, Q-matrix and leakage audits.
- Row-aligned ORCDF/SVGCD external predictions and the external S/H/T registry.
- Generic metric and external-win helpers.

## Excluded model history

The r26, r28 and r29 model searches are archived negative experiments. Their
completion models, marginal predictor, proxy mechanisms, recipes and campaign
runners are not part of this branch and must not be imported into a new model.

There is no internal performance opponent. Model selection uses only the
row-aligned external S/H/T opponents in
`configs/external_benchmark_registry.json`. Internal `w/o Module` and
capacity-matched controls exist only to attribute a paper module.

## Model boundary

A new model must consume explicit TKC and UKC masks. TKC state comes from
train-history response evidence. A candidate UKC module uniquely produces UKC
state from permitted train-only inputs. The final state is assembled explicitly
before a shared standard diagnosis function. Global student summaries, legacy
states, residual prediction heads and dataset-specific routing may not bypass
the UKC path.

TKC/UKC is the preferred entry hypothesis because the repository already has
explicit masks, protocols and failure evidence for it. It is not a mandatory
final narrative: a model discovered through another data flow may be retained
when it satisfies the external-win and clean-ablation contract in
`docs/research_goal.md`. The final paper problem and contribution may then be
formulated from the validated failure mechanism and measured gains rather than
imposed in advance.

## Active validated exception (2026-07-21)

RCPK is the first model to exercise the non-TKC/UKC exception above. Its
concept-holdout route was rejected, while its declared standard-representation
responsibility passed frozen validation, clean controls, and standard-test
confirmation. The current active paper route is therefore relation-aware
standard cognitive diagnosis, with RCPK as the sole claimed contribution
module. See
`docs/experiments/2026-07-21-rcpk-standard-test-confirmation-result.md`.

Do not restart r26/r28/r29 completion searches or relabel RCPK as UKC
completion. Do not force a second contribution module: the matched History Set
audit failed and is retained only as a negative experiment. New structural
work requires a separate, evidence-backed responsibility and preregistered
control; otherwise proceed with paper validation, external comparisons, and
analysis of the qualified RCPK mechanism.

## Target-conditioning boundary (2026-07-23)

The clean student-global control establishes that target-conditioned relation
aggregation is functionally important on ASSIST09 and NIPS34 under identical
inputs, capacity and initialization. It does not validate the proposed
low-relevance-share dilution explanation; that natural diagnostic failed. New
sessions may use the clean mechanism result as attribution evidence, but must
not restore relational dilution as the problem premise or relabel it as
TKC/UKC completion. See
`docs/experiments/2026-07-23-rcpk-target-conditioning-gate-result.md`.
