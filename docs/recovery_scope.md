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
