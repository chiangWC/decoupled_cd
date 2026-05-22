# Experiment 104: Single-checkpoint dual CDM ensemble

## Status

`single_checkpoint_0778_target_cleared`, current best pure-CDM default-training candidate.

## Verdict

A single training run with a two-tower `DecoupledCDMEnsemble` plus branch-level BCE clears the user's practical `0.778` AUC target across four seeds without fixed checkpoint averaging, validation-trained combiners, or hybrid train-history features.

Best configuration:

After runner cleanup, this configuration is encoded by default in
`scripts/run_assist09_history_alignment_trial.sh`. The expanded command below
is kept as the historical exact configuration record.

```bash
bash scripts/run_assist09_history_alignment_trial.sh \
  --dual-cdm-ensemble \
  --dual-cdm-secondary-concept-dim 80 \
  --dual-cdm-branch-bce-weight 0.10 \
  --history-evidence-cognitive-alignment-final-weight 0.0881 \
  --history-evidence-cognitive-alignment-anneal-start-epoch 170 \
  --history-evidence-cognitive-alignment-anneal-end-epoch 230 \
  --concept-evidence-prior-residual \
  --concept-evidence-prior-min-count 1 \
  --concept-evidence-prior-min-seen-ratio 1.0 \
  --concept-evidence-prior-max-logit 0.30 \
  --concept-evidence-prior-min-confidence 0.75 \
  --concept-evidence-prior-min-abs-mastery 0.5 \
  --concept-evidence-prior-apply-mode train_only \
  --concept-evidence-prior-train-start-epoch 135
```

Four-seed AUC is `0.778773/0.778250/0.778508/0.777948`, mean `0.778370`, population stdev `0.000306`, and mean ECE `0.035412`.

This is the first documented pure-CDM single-checkpoint route in this branch that reaches the `0.778` mean target. It remains below experiment 102's fixed checkpoint-average mean `0.778872`, but it satisfies the user's default-training constraint better because it is one training run and one checkpoint artifact.

## Base

- Branch: `exp/pure-cdm-default-promotion`
- Runner: `scripts/run_assist09_history_alignment_trial.sh`
- Base single-checkpoint candidate: late-window cognitive alignment plus train-only concept evidence prior refinement
- Base four-seed AUC before dual tower: `0.778379/0.776930/0.775065/0.776569`, mean `0.776736`
- Rejected comparison path: experiment 102 fixed probability average remains stronger in raw mean but is a two-checkpoint evaluator, not the accepted default-training answer.

## Model And Training Change

- Model structure: add `DecoupledCDMEnsemble`, a single-checkpoint two-tower wrapper over `DecoupledCDM`.
- Primary tower: `concept_dim=64`.
- Secondary tower: `concept_dim=80`.
- Inference output: fixed equal-weight average of branch `probs`, `cognitive_probs`, `guess_probs`, and `slip_probs`.
- Training objective: add branch BCE auxiliary loss through `dual_cdm_branch_bce_weight`, so both primary and secondary towers are directly supervised before probability mixing.

The key growth signal is not capacity alone. Equal-weight dual tower without branch BCE mainly improved variance, while branch BCE made both towers individually useful enough for the averaged output to clear the target.

## Results

| seed | AUC | best val AUC | best epoch | ECE | result file |
|---:|---:|---:|---:|---:|---|
| 2024 | 0.778773 | 0.784256 | 167 | 0.035794 | `results/pure_cdm_default_promotion/seed2024_dual64x80_branchbce010_late170to230_eprior_trainonly_conf075_abs05_max030_start135_300ep.json` |
| 2025 | 0.778250 | 0.783253 | 157 | 0.038834 | `results/pure_cdm_default_promotion/seed2025_dual64x80_branchbce010_late170to230_eprior_trainonly_conf075_abs05_max030_start135_300ep.json` |
| 2026 | 0.778508 | 0.783970 | 155 | 0.035186 | `results/pure_cdm_default_promotion/seed2026_dual64x80_branchbce010_late170to230_eprior_trainonly_conf075_abs05_max030_start135_300ep.json` |
| 2027 | 0.777948 | 0.782306 | 146 | 0.031834 | `results/pure_cdm_default_promotion/seed2027_dual64x80_branchbce010_late170to230_eprior_trainonly_conf075_abs05_max030_start135_300ep.json` |

Aggregate:

| metric | value |
|---|---:|
| AUC mean | 0.7783698724 |
| AUC population stdev | 0.0003059041 |
| mean ECE | 0.0354122913 |
| delta vs pre-dual single-checkpoint base | +0.001634 |
| delta vs experiment 95 mean `0.776279` | +0.002091 |
| delta vs experiment 102 fixed-average mean `0.778872` | -0.000502 |

## Validation Path

- Local `py_compile` passed for the modified model, train, analysis, trainer, and focused test files.
- Remote focused unittest suite passed after the branch BCE implementation commit.
- Final four-seed metric aggregation was read from remote result JSON files.

## Rejected Or Weaker Follow-Ups

| variant | seeds | AUCs | verdict |
|---|---|---:|---|
| dual `64x80`, no branch BCE | 2024/2025/2026/2027 | `0.777419/0.776724/0.776681/0.776863` | stable but below target, mean `0.776922` |
| post-hoc secondary weight sweep on no-branch-BCE checkpoints | 2024/2025/2026/2027 | best at weight `0.5` | individual towers were weak when only the averaged output was supervised |
| train secondary weight `0.40` | 2024/2026 | `0.777063/0.776730` | no growth over equal weighting |
| checkpoint selection start epoch `170` | 2024/2026 | `0.777823/0.776619` | mixed; not enough |
| EMA start `170`, decay `0.99` | 2024/2026 | `0.777379/0.776614` | no growth |

## Decision

Promote this as the current best pure-CDM single-checkpoint candidate for the `0.778` sprint. Future work should compare any new pure-CDM idea against this four-seed mean, not against experiment 103 or the older experiment 95 cog-only baseline alone.
