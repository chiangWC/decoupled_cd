# Experiment 95: History Alignment CF-Risk Ablation

## Status

`trial_candidate_refined`, not promoted to default baseline script.

## Verdict

The stable history-alignment gain is not mainly a student/item CF shortcut. Removing direct student and exercise history terms improves the four-seed mean over the full experiment 94 runner, while the student/exercise-only target is weaker and worsens calibration.

The named trial runner on `exp/trellis-trial` is now the cleaner `cogonly loss_only` configuration:

- runner: `scripts/run_assist09_history_alignment_trial.sh`
- default baseline script unchanged: `scripts/run_assist09_baseline.sh`
- `history_evidence_logit_prior_location = loss_only`
- student direct history weight: `0.0`
- exercise direct history weight: `0.0`
- target-concept weight: `0.44`
- global-concept weight: `0.22`
- mastery weight: `0.22`
- alignment weight: `0.05`

## Question

Does the stable experiment 94 signal come mainly from CF-style direct student/exercise history terms, or from concept/mastery-level cognitive evidence?

## Base And References

Base: `pseudo_mainline_exp81_matched_seeds_2024_2025_2026_2027`

Branch: `exp/trellis-trial`

Parent detail:

- experiment 93: single-seed high-water `loss_only` cognitive alignment signal
- experiment 94: first four-seed lower-strength trial validation with full target weights
- detail doc: [093_history_evidence_cognitive_alignment.md](./093_history_evidence_cognitive_alignment.md)

Matched baselines:

- seed2024: `results/pure_cdm_hybrid_signal/seed2024_exp81_baseline_for_alignment_trial_300ep.json`
- seed2025: `results/pure_cdm_hybrid_signal/seed2025_exp81_baseline_for_alignment_trial_300ep.json`
- seed2026: `results/pure_cdm_hybrid_signal/seed2026_exp81_baseline_for_alignment_trial_300ep.json`
- seed2027: `results/expert_output_modulation/assist_09_seed2027_exp81_baseline_300ep.json`

## Compared Targets

All variants use:

- `--history-evidence-logit-prior-residual`
- `--history-evidence-logit-prior-location loss_only`
- `--history-evidence-logit-prior-min-count 1`
- `--history-evidence-logit-prior-min-seen-ratio 0.0`
- `--history-evidence-logit-prior-max-logit 4.0`
- `--history-evidence-logit-prior-component-cap 4.0`
- `--history-evidence-cognitive-alignment-weight 0.05`

`full` from experiment 94:

- output pattern: `results/pure_cdm_hybrid_signal/seed{seed}_history_alignment_tc2w005_300ep.json`
- student: `0.22`
- exercise: `0.22`
- target concept: `0.44`
- global concept: `0.22`
- mastery: `0.22`
- mean AUC delta: `+0.004828`
- mean ECE delta: `-0.001312`

`cogonly` removes the strongest CF-flavored terms:

- output pattern: `results/pure_cdm_hybrid_signal/seed{seed}_history_alignment_cogonly_w005_300ep.json`
- student: `0.0`
- exercise: `0.0`
- target concept: `0.44`
- global concept: `0.22`
- mastery: `0.22`

`cfonly` keeps only the CF-flavored student/exercise terms:

- output pattern: `results/pure_cdm_hybrid_signal/seed{seed}_history_alignment_cfonly_w005_300ep.json`
- student: `0.22`
- exercise: `0.22`
- target concept: `0.0`
- global concept: `0.0`
- mastery: `0.0`

## Cogonly Results

| seed | baseline auc | cogonly auc | auc delta | acc delta | rmse delta | brier delta | ece delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 0.767478 | 0.777843 | +0.010365 | +0.001009 | -0.003563 | -0.003020 | -0.001201 |
| 2025 | 0.771661 | 0.776440 | +0.004779 | +0.003083 | -0.002497 | -0.002114 | -0.005367 |
| 2026 | 0.771263 | 0.774669 | +0.003407 | +0.002360 | -0.002254 | -0.001909 | -0.003431 |
| 2027 | 0.772682 | 0.776163 | +0.003481 | +0.005081 | -0.002448 | -0.002071 | -0.006565 |
| mean | 0.770771 | 0.776279 | +0.005508 | +0.002883 | -0.002690 | -0.002279 | -0.004141 |

## CF-Only Results

| seed | baseline auc | cfonly auc | auc delta | acc delta | rmse delta | brier delta | ece delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 0.767478 | 0.775300 | +0.007822 | -0.001808 | -0.001443 | -0.001226 | +0.006390 |
| 2025 | 0.771661 | 0.774789 | +0.003128 | +0.003273 | -0.001521 | -0.001289 | +0.001953 |
| 2026 | 0.771263 | 0.772312 | +0.001049 | +0.001998 | -0.000465 | -0.000395 | +0.004550 |
| 2027 | 0.772682 | 0.772440 | -0.000241 | +0.001656 | -0.000007 | -0.000006 | +0.002710 |
| mean | 0.770771 | 0.773710 | +0.002939 | +0.001280 | -0.000859 | -0.000729 | +0.003901 |

## Interpretation

The ablation rejects the strongest shortcut concern:

- `cogonly` is stronger than `full`: mean AUC delta `+0.005508` vs `+0.004828`.
- `cogonly` is cleaner on calibration: mean ECE delta `-0.004141` vs full `-0.001312`.
- `cfonly` is weaker and has a negative AUC tail on seed2027.
- `cfonly` worsens calibration: mean ECE delta `+0.003901`.

The useful signal is better described as a training-time concept/mastery cognitive alignment objective, not an inference-time output prior or direct student/item collaborative-filtering shortcut.

Do not add student/exercise direct history terms back to the named trial runner unless a later experiment gives a new mechanism-level reason and passes matched multi-seed calibration checks.
