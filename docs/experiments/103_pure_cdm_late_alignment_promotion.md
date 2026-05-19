# Experiment 103: Pure CDM late alignment promotion

## Status

`single_checkpoint_candidate_refined`, not a `0.778` default promotion.

## Verdict

Late-window annealing of the experiment 95 cog-only cognitive alignment loss gives the best current single-checkpoint pure CDM runner signal, without fixed checkpoint averaging or a validation-trained combiner.

Best configuration:

```bash
bash scripts/run_assist09_history_alignment_trial.sh \
  --history-evidence-cognitive-alignment-final-weight 0.0881 \
  --history-evidence-cognitive-alignment-anneal-start-epoch 170 \
  --history-evidence-cognitive-alignment-anneal-end-epoch 230
```

Four-seed AUC is `0.778242/0.776913/0.775059/0.776280`, mean `0.776623`,
which is `+0.000345` over experiment 95's four-seed mean `0.776279`.
This is a real single-run growth signal, but it is still below the user's
practical `0.778` target and below experiment 102's fixed checkpoint-average
mean `0.778872`. Keep it as the current single-checkpoint pure CDM candidate,
not as a completed default promotion.

## Base

- Branch: `exp/pure-cdm-default-promotion`
- Base runner: experiment 95 cog-only `loss_only` history evidence cognitive alignment
- Base four-seed AUC: `0.777843/0.776440/0.774669/0.776163`, mean `0.776279`
- Rejected acceptance path: experiment 102 fixed probability average remains an evaluator result, not a single-checkpoint default-training reproduction path.

## Results

| variant | seeds | AUCs | mean | verdict |
|---|---|---:|---:|---|
| `late170to230`, `0.05 -> 0.0881` | 2024/2025/2026/2027 | `0.778242/0.776913/0.775059/0.776280` | `0.776623` | best single-checkpoint candidate |
| `late165to225`, `0.05 -> 0.0881` | 2024/2025/2026/2027 | `0.778192/0.776855/0.775052/0.776293` | `0.776598` | stable but lower mean |
| `late170to220`, `0.05 -> 0.0881` | 2024/2025/2026/2027 | `0.778211/0.776833/0.775064/0.776315` | `0.776606` | better seed2027/secondary metrics, lower mean |
| `late175to235`, `0.05 -> 0.0881` | 2024/2027 | `0.778255/0.776233` | n/a | no stronger anchor; worse ECE |
| `late170to240`, `0.05 -> 0.0881` | 2024/2027 | `0.778245/0.776118` | n/a | hurts seed2027 |

Endpoint checks on `late170to230`:

| final weight | seeds | AUCs | verdict |
|---:|---|---:|---|
| `0.086` | 2024/2027 | `0.778229/0.776278` | slightly lower AUC, cleaner secondary metrics |
| `0.090` | 2024/2027 | `0.778242/0.776281` | essentially same AUC, worse calibration |

## Rejected Follow-Ups

- Capacity line: `dim72` and `dim80` late-window variants can lift seed2027 near or above `0.778`, but they damage seed2024, so they are not default candidates.
- Output alignment with late-window capacity keeps the same seed tradeoff and does not fix the weak-seed tail.
- Rank alignment on top of `late170to230` improves secondary metrics but reduces AUC; `rank_weight=0.002` even makes seed2027 negative versus experiment 95.
- Cognitive readout head averaging (`head_count=3`) strongly hurts both checked seeds.
- Direct cognitive-location history prior collapses or sharply degrades AUC.
- Concept evidence calibrated readout is only a tiny seed2024 gain and hurts seed2027.
- Target-heavy prior mix `0.52/0.18/0.18` hurts both anchors.
- Global-heavy prior mix `0.36/0.26/0.26` improves seed2027 but collapses seed2026 to `0.504274`; milder `0.40/0.24/0.24` still collapses seed2026.
- SWA and fixed checkpoint averaging are not accepted as the route for this single-run/default-training question.

## Decision

Use `late170to230` as the next single-checkpoint pure CDM baseline for this task.
It is a valid growth signal because every seed improves over experiment 95, but
it does not reproduce experiment 102's `0.778+` fixed-average performance and
should not be described as having solved default promotion.
