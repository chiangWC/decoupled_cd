# Experiment 96: Smooth Cognitive Alignment Loss

## Status

`rejected_single_seed`, not promoted.

## Verdict

Changing the experiment 95 `cogonly loss_only` alignment objective from standardized MSE to smoother bounded losses does not improve the seed2027 high-water check. Both `standardized_smooth_l1` and `correlation` stay around `test_auc ~= 0.7747`, below experiment 95's seed2027 `0.776163` and below the corrected stop threshold `0.776682`.

Do not expand these loss-shape variants to multi-seed unless a later mechanism-level change gives a new reason. The useful next step is not another local loss replacement at the same weight; move toward a materially different target construction, reliability weighting, or representation-level mechanism.

## Question

Can a smoother or bounded alignment loss preserve experiment 95's stable `cogonly` signal while widening the safe training window?

## Base And References

Base: `pseudo_mainline_exp81_high_water_seed2027`

Branch: `exp/smooth-cognitive-alignment`

Parent detail:

- experiment 93: first `loss_only` cognitive alignment signal
- experiment 94: multi-seed validation and hot-weight rejection
- experiment 95: current `cogonly` trial candidate

Corrected high-water reference:

- output: `results/expert_output_modulation/assist_09_seed2027_exp81_baseline_300ep.json`
- `test_auc = 0.7726816125195809`
- `test_acc = 0.7328207958286551`
- `test_rmse = 0.4243497503536669`
- `test_brier = 0.18007271062521943`
- `test_ece = 0.050954005791600435`

Experiment 95 seed2027 reference:

- output: `results/pure_cdm_hybrid_signal/seed2027_history_alignment_cogonly_w005_300ep.json`
- `test_auc = 0.776163`
- matched AUC delta vs high-water: `+0.003481`

## Implementation

Added a default-off alignment loss selector:

- `--history-evidence-cognitive-alignment-loss standardized_mse`
- `--history-evidence-cognitive-alignment-loss standardized_smooth_l1`
- `--history-evidence-cognitive-alignment-loss correlation`

The default remains `standardized_mse`, so existing experiment 95 runner semantics are unchanged unless the new flag is passed.

Smoke and focused verification:

- `python -m py_compile trainers/engine.py scripts/train.py`
- `python -m unittest tests.test_training_modes`
- `bash scripts/run_assist09_history_alignment_smooth_trial.sh --epochs 1 --max-rows 2000 --output results/pure_cdm_hybrid_signal/smoke_history_alignment_smoothl1_seed2024.json`

## Seed2027 Results

All candidates use the experiment 95 `cogonly` target:

- student: `0.0`
- exercise: `0.0`
- target concept: `0.44`
- global concept: `0.22`
- mastery: `0.22`
- alignment weight: `0.05`
- `history_evidence_logit_prior_location = loss_only`

| loss mode | output | test_auc | auc delta vs high-water | acc delta | rmse delta | brier delta | ece delta | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `standardized_smooth_l1` | `results/pure_cdm_hybrid_signal/seed2027_history_alignment_cogonly_smoothl1_w005_300ep.json` | 0.774677 | +0.001995 | +0.001313 | -0.001230 | -0.001042 | -0.004048 | below exp95; do not expand |
| `correlation` | `results/pure_cdm_hybrid_signal/seed2027_history_alignment_cogonly_corr_w005_300ep.json` | 0.774699 | +0.002018 | +0.001713 | -0.000794 | -0.000673 | -0.000163 | below exp95; do not expand |

## Interpretation

The smoother losses do train and keep secondary metrics mostly non-destructive versus the exp81 high-water baseline, but they give up too much AUC relative to the current experiment 95 MSE objective. This suggests the current bottleneck is not just MSE outlier sensitivity at `alignment=0.05`.

The experiment narrows the next route: avoid more same-target loss-shape sweeps and prefer a different evidence target, reliability-aware sample weighting, or a representation-level way to consume the history signal.
