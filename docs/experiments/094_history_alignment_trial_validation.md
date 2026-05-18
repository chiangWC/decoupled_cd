# Experiment 94: History Alignment Trial Validation

## Status

`trial_candidate_confirmed`; superseded by experiment 95's cleaner `cogonly` runner.

## Verdict

The hot experiment 93 configuration proves headroom but is too brittle: seed2026 collapses at `alignment=0.08810`. Lowering alignment to `0.05` with the full target-concept-doubled evidence target gives a four-seed matched positive result and becomes the first stable trial candidate. Experiment 95 later refines this candidate by removing direct student/exercise history terms.

## Question

Does the experiment 93 high-water `loss_only` cognitive alignment signal survive matched multi-seed validation?

## Base And References

Base: `pseudo_mainline_exp81_matched_seeds_2024_2025_2026_2027`

Branch: `exp/evidence-prior-calibrated-readout`

Parent detail: [093_history_evidence_cognitive_alignment.md](./093_history_evidence_cognitive_alignment.md)

Follow-up refinement: [095_history_alignment_cf_risk_ablation.md](./095_history_alignment_cf_risk_ablation.md)

Matched baselines:

- seed2024: `results/pure_cdm_hybrid_signal/seed2024_exp81_baseline_for_alignment_trial_300ep.json`
- seed2025: `results/pure_cdm_hybrid_signal/seed2025_exp81_baseline_for_alignment_trial_300ep.json`
- seed2026: `results/pure_cdm_hybrid_signal/seed2026_exp81_baseline_for_alignment_trial_300ep.json`
- seed2027: `results/expert_output_modulation/assist_09_seed2027_exp81_baseline_300ep.json`

## Hot Config Rejection

The first over-threshold configuration, `target-concept doubled` with alignment `0.08810`, does not pass multi-seed admission because seed2026 collapses:

| seed | baseline auc | candidate auc | auc delta | acc delta | rmse delta | brier delta | ece delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 0.767478 | 0.777905 | +0.010427 | +0.004015 | -0.004757 | -0.004027 | -0.002236 |
| 2025 | 0.771661 | 0.777100 | +0.005439 | +0.004168 | -0.003354 | -0.002837 | -0.007843 |
| 2026 | 0.771263 | 0.504198 | -0.267064 | -0.209994 | +0.090136 | +0.084680 | +0.114341 |
| 2027 | 0.772682 | 0.776868 | +0.004187 | +0.004643 | -0.003158 | -0.002670 | -0.009771 |

## Stable Full Target

Lower-strength fixed configuration:

- runner at the time: `scripts/run_assist09_history_alignment_trial.sh`
- output pattern: `results/pure_cdm_hybrid_signal/seed{seed}_history_alignment_tc2w005_300ep.json`
- `history_evidence_logit_prior_location = loss_only`
- student: `0.22`
- exercise: `0.22`
- target concept: `0.44`
- global concept: `0.22`
- mastery: `0.22`
- alignment weight: `0.05`

| seed | baseline auc | candidate auc | auc delta | acc delta | rmse delta | brier delta | ece delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 0.767478 | 0.777399 | +0.009921 | +0.002265 | -0.004001 | -0.003389 | -0.001980 |
| 2025 | 0.771661 | 0.776005 | +0.004344 | +0.002778 | -0.002116 | -0.001792 | -0.001835 |
| 2026 | 0.771263 | 0.773702 | +0.002439 | +0.002379 | -0.001364 | -0.001156 | +0.001247 |
| 2027 | 0.772682 | 0.775291 | +0.002609 | +0.003939 | -0.001748 | -0.001480 | -0.002681 |
| mean | 0.770771 | 0.775599 | +0.004828 | +0.002840 | -0.002307 | -0.001955 | -0.001312 |

## Interpretation

The lower-strength full configuration does not hit the corrected high-water stop line on seed2027 alone, but it clears `+0.004` on the four-seed matched mean and removes the seed2026 collapse seen in the hot config.

This establishes `loss_only` history-evidence cognitive alignment as a viable trial family. However, the full target still includes direct student/exercise history terms, so experiment 95 should be used as the current trial reference.
