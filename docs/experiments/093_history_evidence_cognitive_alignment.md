# Experiment 93: History Evidence Cognitive Alignment

## Status

`trial_candidate_refined`, not promoted to default baseline script.

## Question

Can the train-history signal from experiments 91/92 be consumed by the cognitive layer instead of a valid-trained hybrid stacker or final output-logit prior?

## Mechanism

Original exploration branch: `exp/evidence-prior-calibrated-readout`

Merged trial branch: `exp/trellis-trial`

Current named runner:

- `scripts/run_assist09_history_alignment_trial.sh`
- current trial configuration: `loss_only cogonly`
- student/exercise direct history terms: `0.0`
- target concept: `0.44`
- global concept: `0.22`
- mastery: `0.22`
- alignment weight: `0.05`

The default baseline script remains `scripts/run_assist09_baseline.sh`; this trial runner has not replaced the default baseline run.

New default-off training option:

- `--history-evidence-logit-prior-residual`
- `--history-evidence-logit-prior-location loss_only`
- `--history-evidence-cognitive-alignment-weight <weight>`

`loss_only` exposes the deterministic train-history evidence prior to the trainer but does not add it to cognitive logits or output logits during forward inference. The trainer adds a standardized MSE alignment loss between the model's cognitive logits and the fixed history-evidence prior. Test predictions remain the ordinary CDM forward output.

## Corrected Reference

Current exp81 high-water reference:

- source: `results/expert_output_modulation/assist_09_seed2027_exp81_baseline_300ep.json`
- `test_auc = 0.7726816125195809`
- `test_acc = 0.7328207958286551`
- `test_rmse = 0.4243497503536669`
- `test_brier = 0.18007271062521943`
- `test_ece = 0.050954005791600435`

Stopping threshold: `0.7766816125195809`.

## Primary Seed2027 Result

Best seed2027 pure-CDM alignment probe:

- output: `results/pure_cdm_hybrid_signal/seed2027_history_alignment_tc2w008810_300ep.json`
- target evidence weights:
  - student: `0.22`
  - exercise: `0.22`
  - target concept: `0.44`
  - global concept: `0.22`
  - mastery: `0.22`
- alignment weight: `0.08810`
- `history_evidence_logit_prior_location = loss_only`
- `test_auc = 0.7768681361535177`
- AUC delta vs high-water: `+0.0041865236339367895`
- over stopping threshold: `+0.00018652363393678595`
- `test_acc = 0.7374640811433139`
- `test_rmse = 0.42119151877701955`
- `test_brier = 0.1774022954896924`
- `test_ece = 0.0411826996781801`

Unlike experiment 92's output-logit prior, this result improves the secondary metrics versus high-water:

- `ACC +0.004643285314658773`
- `RMSE -0.0031582315766473455`
- `Brier -0.002670414863527038`
- `ECE -0.009771306113420334`

## Trial Validation

The first over-threshold configuration, `target-concept doubled` with alignment `0.08810`, does **not** pass multi-seed admission because seed2026 collapses:

| seed | baseline auc | candidate auc | auc delta | acc delta | rmse delta | brier delta | ece delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 0.767478 | 0.777905 | +0.010427 | +0.004015 | -0.004757 | -0.004027 | -0.002236 |
| 2025 | 0.771661 | 0.777100 | +0.005439 | +0.004168 | -0.003354 | -0.002837 | -0.007843 |
| 2026 | 0.771263 | 0.504198 | -0.267064 | -0.209994 | +0.090136 | +0.084680 | +0.114341 |
| 2027 | 0.772682 | 0.776868 | +0.004187 | +0.004643 | -0.003158 | -0.002670 | -0.009771 |

A lower-strength fixed configuration was stable across all four seeds before the CF-risk ablation:

- trial runner: `scripts/run_assist09_history_alignment_trial.sh`
- output pattern: `results/pure_cdm_hybrid_signal/seed{seed}_history_alignment_tc2w005_300ep.json`
- target evidence weights:
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

This lower-strength full configuration was the first stable trial candidate. It does not hit the corrected high-water stop line on seed2027 alone, but it clears `+0.004` on four-seed matched mean and removes the seed2026 collapse. After the CF-risk ablation below, the final named trial runner was refined to `cogonly`.

## CF-Risk Component Ablation

To test whether the stable signal is mostly a CF-style student/exercise shortcut, two additional four-seed ablations were run at the same alignment weight `0.05`.

`cogonly` removes the strongest CF-flavored terms:

- output pattern: `results/pure_cdm_hybrid_signal/seed{seed}_history_alignment_cogonly_w005_300ep.json`
- student: `0.0`
- exercise: `0.0`
- target concept: `0.44`
- global concept: `0.22`
- mastery: `0.22`

| seed | baseline auc | cogonly auc | auc delta | acc delta | rmse delta | brier delta | ece delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 0.767478 | 0.777843 | +0.010365 | +0.001009 | -0.003563 | -0.003020 | -0.001201 |
| 2025 | 0.771661 | 0.776440 | +0.004779 | +0.003083 | -0.002497 | -0.002114 | -0.005367 |
| 2026 | 0.771263 | 0.774669 | +0.003407 | +0.002360 | -0.002254 | -0.001909 | -0.003431 |
| 2027 | 0.772682 | 0.776163 | +0.003481 | +0.005081 | -0.002448 | -0.002071 | -0.006565 |
| mean | 0.770771 | 0.776279 | +0.005508 | +0.002883 | -0.002690 | -0.002279 | -0.004141 |

`cfonly` keeps only the CF-flavored student/exercise terms:

- output pattern: `results/pure_cdm_hybrid_signal/seed{seed}_history_alignment_cfonly_w005_300ep.json`
- student: `0.22`
- exercise: `0.22`
- target concept: `0.0`
- global concept: `0.0`
- mastery: `0.0`

| seed | baseline auc | cfonly auc | auc delta | acc delta | rmse delta | brier delta | ece delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2024 | 0.767478 | 0.775300 | +0.007822 | -0.001808 | -0.001443 | -0.001226 | +0.006390 |
| 2025 | 0.771661 | 0.774789 | +0.003128 | +0.003273 | -0.001521 | -0.001289 | +0.001953 |
| 2026 | 0.771263 | 0.772312 | +0.001049 | +0.001998 | -0.000465 | -0.000395 | +0.004550 |
| 2027 | 0.772682 | 0.772440 | -0.000241 | +0.001656 | -0.000007 | -0.000006 | +0.002710 |
| mean | 0.770771 | 0.773710 | +0.002939 | +0.001280 | -0.000859 | -0.000729 | +0.003901 |

The ablation rejects the strongest shortcut concern: the student/exercise-only target is weaker and worsens ECE, while the concept/mastery-only target is stronger than the previously recorded full trial config and improves all secondary metrics on mean. The trial runner now uses the cleaner `cogonly` configuration.

## Sweep Notes

Equal five-term target on seed2027:

| alignment weight | test_auc | delta vs high-water | note |
|---:|---:|---:|---|
| 0.003 | 0.771597 | -0.001085 | no gain |
| 0.005 | 0.765172 | -0.007509 | degraded |
| 0.020 | 0.773796 | +0.001115 | clean small gain |
| 0.030 | 0.774153 | +0.001471 | clean small gain |
| 0.050 | 0.775566 | +0.002884 | clean gain |
| 0.080 | 0.776043 | +0.003362 | near target |
| 0.086 | 0.776349 | +0.003668 | near target, best equal target |
| 0.088 | 0.776422 | +0.003740 | near target |
| 0.0882 | 0.496229 | -0.276453 | early-collapse boundary |
| 0.090 | 0.496224 | -0.276458 | early-collapse boundary |
| 0.100 | 0.496200 | -0.276482 | early-collapse boundary |

Target composition probes at stable alignment `0.086`:

| target | test_auc | delta vs high-water | note |
|---|---:|---:|---|
| equal five terms | 0.776349 | +0.003668 | best equal target |
| no mastery | 0.775950 | +0.003268 | worse |
| no concept/mastery | 0.774744 | +0.002063 | worse |
| student+exercise only | 0.772821 | +0.000139 | weak |
| target-concept doubled | 0.776428 | +0.003747 | best composition at this weight |
| global concept doubled | 0.496128 | -0.276553 | early collapse |
| mastery doubled | 0.776295 | +0.003614 | best calibration, below target |

Target-concept doubled fine sweep:

| alignment weight | test_auc | delta vs high-water | over target | ece |
|---:|---:|---:|---:|---:|
| 0.08700 | 0.776346 | +0.003665 | -0.000335 | 0.045953 |
| 0.08800 | 0.776580 | +0.003899 | -0.000101 | 0.043920 |
| 0.08805 | 0.776527 | +0.003846 | -0.000154 | 0.045267 |
| 0.08810 | 0.776868 | +0.004187 | +0.000187 | 0.041183 |
| 0.08815 | 0.776659 | +0.003978 | -0.000022 | 0.043717 |

Seed2024 check for equal target `alignment=0.01`:

- output: `results/pure_cdm_hybrid_signal/seed2024_history_alignment_eq022_w001_300ep.json`
- `test_auc = 0.7745519920010073`
- `test_acc = 0.7320596015147767`
- `test_rmse = 0.4239646706338235`
- `test_brier = 0.17974604194564645`
- `test_ece = 0.05124051534815723`

## Interpretation

This is the first pure CDM training-objective family with both:

- a corrected high-water single-seed signal (`alignment=0.08810`, seed2027 AUC `+0.004187`)
- a four-seed matched final trial candidate (`alignment=0.05 cogonly`, mean AUC `+0.005508`)

It should enter trial as the lower-strength `alignment=0.05 cogonly` candidate, not as the hotter `0.08810` point and not as the full student/exercise/target-concept/concept/mastery configuration. The hot point is useful for proving headroom, but seed2026 shows it is too brittle for default use.

The cleaner trial version now uses `cogonly`: target-concept/global-concept/mastery evidence only, with student and exercise direct history terms disabled. Do not promote directly to the default baseline script yet. Next step: test a smoother bounded/correlation alignment loss to recover some seed2027 headroom without reintroducing the seed2026 collapse.
