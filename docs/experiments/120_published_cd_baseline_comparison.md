# Experiment 120: Published CD Baseline Comparison

## Purpose

Record the paper-facing comparison between the current best observed DecoupledCDM runs and external published CD baselines on the datasets/splits exercised in the exp116-exp122 evaluation round.

This is a comparison ledger. It uses AUC as the primary metric because that is the main paper metric for these experiments. Full JSON/log artifacts remain on the remote host; this document keeps only the compact table and the exact run口径 needed to recover the numbers.

## Main AUC Table

`Ours best` means the best observed DecoupledCDM AUC among the runs listed in [Ours Best Run Ledger](#ours-best-run-ledger). It is not a same-seed mean/std table.

| Dataset split | Ours best | Source | DINA | IRT | MIRT | NCD | RCD | SVGCD | SCD | Best external | Gap |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| assist09_standard | 0.7793 | exp110 seed2027 | 0.7008 | 0.7021 | 0.6964 | 0.7568 | 0.7730 | 0.7830 | 0.7727 | SVGCD 0.7830 | -0.0037 |
| assist09_holdout | 0.7587 | exp110 seed2026 | 0.6393 | 0.6938 | 0.6856 | 0.7228 | 0.7634 | 0.7684 | 0.7631 | SVGCD 0.7684 | -0.0097 |
| assist17_standard | 0.7823 | tune dim80x96 | 0.6767 | 0.6905 | 0.7520 | 0.7606 | 0.7840 | 0.7867 | 0.7875 | SCD 0.7875 | -0.0052 |
| assist17_holdout | 0.7637 | exp110 seed2028 | 0.6407 | 0.6872 | 0.7526 | 0.7363 | 0.7830 | 0.7836 | 0.7866 | SCD 0.7866 | -0.0229 |
| nips34_standard | 0.7862 | dim80x96 seed2026 | 0.7058 | 0.6714 | 0.7634 | 0.7718 | 0.7789 | 0.7829 | 0.7787 | SVGCD 0.7829 | +0.0033 |
| nips34_holdout | 0.7728 | dim80_lr2e4 seed2024 | 0.6408 | 0.6689 | 0.7584 | 0.7251 | 0.7732 | 0.7735 | 0.7732† | SVGCD 0.7735 | -0.0007 |
| junyi_sample_standard | 0.8029 | exp110 seed2027 | 0.7221 | 0.7714 | 0.8085 | 0.7880 | 0.8180 | 0.8149 | 0.8120 | RCD 0.8180 | -0.0151 |
| junyi_sample_holdout | 0.7977 | exp110 seed2025 | 0.6921 | 0.7672 | 0.8062 | 0.7825 | 0.8165 | 0.8132 | 0.8110 | RCD 0.8165 | -0.0188 |

`Gap = Ours best - Best external`.

SCD completed for all listed standard splits and for ASSIST09/ASSIST17/Junyi holdout splits. `†` `nips34_holdout` SCD is a batch-size memory-rescue partial result: the original copied SCD default `batch_size=256` failed with DGL CUDA OOM, while `nips34_holdout_bs128` reached epoch 3 with AUC `0.773241`, ACC `0.704620`, RMSE `0.439652`; epoch 4 was only partially logged and has no metric. Do not report this as a full 5-epoch default-SCD run.

## Ours Best Run Ledger

The `Ours best` column can be recovered from the following remote result files under `/home/xph/jwc/research/decoupled_cd`.

| Dataset split | AUC | Config口径 | Result JSON |
|---|---:|---|---|
| assist09_standard | 0.779321 | exp110, seed 2027, original ASSIST09 standard split | `results/pure_cdm_exp110_ablation/seed2027_baseline_reproduce.json` |
| assist09_holdout | 0.758717 | exp110, seed 2026, ASSIST09 student-concept holdout split | `results/coverage_holdout_exp116/seed2026_exp110_full.json` |
| assist17_standard | 0.782313 | exp121 tuning, `concept_dim=80`, secondary tower dim `96`, seed 2024 | `results/exp121_dataset_tuning/seed2024_assist17_standard_dim80x96.json` |
| assist17_holdout | 0.763708 | exp110, seed 2028, ASSIST17 student-concept holdout split | `results/exp122_best_seed_sweep/seed2028_assist17_holdout_exp110.json` |
| nips34_standard | 0.786190 | exp121 tuning, `dim80x96`, seed 2026 | `results/exp121_dataset_tuning/seed2026_nips34_standard_dim80x96.json` |
| nips34_holdout | 0.772800 | exp122 sweep, `dim80_lr2e4`, seed 2024 | `results/exp121_dataset_tuning/seed2024_nips34_holdout_dim80_lr2e4.json` |
| junyi_sample_standard | 0.802921 | exp110, seed 2027, 39-concept `junyi_sample` standard split | `results/exp122_best_seed_sweep/seed2027_junyi_sample_standard_exp110.json` |
| junyi_sample_holdout | 0.797678 | exp110, seed 2025, 39-concept `junyi_sample` holdout split | `results/exp122_best_seed_sweep/seed2025_junyi_sample_holdout_exp110.json` |

### Exp110口径

Unless a row explicitly names a tuning variant, `exp110` means the current pure-CDM trial口径:

- single graph mode;
- dual CDM ensemble with secondary concept dim `80`;
- branch BCE weight `0.18`;
- `concept_dim=64`;
- `learning_rate=3e-4`;
- `training_mode=recompute_minibatch`;
- `batch_size=65536`;
- history-evidence prior at `loss_only`;
- concept-evidence prior `train_only`, starting at epoch 135.

The tuned `dim80x96` rows keep the same exp110 feature set but use `concept_dim=80` and secondary tower dim `96`. The tuned `dim80_lr2e4` NIPS row additionally uses `learning_rate=2e-4`.

## External Baseline Artifacts

External baseline results are stored under:

- PyEdmine DINA/IRT/MIRT/NCD/RCD: `/home/xph/jwc/research/local_data/pyedmine_cd_baselines/job_outputs`
- PyEdmine `junyi_sample`: `/home/xph/jwc/research/local_data/pyedmine_cd_baselines_junyi_sample/job_outputs`
- SVGCD: `/home/xph/jwc/research/local_data/svgcd_baselines/job_outputs`
- SCD: `/home/xph/jwc/research/local_data/scd_baselines/runs/<dataset_split>/result/scd_model_val.txt`
- SCD `nips34_holdout` bs128 partial: `/home/xph/jwc/research/local_data/scd_baselines/runs/nips34_holdout_bs128/result/scd_model_val.txt`

SCD values in the table use the best AUC written across recorded epochs. PyEdmine/SVGCD values use their reported test AUC from the corresponding JSON output.

The SVGCD JSON files above were overwritten on 2026-06-08 with the new uploaded script `/home/xph/jwc/svgcd.py`. The run logs and source JSONs are under `/home/xph/jwc/research/local_data/svgcd_new_baselines`; the previous SVGCD JSONs were backed up to `/home/xph/jwc/research/local_data/svgcd_baselines/job_outputs_backup_before_svgcd_py_20260608_110751`.

## Holdout Coverage Slice Supplement

This supplement reuses already trained PyEdmine NCD/RCD checkpoints and does
not retrain them. It exports row-level test probabilities, aligns them with the
same holdout test rows, and computes target coverage from the same train split
and Q-matrix:

```text
target_coverage(s, e) = |Q_e ∩ Seen_s(train)| / |Q_e|
low_coverage = target_coverage < 0.5, including zero coverage
coverage_gap = full_coverage_auc - low_coverage_auc
```

The strongest paper-safe read is:

- Published NCD shows large coverage-induced degradation on all three holdout
  splits.
- Exp110 improves substantially over NCD on low-coverage AUC and coverage gap.
- RCD is a strong graph-based published baseline. It often has higher
  low-coverage AUC than Exp110, so do not claim that Exp110 dominates all
  published baselines on coverage slices.

| Dataset | Model | Overall AUC | Low AUC | Full AUC | Gap | Low Brier | Low ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| assist09_holdout | NCD | 0.722748 | 0.667339 | 0.760785 | 0.093446 | 0.203640 | 0.058475 |
| assist09_holdout | RCD | 0.763432 | 0.749492 | 0.773860 | 0.024369 | 0.179028 | 0.033909 |
| assist09_holdout | Exp110 full mean | 0.755054 | 0.733920 | 0.776789 | 0.042869 | 0.190867 | 0.081303 |
| assist17_holdout | NCD | 0.736298 | 0.715705 | 0.761994 | 0.046290 | 0.212834 | 0.027060 |
| assist17_holdout | RCD | 0.782990 | 0.781943 | 0.792116 | 0.010173 | 0.188007 | 0.012530 |
| assist17_holdout | Exp110 full | 0.755751 | 0.743160 | 0.780608 | 0.037447 | 0.213228 | 0.097672 |
| nips34_holdout | NCD | 0.725060 | 0.672518 | 0.770234 | 0.097716 | 0.227390 | 0.024784 |
| nips34_holdout | RCD | 0.773228 | 0.773302 | 0.773401 | 0.000099 | 0.193702 | 0.036511 |
| nips34_holdout | Exp110 full | 0.772583 | 0.766900 | 0.780103 | 0.013203 | 0.195253 | 0.025829 |

`assist09_holdout` uses the Exp110 four-seed mean from experiment 116. The
ASSIST17/NIPS34 Exp110 rows use the seed2024 cross-dataset extension from the
same holdout protocol. PyEdmine NCD/RCD rows are single seed2024 runs.

Artifacts:

```text
/home/xph/jwc/research/local_data/pyedmine_cd_baselines/coverage_slices
/home/xph/jwc/research/local_data/pyedmine_cd_baselines/coverage_slices/*_ncd_rcd_coverage_summary.csv
/home/xph/jwc/research/local_data/pyedmine_cd_baselines/coverage_slices/predictions
```

Extractor:

```text
scripts/pyedmine_coverage_slice.py
```

## RCD History-Hiding Supplement

This supplement answers whether the graph-based RCD baseline can be evaluated
under the same hidden-history stress idea. It does not retrain RCD. For each
mask seed and hide ratio, it hides a proportion of each student's train
interactions, rebuilds RCD's train-history graph files from the masked train
file, and evaluates the fixed RCD checkpoint through that masked graph
configuration.

Protocol:

- checkpoint weights fixed;
- `hide_ratio=0.2/0.4/0.6/0.8`, mask seeds `11/13/17`;
- non-Junyi holdout splits only;
- RCD graph files rebuilt for the masked train history before evaluation.

This is an inference-time graph/history perturbation stress test. It is useful
for the PPT/diagnostic story, but it should not be mixed with the main AUC
table as a retrained fair-comparison number. The table reports means over mask
seeds `11/13/17`.

| Dataset | Model | Hide | Original AUC | Hidden AUC | Delta AUC | Hidden Brier | Hidden ECE |
|---|---|---:|---:|---:|---:|---:|---:|
| assist09_holdout | RCD | 0.2 | 0.763430 | 0.761432 | 0.001998 | 0.180605 | 0.025000 |
| assist09_holdout | RCD | 0.4 | 0.763430 | 0.758015 | 0.005415 | 0.182206 | 0.030221 |
| assist09_holdout | RCD | 0.6 | 0.763430 | 0.751735 | 0.011695 | 0.185324 | 0.039644 |
| assist09_holdout | RCD | 0.8 | 0.763430 | 0.736126 | 0.027304 | 0.193344 | 0.060353 |
| assist09_holdout | Exp110 full mean | 0.8 | 0.755054 | 0.716900 | 0.038154 | 0.199971 | 0.068623 |
| assist17_holdout | RCD | 0.2 | 0.782990 | 0.782251 | 0.000739 | 0.187391 | 0.006015 |
| assist17_holdout | RCD | 0.4 | 0.782990 | 0.780861 | 0.002129 | 0.187972 | 0.006870 |
| assist17_holdout | RCD | 0.6 | 0.782990 | 0.777792 | 0.005198 | 0.189314 | 0.010024 |
| assist17_holdout | RCD | 0.8 | 0.782990 | 0.770384 | 0.012606 | 0.192810 | 0.022959 |
| assist17_holdout | Exp110 full | 0.8 | 0.755751 | 0.715649 | 0.040103 | 0.215247 | 0.058250 |
| nips34_holdout | RCD | 0.2 | 0.773230 | 0.773125 | 0.000105 | 0.193685 | 0.035285 |
| nips34_holdout | RCD | 0.4 | 0.773230 | 0.773038 | 0.000192 | 0.193713 | 0.035119 |
| nips34_holdout | RCD | 0.6 | 0.773230 | 0.772778 | 0.000452 | 0.193829 | 0.035154 |
| nips34_holdout | RCD | 0.8 | 0.773230 | 0.772018 | 0.001212 | 0.194151 | 0.034664 |
| nips34_holdout | Exp110 full | 0.8 | 0.772583 | 0.739493 | 0.033091 | 0.205920 | 0.021949 |

Read:

- ASSIST09/ASSIST17 show a monotonic RCD degradation curve as train-history
  graph evidence is hidden, so the stress-test construction is meaningful for
  graph baselines.
- RCD remains stronger than Exp110 on hidden AUC and Brier on these holdout
  splits. This result should be presented honestly; it is not evidence that
  Exp110 dominates RCD.
- NIPS34 RCD is almost invariant to this masking, suggesting that its signal on
  NIPS34 mostly comes from exercise-concept/concept graph structure or dense
  remaining evidence rather than the masked part of the student-exercise graph.
- Exp110 still has better hidden ECE than RCD on NIPS34, so calibration can be
  discussed separately from hidden AUC.

Artifacts:

```text
/home/xph/jwc/research/local_data/pyedmine_cd_baselines/history_hiding
/home/xph/jwc/research/local_data/pyedmine_cd_baselines/history_hiding/*_rcd_hide_multi_report_summary.csv
```

Runner:

```text
scripts/pyedmine_rcd_history_hiding.py
```

## Dataset And Reporting Caveats

- This compact table intentionally excludes the 706-concept one-exercise-one-concept `junyi` rows. That encoding is not comparable to the 39-concept `junyi_sample` rows and should not be mixed into the same paper table.
- `junyi_sample` is the 39-concept variant from `ConceptSkillCDM/data/junyi_sample`, converted to the current standard/holdout split format.
- Original PyEdmine RCD constructs a graph from the PyEdmine preprocessed data file. Treat RCD as a published-baseline diagnostic; if a final table depends critically on RCD graph construction, rerun train-only graph construction for that dataset/split.
- The main table reports best observed single-run AUC for our model. A stricter paper table can instead report mean/std by seed, but that is a different comparison口径.
- The coverage slice supplement is diagnostic, not a replacement for the main
  AUC table. It supports the coverage-bias story against NCD, while showing
  that RCD is a strong graph baseline that should be discussed separately.
- The RCD history-hiding supplement is also diagnostic. It changes only the
  evaluation graph/history evidence for a fixed RCD checkpoint, so it should be
  described as a stress test rather than a retrained baseline comparison.
