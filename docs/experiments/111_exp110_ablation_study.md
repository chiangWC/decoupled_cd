# Experiment 111: exp110 pure-CDM ablation study

## Status

`ablation_completed_batch131072_mean_auc_candidate`

## Verdict

Using the experiment 110 single-checkpoint pure-CDM route as the baseline,
the strongest four-seed mean-AUC variant in this ablation is:

```text
b_batch131072_lr3e4: mean AUC 0.7785364000, delta +0.0002150566 vs exp110 reproduce
```

This is a small AUC lift, but it raises peak CUDA from `6.190541GB` to
`8.759760GB`. It is therefore a higher-AUC candidate, not the best low-memory
candidate.

The strongest single-seed AUC in this full matrix is
`c_prior_start001 seed2027 = 0.7793765475`, but its four-seed mean is
`0.7783169685`, essentially tied with the exp110 baseline and not a promoted
route.

The default low-memory route remains exp110 itself:

```text
baseline_reproduce: mean AUC 0.7783213434, peak CUDA 6.190541GB
```

Lower-memory alternatives such as `b_batch32768_lr3e4` (`4.870220GB`) and
`d_secondary_dim032` (`4.527024GB`) lose too much AUC to replace exp110.

No checkpoint inference average, hybrid stacker, validation-trained combiner,
or train-history tabular inference side channel was used. Every row is a
single training run with a single selected checkpoint per seed.

## Base

- Branch: `exp/trellis-trial`
- Key commit for the code path: `71e9898`
- Baseline: experiment 110 `dual64x80 + branchBCE=0.18`, `recompute_minibatch`,
  `batch_size=65536`, `learning_rate=0.0003`, `epochs=300`
- Result directory on remote host: `results/pure_cdm_exp110_ablation/`
- Summary file on remote host: `results/pure_cdm_exp110_ablation/summary.jsonl`
- Per-run result path pattern:
  `results/pure_cdm_exp110_ablation/seed{seed}_{variant}.json`
- Seeds: `2024`, `2025`, `2026`, `2027`

## Command Template

Each run used this base command shape, with `${SEED}`, `${VARIANT}`, and the
variant overlay below substituted:

```bash
python scripts/train.py \
  --dataset assist_09 \
  --train-interactions data/assist_09_ordered/train.csv \
  --valid-interactions data/assist_09_ordered/valid.csv \
  --test-interactions data/assist_09_ordered/test.csv \
  --q-matrix data/assist_09_ordered/Q_matrix.csv \
  --concept-graph data/assist_09_ordered/transition_graph/propagation_graph.csv \
  --graph-mode single \
  --concept-dim 64 \
  --gs-mode conditional \
  --high-concept-logit-adapter --high-concept-logit-min-count 2 \
  --pairwise-history-interaction-adapter --pairwise-history-interaction-min-count 2 \
  --gs-difficulty-adapter \
  --interpretable-readout-expert-adapter --interpretable-readout-expert-count 3 \
  --student-conditioned-ukc-readout-residual \
  --concept-evidence-readout-residual \
  --concept-evidence-readout-min-count 1 \
  --concept-evidence-readout-max-count 1 \
  --concept-evidence-readout-min-seen-ratio 1.0 \
  --concept-evidence-readout-max-logit 0.5 \
  --concept-evidence-prior-residual \
  --concept-evidence-prior-min-count 1 \
  --concept-evidence-prior-min-seen-ratio 1.0 \
  --concept-evidence-prior-max-logit 0.3 \
  --concept-evidence-prior-min-confidence 0.75 \
  --concept-evidence-prior-min-abs-mastery 0.5 \
  --concept-evidence-prior-apply-mode train_only \
  --concept-evidence-prior-train-start-epoch 135 \
  --history-evidence-logit-prior-residual \
  --history-evidence-logit-prior-location loss_only \
  --history-evidence-logit-prior-min-count 1 \
  --history-evidence-logit-prior-min-seen-ratio 0.0 \
  --history-evidence-logit-prior-max-logit 4.0 \
  --history-evidence-logit-prior-component-cap 4.0 \
  --history-evidence-logit-prior-weight-student 0.0 \
  --history-evidence-logit-prior-weight-exercise 0.0 \
  --history-evidence-logit-prior-weight-target-concept 0.44 \
  --history-evidence-logit-prior-weight-concept 0.22 \
  --history-evidence-logit-prior-weight-mastery 0.22 \
  --history-evidence-cognitive-alignment-weight 0.05 \
  --history-evidence-cognitive-alignment-final-weight 0.0881 \
  --history-evidence-cognitive-alignment-anneal-start-epoch 170 \
  --history-evidence-cognitive-alignment-anneal-end-epoch 230 \
  --dual-cdm-ensemble \
  --dual-cdm-secondary-concept-dim 80 \
  --dual-cdm-branch-bce-weight 0.18 \
  --training-mode recompute_minibatch \
  --batch-size 65536 \
  --learning-rate 0.0003 \
  --epochs 300 \
  --checkpoint-selection-metric auc \
  --seed "${SEED}" \
  --output "results/pure_cdm_exp110_ablation/seed${SEED}_${VARIANT}.json"
```

Variant overlays are exact replacements or removals against the base command:

| variant | overlay |
|---|---|
| `baseline_reproduce` | none |
| `a_no_branch_bce` | replace `--dual-cdm-branch-bce-weight 0.18` with `0.0` |
| `a_no_concept_prior` | remove all `--concept-evidence-prior-*` flags and `--concept-evidence-prior-residual` |
| `a_constant_cog_align` | replace final alignment weight `0.0881` with `0.05` |
| `a_no_cog_align` | replace both cognitive alignment weights with `0.0` |
| `a_no_dual_tower` | remove `--dual-cdm-ensemble`, `--dual-cdm-secondary-concept-dim`, and branch BCE flags |
| `b_fullbatch_lr3e4` | replace `--training-mode recompute_minibatch` with `full_batch`; remove `--batch-size` |
| `b_recompute_lr1e3` | replace `--learning-rate 0.0003` with `0.001` |
| `b_batch32768_lr3e4` | replace `--batch-size 65536` with `32768` |
| `b_batch131072_lr3e4` | replace `--batch-size 65536` with `131072` |
| `b_brier_select` | replace checkpoint metric `auc` with `brier` |
| `c_lr4e4` | replace learning rate with `0.0004` |
| `c_lr5e4` | replace learning rate with `0.0005` |
| `c_branchbce010` | replace branch BCE weight with `0.10` |
| `c_branchbce015` | replace branch BCE weight with `0.15` |
| `c_branchbce020` | replace branch BCE weight with `0.20` |
| `c_prior_start001` | replace concept prior train start epoch with `1` |
| `c_prior_start170` | replace concept prior train start epoch with `170` |
| `d_secondary_dim032` | replace secondary dim with `32` |
| `d_secondary_dim064` | replace secondary dim with `64` |
| `d_secondary_dim096` | replace secondary dim with `96` |

## Matrix

All variants use ASSIST09 ordered split, single graph mode, the experiment 110
mainline adapters, cog-only history evidence prior, and AUC checkpoint
selection unless noted by the variant name.

| variant | n | mean AUC | delta vs exp110 | min AUC | max AUC | mean ACC | mean RMSE | mean Brier | mean ECE | peak CUDA GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `b_batch131072_lr3e4` | 4 | 0.7785364000 | +0.0002150566 | 0.7779989939 | 0.7789142013 | 0.7384821785 | 0.4188992774 | 0.1754766572 | 0.0260068309 | 8.759760 |
| `c_lr5e4` | 4 | 0.7784570231 | +0.0001356797 | 0.7778067775 | 0.7788754381 | 0.7385154808 | 0.4196520129 | 0.1761083176 | 0.0341779091 | 6.190541 |
| `c_lr4e4` | 4 | 0.7783770644 | +0.0000557210 | 0.7778143716 | 0.7790242497 | 0.7384916935 | 0.4194513664 | 0.1759395656 | 0.0326511047 | 6.190541 |
| `a_no_concept_prior` | 4 | 0.7783249121 | +0.0000035687 | 0.7774380651 | 0.7793381479 | 0.7385487830 | 0.4194613010 | 0.1759478698 | 0.0321619243 | 6.190541 |
| `c_prior_start170` | 4 | 0.7783249121 | +0.0000035687 | 0.7774380651 | 0.7793381479 | 0.7385487830 | 0.4194613010 | 0.1759478698 | 0.0321619243 | 6.190541 |
| `baseline_reproduce` | 4 | 0.7783213434 | +0.0000000000 | 0.7774488561 | 0.7793212012 | 0.7386486898 | 0.4192017836 | 0.1757302240 | 0.0288909961 | 6.190541 |
| `a_constant_cog_align` | 4 | 0.7783213434 | +0.0000000000 | 0.7774488561 | 0.7793212012 | 0.7386486898 | 0.4192017836 | 0.1757302240 | 0.0288909961 | 6.190541 |
| `c_prior_start001` | 4 | 0.7783169685 | -0.0000043749 | 0.7773400929 | 0.7793765475 | 0.7391672534 | 0.4189659708 | 0.1755326596 | 0.0269394177 | 6.190541 |
| `c_branchbce020` | 4 | 0.7782656714 | -0.0000556720 | 0.7773409717 | 0.7791980238 | 0.7384964509 | 0.4191051581 | 0.1756493231 | 0.0278694713 | 6.190541 |
| `c_branchbce015` | 4 | 0.7781814737 | -0.0001398698 | 0.7774490673 | 0.7790619825 | 0.7385297532 | 0.4191201800 | 0.1756618548 | 0.0283644608 | 6.190541 |
| `d_secondary_dim096` | 4 | 0.7781119619 | -0.0002093816 | 0.7767949764 | 0.7790006481 | 0.7382110031 | 0.4194529476 | 0.1759408953 | 0.0309957778 | 6.745297 |
| `c_branchbce010` | 4 | 0.7780580394 | -0.0002633040 | 0.7772500211 | 0.7788946028 | 0.7385059659 | 0.4193660534 | 0.1758679850 | 0.0299150535 | 6.190541 |
| `b_recompute_lr1e3` | 4 | 0.7780195510 | -0.0003017925 | 0.7773301324 | 0.7785741353 | 0.7376876820 | 0.4200865733 | 0.1764728599 | 0.0355895959 | 6.190541 |
| `b_fullbatch_lr3e4` | 4 | 0.7777226116 | -0.0005987319 | 0.7772442661 | 0.7780855955 | 0.7383299397 | 0.4188615020 | 0.1754451725 | 0.0199303483 | 11.756714 |
| `b_batch32768_lr3e4` | 4 | 0.7777145762 | -0.0006067672 | 0.7768848974 | 0.7785427525 | 0.7383584845 | 0.4195273800 | 0.1760033257 | 0.0306232670 | 4.870220 |
| `d_secondary_dim064` | 4 | 0.7775755859 | -0.0007457576 | 0.7769768526 | 0.7781986689 | 0.7381253687 | 0.4195462302 | 0.1760191350 | 0.0292269540 | 5.624450 |
| `d_secondary_dim032` | 4 | 0.7772125373 | -0.0011088062 | 0.7764662205 | 0.7781171024 | 0.7386201450 | 0.4193453491 | 0.1758505442 | 0.0253433008 | 4.527024 |
| `b_brier_select` | 4 | 0.7771751415 | -0.0011462020 | 0.7760954579 | 0.7777736774 | 0.7380111896 | 0.4192074620 | 0.1757349650 | 0.0242063705 | 6.190541 |
| `a_no_branch_bce` | 4 | 0.7764808739 | -0.0018404696 | 0.7758484838 | 0.7769995930 | 0.7369883347 | 0.4209897632 | 0.1772326888 | 0.0378837222 | 6.190541 |
| `a_no_dual_tower` | 4 | 0.7746567545 | -0.0036645889 | 0.7716493253 | 0.7763614701 | 0.7363698643 | 0.4215082937 | 0.1776696615 | 0.0355190740 | 3.346300 |
| `a_no_cog_align` | 4 | 0.7743975282 | -0.0039238153 | 0.7740247342 | 0.7753229118 | 0.7368360958 | 0.4213273495 | 0.1775167763 | 0.0346632346 | 6.190754 |

## Highest Single-Seed Rows

| rank | variant | seed | AUC | ACC | RMSE | Brier | ECE | best epoch | peak CUDA GB | result JSON |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `c_prior_start001` | 2027 | 0.7793765475 | 0.7391957982 | 0.4183869026 | 0.1750476002 | 0.0258953873 | 149 | 6.190541 | `results/pure_cdm_exp110_ablation/seed2027_c_prior_start001.json` |
| 2 | `a_no_concept_prior` | 2027 | 0.7793381479 | 0.7378066186 | 0.4190944180 | 0.1756401312 | 0.0309486797 | 149 | 6.190541 | `results/pure_cdm_exp110_ablation/seed2027_a_no_concept_prior.json` |
| 3 | `c_prior_start170` | 2027 | 0.7793381479 | 0.7378066186 | 0.4190944180 | 0.1756401312 | 0.0309486797 | 149 | 6.190541 | `results/pure_cdm_exp110_ablation/seed2027_c_prior_start170.json` |
| 4 | `baseline_reproduce` | 2027 | 0.7793212012 | 0.7379969172 | 0.4188889911 | 0.1754679869 | 0.0282077546 | 149 | 6.190541 | `results/pure_cdm_exp110_ablation/seed2027_baseline_reproduce.json` |
| 5 | `c_branchbce020` | 2027 | 0.7791980238 | 0.7388152011 | 0.4184603221 | 0.1751090412 | 0.0260116735 | 149 | 6.190541 | `results/pure_cdm_exp110_ablation/seed2027_c_branchbce020.json` |
| 6 | `c_branchbce015` | 2027 | 0.7790619825 | 0.7386629622 | 0.4186196060 | 0.1752423745 | 0.0267440943 | 149 | 6.190541 | `results/pure_cdm_exp110_ablation/seed2027_c_branchbce015.json` |
| 7 | `c_lr4e4` | 2027 | 0.7790242497 | 0.7377875887 | 0.4190479863 | 0.1756012148 | 0.0318735928 | 123 | 6.190541 | `results/pure_cdm_exp110_ablation/seed2027_c_lr4e4.json` |
| 8 | `d_secondary_dim096` | 2024 | 0.7790006481 | 0.7375402006 | 0.4192718506 | 0.1757888847 | 0.0305074904 | 140 | 6.745297 | `results/pure_cdm_exp110_ablation/seed2024_d_secondary_dim096.json` |
| 9 | `b_batch131072_lr3e4` | 2024 | 0.7789142013 | 0.7376353499 | 0.4189489765 | 0.1755182449 | 0.0265885146 | 211 | 8.759760 | `results/pure_cdm_exp110_ablation/seed2024_b_batch131072_lr3e4.json` |

## Interpretation

- Core mechanism attribution:
  - Branch BCE matters. Removing it drops mean AUC by `-0.001840`.
  - Cognitive alignment is the largest required signal. Removing it drops mean
    AUC by `-0.003924`.
  - Dual tower matters. Removing the secondary tower drops mean AUC by
    `-0.003665`; it saves memory but loses too much performance.
  - Concept prior is not an AUC driver in this exp110 window. Removing it is
    near-neutral on AUC but worsens ECE.
  - Constant cog align is identical to baseline in this run because selected
    checkpoints land before the anneal-final schedule can affect the best
    checkpoint.
- Training protocol:
  - `batch=131072, lr=3e-4` is the only clear four-seed mean-AUC improvement,
    but uses `8.76GB`, outside the original exp110 low-memory target.
  - Full-batch `lr=3e-4` improves calibration (`ECE 0.019930`) but loses AUC
    and uses `11.76GB`.
  - Brier checkpoint selection improves calibration relative to high-ECE runs
    but hurts AUC.
- Local AUC search:
  - `lr=4e-4/5e-4` gives small AUC mean lifts, but both worsen ECE; the lift is
    smaller than `batch=131072`.
  - Branch BCE neighborhood confirms `0.18` remains the best local balance.
    `0.20` gives a high seed2027 point but weaker mean.
  - Prior start timing does not materially improve mean AUC.
- Capacity:
  - Secondary dim `32` and `64` are weaker than `80`.
  - Secondary dim `96` gives occasional single-seed headroom but increases
    memory and lowers mean AUC versus exp110.

## Decision

- Keep exp110 (`batch=65536`, `lr=3e-4`, secondary dim `80`, branch BCE `0.18`)
  as the default low-memory pure-CDM training route.
- Record `b_batch131072_lr3e4` as the best mean-AUC ablation candidate if a
  higher memory band is acceptable.
- Do not promote `c_prior_start001` despite the strongest single-seed AUC,
  because its mean is effectively neutral.
- Do not continue no-BCE, no-cog-align, no-dual-tower, dim32, dim64, or Brier
  selection as AUC routes.
