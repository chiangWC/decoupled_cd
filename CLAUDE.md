# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A decoupled cognitive diagnosis model (CDM) in Python + PyTorch. The model spec (in Chinese) lives in [README_spec.md](README_spec.md): the knowledge space is split per student into tested concepts (TKC, supervised by right/wrong responses) and untested concepts (UKC, informed only by the concept graph); the two are propagated separately over a heterogeneous graph and fused with a learned per-student gate, then passed through a guess/slip layer and trained with BCE.

Runtime environment is the `decoupled_cd` conda env (PyTorch + pandas + scikit-learn). There is no test suite or linter configured; validation is done by running training and comparing metrics.

## Common commands

```bash
# Train with dataset defaults (paths + hyperparameters from configs/defaults.py)
python scripts/train.py --dataset assist_09

# Quick smoke run
python scripts/train.py --dataset assist_09 --epochs 2 --max-rows 5000

# Official ASSIST09 baseline (locks single-graph mode, 300 epochs, full adapter stack;
# SEED and OUTPUT env vars override seed/output path; extra args are appended)
bash scripts/run_assist09_baseline.sh

# Evaluate a checkpoint on train/valid/test splits
python scripts/evaluate.py --help

# Regenerate ASSIST09 data (outputs to data/assist_09_ordered/, which is gitignored)
python scripts/preprocess_assist09_ordered.py --raw-csv <path>
python scripts/build_assist09_transition_graph.py

# Run a command on the GPU host (ssh xph-pc). Requires: current branch pushed to
# origin and local HEAD == origin HEAD; the remote checks out the same branch.
bash scripts/remote_exec.sh python scripts/train.py --dataset assist_09
```

Training writes a summary JSON (`--output`), a `*_best.pt` checkpoint and `*_history.csv` next to it, and appends one row per run to `results/experiment_results.csv`. `logs/`, `results/`, and `data/assist_09_ordered/` are gitignored.

## Architecture

Data flow: `scripts/train.py` → `data.prepare_experiment_split_bundles` → `models.DecoupledCDM` → `trainers.train_model/evaluate_model`.

- **configs/defaults.py** — per-dataset default paths and hyperparameters (`assist_09`, `assist_17`, `junyi`). `assist_17`/`junyi` paths point at the sibling `../ConceptSkillCDM` repo, which is an engineering reference only — do not copy its core model structure.
- **data/** — `pipeline.py` builds a `StepDataBundle` per split: Q-matrix tensor, concept graph, interaction tensors, and history tensors (student TKC/UKC masks, response matrix, per-student-concept and per-exercise evidence counts). Critically, valid/test bundles carry **train-history-only** tensors; `trainers/engine.py::_validate_history_visibility` asserts eval interactions are not inside their own history (leakage guard).
- **models/** — `hetero_propagation.py` implements the decoupled message passing: TKC states receive exercise embeddings weighted by response correctness plus concept-neighbor messages; UKC states receive concept-neighbor messages only. `decoupled_cdm.py` (`DecoupledCDM`) fuses TKC/UKC via a per-student adaptive gate (`--student-fusion-mode` has tkc_only/ukc_only/mean ablations), computes cognitive logits, applies conditional guess/slip, and hosts many **opt-in residual adapters** (high-concept logit, pairwise history interaction, concept-evidence readout/prior, history-evidence logit prior, etc.). `ensemble_cdm.py` is a two-tower variant that averages probabilities inside one checkpoint.
- **trainers/engine.py** — training loop with three modes: `full_batch` (mainline default), `recompute_minibatch`, and `student_recompute_minibatch` (for Junyi-scale data). Handles validation-metric checkpoint selection (`--checkpoint-selection-*`), early stopping, LR plateau scheduling, and optional history-evidence alignment losses.
- **scripts/** — besides train/evaluate: `pyedmine_cd_baselines.py`, `scd_baselines.py`, `svgcd_baselines.py` run published baselines on the same splits; `analyze_prediction_slices.py`, `evaluate_coverage_slice.py`, `evaluate_gate_diagnostic.py`, `evaluate_history_hiding_stress.py` are diagnostic/stress evaluations.

## Conventions

- **Baseline preservation is the core convention.** New model features are added as flag-gated adapters that default off and are zero-initialized, so existing runs reproduce exactly. Follow this pattern when extending the model; don't change defaults that would shift baseline numbers (e.g. `--weight-decay` defaults to 0.0 for this reason).
- The mainline uses `--graph-mode single`; `dual` (prerequisite + similarity graphs) exists only for historical ablations and is rejected by the baseline script.
- Every new train flag is threaded through all of: argparse in `scripts/train.py`, model kwargs, the output JSON, and the `results/experiment_results.csv` summary row — keep these in sync so runs stay comparable.
- Interaction CSVs use columns `stu_id`, `exer_id`, `cpt_seq` (list of concept ids), `label`.
