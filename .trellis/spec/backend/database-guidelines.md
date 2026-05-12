# Database / Data Artifact Guidelines

> This project has no database layer. Treat CSV files, tensors, and result files as the persistence interface.

---

## Overview

There is no ORM, migration system, transaction layer, or application database in this repository. Data enters through CSV artifacts and is transformed into pandas DataFrames and PyTorch tensors.

Current mainline data paths and experiment defaults are defined in [Experiment Protocol](./experiment-protocol.md) and `configs/defaults.py`. The current primary dataset is `data/assist_09_ordered`.

---

## Input Artifact Contracts

- Interaction CSVs must include at least `stu_id`, `exer_id`, and `label`; current ASSIST09 files also carry `cpt_seq`.
- Q-matrix CSVs must include `exer_id` and `cpt_seq`.
- Concept sequences are comma-separated and normalized through `data/q_matrix.py::normalize_concept_sequence`.
- External concept graphs are CSV matrices loaded with `pd.read_csv(path, index_col=0)` and converted to `torch.float32`.
- Ordered ASSIST09 plus `data/assist_09_ordered/transition_graph/propagation_graph.csv` is the current mainline.

Examples:

- `data/readers.py` validates required columns before returning copies.
- `data/pipeline.py` builds unified mappings from train/valid/test plus Q-matrix before creating split bundles.
- `data/concept_graph.py` builds and saves graph matrices as CSV artifacts.

---

## History Visibility Contract

Evaluation bundles must reuse train history for propagation inputs and must not include target split rows in propagation history.

Current pattern:

- `prepare_experiment_split_bundles` builds history tensors from `train_df`.
- Train bundle sets `allow_target_in_history=True`.
- Valid and test bundles set `allow_target_in_history=False`.
- `trainers/engine.py::_validate_history_visibility` rejects evaluation bundles whose target rows overlap with history rows.

Do not rebuild valid/test behavior history from their own interactions unless the experiment explicitly changes the protocol and documents that protocol change.

---

## Result Artifacts

- JSON summaries are written with `utils/io.py::write_json`.
- Training history CSVs are written with `save_history_csv`.
- Aggregate rows append to `results/experiment_results.csv` through `append_summary_csv`.
- Training logs go under `logs/` with timestamp and process ID in the filename.
- Experiment conclusions belong in `docs/experiment_index.jsonl`, `docs/model_improvement_plan.md`, and detail docs under `docs/experiments/` when they affect future decisions.

---

## Naming Conventions

- Keep dataframe columns aligned with existing names: `stu_id`, `exer_id`, `cpt_seq`, `label`.
- Keep output metric keys aligned with existing summaries: `auc`, `acc`, `rmse`, `brier`, `ece`, and `calibration_bins`.
- Keep CLI/output field names consistent between `scripts/train.py` and `scripts/evaluate.py`.

---

## Common Mistakes

- Do not introduce a database or caching layer for experiment artifacts without an explicit task.
- Do not change dataset defaults in only one place; update CLI defaults, run scripts, output summaries, and docs as needed.
- Do not silently reinterpret `valid/test` history semantics.
- Do not report an experiment conclusion without enough artifact paths and ledger updates for the next agent to reproduce the judgment.
