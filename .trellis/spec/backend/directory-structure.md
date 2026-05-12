# Directory Structure

> How Python code is organized in this research repository.

---

## Overview

The project uses a flat repository layout with top-level Python packages, not a `src/` layout. Keep new modules in the existing package that owns the behavior; do not create a new top-level package unless the responsibility is genuinely new and cross-cutting.

---

## Directory Layout

```text
configs/      Dataset defaults and experiment default values.
data/         CSV readers, ID mappings, Q-matrix handling, graph construction, and StepDataBundle assembly.
models/       PyTorch model modules and propagation logic.
trainers/     Training and evaluation loops.
utils/        Device, IO, logging, metrics, and seed helpers.
scripts/      CLI entry points, preprocessing scripts, remote execution, and experiment run scripts.
tests/        unittest-based focused regression tests.
docs/         Project workflow, handoff, experiment ledger, and experiment detail docs.
```

---

## Module Ownership

- Put dataset loading, schema checks, ID mapping, Q-matrix, and tensor-bundle construction in `data/`. Examples: `data/readers.py`, `data/mappings.py`, `data/pipeline.py`.
- Put train/eval loop semantics in `trainers/`. Example: `trainers/engine.py` owns `full_batch` vs `recompute_minibatch`.
- Put model architecture and tensor-level forward logic in `models/`. Examples: `models/decoupled_cdm.py`, `models/hetero_propagation.py`.
- Put reusable non-domain helpers in `utils/`. Examples: `utils/logging.py`, `utils/metrics.py`, `utils/io.py`.
- Put command-line orchestration in `scripts/`. Scripts may insert the project root into `sys.path`; package modules should not.
- Put experiment default paths and hyperparameters in `configs/defaults.py`; when a new default is added, search for corresponding CLI, run-script, output-summary, and docs updates.

---

## Naming Conventions

- Python files and functions use `snake_case`.
- Model classes use `PascalCase`, for example `DecoupledCDM` and `HeterogeneousGraphPropagation`.
- Dataclass result/bundle types end with clear domain nouns such as `StepDataBundle`, `TrainResult`, or `PropagationOutput`.
- Private helper functions use a leading underscore when they are local implementation details, for example `_bundle_tensors` and `_masked_average`.
- CLI flags use kebab case and should map to `argparse` destination names that fit existing Python naming.

---

## Examples To Follow

- `data/pipeline.py`: keyword-only orchestration functions returning explicit bundle objects.
- `trainers/engine.py`: small dataclasses for return values and fail-fast validation before training.
- `scripts/train.py`: CLI parsing plus explicit output payload and summary-row fields.
- `tests/test_history_visibility.py`: focused regression tests for data-flow invariants.

---

## Common Mistakes To Avoid

- Do not create a generic `src/`, `app/`, or `service/` directory for this project.
- Do not put data parsing inside model code or model architecture inside run scripts.
- Do not add a new CLI flag without checking `scripts/train.py`, `scripts/evaluate.py`, run scripts, summaries, and docs for matching changes.
- Do not treat legacy dual-graph paths as the default path; current mainline is single-graph propagation.
