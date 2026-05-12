# Logging Guidelines

> Runtime logging conventions for training, evaluation, and experiment scripts.

---

## Overview

Use the standard library `logging` module through `utils/logging.py::setup_logging` for long-running scripts. It creates a unique logger name, writes both to file and stderr/stdout, and includes timestamp plus process ID in the log filename to avoid collisions during parallel runs.

---

## Log Levels

- Use `INFO` for selected device, graph mode, seed, split paths, and final run metrics.
- Use `ERROR` only when catching and logging an error before re-raising or exiting. Do not add broad catch-and-log wrappers by default.
- Avoid debug-level logging inside tensor-heavy loops unless the task explicitly asks for diagnostic instrumentation.

---

## Log Format And Files

Current format:

```text
%(asctime)s - %(levelname)s - %(message)s
```

Current log path pattern:

```text
logs/<name>_<UTC timestamp>_pid<PID>.log
```

Follow `utils/logging.py` instead of creating ad hoc file handlers in scripts.

---

## What To Log

Training and evaluation scripts should log:

- Resolved device.
- Graph mode.
- Seed.
- Input split paths or single-file input path.
- Final `best_val_auc`, `test_auc`, `test_acc`, `test_rmse`, `test_brier`, and `test_ece` when available.
- Any protocol-level mode that changes result interpretation, such as `training_mode` or graph mode.

---

## What Not To Log

- Do not dump full DataFrames, full tensors, full Q-matrices, or full prediction arrays into logs.
- Do not log secrets, SSH details beyond stable host aliases, or environment credentials.
- Do not make logs the only location of experiment conclusions; durable judgments belong in the docs ledger.
