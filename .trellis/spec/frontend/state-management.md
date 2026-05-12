# State Management

> Not applicable to the current codebase.

---

## Current State

There is no client-side state management layer. Project state lives in explicit Python objects and artifacts:

- `StepDataBundle` for loaded data and tensors.
- PyTorch module parameters and optimizer/scheduler state during training.
- JSON, CSV, log, and checkpoint files under result/log directories.
- Experiment conclusions in `docs/`.

---

## Rule

Do not add global frontend state libraries or browser data caches unless a future UI task explicitly requires them.
