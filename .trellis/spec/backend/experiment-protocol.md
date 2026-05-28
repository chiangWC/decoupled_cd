# Experiment Protocol

> Project operating rules for branches, remote execution, mainline defaults, and experiment records.

---

## Purpose

This file is the Trellis-owned source for project execution constraints. Keep workflow mechanics in `.trellis/workflow.md`, code conventions in the other backend spec files, and experiment evidence in `docs/`.

Do not duplicate long result tables here. Use this file to state rules that future agents must follow while changing code or running experiments.

---

## Local And Remote Roles

- Local worktree: edit code, docs, Trellis tasks, and specs locally.
- Remote host: run training, evaluation, smoke tests, and project test commands on `xph-pc`.
- Remote project path: `/home/xph/jwc/research/decoupled_cd`.
- Remote conda environment: `decoupled_cd`.
- Default `origin`: the remote execution repository on `xph-pc`.
- Do not directly edit remote code unless the user explicitly requests it.

Use:

```bash
bash scripts/remote_exec.sh <command>
```

`scripts/remote_exec.sh` verifies that the remote branch exists, remote branch `HEAD` matches local `HEAD`, and the remote worktree is clean before running the command.

- Do not launch multiple `scripts/remote_exec.sh` commands against the same remote repository at the same time. Each invocation checks git status and runs `git switch` on `xph-pc`; concurrent calls can race on `.git/index.lock` before the actual training or test command starts.

---

## Branch And Commit Rules

- `master` holds only accepted upstream model-mainline state.
- `exp/trellis-trial` is the Trellis-enabled pseudo-mainline for this worktree. Treat it as the default base for future local experiments and Trellis-managed docs/spec/task updates.
- Exploratory experiments use `exp/<short-name>` branches.
- This worktree is Trellis-managed on `exp/trellis-trial` and its descendants. Do not create new experiment branches directly from `master` in this worktree, because `master` may not contain the Trellis files/hooks required for session detection.
- New structure experiments normally continue from `exp/trellis-trial`, the current Trellis-enabled branch, or an `exp/*` branch descended from `exp/trellis-trial`, unless the user explicitly asks for a different base.
- Use `master` as the accepted model-mainline reference, not as the default branch base for Trellis-managed experimentation.
- Any validation that requires code changes should happen on a non-`master`, Trellis-enabled branch first.
- Before remote execution, commit local changes and push the current branch:

```bash
git push origin "$(git branch --show-current)"
```

- Do not push unverified exploratory experiment code directly into `master`.
- Even when exploratory code is not merged, completed experiment conclusions should return to the Trellis-enabled experiment line as doc-only ledger updates when they affect future decisions.
- If those ledger updates are first written on a descendant `exp/*` branch, do not leave them there only. Before wrapping up, sync the doc-only ledger commits back onto `exp/trellis-trial` and push `origin/exp/trellis-trial`, because new sessions bootstrap from `exp/trellis-trial` rather than from arbitrary descendant branches.

---

## Current Pseudo-Mainline Contract

The current Trellis pseudo-mainline is the experiment 70 structure baseline plus experiment 81's single-only concept-evidence readout, promoted after exp70-based multi-seed validation. The accepted `master` model-mainline still uses the experiment 70 structure reference. Preserve this protocol unless the active task explicitly changes it:

- Dataset: `data/assist_09_ordered`.
- Graph: `data/assist_09_ordered/transition_graph/propagation_graph.csv`.
- `graph_mode = single`; dual graph is legacy ablation only.
- `learning_rate = 1e-3`.
- `concept_dim = 64`.
- `gs_mode = conditional`.
- TKC/UKC propagation parameters are decoupled.
- TKC behavior messages use correct and incorrect channels.
- Student-level TKC/UKC fusion uses the adaptive gate.
- `high_concept_logit_adapter` is enabled with `high_concept_logit_min_count = 2`.
- `pairwise_history_interaction_adapter` is enabled with `pairwise_history_interaction_min_count = 2`.
- `gs_difficulty_adapter` is enabled.
- `interpretable_readout_expert_adapter` is enabled with `interpretable_readout_expert_count = 3`.
- `student_conditioned_ukc_readout_residual` is enabled.
- `concept_evidence_readout_residual` is enabled with `min_count = 1`, `max_count = 1`, `min_seen_ratio = 1.0`, `max_logit = 0.5`.
- Structural comparisons default to 300 epochs.

The official run script encodes this mainline:

```bash
bash scripts/run_assist09_baseline.sh
```

---

## Evaluation Semantics

- `valid` and `test` reuse `train` behavior history for propagation inputs.
- Do not rebuild propagation history from valid/test target rows.
- Main reports prioritize `AUC/ACC`.
- `RMSE`, `Brier`, `ECE`, and calibration bins are retained as secondary metrics for error and calibration side effects.

---

## Hybrid Stacker Evaluation Contract

### 1. Scope / Trigger

- Trigger: `scripts/evaluate_ensemble.py` supports opt-in `stack_*` combiners and `--stack-feature-set hybrid`.
- This is an explicitly hybrid evaluator, not a default CDM training or promotion path.

### 2. Signatures

```bash
python3 scripts/evaluate_ensemble.py \
  --summaries <summary.json> [<summary.json> ...] \
  --combiner average|stack_logistic|stack_hist_gradient|stack_gradient_boosting|stack_extra_trees \
  --stack-feature-set predictions|hybrid \
  --average prob|logit \
  --split valid|test \
  --output <result.json>
```

### 3. Contracts

- `--summaries` must point to training summary JSON files with `best_checkpoint_path` and matching split/data paths.
- `average` controls member prediction features: raw probability or logit-transformed probability.
- `stack_*` combiners fit only on the validation split, then evaluate on `--split`.
- `--stack-feature-set hybrid` may add train-history tabular features from student, exercise, student-exercise, concept, and student-concept aggregates.
- Hybrid feature construction must use `train` history only; target split labels are used only as labels for the combiner fit/evaluation, not as feature inputs.

### 4. Validation & Error Matrix

- Missing `best_checkpoint_path` -> `ValueError`.
- Missing checkpoint file -> `FileNotFoundError`.
- Weight count mismatch -> `ValueError`.
- `--stack-feature-set hybrid` without train-history feature state in stack code -> `ValueError`.
- Valid/test history containing target rows -> existing `_validate_history_visibility` `ValueError`.

### 5. Good/Base/Bad Cases

- Good: `stack_hist_gradient --stack-feature-set hybrid` trained on valid and evaluated on test, with all feature values derived from train history plus checkpoint predictions.
- Base: `combiner=average --stack-feature-set predictions`, which is ordinary checkpoint ensembling without learned hybrid features.
- Bad: fitting a combiner on test labels, or deriving student/exercise/concept aggregates from valid/test target rows.

### 6. Tests Required

- Add or keep a unit test proving target labels do not affect hybrid history features.
- Keep remote `py_compile` and focused `unittest` verification before reporting evaluator changes.
- Experiment conclusions from hybrid stackers must record the combiner, feature set, member summaries, output path, and valid/test metrics.

### 7. Wrong vs Correct

Wrong:

```text
Report a hybrid stacker result as if it were the default CDM model or `run_assist09_baseline.sh` output.
```

Correct:

```text
Report it as an opt-in hybrid evaluator candidate and require multi-seed validation or a separate model-integration task before promotion.
```

## Prediction-Only Checkpoint Average Contract

### 1. Scope / Trigger

- Trigger: `scripts/evaluate_checkpoint_average.py` evaluates fixed prediction averages over two or more trained checkpoint summaries.
- This is a pure-CDM evaluator only when every member summary is a pure-CDM checkpoint and averaging weights are fixed by the command, not learned from validation labels.
- It is not a hybrid stacker: it must not add train-history tabular features outside the member checkpoints and must not fit a combiner.

### 2. Signatures

```bash
python scripts/evaluate_checkpoint_average.py \
  --summaries <summary.json> <summary.json> [...] \
  --average prob|logit \
  --split train|valid|test \
  --output <result.json>
```

### 3. Contracts

- `--summaries` must contain at least two training summary JSON files with `best_checkpoint_path`.
- Member summaries must share the same train/valid/test split paths, Q-matrix, concept graph, and graph mode.
- `--average prob` averages probabilities directly.
- `--average logit` averages clipped logits and maps back through sigmoid.
- Valid/test bundles must continue to reuse train-history propagation inputs; target split rows must not enter history tensors.
- Experiment 102 establishes the current checkpoint-average candidate: experiment 95 cog-only checkpoint plus experiment 100 dim80 + output-alignment checkpoint, probability average.

### 4. Validation & Error Matrix

- Fewer than two summaries -> `ValueError`.
- Missing `best_checkpoint_path` -> `ValueError`.
- Missing `concept_dim` -> `ValueError`.
- Mismatched split/data signature -> `ValueError`.
- Unsupported graph mode -> `ValueError`.
- Missing checkpoint file -> standard checkpoint load failure.
- Valid/test history containing target rows -> existing `_validate_history_visibility` `ValueError`.

### 5. Good/Base/Bad Cases

- Good: fixed probability average of two pure-CDM checkpoints, evaluated on test, with no validation fitting.
- Base: single experiment 95 checkpoint evaluated through its training summary.
- Bad: calling a valid-trained stacker a checkpoint average, or adding train-history tabular features outside the member checkpoint predictions.
- Bad: reporting a checkpoint-average result as if it were a single `scripts/run_assist09_baseline.sh` training run.

### 6. Tests Required

- Unit tests for probability/logit averaging and summary compatibility validation.
- Remote `py_compile`, focused `unittest`, and at least one script-level reproduction before reporting evaluator changes.

### 7. Wrong vs Correct

Wrong:

```text
Promote experiment 102 by silently changing the single-checkpoint baseline training script.
```

Correct:

```text
Report experiment 102 as an opt-in pure-CDM checkpoint-average evaluator and keep the single-checkpoint default-training question separate.
```

## Current Pure-CDM Trial Runner Contract

### 1. Scope / Trigger

- Trigger: `scripts/run_assist09_history_alignment_trial.sh` runs the current active single-run/single-checkpoint pure-CDM trial, experiment 110.
- The runner layers experiment 110's low-memory dual CDM ensemble, branch BCE, recompute-minibatch training, late cognitive-alignment anneal, and train-only concept evidence prior on top of the official experiment 81 baseline script.
- Historical probe runners for reliability weighting, output alignment, rank alignment, fusion/linear readouts, evidence-gate variants, and multiseed wrappers have been removed from the active CLI surface. Reproduce old experiments from their detail docs or git history instead of keeping runnable wrappers in `scripts/`.

### 2. Signatures

```bash
bash scripts/run_assist09_history_alignment_trial.sh
```

The runner accepts common `scripts/train.py` smoke/test overrides such as `--epochs`, `--max-rows`, `--device`, `--gpus`, `--seed`, and `--output`.

### 3. Contracts

- `scripts/run_assist09_baseline.sh` remains the official experiment 81 baseline runner.
- `scripts/run_assist09_history_alignment_trial.sh` remains the current active trial runner and encodes experiment 110 by default.
- `scripts/train.py` keeps CLI flags needed by the official baseline and experiment 110; rejected probe-only flags should not be reintroduced without a new task and updated ledger rationale.
- Experiment 110 still uses `history_evidence_logit_prior_location=loss_only`; it must not add an inference-time output-logit sidecar, valid-trained combiner, or hybrid tabular features.
- Concept evidence prior uses `apply_mode=train_only` and starts at epoch 135 in the current runner.
- Dual tower branch BCE uses `dual_cdm_branch_bce_weight=0.18`; the secondary tower concept dimension is 80.
- Training mode is `recompute_minibatch` with `batch_size=65536` and `learning_rate=0.0003`.
- Experiment 104 remains the previous heavy full-batch dual-tower reference (`branchBCE=0.10`), and experiment 106 remains the heavy full-batch `branchBCE=0.18` refinement that experiment 110 builds on.

### 4. Validation & Error Matrix

- Negative `history_evidence_cognitive_alignment_weight` -> `ValueError`.
- Negative `weight_decay` -> `ValueError`.
- Positive alignment weight without `history_evidence_logit_prior_residual` -> `ValueError`.
- Positive alignment weight unless `history_evidence_logit_prior_location=loss_only` -> `ValueError`.
- Negative `dual_cdm_branch_bce_weight` -> `ValueError`.
- Non-positive `dual_cdm_secondary_concept_dim` -> `ValueError`.
- Invalid concept-evidence prior/readout counts, ratios, or logit caps -> `ValueError`.

### 5. Good/Base/Bad Cases

- Good: compare a new pure-CDM single-checkpoint idea against the current active runner, experiment 110's four-seed mean `0.778321`.
- Good: use `scripts/run_assist09_baseline.sh` for exp81 baseline checks and `scripts/run_assist09_history_alignment_trial.sh` for exp110 trial checks.
- Base: exp81 baseline runner.
- Bad: reporting experiment 95/100/103 historical runner behavior as the current trial after this cleanup.
- Bad: reintroducing removed probe-only runner scripts or train CLI flags because an old detail doc mentions them.
- Bad: changing `scripts/run_assist09_baseline.sh` to experiment 104, 106, or 110; baseline and trial remain separate.
- Bad: reporting exp111's `c_prior_start001` single-seed near-tie as the current default; exp110 baseline remains the default.

### 6. Tests Required

- `python3 -m py_compile scripts/train.py`.
- Shell syntax check for active runner scripts.
- Remote smoke test for both baseline and exp110 runner after CLI or runner cleanup.
- Remote full exp110 run when the user asks to verify current active trial metrics.

### 7. Wrong vs Correct

Wrong:

```text
Recreate `run_assist09_history_output_alignment_trial.sh` for a quick old probe without documenting why the route is active again.
```

Correct:

```text
Keep the active script surface small: baseline runner plus exp110 trial runner.
```

Wrong:

```text
Treat checkpoint-average, hybrid stacker, or a single-seed exp111 ablation result as the default training runner.
```

Correct:

```text
Report those as evaluator/diagnostic routes and keep experiment 110 as the default single-run pure-CDM trial until an explicit promotion changes the runner contract.
```

## Current CDM Model Surface Contract

### 1. Scope / Trigger

- Trigger: changes touch `DecoupledCDM`, `DecoupledCDMEnsemble`, the retained evaluation loaders, or focused CDM unit tests.
- Current HEAD is only required to preserve the official exp81 baseline path and the current exp110 pure-CDM trial path.
- Constructor knobs and helper branches that no longer have an active `scripts/train.py` CLI or runner are not part of the supported surface.

### 2. Contracts

- Keep baseline model paths: graph propagation, high-concept adapter, pairwise history interaction adapter, GS difficulty adapter, interpretable readout expert adapter, student-conditioned UKC readout residual, and concept evidence readout residual.
- Keep exp110 model/training paths: dual CDM ensemble, equal-weight tower averaging, branch BCE supervision, concept evidence prior residual, history evidence logit prior residual with `loss_only`, cognitive alignment support, and `recompute_minibatch` training.
- `DecoupledCDMEnsemble` uses a fixed equal-weight average across towers; there is no active `secondary_weight` runtime knob.
- Removed probe-only model routes should stay out of active code and active tests unless a new task explicitly restores them with updated docs and runner semantics.

### 3. Bad Cases

- Reintroduce deleted constructor flags because an old experiment detail doc still mentions them.
- Keep evaluation/model loading shims for rejected probes after the active CLI and runner surface has already dropped them.
- Treat historical reproduction as a reason to preserve every retired branch in current HEAD.

## Student-Subset Propagation Training Mode

### 1. Scope / Trigger

- Trigger: `scripts/train.py --training-mode student_recompute_minibatch`.
- Purpose: runtime optimization for large student-concept grids such as Junyi,
  where interaction minibatches repeatedly recompute dense all-student
  propagation.

### 2. Signatures

```bash
python scripts/train.py \
  --training-mode student_recompute_minibatch \
  --student-batch-size <positive-int> \
  [normal train.py model/data flags]
```

### 3. Contracts

- `student_recompute_minibatch` groups optimizer steps by student IDs, then
  trains on all interactions belonging to the selected student chunk.
- Model forward may use `use_student_subset=True` to compute propagation states
  only for the target students in that step.
- Default `full_batch` and `recompute_minibatch` contracts remain unchanged.
- Result JSON and summary CSV must record `student_batch_size` whenever the
  field exists, so this training mode is not confused with exp110's original
  interaction-minibatch protocol.

### 4. Validation & Error Matrix

- `training_mode=student_recompute_minibatch` without
  `--student-batch-size` -> `ValueError`.
- `training_mode=student_recompute_minibatch` with `--batch-size` ->
  `ValueError`.
- Non-positive `--student-batch-size` -> `ValueError`.
- `full_batch` with `--student-batch-size` -> `ValueError`.
- `recompute_minibatch` with `--student-batch-size` -> `ValueError`.

### 5. Good/Base/Bad Cases

- Good: Junyi smoke or exploratory run records
  `training_mode=student_recompute_minibatch` and `student_batch_size=N`.
- Base: exp110 original ASSIST09 low-memory protocol remains
  `training_mode=recompute_minibatch`, `batch_size=65536`.
- Bad: reporting a `student_recompute_minibatch` run as if it were exp110's
  original interaction-minibatch protocol.

### 6. Tests Required

- Unit test that subset propagation predictions match full propagation for the
  same target interactions.
- Unit test that dual-tower subset forwarding matches full forwarding.
- Training-mode validation tests for required and incompatible flags.
- Script-level smoke that writes a summary with `student_batch_size`.

### 7. Wrong vs Correct

Wrong:

```text
Use student_recompute_minibatch for Junyi and omit the mode from the experiment
record because the model architecture is unchanged.
```

Correct:

```text
Record student_recompute_minibatch as a training-mode/runtime optimization,
including student_batch_size, and compare it separately from exp110's original
recompute_minibatch protocol.
```

---

## Experiment Design Rules

- New hypotheses should control one structural factor first.
- The project is in a low-marginal-gain phase; ordinary small residuals or sidecars are not the default path to the AUC sprint target.
- Default priority is representation-level change: student-state formation, target-conditioned history, constrained graph/Q structure learning, or explicitly labeled hybrid side channels.
- Small changes are useful when they admit or reject a larger hypothesis; they are not a complete default strategy by themselves.
- Allow a small number of already-supported orthogonal combinations, but avoid combination explosion.
- For readout, `q_repr`, target-conditioned history, or student-state changes that may be suppressed by experiment 51's full-trigger readout expert, the first experiment design should include both `B49 seed=2024` and current `Exp70 seed=2024`.
- Run a new structure once first; expand to 2-3 more seeds only when the first run is worth continuing.
- Local improvements should usually reach about `1e-3` in `AUC` or `ACC` before expansion.
- For large structures aimed at `test_auc ~= 0.780`, a single seed should usually show around `AUC +0.002` before expansion, unless slice evidence is unusually strong.

---

## Experiment Ledger Rules

Experiment evidence remains in `docs/`, not in Trellis specs:

- `docs/model_improvement_plan.md`: current snapshot, route decisions, candidates, default next step.
- `docs/experiment_index.jsonl`: machine-readable experiment index.
- `docs/experiments/`: detail docs with commands, result paths, seed metrics, and slice evidence.
- `docs/handoff.md`: current mainline status, durable judgments, branch priorities, and key files.
- `docs/experiment_themes.md`: cross-experiment diagnoses.
- `docs/archive_legacy_experiments.md`: compressed legacy history.

When an experiment finishes:

- Always update `docs/experiment_index.jsonl`.
- Update `docs/model_improvement_plan.md` when route, priority, or repeat-risk changes.
- Add or update a detail doc when seed metrics, diagnostics, commands, or result paths matter.
- Update `docs/handoff.md` only when the current mainline, default priority, or recommended branch changes.
- Keep long metrics in detail docs; keep ledger summaries compressed.
- If any of the ledger files above were updated on a non-`exp/trellis-trial` branch and the new content changes future-session judgment, cherry-pick or otherwise sync those doc-only updates onto `exp/trellis-trial` before considering the experiment wrapped up.
