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

---

## Current Pseudo-Mainline Contract

The current Trellis pseudo-mainline is experiment 78 on top of experiment 76 and experiment 70 as tracked by `docs/model_improvement_plan.md` and `docs/handoff.md`. The accepted `master` model-mainline may still lag behind this worktree state. Preserve this protocol unless the active task explicitly changes it:

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
- `concept_evidence_readout_residual` is enabled with:
  - `concept_evidence_readout_min_count = 1`
  - `concept_evidence_readout_min_seen_ratio = 1.0`
  - `concept_evidence_readout_max_logit = 0.5`
- `concept_evidence_prior_residual` is enabled with:
  - `concept_evidence_prior_min_count = 1`
  - `concept_evidence_prior_min_seen_ratio = 1.0`
  - `concept_evidence_prior_max_logit = 0.5`
  - `concept_evidence_prior_strength = 2.0`
  - `concept_evidence_prior_confidence_cap = 20.0`
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
