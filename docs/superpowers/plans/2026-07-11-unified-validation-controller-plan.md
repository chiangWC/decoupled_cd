# Unified Validation Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace stateless self-hashed authorization with a route-bound, controller-owned, one-time capability state machine whose progress comes from immutable outer campaign proofs.

**Architecture:** A new `scripts/unified_validation_controller.py` owns exclusive initialization, fsynced JSON state, one flock, issued/consumed capability registries, and proof validation. `scripts/run_unified_validation.py` exposes initialization/authorization CLI commands and calls capability consumption as the first `run-split` action. Progress is derived only from consumed records and their fixed outer attempt artifacts.

**Tech Stack:** Python 3.12 standard library (`fcntl`, `hashlib`, `json`, `os`, `secrets`, `subprocess`, `pathlib`), existing Unified V2 manifests/cohort schema, `unittest`.

## Global Constraints

- Work only in the remote `complete_model` worktree; do not push or launch experiments.
- Read no real test data and accept no test path/metric in controller proofs.
- Every transition uses one controller flock plus fsynced exclusive-create or atomic replace.
- Exact current route HEAD must equal the commit registered at initialization.
- Authorization covers only the next ordered cohort dataset pair.
- Dataset success is the conjunction of standard/holdout overall non-regression, weighted DOA non-regression, strict zero-AUC improvement, and strict ordinary-DOA improvement.
- Final global success additionally requires one zero-AUC delta of at least `0.001`.

---

### Task 1: Controller initialization and immutable registered inputs

**Files:**
- Create: `scripts/unified_validation_controller.py`
- Create: `tests/test_unified_validation_controller.py`
- Modify: `scripts/run_unified_validation.py`

**Interfaces:**
- Produces: `initialize_controller(state_dir: Path, repo_root: Path, cohort_path: Path, manifest_path: Path, architecture: str, baseline_rows_path: Path) -> dict[str, object]`.
- Produces: `controller-init` CLI with the same inputs.

- [ ] **Step 1: Write failing tests** for exclusive initialization, route HEAD binding, copied cohort/manifest/baseline artifacts, ordered dataset state, and baseline hash/schema rejection.
- [ ] **Step 2: Run controller tests and verify RED** from the missing module/CLI.
- [ ] **Step 3: Implement fsync helpers and initialization** using atomic `os.mkdir`, `open("x")`, file/parent fsync, exact `git rev-parse HEAD`, verified cohort loading, exact manifest validation, and baseline row validation.
- [ ] **Step 4: Run controller and existing cohort tests; verify GREEN.**

### Task 2: Next-pair issuance and exact one-time consumption

**Files:**
- Modify: `scripts/unified_validation_controller.py`
- Modify: `scripts/run_unified_validation.py`
- Modify: `tests/test_unified_validation_controller.py`
- Modify: `tests/test_unified_validation.py`

**Interfaces:**
- Produces: `authorize_next(state_dir: Path, repo_root: Path, output_path: Path) -> dict[str, object]`.
- Produces: `consume_split_capability(state_dir: Path, repo_root: Path, token_path: Path, dataset_id: str, split_id: str, data_root: Path, raw_output: Path, attempt_dir: Path) -> dict[str, object]`.
- Token fields: controller ID, route commit, monotonic counter, one dataset ID, and exact standard/holdout payloads with independent random nonces; no authentication hash.

- [ ] **Step 1: Write failing tests** proving only dataset 0 is issued, fabricated/recomputed-hash tokens fail registry lookup, standard/holdout consume independently once, reuse fails, stale counters fail, and route mismatch fails.
- [ ] **Step 2: Run each security regression and verify the expected RED failure.**
- [ ] **Step 3: Implement locked issuance** with exact `issued/` registry records and atomic state update.
- [ ] **Step 4: Implement locked consumption** that verifies registry payload and route/counter, records attempt/output/data paths, fsyncs, then atomically renames `issued/` to `consumed/`.
- [ ] **Step 5: Make `_run_split` call consumption first** and replace old stateless authorization inputs with `--controller-state-dir/--repo-root/--capability`.
- [ ] **Step 6: Run focused tests and verify GREEN.**

### Task 3: Immutable outer proof advancement and stop gates

**Files:**
- Modify: `scripts/unified_validation_controller.py`
- Modify: `scripts/run_unified_validation.py`
- Modify: `tests/test_unified_validation_controller.py`

**Interfaces:**
- `authorize_next` advances an active pair only after both capabilities are consumed and their fixed attempt artifacts validate.
- Proof records contain consumed identity, status/summary hashes, exact dataset manifests, metrics/deltas, joint success, and route/cohort/manifest/capability bindings.

- [ ] **Step 1: Write proof fixture helpers** that create standard/holdout summaries and completed outer statuses with exact path/size/SHA records.
- [ ] **Step 2: Write and verify RED tests** for fake decisions, wrong attempt directory, output hash mismatch, route/cohort/manifest/data mismatch, invalid seed/test routing, stale tokens, ordered issuance, and fail-closed reachability.
- [ ] **Step 3: Implement proof loading from consumed records only** and reject caller proof/decision paths.
- [ ] **Step 4: Validate outer status closure** against route, immutable cohort/manifest, expected dataset paths/current hashes, and summary output path/size/SHA.
- [ ] **Step 5: Validate summary capability/schema bindings**, compare the registered baseline row, compute the five-condition joint gate, write a proof, and update progress atomically.
- [ ] **Step 6: Issue only the next pair or finish without issuance**, enforcing reachability and the final `+0.001` global gate.
- [ ] **Step 7: Run controller tests and verify GREEN.**

### Task 4: Remove misleading authorization and update evidence boundaries

**Files:**
- Modify: `scripts/run_unified_validation.py`
- Modify: `tests/test_unified_validation.py`
- Modify: `docs/unified_v2_validation_log.md`
- Modify: `.superpowers/sdd/task-8-fix-report.md`

**Interfaces:**
- Removes: `authorization_sha256`, `--decision`, `--remaining`, `--allow`, and self-authentication wording.
- Split summaries add controller ID, counter, nonce, route commit, cohort SHA, and architecture manifest/fingerprint.

- [ ] **Step 1: Write/adjust runner tests** for consumption before output/work/GPU/data side effects and bound summary schema.
- [ ] **Step 2: Verify RED against any remaining stateless path.**
- [ ] **Step 3: Delete stateless authorization and update runner summary bindings.**
- [ ] **Step 4: Update ledger/report** with registry-backed wording and unchanged exploratory classifications.
- [ ] **Step 5: Run focused tests and verify GREEN.**

### Task 5: Verification and delivery

**Files:**
- Modify: `.superpowers/sdd/task-8-fix-report.md`

- [ ] **Step 1: Run syntax and whitespace checks.**

```bash
python -m compileall scripts/unified_validation_controller.py scripts/unified_cohort.py scripts/run_unified_validation.py tests/test_unified_validation_controller.py tests/test_unified_cohort.py tests/test_unified_validation.py
git diff --check
```

- [ ] **Step 2: Run focused tests.**

```bash
python -m unittest tests.test_unified_validation_controller tests.test_unified_cohort tests.test_unified_validation -v
```

- [ ] **Step 3: Run the full suite.**

```bash
python -m unittest discover -s tests -v
```

- [ ] **Step 4: Update fix report** with RED evidence, exact commands/counts, state guarantees, and no experiment/test-data access.
- [ ] **Step 5: Commit with `chiangWC` identity**, verify exact HEAD/clean tree, rerun focused tests post-commit, and report hashes without pushing.
