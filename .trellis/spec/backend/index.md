# Backend Development Guidelines

> Project conventions for the Python research pipeline in this repository.

---

## Overview

This repository is a Python/PyTorch research codebase for decoupled cognitive diagnosis experiments. It is not a web service backend: there are no API routes, ORM models, migrations, or long-running server processes.

Trellis owns the operational constraints. Use `.trellis/workflow.md` for task flow and [Experiment Protocol](./experiment-protocol.md) for branch, remote-run, mainline, and ledger rules.

---

## Pre-Development Checklist

Before modifying code in this repository:

1. Read [Experiment Protocol](./experiment-protocol.md).
2. Run `git status --short --branch` and confirm the current branch matches the intended work.
3. For experiment code, start from current `master` unless the user explicitly asks to continue an existing experiment branch.
4. Search before changing constants, CLI flags, defaults, data paths, model switches, or metric names.
5. Read the relevant files from this directory:
   - [Directory Structure](./directory-structure.md)
   - [Database / Data Artifacts](./database-guidelines.md)
   - [Error Handling](./error-handling.md)
   - [Experiment Protocol](./experiment-protocol.md)
   - [Logging](./logging-guidelines.md)
   - [Quality](./quality-guidelines.md)
6. If the change touches repeated patterns or cross-layer data flow, also read `.trellis/spec/guides/index.md`.

---

## Guidelines Index

| Guide | Description | Status |
|-------|-------------|--------|
| [Directory Structure](./directory-structure.md) | Python package layout and where new code belongs | Active |
| [Database / Data Artifacts](./database-guidelines.md) | CSV, tensor, result, and experiment artifact conventions | Active |
| [Error Handling](./error-handling.md) | Fail-fast validation and exception patterns | Active |
| [Experiment Protocol](./experiment-protocol.md) | Branch, remote execution, mainline, and experiment ledger rules | Active |
| [Logging](./logging-guidelines.md) | Runtime logging conventions for scripts and training | Active |
| [Quality](./quality-guidelines.md) | Tests, experiment protocol, and forbidden patterns | Active |

---

## Quality Check

Before considering backend work complete:

1. Verify all touched constants and flags are reflected across CLI, defaults, run scripts, summaries, and docs where applicable.
2. Add or update focused `unittest` tests for behavior changes.
3. Do not run project tests locally by default. Commit and push, then use `bash scripts/remote_exec.sh <command>` for tests, smoke tests, training, and evaluation.
4. For experiment conclusions, update the ledger files described in [Experiment Protocol](./experiment-protocol.md).
