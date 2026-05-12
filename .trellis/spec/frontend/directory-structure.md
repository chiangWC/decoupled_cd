# Frontend Directory Structure

> Not applicable to the current repository.

---

## Current State

There is no frontend directory, no `src/` browser app, no `package.json`, and no frontend build system in this repository.

Current top-level code directories are Python research modules: `configs/`, `data/`, `models/`, `trainers/`, `utils/`, `scripts/`, and `tests/`.

---

## Rule

Do not create frontend directories or assets unless a task explicitly asks for a web UI, dashboard, report viewer, or other browser-facing feature.

If such a task is introduced later, create a new task-specific technical design first and then update this file with the actual chosen layout.
