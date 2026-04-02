#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/xph/jwc/research/decoupled_cd"
CONDA_ENV_NAME="decoupled_cd"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found in PATH" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate "${CONDA_ENV_NAME}"
cd "${PROJECT_ROOT}"

if [[ -f "${HOME}/.bashrc" ]]; then
  # Allow user-defined wrappers such as `mycodex` from interactive shell config.
  source "${HOME}/.bashrc"
fi

if command -v mycodex >/dev/null 2>&1; then
  exec mycodex "$@"
fi

if command -v codex >/dev/null 2>&1; then
  exec codex "$@"
fi

echo "Neither 'mycodex' nor 'codex' was found in PATH" >&2
exit 1
