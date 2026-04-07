#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV_NAME="decoupled_cd"
LOCAL_PROJECT_ROOT="/home/jameschiang/work/decoupled_cd"

if [[ "${PROJECT_ROOT}" != "${LOCAL_PROJECT_ROOT}" ]]; then
  echo "scripts/enter_env.sh is for the local workspace only." >&2
  echo "Current project root: ${PROJECT_ROOT}" >&2
  echo "Use this script from ${LOCAL_PROJECT_ROOT}; on remote hosts, sync code first, activate ${CONDA_ENV_NAME}, and run project commands directly." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found in PATH" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

conda activate "${CONDA_ENV_NAME}"
cd "${PROJECT_ROOT}"

if [[ $# -gt 0 ]]; then
  exec "$@"
fi

exec "${SHELL:-bash}" -i
