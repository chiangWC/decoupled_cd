#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_HOST="xph-pc"
REMOTE_PROJECT_ROOT="/home/xph/jwc/research/decoupled_cd"
REMOTE_CONDA_SH="/home/xph/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="decoupled_cd"

if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/remote_exec.sh <command> [args...]" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"

printf -v remote_user_cmd '%q ' "$@"
remote_script="source '${REMOTE_CONDA_SH}' && conda activate '${CONDA_ENV_NAME}' && cd '${REMOTE_PROJECT_ROOT}' && ${remote_user_cmd}"
printf -v remote_script_quoted '%q' "${remote_script}"

ssh "${REMOTE_HOST}" "bash -lc ${remote_script_quoted}"
