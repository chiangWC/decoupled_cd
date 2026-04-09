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

TARGET_BRANCH="$(git branch --show-current)"
if [[ -z "${TARGET_BRANCH}" ]]; then
  echo "Failed to determine the current local branch. Detached HEAD is not supported for remote execution." >&2
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "Git remote 'origin' is not configured." >&2
  exit 1
fi

LOCAL_HEAD="$(git rev-parse HEAD)"
REMOTE_HEAD="$(git ls-remote origin "refs/heads/${TARGET_BRANCH}" | awk '{print $1}')"
if [[ -z "${REMOTE_HEAD}" ]]; then
  echo "Remote branch '${TARGET_BRANCH}' does not exist on origin. Push this branch before remote execution." >&2
  exit 1
fi

if [[ "${LOCAL_HEAD}" != "${REMOTE_HEAD}" ]]; then
  echo "Remote branch '${TARGET_BRANCH}' is not up to date with local HEAD. Push the current branch before remote execution." >&2
  exit 1
fi

printf -v remote_user_cmd '%q ' "$@"
printf -v target_branch_quoted '%q' "${TARGET_BRANCH}"
remote_script="source '${REMOTE_CONDA_SH}' && conda activate '${CONDA_ENV_NAME}' && cd '${REMOTE_PROJECT_ROOT}' && if [[ -n \"\$(git status --porcelain)\" ]]; then echo \"Remote working tree is not clean. Clean tracked changes before switching branches or running commands.\" >&2; exit 1; fi && if ! git show-ref --verify --quiet refs/heads/${target_branch_quoted}; then echo \"Remote branch '${TARGET_BRANCH}' does not exist. Deploy this branch before remote execution.\" >&2; exit 1; fi && git switch --quiet ${target_branch_quoted} && ${remote_user_cmd}"
printf -v remote_script_quoted '%q' "${remote_script}"

ssh "${REMOTE_HOST}" "bash -lc ${remote_script_quoted}"
