#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Local working tree is not clean. Commit or stash changes before deploying." >&2
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "Git remote 'origin' is not configured." >&2
  exit 1
fi

TARGET_BRANCH="$(git ls-remote --symref origin HEAD | awk '/^ref:/ {sub("refs/heads/", "", $2); print $2; exit}')"
if [[ -z "${TARGET_BRANCH}" ]]; then
  echo "Failed to determine the default branch of remote 'origin'." >&2
  exit 1
fi

git push origin "HEAD:refs/heads/${TARGET_BRANCH}"
