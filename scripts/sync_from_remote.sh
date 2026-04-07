#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE_HOST="xph-pc"
REMOTE_PATH="~/jwc/research/decoupled_cd/"

rsync -avz \
  --exclude '__pycache__/' \
  --exclude '.venv/' \
  --exclude 'logs/' \
  --exclude 'results/' \
  "$@" \
  "${REMOTE_HOST}:${REMOTE_PATH}" \
  "${PROJECT_ROOT}/"
