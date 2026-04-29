#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

for arg in "$@"; do
  case "${arg}" in
    --graph-mode|--graph-mode=*|--prerequisite-graph|--prerequisite-graph=*|--similarity-graph|--similarity-graph=*)
      echo "run_assist09_baseline.sh locks the official baseline to single-graph propagation; dual-graph overrides are not allowed." >&2
      exit 1
      ;;
  esac
done

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "Neither python nor python3 was found in PATH." >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/train.py \
  --dataset assist_09 \
  --graph-mode single \
  --epochs 300 \
  --learning-rate 1e-3 \
  --concept-dim 64 \
  --gs-mode conditional \
  --high-concept-logit-adapter \
  --high-concept-logit-min-count 2 \
  --pairwise-history-interaction-adapter \
  --pairwise-history-interaction-min-count 2 \
  --gs-difficulty-adapter \
  --seed "${SEED:-2024}" \
  --output "${OUTPUT:-results/assist_09_tkc_dual_channel_300ep.json}" \
  "$@"
