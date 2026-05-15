#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

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
  --interpretable-readout-expert-adapter \
  --interpretable-readout-expert-count 3 \
  --student-conditioned-ukc-readout-residual \
  --concept-evidence-readout-residual \
  --concept-evidence-readout-min-count "${CONCEPT_EVIDENCE_READOUT_MIN_COUNT:-2}" \
  --concept-evidence-readout-min-seen-ratio "${CONCEPT_EVIDENCE_READOUT_MIN_SEEN_RATIO:-1.0}" \
  --concept-evidence-readout-max-logit "${CONCEPT_EVIDENCE_READOUT_MAX_LOGIT:-0.5}" \
  --seed "${SEED:-2024}" \
  --output "${OUTPUT:-results/concept_evidence_readout/assist_09_seed2024_min2_seen1_max05_300ep.json}" \
  "$@"
