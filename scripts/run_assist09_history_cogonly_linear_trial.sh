#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DEFAULT="results/pure_cdm_default_promotion/seed${SEED:-2027}_history_alignment_cogonly_linear_300ep.json"

OUTPUT="${OUTPUT:-${OUTPUT_DEFAULT}}" \
  bash "${SCRIPT_DIR}/run_assist09_history_alignment_trial.sh" \
    --history-evidence-linear-readout \
    --history-evidence-linear-feature-set cogonly \
    --history-evidence-linear-min-count 1 \
    --history-evidence-linear-min-seen-ratio 0.0 \
    --history-evidence-linear-max-logit "${HISTORY_LINEAR_MAX_LOGIT:-0.25}" \
    "$@"
