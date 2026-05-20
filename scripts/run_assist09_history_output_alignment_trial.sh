#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DEFAULT="results/pure_cdm_default_promotion/seed${SEED:-2027}_history_output_alignment_w001_300ep.json"

OUTPUT="${OUTPUT:-${OUTPUT_DEFAULT}}" \
  bash "${SCRIPT_DIR}/run_assist09_history_alignment_trial.sh" \
    --history-evidence-output-alignment-weight "${HISTORY_OUTPUT_ALIGNMENT_WEIGHT:-0.01}" \
    --history-evidence-output-alignment-confidence-power "${HISTORY_OUTPUT_ALIGNMENT_CONFIDENCE_POWER:-0.0}" \
    --history-evidence-output-alignment-confidence-cap "${HISTORY_OUTPUT_ALIGNMENT_CONFIDENCE_CAP:-20.0}" \
    --history-evidence-output-alignment-confidence-floor "${HISTORY_OUTPUT_ALIGNMENT_CONFIDENCE_FLOOR:-0.0}" \
    "$@"
