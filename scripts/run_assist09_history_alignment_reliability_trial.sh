#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DEFAULT="results/pure_cdm_runner/seed${SEED:-2027}_history_alignment_cogonly_confpow${CONFIDENCE_POWER:-1p0}_w005_300ep.json"

OUTPUT="${OUTPUT:-${OUTPUT_DEFAULT}}" \
  bash "${SCRIPT_DIR}/run_assist09_history_alignment_trial.sh" \
    --history-evidence-cognitive-alignment-confidence-power "${CONFIDENCE_POWER:-1.0}" \
    --history-evidence-cognitive-alignment-confidence-cap "${CONFIDENCE_CAP:-20.0}" \
    --history-evidence-cognitive-alignment-confidence-floor "${CONFIDENCE_FLOOR:-0.2}" \
    "$@"
