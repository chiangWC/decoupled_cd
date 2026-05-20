#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DEFAULT="results/pure_cdm_runner/seed${SEED:-2024}_history_alignment_cogonly_fusion_300ep.json"

OUTPUT="${OUTPUT:-${OUTPUT_DEFAULT}}" \
  bash "${SCRIPT_DIR}/run_assist09_baseline.sh" \
    --history-evidence-logit-prior-residual \
    --history-evidence-logit-prior-location loss_only \
    --history-evidence-logit-prior-min-count 1 \
    --history-evidence-logit-prior-min-seen-ratio 0.0 \
    --history-evidence-logit-prior-max-logit 4.0 \
    --history-evidence-logit-prior-component-cap 4.0 \
    --history-evidence-logit-prior-weight-student 0.0 \
    --history-evidence-logit-prior-weight-exercise 0.0 \
    --history-evidence-logit-prior-weight-target-concept 0.44 \
    --history-evidence-logit-prior-weight-concept 0.22 \
    --history-evidence-logit-prior-weight-mastery 0.22 \
    --history-evidence-cognitive-alignment-weight 0.05 \
    --history-evidence-fusion-readout \
    --history-evidence-fusion-feature-set cogonly \
    --history-evidence-fusion-min-count 1 \
    --history-evidence-fusion-min-seen-ratio 0.0 \
    --history-evidence-fusion-max-logit 0.35 \
    "$@"
