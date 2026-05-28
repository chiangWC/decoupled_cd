#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTPUT_DEFAULT="results/pure_cdm_default_promotion/seed${SEED:-2024}_exp110_dual64x80_branchbce018_recompute65536_lr3e4_300ep.json"

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
    --history-evidence-cognitive-alignment-final-weight 0.0881 \
    --history-evidence-cognitive-alignment-anneal-start-epoch 170 \
    --history-evidence-cognitive-alignment-anneal-end-epoch 230 \
    --concept-evidence-prior-residual \
    --concept-evidence-prior-min-count 1 \
    --concept-evidence-prior-min-seen-ratio 1.0 \
    --concept-evidence-prior-max-logit 0.30 \
    --concept-evidence-prior-min-confidence 0.75 \
    --concept-evidence-prior-min-abs-mastery 0.5 \
    --concept-evidence-prior-apply-mode train_only \
    --concept-evidence-prior-train-start-epoch 135 \
    --dual-cdm-ensemble \
    --dual-cdm-secondary-concept-dim 80 \
    --dual-cdm-branch-bce-weight 0.18 \
    --training-mode recompute_minibatch \
    --batch-size 65536 \
    --learning-rate 0.0003 \
    "$@"
