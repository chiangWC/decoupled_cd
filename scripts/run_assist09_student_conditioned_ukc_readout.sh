#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

OUTPUT="${OUTPUT:-results/student_conditioned_ukc_readout/assist_09_seed${SEED:-2024}_300ep.json}" \
  bash "${SCRIPT_DIR}/run_assist09_baseline.sh" \
    --student-conditioned-ukc-readout-residual \
    "$@"
