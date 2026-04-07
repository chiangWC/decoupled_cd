#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -n "${SEEDS:-}" ]]; then
  read -r -a seed_list <<< "${SEEDS}"
else
  seed_list=(2024 2025 2026)
fi

for seed in "${seed_list[@]}"; do
  output_path="${OUTPUT_DIR:-results/multiseed}/assist_09_tkc_ukc_separate_seed${seed}_300ep.json"
  echo "Running ASSIST09 official baseline with seed=${seed}"
  bash "${SCRIPT_DIR}/run_assist09_baseline.sh" \
    --seed "${seed}" \
    --output "${output_path}" \
    "$@"
done
