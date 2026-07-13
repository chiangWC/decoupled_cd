from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


DATASET_LAYOUTS = {
    "assist_17": {"standard": "assist_17", "holdout": "assist_17_chold_v2"},
    "moocradar": {"standard": "moocradar", "holdout": "moocradar_chold_v2"},
    "xes3g5m": {"standard": "xes3g5m", "holdout": "xes3g5m_chold_v2"},
}
DATASET_FLAGS = {
    "assist_17": [
        "--v2-mastery-aux-weight", "1.0",
        "--v2-consistency-weight", "0.5",
        "--v2-consistency-adaptive",
    ],
    "moocradar": [
        "--v2-mastery-aux-weight", "1.0",
        "--v2-consistency-weight", "0.0",
        "--concept-dim", "256",
    ],
    "xes3g5m": [
        "--v2-mastery-aux-weight", "1.0",
        "--v2-consistency-weight", "0.0",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the missing raw/static-state + pooled-NCF factorial cell.")
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_LAYOUTS), required=True)
    parser.add_argument("--dataset-variant", choices=["standard", "holdout"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    legacy_root = Path(args.legacy_root).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    valid_path = data_root / DATASET_LAYOUTS[args.dataset][args.dataset_variant] / "valid.csv"
    if not valid_path.exists():
        raise FileNotFoundError(valid_path)
    train_script = legacy_root / "scripts/train.py"
    coverage_script = legacy_root / "scripts/evaluate_coverage_slice.py"
    if not train_script.exists() or not coverage_script.exists():
        raise FileNotFoundError("Legacy r27 training/evaluation scripts are unavailable.")

    summary_path = output_dir / "summary.json"
    command = [
        sys.executable,
        str(train_script),
        "--dataset", args.dataset,
        "--dataset-variant", args.dataset_variant,
        "--data-root", str(data_root),
        # The legacy trainer always evaluates a test_bundle. Point it to valid
        # so this diagnostic never reads a new test row.
        "--test-interactions", str(valid_path),
        "--seed", "42",
        "--model", "v2",
        "--v2-evidence-mode", "raw_concept",
        "--v2-hybrid-readout",
        "--v2-monotonic-readout",
        "--v2-hybrid-prediction-mode", "pooled_ncf_only",
        "--output", str(summary_path),
        "--log-dir", str(output_dir / "logs"),
        "--device", args.device,
        *DATASET_FLAGS[args.dataset],
    ]
    if args.gpus:
        command.extend(["--gpus", args.gpus])
    print(" ".join(command), flush=True)
    if args.dry_run:
        return
    subprocess.run(command, cwd=legacy_root, check=True)

    coverage_command = [
        sys.executable,
        str(coverage_script),
        "--dataset-name", args.dataset,
        "--summary", str(summary_path),
        "--model-name", "r28_raw_static_pooled_factorial",
        "--split", "valid",
        "--output", str(output_dir / "coverage.json"),
        "--slice-csv", str(output_dir / "coverage.csv"),
        "--device", args.device,
    ]
    if args.gpus:
        coverage_command.extend(["--gpus", args.gpus])
    subprocess.run(coverage_command, cwd=legacy_root, check=True)

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["r28_factorial_diagnostic"] = {
        "cell": "raw_static_state__pooled_ncf",
        "test_rows_read": False,
        "legacy_test_argument_redirected_to": str(valid_path),
        "seed": 42,
    }
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
