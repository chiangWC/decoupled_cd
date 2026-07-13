from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_r28 import load_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select strongest row-aligned external S/H/T axes.")
    parser.add_argument("--registry", default="configs/r29_registry.json")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--model", action="append", choices=["orcdf", "svgcd"])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def verify_summary(path: Path) -> dict[str, Any]:
    summary = load_json(path)
    if summary["seed"] != 42:
        raise RuntimeError(f"External result is not seed42: {path}")
    for split in ("valid", "test"):
        payload = summary["splits"][split]
        prediction = Path(payload["prediction_path"])
        if not payload.get("row_aligned") or not prediction.exists():
            raise RuntimeError(f"External result is not row-aligned: {path}/{split}")
        if sha256_file(prediction) != payload["prediction_sha256"]:
            raise RuntimeError(f"External prediction fingerprint changed: {prediction}")
    return summary


def axis_payload(summary: dict[str, Any], *, target: bool) -> dict[str, Any]:
    def portable(path: str) -> str:
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(PROJECT_ROOT.resolve()))
        except ValueError:
            return str(resolved)

    valid = summary["splits"]["valid"]
    test = summary["splits"]["test"]
    valid_metric = valid["target_metrics"] if target else valid["metrics"]
    test_metric = test["target_metrics"] if target else test["metrics"]
    return {
        "model": summary["external_model"],
        "seed": 42,
        "validation_auc": float(valid_metric["auc"]),
        "test_auc": float(test_metric["auc"]),
        "validation_target_auc": float(valid_metric["auc"]) if target else None,
        "test_target_auc": float(test_metric["auc"]) if target else None,
        "validation_prediction_path": portable(valid["prediction_path"]),
        "validation_prediction_sha256": valid["prediction_sha256"],
        "validation_rows": valid["rows"],
        "test_prediction_path": portable(test["prediction_path"]),
        "test_prediction_sha256": test["prediction_sha256"],
        "test_rows": test["rows"],
        "checkpoint": portable(summary["checkpoint"]),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "source_hashes": summary["external_source"]["source_hashes"],
    }


def main() -> None:
    args = parse_args()
    registry = load_json(args.registry)
    models = args.model or ["orcdf", "svgcd"]
    root = Path(args.results_root)
    decisions: dict[str, Any] = {}
    for dataset in args.dataset:
        candidates: dict[str, dict[str, dict[str, Any]]] = {}
        for model in models:
            candidates[model] = {
                variant: verify_summary(root / model / dataset / variant / "summary.json")
                for variant in ("standard", "holdout")
            }
        standard_winner = max(
            models,
            key=lambda model: candidates[model]["standard"]["splits"]["valid"]["metrics"]["auc"],
        )
        holdout_winner = max(
            models,
            key=lambda model: candidates[model]["holdout"]["splits"]["valid"]["metrics"]["auc"],
        )
        target_winner = max(
            models,
            key=lambda model: candidates[model]["holdout"]["splits"]["valid"]["target_metrics"]["auc"],
        )
        axes = {
            "S": axis_payload(candidates[standard_winner]["standard"], target=False),
            "H": axis_payload(candidates[holdout_winner]["holdout"], target=False),
            "T": axis_payload(candidates[target_winner]["holdout"], target=True),
        }
        registry["datasets"][dataset]["external_axes"] = axes
        registry["datasets"][dataset]["status"] = "active"
        if dataset not in registry["active_pool"]:
            registry["active_pool"].append(dataset)
        registry["provisional_pool"] = [
            value for value in registry["provisional_pool"] if value != dataset
        ]
        decisions[dataset] = {
            "S": {"model": standard_winner, "validation_auc": axes["S"]["validation_auc"]},
            "H": {"model": holdout_winner, "validation_auc": axes["H"]["validation_auc"]},
            "T": {"model": target_winner, "validation_auc": axes["T"]["validation_auc"]},
        }
    payload = {"schema_version": 2, "decisions": decisions, "registry": registry}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(decisions, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
