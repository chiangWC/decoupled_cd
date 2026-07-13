from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_r28 import load_json
from utils import write_json
from utils.r29_evaluation import classify_external_win


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the non-contribution r29 marginal anchor over the live pool."
    )
    parser.add_argument(
        "--result-root",
        action="append",
        required=True,
        metavar="DATASET=PATH",
        help="Campaign root containing DATASET__{standard,holdout}__marginal_anchor__none.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def parse_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected DATASET=PATH, got {value!r}")
        dataset, path = value.split("=", 1)
        if not dataset or dataset in roots:
            raise ValueError(f"Missing or duplicate dataset in {value!r}")
        roots[dataset] = Path(path)
    return roots


def summary(root: Path, dataset: str, variant: str) -> dict[str, Any]:
    path = root / f"{dataset}__{variant}__marginal_anchor__none" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = load_json(path)
    if payload["state_completer"] != "marginal_anchor" or payload["seed"] != 42:
        raise RuntimeError(f"Unexpected anchor result: {path}")
    if payload["evaluation_stage"] != "validation":
        raise RuntimeError(f"Anchor report accepts validation results only: {path}")
    return payload


def metric_fields(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    return {
        f"{prefix}_{name}": float(metrics[name])
        for name in ("auc", "acc", "rmse", "brier", "ece")
    }


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for dataset, root in parse_roots(args.result_root).items():
        standard = summary(root, dataset, "standard")
        holdout = summary(root, dataset, "holdout")
        if standard["recipe_sha256"] != holdout["recipe_sha256"]:
            raise RuntimeError(f"{dataset} standard/holdout recipes differ.")
        if standard["architecture_fingerprint"] != holdout["architecture_fingerprint"]:
            raise RuntimeError(f"{dataset} standard/holdout topologies differ.")
        fingerprints.add(standard["architecture_fingerprint"])
        margins = {
            "S": float(standard["overall_external_margin"]),
            "H": float(holdout["overall_external_margin"]),
            "T": float(holdout["target_external_margin"]),
        }
        ordinary, strict = classify_external_win(margins)
        doa = holdout.get("doa") or {}
        rows.append(
            {
                "dataset": dataset,
                **metric_fields("S", standard["metrics"]),
                **metric_fields("H", holdout["metrics"]),
                **metric_fields("T", holdout["target_metrics"]),
                **{f"margin_{axis}": value for axis, value in margins.items()},
                "ordinary_win": ordinary,
                "strict_win": strict,
                "doa": doa.get("doa"),
                "doa_weighted": doa.get("doa_weighted"),
                "doa_ci_low": doa.get("doa_ci_low"),
                "doa_ci_high": doa.get("doa_ci_high"),
                "doa_spearman": doa.get("doa_spearman"),
                "architecture_fingerprint": standard["architecture_fingerprint"],
                "standard_prediction_sha256": standard["prediction_sha256"],
                "holdout_prediction_sha256": holdout["prediction_sha256"],
            }
        )
    if len(fingerprints) != 1:
        raise RuntimeError(f"Anchor results use multiple fingerprints: {sorted(fingerprints)}")
    rows.sort(key=lambda row: row["dataset"])
    payload = {
        "schema_version": 1,
        "stage": "validation",
        "seed": 42,
        "model": "r29_completion/marginal_anchor",
        "paper_module": False,
        "architecture_fingerprint": next(iter(fingerprints)),
        "ordinary_wins": [row["dataset"] for row in rows if row["ordinary_win"]],
        "strict_wins": [row["dataset"] for row in rows if row["strict_win"]],
        "rows": rows,
    }
    output = Path(args.output)
    write_json(payload, output)
    pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
