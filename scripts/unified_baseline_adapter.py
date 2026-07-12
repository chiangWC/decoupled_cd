from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_coverage_slice import add_target_coverage, compute_slice_rows
from scripts.unified_dataset_audit import (
    canonical_sha256,
    canonical_split_source_hashes,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _verify_canonical(payload: Mapping[str, Any], label: str) -> None:
    stored = payload.get("audit_sha256")
    unhashed = dict(payload)
    unhashed.pop("audit_sha256", None)
    if stored != canonical_sha256(unhashed):
        raise ValueError(f"{label} canonical SHA-256 mismatch")


def validate_job_manifest(
    manifest: Mapping[str, Any], *, valid_path: Path
) -> None:
    if (
        manifest.get("seed") != 42
        or manifest.get("split_seed") != 2024
        or manifest.get("evaluation_role") != "validation_alias"
    ):
        raise ValueError("job manifest must declare seed42 validation_alias protocol")
    try:
        declared_valid = Path(str(manifest["valid_file"])).resolve()
        declared_test = Path(str(manifest["test_file"])).resolve()
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        raise ValueError("job manifest validation paths are invalid") from error
    if declared_valid != valid_path.resolve() or declared_test != valid_path.resolve():
        raise ValueError("validation_alias requires test_file to equal audited valid_file")


def _write_bound_predictions(
    *, valid_path: Path, prediction_path: Path, output_path: Path
) -> pd.DataFrame:
    valid = pd.read_csv(valid_path).reset_index(drop=True)
    predictions = pd.read_csv(prediction_path).reset_index(drop=True)
    required_prediction_fields = {"prob", "label"}
    if not required_prediction_fields.issubset(predictions.columns):
        raise ValueError("prediction artifact must contain prob,label columns")
    if len(predictions) != len(valid):
        raise ValueError("prediction row count does not match audited valid order")
    expected_labels = valid["label"].astype(int).to_numpy()
    actual_labels = predictions["label"].round().astype(int).to_numpy()
    if not (expected_labels == actual_labels).all():
        raise ValueError("prediction labels do not match audited valid order")
    probabilities = predictions["prob"].astype(float)
    if not probabilities.map(math.isfinite).all():
        raise ValueError("prediction probabilities must be finite")
    bound = valid.copy()
    bound["label"] = actual_labels
    bound["prob"] = probabilities.to_numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", newline="", encoding="utf-8") as handle:
        bound.to_csv(handle, index=False)
    return bound


def finalize_baseline_rows(
    *,
    dataset_id: str,
    split_id: str,
    model: str,
    dataset_audit_path: Path,
    manifest_path: Path,
    prediction_path: Path,
    checkpoint_path: Path,
    bound_prediction_path: Path,
    output_path: Path,
) -> list[dict[str, Any]]:
    audit = _load_json(dataset_audit_path)
    if not isinstance(audit, Mapping):
        raise ValueError("dataset audit must be a JSON object")
    _verify_canonical(audit, "dataset audit")
    datasets = audit.get("datasets")
    record = datasets.get(dataset_id) if isinstance(datasets, Mapping) else None
    if not isinstance(record, Mapping) or record.get("eligible") is not True:
        raise ValueError(f"dataset is not eligible: {dataset_id}")
    _verify_canonical(record, f"dataset record {dataset_id}")
    split = record.get(split_id)
    if not isinstance(split, Mapping):
        raise ValueError(f"dataset split audit is missing: {dataset_id}/{split_id}")
    train_path = Path(str(split["train_path"])).resolve()
    valid_path = Path(str(split["valid_path"])).resolve()
    q_path = Path(str(split["q_path"])).resolve()
    actual_hashes = canonical_split_source_hashes(
        train_path=train_path,
        valid_path=valid_path,
        q_path=q_path,
    )
    for field, actual in actual_hashes.items():
        if split.get(field) != actual:
            raise ValueError(
                f"audited {dataset_id}/{split_id} {field} SHA-256 mismatch"
            )
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("job manifest must be a JSON object")
    validate_job_manifest(manifest, valid_path=valid_path)
    if (
        manifest.get("dataset_id") != dataset_id
        or manifest.get("split") != split_id
        or manifest.get("model") != model
        or Path(str(manifest.get("train_file"))).resolve() != train_path
    ):
        raise ValueError("job manifest identity does not match requested audit row")
    if not checkpoint_path.is_file():
        raise ValueError("checkpoint artifact does not exist")

    bound = _write_bound_predictions(
        valid_path=valid_path,
        prediction_path=prediction_path,
        output_path=bound_prediction_path,
    )
    q_matrix = pd.read_csv(q_path)
    enriched = add_target_coverage(
        bound,
        train_frame=pd.read_csv(train_path),
        q_matrix=q_matrix,
    )
    slices = compute_slice_rows(
        enriched, dataset_name=dataset_id, model_name=model
    )
    by_scope = {row["scope"]: row for row in slices}
    try:
        metrics = [
            ("auc", float(by_scope["overall"]["auc"])),
            ("zero_auc", float(by_scope["bucket:zero"]["auc"])),
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("validation predictions have no finite overall/zero AUC") from error
    if not all(math.isfinite(value) for _, value in metrics):
        raise ValueError("validation predictions have non-finite overall/zero AUC")

    common = {
        "dataset_id": dataset_id,
        "model": model,
        "seed": 42,
        "split_seed": 2024,
        "split": split_id,
        "data_sha256": split["data_sha256"],
        "q_sha256": split["q_sha256"],
        "prediction_sha256": _file_sha256(bound_prediction_path),
        "prediction_order_sha256": split["prediction_order_sha256"],
        "config_sha256": _file_sha256(manifest_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "source_path": str(output_path.resolve()),
        "evaluation_role": "validation_alias",
        "valid_path": str(valid_path),
        "bound_prediction_path": str(bound_prediction_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "config_path": str(manifest_path.resolve()),
    }
    rows = [{**common, "metric": metric, "value": value} for metric, value in metrics]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return rows


def _write_manifest(args: argparse.Namespace) -> None:
    configuration: dict[str, str] = {}
    for item in args.config:
        key, separator, value = item.partition("=")
        if not separator or not key or key in configuration:
            raise ValueError("--config must be repeated as unique key=value entries")
        configuration[key] = value
    payload = {
        "schema_version": 1,
        "model": args.model,
        "dataset_id": args.dataset_id,
        "split": args.split_id,
        "seed": 42,
        "split_seed": 2024,
        "evaluation_role": "validation_alias",
        "train_file": str(args.train_file.resolve()),
        "valid_file": str(args.valid_file.resolve()),
        "test_file": str(args.valid_file.resolve()),
        "configuration": configuration,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _finalize(args: argparse.Namespace) -> None:
    finalize_baseline_rows(
        dataset_id=args.dataset_id,
        split_id=args.split_id,
        model=args.model,
        dataset_audit_path=args.dataset_audit,
        manifest_path=args.manifest,
        prediction_path=args.predictions,
        checkpoint_path=args.checkpoint,
        bound_prediction_path=args.bound_predictions,
        output_path=args.output,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bind validation-only baseline provenance.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--model", required=True)
    manifest.add_argument("--dataset-id", required=True)
    manifest.add_argument("--split-id", choices=("standard", "holdout"), required=True)
    manifest.add_argument("--train-file", type=Path, required=True)
    manifest.add_argument("--valid-file", type=Path, required=True)
    manifest.add_argument("--config", action="append", default=[])
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(handler=_write_manifest)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--model", required=True)
    finalize.add_argument("--dataset-id", required=True)
    finalize.add_argument("--split-id", choices=("standard", "holdout"), required=True)
    finalize.add_argument("--dataset-audit", type=Path, required=True)
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--predictions", type=Path, required=True)
    finalize.add_argument("--checkpoint", type=Path, required=True)
    finalize.add_argument("--bound-predictions", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.set_defaults(handler=_finalize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
