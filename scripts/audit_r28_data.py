from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.q_matrix import normalize_concept_sequence
from scripts.evaluate_coverage_slice import add_target_coverage
from utils import compute_metrics, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit r28 data, target slices and external validation rows.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--external-root", default=None)
    parser.add_argument("--registry", default="configs/r28_registry.json")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--output", default="results/r28/data_audit.json")
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_hashes(frame: pd.DataFrame) -> set[int]:
    normalized = frame.fillna("<NA>").astype(str)
    return set(int(value) for value in pd.util.hash_pandas_object(normalized, index=False))


def scope_frame(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope.startswith("bucket:"):
        return frame[frame["coverage_bucket"] == scope.split(":", 1)[1]]
    if scope == "low_coverage":
        return frame[frame["coverage_group"] == "low_coverage"]
    raise ValueError(f"Unknown target scope {scope!r}.")


def verify_q_alignment(frames: dict[str, pd.DataFrame], q_matrix: pd.DataFrame) -> dict[str, int]:
    q_by_exercise: dict[str, set[str]] = {}
    for row in q_matrix[["exer_id", "cpt_seq"]].drop_duplicates().itertuples(index=False):
        q_by_exercise.setdefault(str(row.exer_id), set()).update(normalize_concept_sequence(row.cpt_seq))
    missing = 0
    mismatched = 0
    for frame in frames.values():
        for row in frame[["exer_id", "cpt_seq"]].drop_duplicates().itertuples(index=False):
            expected = q_by_exercise.get(str(row.exer_id))
            actual = set(normalize_concept_sequence(row.cpt_seq))
            if expected is None:
                missing += 1
            elif not actual or not actual.issubset(expected):
                mismatched += 1
    if missing or mismatched:
        raise RuntimeError(f"Q alignment failed: missing={missing}, mismatched={mismatched}")
    concepts = set().union(*q_by_exercise.values()) if q_by_exercise else set()
    return {"exercises": len(q_by_exercise), "concepts": len(concepts)}


def read_variant(
    *,
    root: Path,
    directory: str,
    expected: dict[str, str],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    files: dict[str, Any] = {}
    for filename in ("train.csv", "valid.csv", "test.csv", "Q_matrix.csv"):
        path = root / directory / filename
        if not path.exists():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != expected[filename]:
            raise RuntimeError(f"Fingerprint mismatch: {path}")
        frame = pd.read_csv(path)
        key = filename.removesuffix(".csv").lower()
        frames[key] = frame
        files[filename] = {"path": str(path), "sha256": digest, "rows": len(frame)}
    for split in ("train", "valid", "test"):
        required = {"stu_id", "exer_id", "cpt_seq", "label"}
        if not required.issubset(frames[split].columns):
            raise RuntimeError(f"{split} is missing {sorted(required - set(frames[split].columns))}")
        labels = set(pd.to_numeric(frames[split]["label"], errors="raise").unique())
        if labels != {0, 1}:
            raise RuntimeError(f"{split} does not contain both binary labels: {labels}")
    hashes = {split: row_hashes(frames[split]) for split in ("train", "valid", "test")}
    overlap = {
        "train_valid": len(hashes["train"] & hashes["valid"]),
        "train_test": len(hashes["train"] & hashes["test"]),
        "valid_test": len(hashes["valid"] & hashes["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Exact target/history overlap: {overlap}")
    q_alignment = verify_q_alignment(
        {split: frames[split] for split in ("train", "valid", "test")},
        frames["q_matrix"],
    )
    return frames, {"files": files, "exact_row_overlap": overlap, "q_alignment": q_alignment}


def verify_external(
    *,
    comparator: dict[str, Any],
    external_root: Path,
    validation: pd.DataFrame,
    target: pd.DataFrame | None,
) -> dict[str, Any]:
    path = external_root / comparator["validation_prediction_path"]
    if not path.exists():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest != comparator["validation_prediction_sha256"]:
        raise RuntimeError(f"External validation fingerprint mismatch: {path}")
    predictions = pd.read_csv(path)
    if len(predictions) != len(validation) or len(predictions) != int(comparator["validation_rows"]):
        raise RuntimeError(f"External validation row count mismatch: {path}")
    labels = pd.to_numeric(predictions["label"], errors="raise").to_numpy()
    if not (labels == validation["label"].to_numpy()).all():
        raise RuntimeError(f"External validation labels are not row-aligned: {path}")
    probs = pd.to_numeric(predictions["prob"], errors="raise").to_numpy()
    overall_auc = float(compute_metrics(labels, probs)["auc"])
    if abs(overall_auc - float(comparator["validation_auc"])) > 5e-5:
        raise RuntimeError(f"External validation AUC mismatch: {path}")
    result: dict[str, Any] = {
        "path": str(path),
        "sha256": digest,
        "rows": len(predictions),
        "labels_row_aligned": True,
        "auc": overall_auc,
    }
    if target is not None:
        target_auc = float(compute_metrics(labels[target.index], probs[target.index])["auc"])
        if abs(target_auc - float(comparator["validation_target_auc"])) > 5e-5:
            raise RuntimeError(f"External validation target AUC mismatch: {path}")
        result["target_auc"] = target_auc
    return result


def audit_dataset(
    *,
    dataset: str,
    spec: dict[str, Any],
    data_root: Path,
    external_root: Path | None,
) -> dict[str, Any]:
    standard, standard_report = read_variant(
        root=data_root,
        directory=spec["standard_dir"],
        expected=spec["fingerprints"]["standard"],
    )
    holdout, holdout_report = read_variant(
        root=data_root,
        directory=spec["holdout_dir"],
        expected=spec["fingerprints"]["holdout"],
    )
    targets: dict[str, Any] = {}
    target_frames: dict[str, pd.DataFrame] = {}
    for split in ("valid", "test"):
        enriched = add_target_coverage(
            holdout[split],
            train_frame=holdout["train"],
            q_matrix=holdout["q_matrix"],
        )
        target = scope_frame(enriched, spec["target_scope"])
        labels = set(pd.to_numeric(target["label"], errors="raise").unique())
        if len(target) < 100 or labels != {0, 1}:
            raise RuntimeError(f"Invalid {dataset}/{split} target: rows={len(target)}, labels={labels}")
        targets[split] = {
            "rows": int(len(target)),
            "labels": sorted(int(label) for label in labels),
            "positive_rate": float(target["label"].mean()),
        }
        target_frames[split] = target

    external: dict[str, Any] = {"verified": False}
    if external_root is not None:
        external = {
            "verified": True,
            "standard": verify_external(
                comparator=spec["standard_external"],
                external_root=external_root,
                validation=standard["valid"],
                target=None,
            ),
            "holdout": verify_external(
                comparator=spec["holdout_external"],
                external_root=external_root,
                validation=holdout["valid"],
                target=target_frames["valid"],
            ),
        }
    return {
        "dataset": dataset,
        "status": "passed",
        "data_version": spec["data_version"],
        "standard": standard_report,
        "holdout": holdout_report,
        "target_scope": spec["target_scope"],
        "targets": targets,
        "external_validation": external,
    }


def main() -> None:
    args = parse_args()
    registry = load_json(args.registry)
    requested = args.dataset or registry["active_pool"]
    reports = []
    for dataset in requested:
        spec = registry["datasets"][dataset]
        if spec.get("status") != "active":
            raise ValueError(f"Dataset {dataset!r} is not active.")
        reports.append(
            audit_dataset(
                dataset=dataset,
                spec=spec,
                data_root=Path(args.data_root),
                external_root=Path(args.external_root) if args.external_root else None,
            )
        )
    payload = {"schema_version": 1, "seed": 42, "reports": reports}
    write_json(payload, args.output)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
