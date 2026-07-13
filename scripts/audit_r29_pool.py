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
from data.r29_protocol import exact_row_overlap, group_overlap, sha256_file
from scripts.evaluate_coverage_slice import add_target_coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit prepared r29 pool datasets before external admission.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--registry", default="configs/r29_registry.json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _row_identity_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for value in frame["source_row_id"].astype(str):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_variant(directory: Path, expected: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    files: dict[str, Any] = {}
    for name in ("train.csv", "valid.csv", "test.csv", "Q_matrix.csv"):
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        expected_digest = expected["files"][name]["sha256"]
        if digest != expected_digest:
            raise RuntimeError(f"Fingerprint mismatch for {path}: {digest} != {expected_digest}")
        frame = pd.read_csv(path, dtype={"stu_id": str, "exer_id": str, "cpt_seq": str})
        frames[name.removesuffix(".csv")] = frame
        files[name] = {"sha256": digest, "rows": len(frame)}
    splits = {name: frames[name] for name in ("train", "valid", "test")}
    row_overlap = exact_row_overlap(splits)
    atomic_overlap = group_overlap(splits)
    if any(row_overlap.values()):
        raise RuntimeError(f"Exact rows cross split boundaries: {row_overlap}")
    q_by_exercise = {
        str(row.exer_id): set(normalize_concept_sequence(row.cpt_seq))
        for row in frames["Q_matrix"].itertuples(index=False)
    }
    missing_q = 0
    mismatched_q = 0
    for frame in splits.values():
        for row in frame[["exer_id", "cpt_seq"]].drop_duplicates().itertuples(index=False):
            expected_concepts = q_by_exercise.get(str(row.exer_id))
            concepts = set(normalize_concept_sequence(row.cpt_seq))
            if expected_concepts is None:
                missing_q += 1
            elif concepts != expected_concepts:
                mismatched_q += 1
    if missing_q or mismatched_q:
        raise RuntimeError(f"Q alignment failed: missing={missing_q}, mismatched={mismatched_q}")
    return frames, {
        "directory": str(directory),
        "files": files,
        "row_identity_hashes": {
            split: _row_identity_hash(frames[split]) for split in ("train", "valid", "test")
        },
        "exact_row_overlap": row_overlap,
        "student_exercise_group_overlap": atomic_overlap,
        "q_alignment": {
            "exercises": len(q_by_exercise),
            "concepts": len(set().union(*q_by_exercise.values())),
            "missing": missing_q,
            "mismatched": mismatched_q,
        },
    }


def _scope_report(frame: pd.DataFrame, *, train: pd.DataFrame, q_matrix: pd.DataFrame) -> dict[str, Any]:
    enriched = add_target_coverage(frame, train_frame=train, q_matrix=q_matrix)
    reports: dict[str, Any] = {}
    scopes = {
        "bucket:zero": enriched[enriched["coverage_bucket"] == "zero"],
        "low_coverage": enriched[enriched["coverage_group"] == "low_coverage"],
    }
    for name, target in scopes.items():
        labels = sorted(pd.to_numeric(target["label"], errors="raise").unique().tolist())
        reports[name] = {
            "rows": len(target),
            "labels": labels,
            "positive_rate": float(target["label"].mean()) if len(target) else None,
            "eligible": len(target) >= 100 and labels == [0, 1],
        }
    return reports


def audit_dataset(
    dataset: str,
    *,
    spec: dict[str, Any],
    manifest: dict[str, Any],
    data_root: Path,
) -> dict[str, Any]:
    dataset_manifest = manifest["datasets"][dataset]
    standard, standard_report = _read_variant(
        data_root / spec["standard_dir"], dataset_manifest["standard"]
    )
    holdout, holdout_report = _read_variant(
        data_root / spec["holdout_dir"], dataset_manifest["holdout"]
    )
    if dataset == "ednet" and any(standard_report["student_exercise_group_overlap"].values()):
        raise RuntimeError("EdNet standard split is not atomic by student-exercise group.")
    if dataset == "ednet" and any(holdout_report["student_exercise_group_overlap"].values()):
        raise RuntimeError("EdNet holdout split is not atomic by student-exercise group.")
    scopes = {
        split: _scope_report(
            holdout[split], train=holdout["train"], q_matrix=holdout["Q_matrix"]
        )
        for split in ("valid", "test")
    }
    exact_zero_eligible = all(scopes[split]["bucket:zero"]["eligible"] for split in ("valid", "test"))
    low_eligible = all(scopes[split]["low_coverage"]["eligible"] for split in ("valid", "test"))
    if dataset == "junyi":
        target_scope = "bucket:zero"
        if not exact_zero_eligible:
            raise RuntimeError("Junyi exact-zero target is not eligible.")
    elif exact_zero_eligible:
        target_scope = "bucket:zero"
    elif low_eligible:
        target_scope = "low_coverage"
    else:
        raise RuntimeError("EdNet has neither an eligible exact-zero nor low-coverage target.")
    return {
        "dataset": dataset,
        "data_status": "passed",
        "admission_status": "external_reproduction_pending",
        "target_scope": target_scope,
        "standard": standard_report,
        "holdout": holdout_report,
        "target_slices": scopes,
        "manifest_metadata": {
            key: value
            for key, value in dataset_manifest.items()
            if key not in {"standard", "holdout"}
        },
    }


def main() -> None:
    args = parse_args()
    registry = load_json(args.registry)
    manifest = load_json(args.manifest)
    if registry["model_seed"] != 42 or registry["split_seed"] != 2024:
        raise RuntimeError("r29 registry seed invariants changed.")
    reports = [
        audit_dataset(
            dataset,
            spec=registry["datasets"][dataset],
            manifest=manifest,
            data_root=Path(args.data_root),
        )
        for dataset in ("junyi", "ednet")
    ]
    payload = {"schema_version": 2, "model_seed": 42, "split_seed": 2024, "reports": reports}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
