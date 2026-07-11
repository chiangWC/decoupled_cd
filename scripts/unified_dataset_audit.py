from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DATASET_LAYOUTS: dict[str, tuple[str, str]] = {
    "ASSIST09": ("assist_09", "assist_09_chold_v2"),
    "ASSIST17": ("assist_17", "assist_17_chold_v2"),
    "NIPS34": ("nips34_clean", "nips34_chold_v2"),
    "MOOCRadar": ("moocradar", "moocradar_chold_v2"),
    "XES3G5M": ("xes3g5m", "xes3g5m_chold_v2"),
    "Junyi": ("junyi", "junyi_chold_v2"),
    "EdNet-ICDM": ("ednet_icdm", "ednet_icdm_chold_v2"),
}

PROVISIONAL_DATASET_IDS = frozenset({"Junyi", "EdNet-ICDM"})
SPLIT_FILES = ("train.csv", "valid.csv", "test.csv", "Q_matrix.csv")


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _concepts(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def _require_columns(fieldnames: Iterable[str] | None, required: set[str]) -> None:
    available = set(fieldnames or ())
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"missing CSV columns: {', '.join(missing)}")


def _q_domain(path: Path) -> tuple[set[str], set[str]]:
    exercises: set[str] = set()
    concepts: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, {"exer_id", "cpt_seq"})
        for row in reader:
            exercises.add(row["exer_id"].strip())
            concepts.update(_concepts(row["cpt_seq"]))
    if not exercises or not concepts:
        raise ValueError("Q_matrix.csv has an empty ID domain")
    return exercises, concepts


def _training_history(
    path: Path,
) -> tuple[dict[str, set[str]], set[str], set[str]]:
    history: dict[str, set[str]] = {}
    exercises: set[str] = set()
    concepts: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, {"stu_id", "exer_id", "cpt_seq"})
        for row in reader:
            student_id = row["stu_id"].strip()
            target = _concepts(row["cpt_seq"])
            history.setdefault(student_id, set()).update(target)
            exercises.add(row["exer_id"].strip())
            concepts.update(target)
    return history, exercises, concepts


def _validation_coverage(
    path: Path,
    history: dict[str, set[str]],
) -> tuple[dict[str, int], set[str], set[str]]:
    counts = {
        "validation_rows": 0,
        "exact_zero_validation_rows": 0,
        "partial_unseen_validation_rows": 0,
        "rows_with_unseen_target_concepts": 0,
        "unseen_target_concepts": 0,
        "empty_target_rows": 0,
    }
    exercises: set[str] = set()
    concepts: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, {"stu_id", "exer_id", "cpt_seq"})
        for row in reader:
            counts["validation_rows"] += 1
            target = _concepts(row["cpt_seq"])
            seen = history.get(row["stu_id"].strip(), set())
            exercises.add(row["exer_id"].strip())
            concepts.update(target)
            if not target:
                counts["empty_target_rows"] += 1
                continue
            unseen = target - seen
            if len(unseen) == len(target):
                counts["exact_zero_validation_rows"] += 1
            elif unseen:
                counts["partial_unseen_validation_rows"] += 1
            if unseen:
                counts["rows_with_unseen_target_concepts"] += 1
                counts["unseen_target_concepts"] += len(unseen)
    return counts, exercises, concepts


def _split_record(path: Path) -> tuple[dict[str, Any], tuple[set[str], set[str]]]:
    q_path = path / "Q_matrix.csv"
    q_exercises, q_concepts = _q_domain(q_path)
    history, train_exercises, train_concepts = _training_history(path / "train.csv")
    coverage, valid_exercises, valid_concepts = _validation_coverage(
        path / "valid.csv", history
    )
    data_exercises = train_exercises | valid_exercises
    data_concepts = train_concepts | valid_concepts
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "train_path": str((path / "train.csv").resolve()),
        "valid_path": str((path / "valid.csv").resolve()),
        "test_path": str((path / "test.csv").resolve()),
        "q_path": str(q_path.resolve()),
        "q_sha256": _file_sha256(q_path),
        "test_sha256": _file_sha256(path / "test.csv"),
        "test_content_hash_only": True,
        "exercise_id_count": len(q_exercises),
        "concept_id_count": len(q_concepts),
        "data_ids_within_q_domain": (
            data_exercises <= q_exercises and data_concepts <= q_concepts
        ),
        **coverage,
    }
    return record, (q_exercises, q_concepts)


def _missing_files(root: Path, standard_name: str, holdout_name: str) -> list[str]:
    missing: list[str] = []
    for split_name, directory_name in (
        ("standard", standard_name),
        ("holdout", holdout_name),
    ):
        for filename in SPLIT_FILES:
            if not (root / directory_name / filename).is_file():
                missing.append(f"{split_name}/{filename}")
    if not (root / holdout_name / "split_summary.json").is_file():
        missing.append("holdout/split_summary.json")
    return missing


def audit_dataset(root: Path, dataset_id: str) -> dict[str, Any]:
    if dataset_id not in DATASET_LAYOUTS:
        raise KeyError(f"unknown dataset ID: {dataset_id}")
    standard_name, holdout_name = DATASET_LAYOUTS[dataset_id]
    standard_path = root / standard_name
    holdout_path = root / holdout_name
    missing = _missing_files(root, standard_name, holdout_name)
    record: dict[str, Any] = {
        "dataset_id": dataset_id,
        "standard_path": str(standard_path.resolve()),
        "holdout_path": str(holdout_path.resolve()),
        "test_content_hash_only": True,
        "missing_files": missing,
        "assets_present": not missing,
        "asset_ready": False,
        "id_domains_match": False,
        "q_hashes_match": False,
        "eligible": False,
        "status": "provisional" if dataset_id in PROVISIONAL_DATASET_IDS else "ineligible",
        "reasons": [],
    }
    if missing:
        record["reasons"].append("missing required assets")
        record["audit_sha256"] = canonical_sha256(record)
        return record

    try:
        standard, standard_domain = _split_record(standard_path)
        holdout, holdout_domain = _split_record(holdout_path)
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        record["reasons"].append(f"invalid training/validation asset: {error}")
        record["audit_sha256"] = canonical_sha256(record)
        return record

    summary_path = holdout_path / "split_summary.json"
    record.update(
        {
            "standard": standard,
            "holdout": holdout,
            "split_summary_path": str(summary_path.resolve()),
            "split_summary_sha256": _file_sha256(summary_path),
            "id_domains_match": standard_domain == holdout_domain,
            "q_hashes_match": standard["q_sha256"] == holdout["q_sha256"],
        }
    )
    record["asset_ready"] = bool(
        record["assets_present"]
        and record["id_domains_match"]
        and record["q_hashes_match"]
        and standard["data_ids_within_q_domain"]
        and holdout["data_ids_within_q_domain"]
    )
    if not record["id_domains_match"]:
        record["reasons"].append("standard and holdout Q ID domains differ")
    if not record["q_hashes_match"]:
        record["reasons"].append("standard and holdout Q hashes differ")
    if not standard["data_ids_within_q_domain"]:
        record["reasons"].append("standard train/validation IDs exceed the Q domain")
    if not holdout["data_ids_within_q_domain"]:
        record["reasons"].append("holdout train/validation IDs exceed the Q domain")
    for split_name, split in (("standard", standard), ("holdout", holdout)):
        if split["exact_zero_validation_rows"] == 0:
            record["reasons"].append(
                f"{split_name} validation has no exact-zero coverage rows"
            )
    record["eligible"] = bool(
        record["asset_ready"]
        and standard["exact_zero_validation_rows"] > 0
        and holdout["exact_zero_validation_rows"] > 0
    )
    record["status"] = "eligible" if record["eligible"] else "ineligible"
    record["audit_sha256"] = canonical_sha256(record)
    return record


def audit_pool(root: str | Path) -> dict[str, Any]:
    resolved_root = Path(root).resolve()
    datasets = {
        dataset_id: audit_dataset(resolved_root, dataset_id)
        for dataset_id in DATASET_LAYOUTS
    }
    audit: dict[str, Any] = {
        "schema_version": 1,
        "root": str(resolved_root),
        "test_content_hash_only": True,
        "coverage_definition": {
            "exact_zero_validation_rows": (
                "target_coverage == 0: every target concept is absent from the "
                "student's training history"
            ),
            "rows_with_unseen_target_concepts": (
                "at least one target concept is absent from the student's training history"
            ),
            "partial_unseen_validation_rows": (
                "some but not all target concepts are absent from the student's "
                "training history"
            ),
        },
        "datasets": datasets,
        "asset_ready_dataset_ids": [
            dataset_id for dataset_id, record in datasets.items() if record["asset_ready"]
        ],
        "eligible_dataset_ids": [
            dataset_id for dataset_id, record in datasets.items() if record["eligible"]
        ],
        "provisional_dataset_ids": [
            dataset_id
            for dataset_id, record in datasets.items()
            if record["status"] == "provisional"
        ],
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Unified V2 dataset eligibility without reading test labels."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit = audit_pool(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
