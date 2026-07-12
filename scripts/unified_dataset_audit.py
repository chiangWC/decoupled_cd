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
ZERO_COUNT_MINIMUM = 1000
ZERO_LABEL_COUNT_MINIMUM = 100


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


def _data_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        encoded_name = path.name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
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
) -> tuple[dict[str, Any], set[str], set[str]]:
    counts = {
        "validation_rows": 0,
        "exact_zero_validation_rows": 0,
        "partial_unseen_validation_rows": 0,
        "rows_with_unseen_target_concepts": 0,
        "unseen_target_concepts": 0,
        "empty_target_rows": 0,
        "zero_count": 0,
        "zero_positive_count": 0,
        "zero_negative_count": 0,
    }
    exercises: set[str] = set()
    concepts: set[str] = set()
    prediction_order: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, {"stu_id", "exer_id", "cpt_seq", "label"})
        for row in reader:
            counts["validation_rows"] += 1
            target = _concepts(row["cpt_seq"])
            seen = history.get(row["stu_id"].strip(), set())
            exercises.add(row["exer_id"].strip())
            concepts.update(target)
            prediction_order.append((row["stu_id"].strip(), row["exer_id"].strip()))
            if not target:
                counts["empty_target_rows"] += 1
                continue
            unseen = target - seen
            if len(unseen) == len(target):
                counts["exact_zero_validation_rows"] += 1
                counts["zero_count"] += 1
                try:
                    label = float(row["label"])
                except (TypeError, ValueError) as error:
                    raise ValueError("validation label is not numeric") from error
                if label == 1.0:
                    counts["zero_positive_count"] += 1
                elif label == 0.0:
                    counts["zero_negative_count"] += 1
                else:
                    raise ValueError("validation labels must be binary")
            elif unseen:
                counts["partial_unseen_validation_rows"] += 1
            if unseen:
                counts["rows_with_unseen_target_concepts"] += 1
                counts["unseen_target_concepts"] += len(unseen)
    counts["prediction_order_sha256"] = canonical_sha256(prediction_order)
    return counts, exercises, concepts


def canonical_split_source_hashes(
    *, train_path: Path, valid_path: Path, q_path: Path
) -> dict[str, str]:
    """Recompute the source bindings stored in a dataset split audit."""
    history, _, _ = _training_history(train_path)
    coverage, _, _ = _validation_coverage(valid_path, history)
    return {
        "data_sha256": _data_sha256((train_path, valid_path)),
        "q_sha256": _file_sha256(q_path),
        "prediction_order_sha256": str(coverage["prediction_order_sha256"]),
    }


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
        **canonical_split_source_hashes(
            train_path=path / "train.csv",
            valid_path=path / "valid.csv",
            q_path=q_path,
        ),
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
            "zero_count": holdout["zero_count"],
            "zero_positive_count": holdout["zero_positive_count"],
            "zero_negative_count": holdout["zero_negative_count"],
            "data_sha256": holdout["data_sha256"],
            "q_sha256": holdout["q_sha256"],
            "prediction_order_sha256": holdout["prediction_order_sha256"],
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
        for count_name, minimum in (
            ("zero_count", ZERO_COUNT_MINIMUM),
            ("zero_positive_count", ZERO_LABEL_COUNT_MINIMUM),
            ("zero_negative_count", ZERO_LABEL_COUNT_MINIMUM),
        ):
            if split[count_name] < minimum:
                record["reasons"].append(
                    f"{split_name} {count_name} below eligibility threshold"
                )
    record["eligible"] = bool(
        record["asset_ready"]
        and all(
            split["zero_count"] >= ZERO_COUNT_MINIMUM
            and split["zero_positive_count"] >= ZERO_LABEL_COUNT_MINIMUM
            and split["zero_negative_count"] >= ZERO_LABEL_COUNT_MINIMUM
            for split in (standard, holdout)
        )
    )
    if record["eligible"]:
        record["status"] = "eligible"
    elif all(
        split["zero_count"] == 0 and split["partial_unseen_validation_rows"] > 0
        for split in (standard, holdout)
    ):
        record["status"] = "partial-only"
    else:
        record["status"] = "ineligible"
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
