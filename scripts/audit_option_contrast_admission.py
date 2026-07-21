from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pool_protocol import sha256_file
from scripts.evaluate_coverage_slice import add_target_coverage


FROZEN_MIN_OPTION_COVERAGE = 0.995
FROZEN_MIN_VALID_RANGE = 1.0
FROZEN_MIN_LABEL_CONSISTENCY = 0.999
FROZEN_MIN_TRAIN_CELL_SUPPORT = 5
FROZEN_MIN_SUPPORTED_TRAIN_FRACTION = 0.90
FROZEN_MIN_SUPPORTED_WRONG_ITEMS = 100
FROZEN_MIN_TARGET_ROWS = 500
FROZEN_MIN_TARGET_STUDENTS = 100
FROZEN_MIN_TARGET_CLASS_ROWS = 100
FROZEN_MIN_HISTORY_ROWS = 10
FROZEN_MIN_TARGET_HISTORY_FRACTION = 0.90
FROZEN_MIN_INCORRECT_HISTORY_ROWS = 3
FROZEN_MIN_TARGET_INCORRECT_HISTORY_FRACTION = 0.80
FROZEN_REQUIRED_DATASETS = 3
DATASETS = ("nips34", "ednet", "enem")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit option-aware protocols before predictive experiments."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "source_row_id": str,
            "stu_id": str,
            "exer_id": str,
            "cpt_seq": str,
        },
    )


def row_multiset(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(str(value) for value in row)
        for row in frame.loc[:, columns].itertuples(index=False, name=None)
    )


def split_partition_report(
    frames: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    columns = ("stu_id", "exer_id", "cpt_seq", "label")
    data_rows = row_multiset(frames["data"], columns)
    split_rows = {
        name: row_multiset(frames[name], columns)
        for name in ("train", "valid", "test")
    }
    overlap_rows = {}
    group_sets = {
        name: set(
            zip(
                frames[name]["stu_id"].astype(str),
                frames[name]["exer_id"].astype(str),
                strict=True,
            )
        )
        for name in ("train", "valid", "test")
    }
    group_overlap = {}
    for left, right in (
        ("train", "valid"),
        ("train", "test"),
        ("valid", "test"),
    ):
        overlap_rows[f"{left}_{right}"] = int(
            sum((split_rows[left] & split_rows[right]).values())
        )
        group_overlap[f"{left}_{right}"] = int(
            len(group_sets[left] & group_sets[right])
        )
    combined = split_rows["train"] + split_rows["valid"] + split_rows["test"]
    return {
        "data_rows": int(sum(data_rows.values())),
        "split_rows": {
            name: int(sum(values.values()))
            for name, values in split_rows.items()
        },
        "overlap_rows": overlap_rows,
        "group_overlap": group_overlap,
        "mutually_disjoint": not any(overlap_rows.values()),
        "student_item_atomic": not any(group_overlap.values()),
        "exhaustive_exact": combined == data_rows,
    }


def verify_train_option_boundary(
    *,
    train: pd.DataFrame,
    train_options: pd.DataFrame,
    audit_options: pd.DataFrame,
) -> dict[str, Any]:
    keys = (
        ["source_row_id"]
        if all(
            "source_row_id" in frame.columns
            for frame in (train, train_options, audit_options)
        )
        else ["stu_id", "exer_id"]
    )
    if train.duplicated(keys).any():
        raise RuntimeError(f"Training interactions are not unique on {keys}.")
    if train_options.duplicated(keys).any():
        raise RuntimeError(f"Training option sidecar is not unique on {keys}.")
    if audit_options.duplicated(keys).any():
        raise RuntimeError(f"Audit option sidecar is not unique on {keys}.")
    train_keys = row_multiset(train, tuple(keys))
    option_keys = row_multiset(train_options, tuple(keys))
    if train_keys != option_keys:
        raise RuntimeError(
            "Training option sidecar is not an exact key match for training rows."
        )
    option_columns = ("selected_option", "correct_option", "option_count")
    comparison = train_options.loc[:, [*keys, *option_columns]].merge(
        audit_options.loc[:, [*keys, *option_columns]],
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("_train", "_audit"),
        indicator=True,
    )
    if not bool((comparison["_merge"] == "both").all()):
        raise RuntimeError("Training option rows are absent from the audit source.")
    exact = pd.Series(True, index=comparison.index)
    for column in option_columns:
        exact &= (
            pd.to_numeric(comparison[f"{column}_train"], errors="raise")
            == pd.to_numeric(comparison[f"{column}_audit"], errors="raise")
        )
    if not bool(exact.all()):
        raise RuntimeError("Training option values differ from the audit source.")
    return {
        "keys": keys,
        "rows": int(len(train_options)),
        "exact_train_key_match": True,
        "exact_audit_value_match": True,
    }


def check_hash(path: Path, expected: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise RuntimeError(f"Fingerprint mismatch for {path}: {digest} != {expected}")


def attach_options(
    frame: pd.DataFrame,
    options: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    if "source_row_id" in frame.columns and "source_row_id" in options.columns:
        keys = ["source_row_id"]
    else:
        keys = ["stu_id", "exer_id"]
    if options.duplicated(keys).any():
        raise RuntimeError(f"Option sidecar is not unique on {keys}.")
    attached = frame.merge(
        options,
        on=keys,
        how="left",
        suffixes=("", "_option"),
        validate="many_to_one",
        indicator=True,
    )
    coverage = float((attached["_merge"] == "both").mean()) if len(attached) else 0.0
    attached = attached.drop(columns="_merge")
    if keys == ["source_row_id"]:
        for column in ("stu_id", "exer_id"):
            option_column = f"{column}_option"
            if option_column in attached.columns:
                matched = attached[option_column].notna()
                identity_equal = (
                    attached.loc[matched, column].astype(str).to_numpy()
                    == attached.loc[matched, option_column].astype(str).to_numpy()
                )
                if not bool(identity_equal.all()):
                    raise RuntimeError(
                        f"Option sidecar source-row identity disagrees on {column}."
                    )
    for column in ("selected_option", "correct_option", "option_count"):
        attached[column] = pd.to_numeric(attached[column], errors="coerce")
    return attached, coverage


def variant_spec(
    dataset: str,
    dataset_spec: dict[str, Any],
    variant: str,
) -> tuple[Path, Path, Path, dict[str, str], str]:
    if dataset == "nips34":
        directory = Path(dataset_spec[f"{variant}_dir"])
        audit_option_path = Path(dataset_spec["option_file"]["path"])
        train_spec = dataset_spec["train_option_files"][variant]
        train_option_path = Path(train_spec["path"])
        train_option_hash = train_spec["sha256"]
        hashes = dataset_spec["protocol_hashes"][variant]
    else:
        report = dataset_spec[variant]
        directory = Path(report["directory"])
        audit_option_path = Path(dataset_spec["option_file"]["path"])
        train_option_path = directory / "train_options.csv"
        hashes = {
            filename: details["sha256"]
            for filename, details in report["files"].items()
        }
        train_option_hash = hashes["train_options.csv"]
    return (
        directory,
        audit_option_path,
        train_option_path,
        hashes,
        train_option_hash,
    )


def load_variant(
    dataset: str,
    dataset_spec: dict[str, Any],
    variant: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    (
        directory,
        audit_option_path,
        train_option_path,
        hashes,
        train_option_hash,
    ) = variant_spec(dataset, dataset_spec, variant)
    option_spec = dataset_spec.get("option_file")
    if not isinstance(option_spec, dict) or option_spec.get("training_access") is not False:
        raise RuntimeError("Full audit option source must be explicitly training-inaccessible.")
    if audit_option_path.resolve() == train_option_path.resolve():
        raise RuntimeError("Full audit options cannot be the model-accessible train sidecar.")
    required = ("data.csv", "train.csv", "valid.csv", "test.csv", "Q_matrix.csv")
    for filename in required:
        if filename not in hashes:
            raise RuntimeError(f"Protocol manifest omits {filename}.")
    for filename, expected in hashes.items():
        check_hash(directory / filename, expected)
    check_hash(
        audit_option_path,
        dataset_spec["option_file"]["sha256"],
    )
    check_hash(train_option_path, train_option_hash)
    audit_options = read_frame(audit_option_path)
    train_options = read_frame(train_option_path)
    frames = {
        name: read_frame(directory / f"{name}.csv")
        for name in ("data", "train", "valid", "test", "Q_matrix")
    }
    option_boundary = verify_train_option_boundary(
        train=frames["train"],
        train_options=train_options,
        audit_options=audit_options,
    )
    data_option, data_coverage = attach_options(
        frames["data"], audit_options
    )
    train_option, train_coverage = attach_options(
        frames["train"], train_options
    )
    return {
        **frames,
        "data_option": data_option,
        "train_option": train_option,
    }, {
        "directory": str(directory.resolve()),
        "audit_option_path": str(audit_option_path.resolve()),
        "train_option_path": str(train_option_path.resolve()),
        "option_coverage": {
            "data_audit": data_coverage,
            "train_accessible": train_coverage,
        },
        "rows": {
            name: int(len(frames[name]))
            for name in ("data", "train", "valid", "test")
        },
        "train_option_boundary": option_boundary,
        "forbidden_option_sidecars_present": sorted(
            filename
            for filename in (
                "valid_options.csv",
                "test_options.csv",
                "options.csv",
            )
            if (directory / filename).exists()
        ),
    }


def option_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    option_columns = ["selected_option", "correct_option", "option_count"]
    available = frame[option_columns].notna().all(axis=1)
    integer = available.copy()
    for column in option_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        integer &= values.notna() & values.mod(1).eq(0)
    valid_range = (
        integer
        & frame["option_count"].between(2, 7)
        & frame["selected_option"].ge(0)
        & frame["correct_option"].ge(0)
        & frame["selected_option"].lt(frame["option_count"])
        & frame["correct_option"].lt(frame["option_count"])
    )
    label = pd.to_numeric(frame["label"], errors="raise").astype(int)
    consistent = valid_range & (
        (frame["selected_option"] == frame["correct_option"]).astype(int)
        == label
    )
    return {
        "available": available,
        "integer": integer,
        "valid_range": valid_range,
        "consistent": consistent,
    }


def option_integrity(frame: pd.DataFrame) -> dict[str, Any]:
    masks = option_masks(frame)
    available = masks["available"]
    valid_range = masks["valid_range"]
    consistent = masks["consistent"]
    valid_denominator = int(available.sum())
    consistency_denominator = int(valid_range.sum())
    return {
        "rows": int(len(frame)),
        "available_fraction": float(available.mean()) if len(frame) else 0.0,
        "valid_range_fraction": (
            float(valid_range.sum() / valid_denominator)
            if valid_denominator
            else 0.0
        ),
        "label_consistency_fraction": (
            float(consistent.sum() / consistency_denominator)
            if consistency_denominator
            else 0.0
        ),
        "option_count_values": sorted(
            int(value)
            for value in frame.loc[valid_range, "option_count"].unique()
        ),
    }


def target_rows(
    frame: pd.DataFrame,
    *,
    train: pd.DataFrame,
    q_matrix: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    enriched = add_target_coverage(
        frame,
        train_frame=train,
        q_matrix=q_matrix,
    )
    if scope == "bucket:zero":
        return enriched.loc[enriched["coverage_bucket"] == "zero"].copy()
    if scope == "low_coverage":
        return enriched.loc[enriched["coverage_group"] == "low_coverage"].copy()
    raise ValueError(f"Unknown target scope: {scope}")


def target_report(
    target: pd.DataFrame,
    *,
    train_options: pd.DataFrame,
) -> dict[str, Any]:
    labels = pd.to_numeric(target["label"], errors="raise").astype(int)
    label_counts = {
        str(value): int((labels == value).sum())
        for value in (0, 1)
    }
    students = set(target["stu_id"].astype(str))
    valid_history = train_options.loc[
        option_masks(train_options)["consistent"]
    ].drop_duplicates(["stu_id", "exer_id"])
    history_counts = valid_history.groupby(
        valid_history["stu_id"].astype(str)
    ).size()
    incorrect_history = valid_history.loc[
        valid_history["selected_option"] != valid_history["correct_option"]
    ]
    incorrect_counts = incorrect_history.groupby(
        incorrect_history["stu_id"].astype(str)
    ).size()
    students_with_history = sum(
        int(history_counts.get(student, 0)) >= FROZEN_MIN_HISTORY_ROWS
        for student in students
    )
    students_with_incorrect_history = sum(
        int(incorrect_counts.get(student, 0))
        >= FROZEN_MIN_INCORRECT_HISTORY_ROWS
        for student in students
    )
    history_fraction = (
        students_with_history / len(students) if students else 0.0
    )
    incorrect_history_fraction = (
        students_with_incorrect_history / len(students)
        if students
        else 0.0
    )
    eligible = (
        len(target) >= FROZEN_MIN_TARGET_ROWS
        and len(students) >= FROZEN_MIN_TARGET_STUDENTS
        and min(label_counts.values()) >= FROZEN_MIN_TARGET_CLASS_ROWS
        and history_fraction >= FROZEN_MIN_TARGET_HISTORY_FRACTION
        and incorrect_history_fraction
        >= FROZEN_MIN_TARGET_INCORRECT_HISTORY_FRACTION
    )
    return {
        "rows": int(len(target)),
        "students": int(len(students)),
        "label_counts": label_counts,
        "students_with_minimum_option_history": int(students_with_history),
        "minimum_option_history_fraction": float(history_fraction),
        "students_with_minimum_incorrect_option_history": int(
            students_with_incorrect_history
        ),
        "minimum_incorrect_option_history_fraction": float(
            incorrect_history_fraction
        ),
        "eligible": bool(eligible),
    }


def train_support_report(train: pd.DataFrame) -> dict[str, Any]:
    valid = option_masks(train)["consistent"]
    valid_train = train.loc[valid].copy()
    counts = valid_train.groupby(
        ["exer_id", "selected_option"]
    )["stu_id"].nunique()
    row_counts = pd.MultiIndex.from_frame(
        train[["exer_id", "selected_option"]]
    ).map(counts).fillna(0)
    supported = valid.to_numpy() & (
        row_counts.to_numpy() >= FROZEN_MIN_TRAIN_CELL_SUPPORT
    )
    wrong = valid_train.loc[
        valid_train["selected_option"] != valid_train["correct_option"]
    ]
    wrong_counts = wrong.groupby(
        ["exer_id", "selected_option"]
    )["stu_id"].nunique()
    supported_wrong = wrong_counts.ge(FROZEN_MIN_TRAIN_CELL_SUPPORT)
    supported_wrong_options = supported_wrong.groupby(level="exer_id").sum()
    supported_wrong_items = int(supported_wrong_options.ge(2).sum())
    supported_fraction = float(supported.mean()) if len(train) else 0.0
    return {
        "item_option_cells": int(len(counts)),
        "wrong_item_option_cells": int(len(wrong_counts)),
        "minimum_cross_student_support": FROZEN_MIN_TRAIN_CELL_SUPPORT,
        "supported_train_row_fraction": supported_fraction,
        "items_with_two_supported_wrong_options": supported_wrong_items,
        "minimum_supported_wrong_items": FROZEN_MIN_SUPPORTED_WRONG_ITEMS,
        "eligible": bool(
            supported_fraction >= FROZEN_MIN_SUPPORTED_TRAIN_FRACTION
            and supported_wrong_items >= FROZEN_MIN_SUPPORTED_WRONG_ITEMS
        ),
    }


def audit_dataset(
    dataset: str,
    dataset_spec: dict[str, Any],
) -> dict[str, Any]:
    standard, standard_io = load_variant(
        dataset, dataset_spec, "standard"
    )
    holdout, holdout_io = load_variant(
        dataset, dataset_spec, "holdout"
    )
    standard_partition = split_partition_report(standard)
    holdout_partition = split_partition_report(holdout)
    standard_integrity = option_integrity(standard["data_option"])
    holdout_integrity = option_integrity(holdout["data_option"])
    interaction_columns = ("stu_id", "exer_id", "cpt_seq", "label")
    same_rows = row_multiset(
        standard["data"], interaction_columns
    ) == row_multiset(
        holdout["data"], interaction_columns
    )
    q_columns = ("exer_id", "cpt_seq")
    same_q = row_multiset(
        standard["Q_matrix"], q_columns
    ) == row_multiset(
        holdout["Q_matrix"], q_columns
    )
    train_support = {
        "standard": train_support_report(standard["train_option"]),
        "holdout": train_support_report(holdout["train_option"]),
    }
    scope = dataset_spec["target_scope"]
    targets: dict[str, Any] = {}
    for split in ("valid", "test"):
        target = target_rows(
            holdout[split],
            train=holdout["train"],
            q_matrix=holdout["Q_matrix"],
            scope=scope,
        )
        targets[split] = target_report(
            target,
            train_options=holdout["train_option"],
        )
    option_coverage_ok = all(
        fraction >= FROZEN_MIN_OPTION_COVERAGE
        for fraction in [
            *standard_io["option_coverage"].values(),
            *holdout_io["option_coverage"].values(),
        ]
    )
    integrity_ok = all(
        report["available_fraction"] >= FROZEN_MIN_OPTION_COVERAGE
        and report["valid_range_fraction"] >= FROZEN_MIN_VALID_RANGE
        and report["label_consistency_fraction"]
        >= FROZEN_MIN_LABEL_CONSISTENCY
        for report in (standard_integrity, holdout_integrity)
    )
    identity_ok = (
        dataset_spec.get("source_to_protocol_identity_exact") is True
        and isinstance(dataset_spec.get("source_hashes"), dict)
        and bool(dataset_spec["source_hashes"])
        and all(
            isinstance(value, str) and len(value) == 64
            for value in dataset_spec["source_hashes"].values()
        )
    )
    split_integrity_ok = all(
        report["mutually_disjoint"]
        and report["student_item_atomic"]
        and report["exhaustive_exact"]
        for report in (standard_partition, holdout_partition)
    )
    artifact_specs = dataset_spec.get("audit_artifacts", {})
    artifacts_ok = isinstance(artifact_specs, dict) and all(
        isinstance(spec, dict)
        and spec.get("training_access") is False
        and Path(spec["path"]).is_file()
        and sha256_file(spec["path"]) == spec["sha256"]
        for spec in artifact_specs.values()
    )
    leakage_boundary_ok = (
        dataset_spec["option_file"].get("training_access") is False
        and not (
            standard_io["forbidden_option_sidecars_present"]
            or holdout_io["forbidden_option_sidecars_present"]
        )
    )
    passed = (
        identity_ok
        and same_rows
        and same_q
        and split_integrity_ok
        and leakage_boundary_ok
        and option_coverage_ok
        and integrity_ok
        and all(report["eligible"] for report in train_support.values())
        and targets["valid"]["eligible"]
        and artifacts_ok
    )
    return {
        "dataset": dataset,
        "passed": bool(passed),
        "target_scope": scope,
        "identity_exact": identity_ok,
        "standard_holdout_row_multiset_equal": bool(same_rows),
        "standard_holdout_q_equal": bool(same_q),
        "leakage_boundary_ok": bool(leakage_boundary_ok),
        "audit_artifacts_ok": bool(artifacts_ok),
        "route_selection_uses_validation_target_only": True,
        "standard": {
            **standard_io,
            "integrity": standard_integrity,
            "partition": standard_partition,
            "train_support": train_support["standard"],
        },
        "holdout": {
            **holdout_io,
            "integrity": holdout_integrity,
            "partition": holdout_partition,
            "train_support": train_support["holdout"],
            "targets": targets,
        },
    }


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    if manifest["split_seed"] != 2024 or manifest["model_seed"] != 42:
        raise RuntimeError("Seed invariants changed.")
    repository = manifest.get("repository", {})
    current_prepare_hash = sha256_file(
        PROJECT_ROOT / "scripts" / "prepare_option_contrast_pool.py"
    )
    if (
        not isinstance(repository.get("git_commit"), str)
        or len(repository["git_commit"]) != 40
        or repository.get("prepare_script_sha256") != current_prepare_hash
    ):
        raise RuntimeError("Manifest is not bound to the current committed preparer.")
    missing = set(DATASETS) - set(manifest["datasets"])
    if missing:
        raise RuntimeError(f"Admission manifest is missing datasets: {sorted(missing)}")
    reports = [
        audit_dataset(dataset, manifest["datasets"][dataset])
        for dataset in DATASETS
    ]
    passing = sum(report["passed"] for report in reports)
    payload = {
        "schema_version": 1,
        "manifest_path": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "repository": repository,
        "gate": {
            "minimum_option_coverage": FROZEN_MIN_OPTION_COVERAGE,
            "minimum_valid_range": FROZEN_MIN_VALID_RANGE,
            "minimum_label_consistency": FROZEN_MIN_LABEL_CONSISTENCY,
            "minimum_train_cell_support": FROZEN_MIN_TRAIN_CELL_SUPPORT,
            "minimum_supported_train_fraction": (
                FROZEN_MIN_SUPPORTED_TRAIN_FRACTION
            ),
            "minimum_items_with_two_supported_wrong_options": (
                FROZEN_MIN_SUPPORTED_WRONG_ITEMS
            ),
            "minimum_target_rows": FROZEN_MIN_TARGET_ROWS,
            "minimum_target_students": FROZEN_MIN_TARGET_STUDENTS,
            "minimum_target_class_rows": FROZEN_MIN_TARGET_CLASS_ROWS,
            "minimum_history_rows": FROZEN_MIN_HISTORY_ROWS,
            "minimum_target_history_fraction": (
                FROZEN_MIN_TARGET_HISTORY_FRACTION
            ),
            "minimum_incorrect_history_rows": FROZEN_MIN_INCORRECT_HISTORY_ROWS,
            "minimum_target_incorrect_history_fraction": (
                FROZEN_MIN_TARGET_INCORRECT_HISTORY_FRACTION
            ),
            "route_selection_target_split": "validation_only",
            "required_datasets": FROZEN_REQUIRED_DATASETS,
        },
        "passing_datasets": int(passing),
        "route_activated": bool(passing >= FROZEN_REQUIRED_DATASETS),
        "reports": reports,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
