from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


INTERACTION_COLUMNS = ("stu_id", "exer_id", "cpt_seq", "label")
SPLIT_NAMES = ("train", "valid", "test")


def _canonical_id(value: Any) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def _concept_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def canonical_concepts(value: Any) -> str:
    if pd.isna(value):
        return ""
    concepts = {_canonical_id(token) for token in str(value).split(",") if str(token).strip()}
    return ",".join(sorted(concepts, key=_concept_sort_key))


def stable_fraction(*parts: object) -> float:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64)


def canonicalize_interactions(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    missing = set(INTERACTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Interaction data is missing columns: {sorted(missing)}")
    output = frame.loc[:, INTERACTION_COLUMNS].copy()
    output["stu_id"] = output["stu_id"].map(_canonical_id)
    output["exer_id"] = output["exer_id"].map(_canonical_id)
    output["cpt_seq"] = output["cpt_seq"].map(canonical_concepts)
    output["label"] = pd.to_numeric(output["label"], errors="raise").astype(int)
    if not set(output["label"].unique()).issubset({0, 1}):
        raise ValueError("Interaction labels must be binary.")
    if (output["cpt_seq"] == "").any():
        raise ValueError("Every interaction must have at least one concept.")
    before = len(output)
    output = output.drop_duplicates(list(INTERACTION_COLUMNS), keep="first").reset_index(drop=True)
    removed = before - len(output)
    output.insert(
        0,
        "source_row_id",
        [
            hashlib.sha256("\x1f".join(map(str, row)).encode("utf-8")).hexdigest()[:32]
            for row in output.loc[:, INTERACTION_COLUMNS].itertuples(index=False, name=None)
        ],
    )
    if output["source_row_id"].duplicated().any():
        raise RuntimeError("Canonical source_row_id collision detected.")
    return output, removed


def canonicalize_q_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = {"exer_id", "cpt_seq"} - set(frame.columns)
    if missing:
        raise ValueError(f"Q-matrix is missing columns: {sorted(missing)}")
    output = frame.loc[:, ["exer_id", "cpt_seq"]].copy()
    output["exer_id"] = output["exer_id"].map(_canonical_id)
    output["cpt_seq"] = output["cpt_seq"].map(canonical_concepts)
    if (output["cpt_seq"] == "").any():
        raise ValueError("Every Q-matrix exercise must have at least one concept.")
    return output.drop_duplicates(
        ["exer_id", "cpt_seq"],
        keep="first",
    ).reset_index(drop=True)


def derive_q_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    mappings: dict[str, set[str]] = defaultdict(set)
    raw_mappings: dict[str, set[str]] = defaultdict(set)
    for row in frame[["exer_id", "cpt_seq"]].itertuples(index=False):
        exercise = _canonical_id(row.exer_id)
        raw_mappings[exercise].add(canonical_concepts(row.cpt_seq))
        mappings[exercise].update(token for token in canonical_concepts(row.cpt_seq).split(",") if token)
    conflicts = sum(len(values) > 1 for values in raw_mappings.values())
    q_rows = [
        {
            "exer_id": exercise,
            "cpt_seq": ",".join(sorted(concepts, key=_concept_sort_key)),
        }
        for exercise, concepts in mappings.items()
    ]
    q_matrix = pd.DataFrame(q_rows).sort_values(
        "exer_id", key=lambda column: column.map(_concept_sort_key)
    ).reset_index(drop=True)
    return q_matrix, conflicts


def add_split_row_index(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy().reset_index(drop=True)
    output.insert(1, "split_row_index", range(len(output)))
    return output


def assign_atomic_standard_split(frame: pd.DataFrame, *, seed: int = 2024) -> dict[str, pd.DataFrame]:
    group_split: dict[tuple[str, str], str] = {}
    for student, exercise in frame[["stu_id", "exer_id"]].drop_duplicates().itertuples(index=False):
        fraction = stable_fraction(seed, "standard", student, exercise)
        group_split[(student, exercise)] = "train" if fraction < 0.7 else "valid" if fraction < 0.8 else "test"
    assignments = [group_split[(row.stu_id, row.exer_id)] for row in frame.itertuples(index=False)]
    return {
        split: add_split_row_index(frame.loc[[value == split for value in assignments]])
        for split in SPLIT_NAMES
    }


def _student_sets(frame: pd.DataFrame) -> set[str]:
    return set(frame["stu_id"].astype(str))


def _group_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            frame["stu_id"].astype(str),
            frame["exer_id"].astype(str),
            strict=True,
        )
    )


def _select_students(frame: pd.DataFrame, students: set[str]) -> pd.DataFrame:
    return frame.loc[frame["stu_id"].astype(str).isin(students)].copy()


def _filter_query(
    query: pd.DataFrame,
    *,
    support: pd.DataFrame,
    known_exercises: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    before = len(query)
    known_mask = query["exer_id"].astype(str).isin(known_exercises)
    unknown_rows = int((~known_mask).sum())
    unknown_groups = len(_group_set(query.loc[~known_mask]))
    filtered = query.loc[known_mask].copy()

    support_groups = _group_set(support)
    overlap_mask = [
        (str(student), str(exercise)) in support_groups
        for student, exercise in filtered[["stu_id", "exer_id"]].itertuples(index=False)
    ]
    overlap_rows = int(sum(overlap_mask))
    overlap_groups = len(_group_set(filtered.loc[overlap_mask]))
    filtered = filtered.loc[[not value for value in overlap_mask]].copy()
    return filtered.reset_index(drop=True), {
        "raw_query_rows": int(before),
        "unknown_exercise_rows_removed": unknown_rows,
        "unknown_exercise_groups_removed": int(unknown_groups),
        "support_query_overlap_rows_removed": overlap_rows,
        "support_query_overlap_groups_removed": int(overlap_groups),
    }


def _student_group_counts(frame: pd.DataFrame) -> pd.Series:
    keys = pd.DataFrame(
        {
            "student": frame["stu_id"].astype(str),
            "exercise": frame["exer_id"].astype(str),
        }
    ).drop_duplicates()
    return keys.groupby("student").size()


def _split_statistics(
    *,
    support: pd.DataFrame,
    query: pd.DataFrame,
    assigned_students: set[str],
    retained_students: set[str],
    known_exercises: set[str],
    filter_stats: dict[str, int],
) -> dict[str, Any]:
    support_row_counts = support.groupby(
        support["stu_id"].astype(str)
    ).size()
    query_row_counts = query.groupby(query["stu_id"].astype(str)).size()
    support_group_counts = _student_group_counts(support)
    query_group_counts = _student_group_counts(query)

    def _count_stats(values: pd.Series) -> dict[str, float | int | None]:
        if values.empty:
            return {"min": None, "median": None, "max": None}
        return {
            "min": int(values.min()),
            "median": float(np.median(values.to_numpy())),
            "max": int(values.max()),
        }

    return {
        "assigned_students": int(len(assigned_students)),
        "retained_students": int(len(retained_students)),
        "excluded_students": int(len(assigned_students - retained_students)),
        "excluded_student_ids": sorted(assigned_students - retained_students),
        "support_rows": int(len(support)),
        "query_rows": int(len(query)),
        "support_groups": int(len(_group_set(support))),
        "query_groups": int(len(_group_set(query))),
        "support_row_length": _count_stats(support_row_counts),
        "query_row_length": _count_stats(query_row_counts),
        "support_group_length": _count_stats(support_group_counts),
        "query_group_length": _count_stats(query_group_counts),
        "support_unknown_exercise_rows": int(
            (~support["exer_id"].astype(str).isin(known_exercises)).sum()
        ),
        "query_positive_rate": float(query["label"].mean()) if len(query) else None,
        **filter_stats,
    }


def _coverage_bucket_audit(
    *,
    support: pd.DataFrame,
    query: pd.DataFrame,
    q_matrix: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    q_concepts: dict[str, set[str]] = defaultdict(set)
    for row in q_matrix.itertuples(index=False):
        q_concepts[str(row.exer_id)].update(
            canonical_concepts(row.cpt_seq).split(",")
        )
    seen_by_student: dict[str, set[str]] = defaultdict(set)
    for row in support.itertuples(index=False):
        exercise = str(row.exer_id)
        if exercise not in q_concepts:
            raise ValueError(f"Support exercise {exercise} is absent from Q-matrix.")
        seen_by_student[str(row.stu_id)].update(q_concepts[exercise])

    buckets: dict[str, dict[str, Any]] = {
        name: {"rows": 0, "label_counts": {"0": 0, "1": 0}}
        for name in ("exact_zero", "partial", "full")
    }
    for row in query.itertuples(index=False):
        exercise = str(row.exer_id)
        if exercise not in q_concepts:
            raise ValueError(f"Query exercise {exercise} is absent from Q-matrix.")
        concepts = q_concepts[exercise]
        seen = len(concepts & seen_by_student.get(str(row.stu_id), set()))
        if seen == 0:
            bucket = "exact_zero"
        elif seen == len(concepts):
            bucket = "full"
        else:
            bucket = "partial"
        buckets[bucket]["rows"] += 1
        buckets[bucket]["label_counts"][str(int(row.label))] += 1
    return buckets


def assign_student_disjoint_support_query(
    *,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    q_matrix: pd.DataFrame,
    seed: int = 2024,
    train_fraction: float = 0.70,
    valid_fraction: float = 0.10,
    min_support_groups: int = 10,
    min_query_groups: int = 3,
    min_target_label_count: int = 30,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build a student-disjoint support-to-query protocol from an existing split.

    Student roles are assigned by a stable hash. The original split semantics are
    preserved: optimizer rows and both support sets come only from the original
    train split, while validation/test queries stay in their original split.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 < valid_fraction < 1.0 - train_fraction:
        raise ValueError("valid_fraction must leave a non-empty test fraction.")
    if min_support_groups < 1 or min_query_groups < 1:
        raise ValueError("min_support_groups and min_query_groups must be positive.")
    if min_target_label_count < 1:
        raise ValueError("min_target_label_count must be positive.")

    common_students = (
        _student_sets(train_frame)
        & _student_sets(valid_frame)
        & _student_sets(test_frame)
    )
    assignments: dict[str, set[str]] = {
        "train": set(),
        "valid": set(),
        "test": set(),
    }
    for student in common_students:
        fraction = stable_fraction(seed, "student-disjoint", student)
        role = (
            "train"
            if fraction < train_fraction
            else "valid"
            if fraction < train_fraction + valid_fraction
            else "test"
        )
        assignments[role].add(student)

    optimizer_train = _select_students(train_frame, assignments["train"])
    known_exercises = set(optimizer_train["exer_id"].astype(str))
    splits: dict[str, pd.DataFrame] = {
        "train": add_split_row_index(optimizer_train),
    }
    audit: dict[str, Any] = {
        "seed": int(seed),
        "assignment": {
            "train_fraction": float(train_fraction),
            "valid_fraction": float(valid_fraction),
            "test_fraction": float(1.0 - train_fraction - valid_fraction),
            "input_students": {
                "train": int(train_frame["stu_id"].nunique()),
                "valid": int(valid_frame["stu_id"].nunique()),
                "test": int(test_frame["stu_id"].nunique()),
            },
            "common_students": int(len(common_students)),
            "assigned_students": {
                role: int(len(students)) for role, students in assignments.items()
            },
        },
        "min_support_groups": int(min_support_groups),
        "min_query_groups": int(min_query_groups),
        "min_target_label_count": int(min_target_label_count),
        "known_train_exercises": int(len(known_exercises)),
        "splits": {},
    }

    for role, query_source in (("valid", valid_frame), ("test", test_frame)):
        assigned = assignments[role]
        support_source = _select_students(train_frame, assigned)
        support_known_mask = support_source["exer_id"].astype(str).isin(
            known_exercises
        )
        support_unknown = support_source.loc[~support_known_mask]
        raw_support = support_source.loc[support_known_mask].copy()
        raw_query = _select_students(query_source, assigned)
        filtered_query, filter_stats = _filter_query(
            raw_query,
            support=raw_support,
            known_exercises=known_exercises,
        )
        filter_stats.update(
            {
                "raw_support_rows": int(len(support_source)),
                "unknown_support_rows_removed": int(len(support_unknown)),
                "unknown_support_groups_removed": int(
                    len(_group_set(support_unknown))
                ),
            }
        )
        support_counts = _student_group_counts(raw_support)
        query_counts = _student_group_counts(filtered_query)
        retained = {
            student
            for student in assigned
            if int(support_counts.get(student, 0)) >= min_support_groups
            and int(query_counts.get(student, 0)) >= min_query_groups
        }
        support = _select_students(raw_support, retained).reset_index(drop=True)
        query = _select_students(filtered_query, retained).reset_index(drop=True)
        label_counts = {
            str(label): int((query["label"] == label).sum())
            for label in (0, 1)
        }
        if not all(label_counts.values()):
            raise RuntimeError(
                f"{role} query must contain both binary labels; got {label_counts}."
            )
        splits[f"{role}_support"] = add_split_row_index(support)
        splits[f"{role}_query"] = add_split_row_index(query)
        split_audit = _split_statistics(
            support=support,
            query=query,
            assigned_students=assigned,
            retained_students=retained,
            known_exercises=known_exercises,
            filter_stats=filter_stats,
        )
        split_audit["label_counts"] = label_counts
        coverage_buckets = _coverage_bucket_audit(
            support=support,
            query=query,
            q_matrix=q_matrix,
        )
        exact_zero_labels = coverage_buckets["exact_zero"]["label_counts"]
        if min(exact_zero_labels.values()) < min_target_label_count:
            raise RuntimeError(
                f"{role} exact-zero query must contain both binary labels; "
                f"got {exact_zero_labels}; each class requires "
                f"{min_target_label_count}."
            )
        split_audit["coverage_buckets"] = coverage_buckets
        audit["splits"][role] = split_audit

    output_student_sets = {
        "train": _student_sets(splits["train"]),
        "valid": _student_sets(splits["valid_query"]),
        "test": _student_sets(splits["test_query"]),
    }
    audit["student_overlap"] = {
        "train_valid": len(output_student_sets["train"] & output_student_sets["valid"]),
        "train_test": len(output_student_sets["train"] & output_student_sets["test"]),
        "valid_test": len(output_student_sets["valid"] & output_student_sets["test"]),
    }
    audit["support_query_group_overlap"] = {
        role: len(
            _group_set(splits[f"{role}_support"])
            & _group_set(splits[f"{role}_query"])
        )
        for role in ("valid", "test")
    }
    return splits, audit


def _student_group_rows(student_frame: pd.DataFrame) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, exercise in zip(student_frame.index, student_frame["exer_id"], strict=True):
        groups[str(exercise)].append(int(index))
    return dict(groups)


def _choose_holdout_groups(
    student_frame: pd.DataFrame,
    *,
    student: str,
    seed: int,
    target_ratio: float = 0.30,
    min_ratio: float = 0.15,
    max_ratio: float = 0.32,
    min_train_rows: int = 10,
) -> tuple[set[str], set[str]]:
    groups = _student_group_rows(student_frame)
    concepts_by_group = {
        exercise: set(student_frame.loc[indices, "cpt_seq"].str.split(",").explode())
        for exercise, indices in groups.items()
    }
    all_concepts = set().union(*concepts_by_group.values()) if concepts_by_group else set()
    ordered = sorted(all_concepts, key=lambda concept: stable_fraction(seed, "concept", student, concept))
    target_rows = round(target_ratio * len(student_frame))
    min_rows = round(min_ratio * len(student_frame))
    max_rows = round(max_ratio * len(student_frame))
    held_concepts: set[str] = set()
    held_groups: set[str] = set()
    for concept in ordered:
        candidate_groups = held_groups | {
            exercise for exercise, concepts in concepts_by_group.items() if concept in concepts
        }
        candidate_rows = sum(len(groups[exercise]) for exercise in candidate_groups)
        if candidate_rows <= max_rows and len(student_frame) - candidate_rows >= min_train_rows:
            held_concepts.add(concept)
            held_groups = candidate_groups
        if candidate_rows >= target_rows:
            break
    held_rows = sum(len(groups[exercise]) for exercise in held_groups)
    if held_rows < min_rows:
        return set(), set()
    return held_concepts, held_groups


def assign_atomic_holdout_split(
    frame: pd.DataFrame,
    *,
    seed: int = 2024,
    holdout_student_fraction: float = 0.5,
    min_student_interactions: int = 30,
    min_student_concepts: int = 5,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    split_by_group: dict[tuple[str, str], str] = {}
    assignment_rows: list[dict[str, Any]] = []
    for student, student_frame in frame.groupby("stu_id", sort=False):
        student = str(student)
        groups = _student_group_rows(student_frame)
        concepts = set(student_frame["cpt_seq"].str.split(",").explode())
        eligible = len(student_frame) >= min_student_interactions and len(concepts) >= min_student_concepts
        strict = eligible and stable_fraction(seed, "holdout-student", student) < holdout_student_fraction
        held_concepts: set[str] = set()
        held_groups: set[str] = set()
        if strict:
            held_concepts, held_groups = _choose_holdout_groups(
                student_frame,
                student=student,
                seed=seed,
            )
        mode = "strict_holdout" if held_groups else "fallback_random" if strict else "random"
        for exercise in groups:
            key = (student, exercise)
            if exercise in held_groups:
                fraction = stable_fraction(seed, "holdout-eval", student, exercise)
                split_by_group[key] = "test" if fraction < 2.0 / 3.0 else "valid"
            else:
                fraction = stable_fraction(seed, "holdout-random", student, exercise)
                split_by_group[key] = "train" if fraction < 0.7 else "valid" if fraction < 0.8 else "test"
        assignment_rows.append(
            {
                "stu_id": student,
                "mode": mode,
                "num_rows": len(student_frame),
                "num_groups": len(groups),
                "num_concepts": len(concepts),
                "holdout_concepts": ",".join(sorted(held_concepts, key=_concept_sort_key)),
                "holdout_groups": len(held_groups),
            }
        )
    assignments = [split_by_group[(str(row.stu_id), str(row.exer_id))] for row in frame.itertuples(index=False)]
    splits = {
        split: add_split_row_index(frame.loc[[value == split for value in assignments]])
        for split in SPLIT_NAMES
    }
    return splits, pd.DataFrame(assignment_rows)


def group_overlap(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    group_sets = {
        name: set(zip(frame["stu_id"].astype(str), frame["exer_id"].astype(str), strict=True))
        for name, frame in splits.items()
    }
    return {
        "train_valid": len(group_sets["train"] & group_sets["valid"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "valid_test": len(group_sets["valid"] & group_sets["test"]),
    }


def exact_row_overlap(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    row_sets = {name: set(frame["source_row_id"]) for name, frame in splits.items()}
    return {
        "train_valid": len(row_sets["train"] & row_sets["valid"]),
        "train_test": len(row_sets["train"] & row_sets["test"]),
        "valid_test": len(row_sets["valid"] & row_sets["test"]),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset_variant(
    output_dir: Path,
    *,
    full_frame: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    q_matrix: pd.DataFrame,
    assignments: pd.DataFrame | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full_frame.to_csv(output_dir / "data.csv", index=False)
    q_matrix.to_csv(output_dir / "Q_matrix.csv", index=False)
    for name, split in splits.items():
        split.to_csv(output_dir / f"{name}.csv", index=False)
    if assignments is not None:
        assignments.to_csv(output_dir / "student_concept_holdout_assignments.csv", index=False)
    filenames: Iterable[str] = ("data.csv", "train.csv", "valid.csv", "test.csv", "Q_matrix.csv")
    return {
        "directory": str(output_dir),
        "files": {
            name: {
                "sha256": sha256_file(output_dir / name),
                "rows": int(len(pd.read_csv(output_dir / name))),
            }
            for name in filenames
        },
        "exact_row_overlap": exact_row_overlap(splits),
        "student_exercise_group_overlap": group_overlap(splits),
    }


def write_student_disjoint_variant(
    output_dir: Path,
    *,
    splits: dict[str, pd.DataFrame],
    q_matrix: pd.DataFrame,
    audit: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = (
        "train.csv",
        "valid_support.csv",
        "valid_query.csv",
        "test_support.csv",
        "test_query.csv",
        "Q_matrix.csv",
    )
    q_matrix.to_csv(output_dir / "Q_matrix.csv", index=False)
    for name in (
        "train",
        "valid_support",
        "valid_query",
        "test_support",
        "test_query",
    ):
        splits[name].to_csv(output_dir / f"{name}.csv", index=False)
    manifest = {
        "schema_version": 1,
        "protocol": "student_disjoint_support_query",
        "directory": str(output_dir.resolve()),
        "files": {
            name: {
                "sha256": sha256_file(output_dir / name),
                "rows": int(len(pd.read_csv(output_dir / name))),
            }
            for name in filenames
        },
        "audit": audit,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        __import__("json").dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
