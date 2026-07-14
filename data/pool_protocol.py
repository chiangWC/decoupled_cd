from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Any, Iterable

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
