from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.q_matrix import normalize_concept_sequence
from data.pool_protocol import stable_fraction


REQUIRED_COLUMNS = ("stu_id", "exer_id", "cpt_seq", "label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build validation-only, quantity-matched concept/random depletion "
            "arms. The source test.csv is deliberately never opened."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--tune-fraction", type=float, default=0.5)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--min-selected", type=int, default=500)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_id(value: Any) -> str:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def normalize_interactions(frame: pd.DataFrame, source: Path | str) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}.")
    result = frame.copy().reset_index(drop=True)
    result["stu_id"] = result["stu_id"].map(_canonical_id)
    result["exer_id"] = result["exer_id"].map(_canonical_id)
    result["label"] = pd.to_numeric(result["label"], errors="raise").astype(int)
    if not result["label"].isin([0, 1]).all():
        raise ValueError(f"{source}: labels must be binary.")
    return result


def q_map_from_frame(
    frame: pd.DataFrame, source: Path | str
) -> dict[str, frozenset[str]]:
    if not {"exer_id", "cpt_seq"}.issubset(frame.columns):
        raise ValueError(f"{source}: Q metadata requires exer_id and cpt_seq.")
    mapping: dict[str, set[str]] = defaultdict(set)
    for row in frame[["exer_id", "cpt_seq"]].itertuples(index=False):
        mapping[_canonical_id(row.exer_id)].update(
            normalize_concept_sequence(row.cpt_seq)
        )
    result = {
        exercise: frozenset(concepts)
        for exercise, concepts in mapping.items()
        if concepts
    }
    if not result:
        raise ValueError(f"{source}: empty Q mapping.")
    return result


def attach_q_union(
    frame: pd.DataFrame, q_map: dict[str, frozenset[str]]
) -> pd.DataFrame:
    output = frame.copy()
    missing = sorted(set(output["exer_id"]).difference(q_map))
    if missing:
        raise ValueError(f"Exercises absent from Q matrix: {missing[:5]}.")
    output["cpt_seq"] = output["exer_id"].map(
        lambda exercise: ",".join(sorted(q_map[str(exercise)]))
    )
    return output


def split_validation(
    valid: pd.DataFrame,
    *,
    dataset: str,
    seed: int,
    tune_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < tune_fraction < 1.0:
        raise ValueError("tune_fraction must be strictly between zero and one.")
    group_is_tune = {
        (str(student), str(exercise)): (
            stable_fraction(
                seed, "concept-depletion-tune", dataset, student, exercise
            )
            < tune_fraction
        )
        for student, exercise in valid[["stu_id", "exer_id"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    tune_mask = [
        group_is_tune[(str(student), str(exercise))]
        for student, exercise in valid[["stu_id", "exer_id"]].itertuples(
            index=False, name=None
        )
    ]
    tune = valid.loc[tune_mask].copy().reset_index(drop=True)
    audit = valid.loc[[not value for value in tune_mask]].copy().reset_index(drop=True)
    if tune.empty or audit.empty:
        raise ValueError("Stable validation split produced an empty arm.")
    return tune, audit


def add_audit_row_ids(audit: pd.DataFrame, *, dataset: str) -> pd.DataFrame:
    output = audit.copy().reset_index(drop=True)
    occurrence: dict[tuple[str, str], int] = defaultdict(int)
    row_ids: list[str] = []
    for row in output.itertuples(index=False):
        key = (str(row.stu_id), str(row.exer_id))
        ordinal = occurrence[key]
        occurrence[key] += 1
        payload = (
            f"{dataset}\x1faudit\x1f{row.stu_id}\x1f{row.exer_id}"
            f"\x1f{ordinal}"
        ).encode("utf-8")
        row_ids.append(hashlib.sha256(payload).hexdigest()[:32])
    output.insert(0, "audit_row_id", row_ids)
    if output["audit_row_id"].duplicated().any():
        raise RuntimeError("audit_row_id collision detected.")
    return output


def seen_concepts(
    frame: pd.DataFrame, q_map: dict[str, frozenset[str]]
) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        seen[str(row.stu_id)].update(q_map[str(row.exer_id)])
    return seen


def build_paired_arms(
    train: pd.DataFrame,
    audit: pd.DataFrame,
    q_map: dict[str, frozenset[str]],
    *,
    dataset: str,
    seed: int,
    min_history: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = train.copy().reset_index(drop=True)
    audit = audit.copy().reset_index(drop=True)
    if "audit_row_id" not in audit:
        raise ValueError("audit must contain audit_row_id.")
    train["_source_index"] = range(len(train))
    by_student = {
        str(student): group.copy()
        for student, group in train.groupby("stu_id", sort=False)
    }
    concept_remove: set[int] = set()
    random_remove: set[int] = set()
    selected_records: list[dict[str, Any]] = []

    for student, candidates in audit.groupby("stu_id", sort=False):
        student_key = str(student)
        history = by_student.get(student_key)
        if history is None or len(history) < min_history + 1:
            continue
        history_rows = [
            (
                int(source_index),
                str(exercise),
                q_map[str(exercise)],
            )
            for source_index, exercise in history[
                ["_source_index", "exer_id"]
            ].itertuples(index=False, name=None)
        ]
        observed = set().union(*(concepts for _, _, concepts in history_rows))
        ordered_candidates = sorted(
            candidates.itertuples(index=False),
            key=lambda row: stable_fraction(
                seed,
                "concept-depletion-target",
                dataset,
                student_key,
                row.exer_id,
                row.audit_row_id,
            ),
        )
        chosen: tuple[Any, frozenset[str], list[int], list[int]] | None = None
        for row in ordered_candidates:
            target = q_map[str(row.exer_id)]
            if not target or not target.issubset(observed):
                continue
            target_rows = [
                index
                for index, _, concepts in history_rows
                if concepts.intersection(target)
            ]
            non_target_rows = [
                index
                for index, _, concepts in history_rows
                if concepts.isdisjoint(target)
            ]
            if (
                not target_rows
                or len(history_rows) - len(target_rows) < min_history
                or len(non_target_rows) < len(target_rows)
            ):
                continue
            ordered_random = sorted(
                non_target_rows,
                key=lambda index: stable_fraction(
                    seed,
                    "concept-depletion-random",
                    dataset,
                    student_key,
                    row.audit_row_id,
                    index,
                ),
            )
            chosen = (
                row,
                target,
                target_rows,
                ordered_random[: len(target_rows)],
            )
            break
        if chosen is None:
            continue
        row, target, target_rows, random_rows = chosen
        concept_remove.update(target_rows)
        random_remove.update(random_rows)
        selected_records.append(
            {
                "audit_row_id": str(row.audit_row_id),
                "stu_id": student_key,
                "exer_id": str(row.exer_id),
                "label": int(row.label),
                "target_concepts": ",".join(sorted(target)),
                "removed_rows": len(target_rows),
                "history_rows_before": len(history_rows),
                "history_rows_after": len(history_rows) - len(target_rows),
            }
        )

    concept = train.loc[
        ~train["_source_index"].isin(concept_remove)
    ].drop(columns="_source_index").reset_index(drop=True)
    random = train.loc[
        ~train["_source_index"].isin(random_remove)
    ].drop(columns="_source_index").reset_index(drop=True)
    selected = pd.DataFrame(selected_records)
    if selected.empty:
        raise ValueError(f"{dataset}: no eligible paired audit rows.")
    if len(concept_remove) != len(random_remove) or len(concept) != len(random):
        raise RuntimeError("Paired arms do not remove identical row counts.")

    concept_seen = seen_concepts(concept, q_map)
    random_seen = seen_concepts(random, q_map)
    for record in selected.itertuples(index=False):
        target = frozenset(str(record.target_concepts).split(","))
        if target.intersection(concept_seen[str(record.stu_id)]):
            raise RuntimeError(
                f"{dataset}: concept arm is not exact-zero for {record.audit_row_id}."
            )
        if not target.issubset(random_seen[str(record.stu_id)]):
            raise RuntimeError(
                f"{dataset}: random arm lost target coverage for {record.audit_row_id}."
            )
        concept_count = int(
            (concept["stu_id"].astype(str) == str(record.stu_id)).sum()
        )
        random_count = int(
            (random["stu_id"].astype(str) == str(record.stu_id)).sum()
        )
        if concept_count != random_count:
            raise RuntimeError(
                f"{dataset}: per-student arm counts differ for {record.stu_id}."
            )

    selected_labels = sorted(selected["label"].unique().tolist())
    audit_summary = {
        "selected_rows": len(selected),
        "selected_students": int(selected["stu_id"].nunique()),
        "labels": selected_labels,
        "positive_rate": float(selected["label"].mean()),
        "train_rows_source": len(train),
        "train_rows_per_arm": len(concept),
        "removed_rows_total": len(concept_remove),
        "mean_removed_rows_per_student": float(selected["removed_rows"].mean()),
        "median_removed_rows_per_student": float(selected["removed_rows"].median()),
        "exact_zero_concept_arm": True,
        "full_coverage_random_arm": True,
        "per_student_counts_matched": True,
    }
    return concept, random, selected, audit_summary


def write_protocol(
    *,
    dataset: str,
    source_dir: Path,
    output_dir: Path,
    train: pd.DataFrame,
    tune: pd.DataFrame,
    audit: pd.DataFrame,
    q_frame: pd.DataFrame,
    concept: pd.DataFrame,
    random: pd.DataFrame,
    selected: pd.DataFrame,
    summary: dict[str, Any],
    seed: int,
    tune_fraction: float,
    min_history: int,
    min_selected: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected_targets.csv"
    selected.to_csv(selected_path, index=False)
    for arm_name, arm_train in (
        ("concept_depleted", concept),
        ("random_depleted", random),
    ):
        arm_dir = output_dir / arm_name
        arm_dir.mkdir(parents=True, exist_ok=True)
        arm_train.to_csv(arm_dir / "train.csv", index=False)
        tune.to_csv(arm_dir / "valid.csv", index=False)
        audit.to_csv(arm_dir / "test.csv", index=False)
        q_frame.to_csv(arm_dir / "Q_matrix.csv", index=False)

    source_hashes = {
        name: sha256_file(source_dir / name)
        for name in ("train.csv", "valid.csv", "Q_matrix.csv")
    }
    generated_hashes: dict[str, dict[str, str]] = {}
    for arm_name in ("concept_depleted", "random_depleted"):
        arm_dir = output_dir / arm_name
        generated_hashes[arm_name] = {
            name: sha256_file(arm_dir / name)
            for name in ("train.csv", "valid.csv", "test.csv", "Q_matrix.csv")
        }
    payload = {
        "schema_version": 1,
        "protocol": "paired_concept_depletion_gate_b",
        "dataset": dataset,
        "source_dir": str(source_dir.resolve()),
        "source_test_opened": False,
        "seed": seed,
        "tune_fraction": tune_fraction,
        "min_history": min_history,
        "min_selected": min_selected,
        "qualified_for_screen": (
            summary["selected_rows"] >= min_selected
            and summary["labels"] == [0, 1]
        ),
        "rows": {
            "source_train": len(train),
            "early_stop_valid": len(tune),
            "audit": len(audit),
            **summary,
        },
        "source_hashes": source_hashes,
        "generated_hashes": generated_hashes,
        "selected_targets_sha256": sha256_file(selected_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed != 2024:
        raise ValueError("Gate B protocol construction is frozen at seed=2024.")
    train_path = args.data_dir / "train.csv"
    valid_path = args.data_dir / "valid.csv"
    q_path = args.data_dir / "Q_matrix.csv"
    train = normalize_interactions(pd.read_csv(train_path), train_path)
    valid = normalize_interactions(pd.read_csv(valid_path), valid_path)
    q_frame = pd.read_csv(q_path)
    q_map = q_map_from_frame(q_frame, q_path)
    train = attach_q_union(train, q_map)
    valid = attach_q_union(valid, q_map)
    tune, audit = split_validation(
        valid,
        dataset=args.dataset,
        seed=args.seed,
        tune_fraction=args.tune_fraction,
    )
    audit = add_audit_row_ids(audit, dataset=args.dataset)
    concept, random, selected, summary = build_paired_arms(
        train,
        audit,
        q_map,
        dataset=args.dataset,
        seed=args.seed,
        min_history=args.min_history,
    )
    return write_protocol(
        dataset=args.dataset,
        source_dir=args.data_dir,
        output_dir=args.output_dir,
        train=train,
        tune=tune,
        audit=audit,
        q_frame=q_frame,
        concept=concept,
        random=random,
        selected=selected,
        summary=summary,
        seed=args.seed,
        tune_fraction=args.tune_fraction,
        min_history=args.min_history,
        min_selected=args.min_selected,
    )


def main() -> None:
    args = parse_args()
    payload = prepare(args)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
