from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an approximate 70/10/20 student-concept holdout split. "
            "A configurable fraction of eligible students receives strict concept holdout; "
            "the remaining students use the existing per-student random interaction split."
        )
    )
    parser.add_argument("--source-dir", default="data/assist_09_ordered")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--holdout-student-frac",
        type=float,
        default=0.5,
        help="Fraction of eligible students assigned to strict student-concept holdout.",
    )
    parser.add_argument("--target-eval-ratio", type=float, default=0.30)
    parser.add_argument("--min-eval-ratio", type=float, default=0.15)
    parser.add_argument("--max-eval-ratio", type=float, default=0.32)
    parser.add_argument("--test-ratio-within-eval", type=float, default=2.0 / 3.0)
    parser.add_argument("--random-valid-ratio", type=float, default=0.10)
    parser.add_argument("--random-test-ratio", type=float, default=0.20)
    parser.add_argument("--min-student-interactions", type=int, default=30)
    parser.add_argument("--min-student-concepts", type=int, default=5)
    parser.add_argument("--min-train-interactions", type=int, default=10)
    parser.add_argument(
        "--copy-transition-graph",
        action="store_true",
        help=(
            "Copy the source transition_graph into the holdout split. Default is NOT to copy: "
            "the source graph is built from data that includes the held-out interactions' labels, "
            "so reusing it leaks evaluation signal. Rebuild the graph from this split's train.csv "
            "via scripts/build_assist09_transition_graph.py instead."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_concepts(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return tuple(int(token) for token in str(value).split(",") if token.strip())


def stringify_concepts(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["cpt_seq"] = output["cpt_seq"].map(lambda seq: ",".join(str(item) for item in parse_concepts(seq)))
    return output


def random_split_indices(n_rows: int, *, rng: random.Random, valid_ratio: float, test_ratio: float) -> tuple[set[int], set[int]]:
    indices = list(range(n_rows))
    rng.shuffle(indices)
    n_test = round(test_ratio * n_rows)
    n_valid = round(valid_ratio * n_rows)
    test_indices = set(indices[:n_test])
    valid_indices = set(indices[n_test : n_test + n_valid])
    return valid_indices, test_indices


def choose_holdout_rows(
    *,
    row_concepts: list[set[int]],
    concepts: list[int],
    rng: random.Random,
    target_eval_ratio: float,
    min_eval_ratio: float,
    max_eval_ratio: float,
    min_train_interactions: int,
) -> tuple[set[int], set[int]]:
    n_rows = len(row_concepts)
    target_eval_rows = round(target_eval_ratio * n_rows)
    min_eval_rows = round(min_eval_ratio * n_rows)
    max_eval_rows = round(max_eval_ratio * n_rows)

    concept_order = concepts[:]
    rng.shuffle(concept_order)

    holdout_concepts: set[int] = set()
    holdout_rows: set[int] = set()
    for concept in concept_order:
        concept_rows = {index for index, values in enumerate(row_concepts) if concept in values}
        candidate_rows = holdout_rows | concept_rows
        if len(candidate_rows) <= max_eval_rows and n_rows - len(candidate_rows) >= min_train_interactions:
            holdout_concepts.add(concept)
            holdout_rows = candidate_rows
        if len(holdout_rows) >= target_eval_rows:
            break

    if len(holdout_rows) < min_eval_rows:
        return set(), set()
    return holdout_concepts, holdout_rows


def split_student_rows(
    *,
    student_id: Any,
    student_frame: pd.DataFrame,
    holdout_enabled: bool,
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
    row_indices = list(student_frame.index)
    row_concepts = [set(parse_concepts(value)) for value in student_frame["cpt_seq"].tolist()]
    concepts = sorted(set().union(*row_concepts)) if row_concepts else []

    metadata: dict[str, Any] = {
        "stu_id": student_id,
        "num_rows": len(row_indices),
        "num_concepts": len(concepts),
        "mode": "random",
        "holdout_concepts": "",
        "holdout_rows": 0,
    }

    eligible = (
        holdout_enabled
        and len(row_indices) >= args.min_student_interactions
        and len(concepts) >= args.min_student_concepts
    )

    if eligible:
        holdout_concepts, holdout_rows = choose_holdout_rows(
            row_concepts=row_concepts,
            concepts=concepts,
            rng=rng,
            target_eval_ratio=args.target_eval_ratio,
            min_eval_ratio=args.min_eval_ratio,
            max_eval_ratio=args.max_eval_ratio,
            min_train_interactions=args.min_train_interactions,
        )
        if holdout_rows:
            eval_positions = list(holdout_rows)
            rng.shuffle(eval_positions)
            n_test = round(args.test_ratio_within_eval * len(eval_positions))
            test_positions = set(eval_positions[:n_test])
            valid_positions = set(eval_positions[n_test:])

            train_indices: list[int] = []
            valid_indices: list[int] = []
            test_indices: list[int] = []
            for position, row_index in enumerate(row_indices):
                if position in test_positions:
                    test_indices.append(row_index)
                elif position in valid_positions:
                    valid_indices.append(row_index)
                else:
                    train_indices.append(row_index)

            metadata.update(
                {
                    "mode": "strict_holdout",
                    "holdout_concepts": ",".join(str(item) for item in sorted(holdout_concepts)),
                    "holdout_rows": len(holdout_rows),
                }
            )
            return train_indices, valid_indices, test_indices, metadata

        metadata["mode"] = "fallback_random"

    valid_positions, test_positions = random_split_indices(
        len(row_indices),
        rng=rng,
        valid_ratio=args.random_valid_ratio,
        test_ratio=args.random_test_ratio,
    )
    train_indices = []
    valid_indices = []
    test_indices = []
    for position, row_index in enumerate(row_indices):
        if position in test_positions:
            test_indices.append(row_index)
        elif position in valid_positions:
            valid_indices.append(row_index)
        else:
            train_indices.append(row_index)
    return train_indices, valid_indices, test_indices, metadata


def add_overlap_stats(train_frame: pd.DataFrame, eval_frame: pd.DataFrame) -> tuple[Counter[str], Counter[str]]:
    student_concepts: dict[Any, set[int]] = defaultdict(set)
    for row in train_frame.itertuples(index=False):
        student_concepts[row.stu_id].update(parse_concepts(row.cpt_seq))

    overlap_counter: Counter[str] = Counter()
    concept_count_counter: Counter[str] = Counter()
    for row in eval_frame.itertuples(index=False):
        concepts = set(parse_concepts(row.cpt_seq))
        seen_count = len(concepts & student_concepts[row.stu_id])
        if seen_count == 0:
            overlap_counter["none_seen"] += 1
        elif seen_count == len(concepts):
            overlap_counter["all_seen"] += 1
        else:
            overlap_counter["partial_seen"] += 1

        concept_count = len(concepts)
        concept_count_counter[str(concept_count) if concept_count < 4 else "4+"] += 1

    return overlap_counter, concept_count_counter


def counter_with_ratio(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    keys = sorted(counter.keys())
    return {
        key: {
            "count": int(counter[key]),
            "ratio": float(counter[key] / total) if total > 0 else 0.0,
        }
        for key in keys
    }


def build_split(full_frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(args.seed)
    grouped = list(full_frame.groupby("stu_id", sort=False))

    eligible_students = [
        student_id
        for student_id, student_frame in grouped
        if len(student_frame) >= args.min_student_interactions
        and len(set().union(*(set(parse_concepts(value)) for value in student_frame["cpt_seq"].tolist())))
        >= args.min_student_concepts
    ]
    rng.shuffle(eligible_students)
    holdout_student_count = round(args.holdout_student_frac * len(eligible_students))
    holdout_students = set(eligible_students[:holdout_student_count])

    train_indices: list[int] = []
    valid_indices: list[int] = []
    test_indices: list[int] = []
    assignment_rows: list[dict[str, Any]] = []

    for student_id, student_frame in grouped:
        train_part, valid_part, test_part, metadata = split_student_rows(
            student_id=student_id,
            student_frame=student_frame,
            holdout_enabled=student_id in holdout_students,
            rng=rng,
            args=args,
        )
        train_indices.extend(train_part)
        valid_indices.extend(valid_part)
        test_indices.extend(test_part)
        assignment_rows.append(metadata)

    train_frame = full_frame.loc[train_indices].reset_index(drop=True)
    valid_frame = full_frame.loc[valid_indices].reset_index(drop=True)
    test_frame = full_frame.loc[test_indices].reset_index(drop=True)
    assignments = pd.DataFrame(assignment_rows)
    return train_frame, valid_frame, test_frame, assignments


def build_summary(
    *,
    source_dir: Path,
    output_dir: Path,
    full_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    assignments: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    valid_overlap, valid_concept_counts = add_overlap_stats(train_frame, valid_frame)
    test_overlap, test_concept_counts = add_overlap_stats(train_frame, test_frame)

    total_rows = len(full_frame)
    mode_counts = assignments["mode"].value_counts().to_dict()
    strict_assignments = assignments[assignments["mode"] == "strict_holdout"]

    return {
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": args.seed,
        "holdout_student_frac": args.holdout_student_frac,
        "target_eval_ratio": args.target_eval_ratio,
        "min_eval_ratio": args.min_eval_ratio,
        "max_eval_ratio": args.max_eval_ratio,
        "test_ratio_within_eval": args.test_ratio_within_eval,
        "random_valid_ratio": args.random_valid_ratio,
        "random_test_ratio": args.random_test_ratio,
        "min_student_interactions": args.min_student_interactions,
        "min_student_concepts": args.min_student_concepts,
        "min_train_interactions": args.min_train_interactions,
        "transition_graph_policy": "copied_external_prior"
        if args.copy_transition_graph
        else "not_copied_rebuild_from_train",
        "rows": {
            "data": int(total_rows),
            "train": int(len(train_frame)),
            "valid": int(len(valid_frame)),
            "test": int(len(test_frame)),
        },
        "row_ratios": {
            "train": float(len(train_frame) / total_rows),
            "valid": float(len(valid_frame) / total_rows),
            "test": float(len(test_frame) / total_rows),
        },
        "students": {
            "total": int(full_frame["stu_id"].nunique()),
            "strict_holdout": int(mode_counts.get("strict_holdout", 0)),
            "fallback_random": int(mode_counts.get("fallback_random", 0)),
            "random": int(mode_counts.get("random", 0)),
        },
        "strict_holdout": {
            "mean_holdout_rows": float(strict_assignments["holdout_rows"].mean())
            if len(strict_assignments) > 0
            else 0.0,
            "min_holdout_rows": int(strict_assignments["holdout_rows"].min())
            if len(strict_assignments) > 0
            else 0,
            "max_holdout_rows": int(strict_assignments["holdout_rows"].max())
            if len(strict_assignments) > 0
            else 0,
        },
        "valid_overlap": counter_with_ratio(valid_overlap, len(valid_frame)),
        "test_overlap": counter_with_ratio(test_overlap, len(test_frame)),
        "valid_concept_count": counter_with_ratio(valid_concept_counts, len(valid_frame)),
        "test_concept_count": counter_with_ratio(test_concept_counts, len(test_frame)),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.holdout_student_frac <= 1.0:
        raise ValueError("--holdout-student-frac must be in [0, 1].")
    if args.min_eval_ratio > args.target_eval_ratio or args.target_eval_ratio > args.max_eval_ratio:
        raise ValueError("Require min_eval_ratio <= target_eval_ratio <= max_eval_ratio.")
    if args.random_valid_ratio + args.random_test_ratio >= 1.0:
        raise ValueError("random valid/test ratios must leave non-empty train mass.")

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{output_dir} exists and is not empty. Use --overwrite or choose another output dir.")

    full_frame = pd.read_csv(source_dir / "data.csv")
    train_frame, valid_frame, test_frame, assignments = build_split(full_frame, args)
    summary = build_summary(
        source_dir=source_dir,
        output_dir=output_dir,
        full_frame=full_frame,
        train_frame=train_frame,
        valid_frame=valid_frame,
        test_frame=test_frame,
        assignments=assignments,
        args=args,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stringify_concepts(full_frame).to_csv(output_dir / "data.csv", index=False, encoding="utf8")
    stringify_concepts(train_frame).to_csv(output_dir / "train.csv", index=False, encoding="utf8")
    stringify_concepts(valid_frame).to_csv(output_dir / "valid.csv", index=False, encoding="utf8")
    stringify_concepts(test_frame).to_csv(output_dir / "test.csv", index=False, encoding="utf8")
    shutil.copy2(source_dir / "Q_matrix.csv", output_dir / "Q_matrix.csv")
    assignments.to_csv(output_dir / "student_concept_holdout_assignments.csv", index=False, encoding="utf8")
    with open(output_dir / "split_summary.json", "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    source_graph_dir = source_dir / "transition_graph"
    if args.copy_transition_graph and source_graph_dir.exists():
        shutil.copytree(source_graph_dir, output_dir / "transition_graph")


if __name__ == "__main__":
    main()
