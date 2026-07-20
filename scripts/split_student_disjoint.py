from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pool_protocol import (
    assign_student_disjoint_support_query,
    canonical_concepts,
    canonicalize_interactions,
    canonicalize_q_matrix,
    sha256_file,
    write_student_disjoint_variant,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic student-disjoint support-to-query protocol "
            "while preserving an existing standard train/valid/test split."
        )
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--min-support-groups", type=int, default=10)
    parser.add_argument("--min-query-groups", type=int, default=3)
    parser.add_argument("--min-target-label-count", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_output_location(
    source_dir: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    source = source_dir.resolve()
    output = output_dir.resolve()
    if (
        source == output
        or source.is_relative_to(output)
        or output.is_relative_to(source)
    ):
        raise ValueError(
            "Output directory must be separate from the source directory; "
            "equal, ancestor, and descendant paths are refused."
        )
    return source, output


def _read_source(source_dir: Path) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    frames: dict[str, pd.DataFrame] = {}
    removed: dict[str, int] = {}
    for split in ("train", "valid", "test"):
        frame, count = canonicalize_interactions(
            pd.read_csv(source_dir / f"{split}.csv")
        )
        frames[split] = frame
        removed[split] = int(count)

    row_sets = {
        name: set(frame["source_row_id"].astype(str))
        for name, frame in frames.items()
    }
    overlaps = {
        "train_valid": len(row_sets["train"] & row_sets["valid"]),
        "train_test": len(row_sets["train"] & row_sets["test"]),
        "valid_test": len(row_sets["valid"] & row_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(
            "Source standard split contains exact rows in multiple splits: "
            f"{overlaps}"
        )
    group_sets = {
        name: set(
            zip(
                frame["stu_id"].astype(str),
                frame["exer_id"].astype(str),
                strict=True,
            )
        )
        for name, frame in frames.items()
    }
    group_overlaps = {
        "train_valid": len(group_sets["train"] & group_sets["valid"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "valid_test": len(group_sets["valid"] & group_sets["test"]),
    }
    if any(group_overlaps.values()):
        raise RuntimeError(
            "Source standard split is not atomic by student-exercise group: "
            f"{group_overlaps}"
        )
    return frames, removed


def main() -> None:
    args = parse_args()
    if args.split_seed != 2024:
        raise ValueError("The student-disjoint protocol fixes split_seed=2024.")

    source_dir, output_dir = validate_output_location(
        Path(args.source_dir),
        Path(args.output_dir),
    )
    source, duplicates_removed = _read_source(source_dir)
    source_q_path = source_dir / "Q_matrix.csv"
    q_matrix = canonicalize_q_matrix(pd.read_csv(source_q_path))
    q_lookup: dict[str, set[str]] = defaultdict(set)
    for row in q_matrix.itertuples(index=False):
        q_lookup[str(row.exer_id)].update(
            canonical_concepts(row.cpt_seq).split(",")
        )
    q_mismatches: list[tuple[str, str, list[str] | None]] = []
    for split_name, frame in source.items():
        for row in frame[["exer_id", "cpt_seq"]].itertuples(index=False):
            expected = q_lookup.get(str(row.exer_id))
            actual = set(canonical_concepts(row.cpt_seq).split(","))
            if expected is None or not actual.issubset(expected):
                q_mismatches.append(
                    (
                        split_name,
                        str(row.exer_id),
                        None if expected is None else sorted(expected),
                    )
                )
                if len(q_mismatches) >= 10:
                    break
        if len(q_mismatches) >= 10:
            break
    if q_mismatches:
        raise RuntimeError(
            "Source interaction concept sequences disagree with Q_matrix.csv; "
            f"first mismatches={q_mismatches}"
        )

    splits, audit = assign_student_disjoint_support_query(
        train_frame=source["train"],
        valid_frame=source["valid"],
        test_frame=source["test"],
        q_matrix=q_matrix,
        seed=args.split_seed,
        min_support_groups=args.min_support_groups,
        min_query_groups=args.min_query_groups,
        min_target_label_count=args.min_target_label_count,
    )
    required_nonempty = (
        "train",
        "valid_support",
        "valid_query",
        "test_support",
        "test_query",
    )
    empty = [name for name in required_nonempty if splits[name].empty]
    if empty:
        raise RuntimeError(f"Student-disjoint protocol produced empty splits: {empty}")

    source_files = {
        f"{name}.csv": {
            "sha256": sha256_file(source_dir / f"{name}.csv"),
            "rows": int(len(pd.read_csv(source_dir / f"{name}.csv"))),
            "exact_duplicates_removed": duplicates_removed[name],
        }
        for name in ("train", "valid", "test")
    }
    source_files["Q_matrix.csv"] = {
        "sha256": sha256_file(source_q_path),
        "rows": int(len(pd.read_csv(source_q_path))),
    }
    audit.update(
        {
            "source_directory": str(source_dir.resolve()),
            "source_files": source_files,
            "q_mapping_conflicts": 0,
            "q_interaction_mismatches": 0,
            "source_student_exercise_group_overlap": {
                "train_valid": 0,
                "train_test": 0,
                "valid_test": 0,
            },
            "preserves_standard_roles": True,
        }
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} is non-empty; pass --overwrite to rebuild it."
            )
        shutil.rmtree(output_dir)

    manifest = write_student_disjoint_variant(
        output_dir,
        splits=splits,
        q_matrix=q_matrix,
        audit=audit,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
