from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.r29_protocol import (
    SPLIT_NAMES,
    add_split_row_index,
    assign_atomic_holdout_split,
    assign_atomic_standard_split,
    canonicalize_interactions,
    derive_q_matrix,
    write_dataset_variant,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reproducible r29 Junyi and EdNet protocols.")
    parser.add_argument("--dataset", choices=["junyi", "ednet", "all"], default="all")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--junyi-holdout-source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split-seed", type=int, default=2024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def _replace_directory(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} is non-empty; pass --overwrite to rebuild it.")
        shutil.rmtree(path)


def _canonical_source_splits(source: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], int]:
    frames: dict[str, pd.DataFrame] = {}
    removed = 0
    for split in SPLIT_NAMES:
        frame, split_removed = canonicalize_interactions(pd.read_csv(source / f"{split}.csv"))
        frames[split] = add_split_row_index(frame)
        removed += split_removed
    full, cross_split_removed = canonicalize_interactions(
        pd.concat([frames[name].drop(columns="split_row_index") for name in SPLIT_NAMES], ignore_index=True)
    )
    if cross_split_removed:
        raise RuntimeError(f"Source contains {cross_split_removed} exact rows in multiple Junyi splits.")
    return full, frames, removed


def prepare_junyi(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    standard_dir = output_root / "junyi_standard_v2"
    holdout_dir = output_root / "junyi_chold_v2"
    _replace_directory(standard_dir, overwrite=args.overwrite)
    _replace_directory(holdout_dir, overwrite=args.overwrite)

    standard_full, standard_splits, removed = _canonical_source_splits(data_root / "junyi")
    holdout_full, holdout_splits, holdout_removed = _canonical_source_splits(
        Path(args.junyi_holdout_source)
    )
    if set(standard_full["source_row_id"]) != set(holdout_full["source_row_id"]):
        raise RuntimeError("Junyi standard and holdout sources do not contain the same canonical rows.")
    q_matrix, conflicts = derive_q_matrix(standard_full)
    one_to_one = bool((q_matrix["cpt_seq"].str.count(",") == 0).all())
    if conflicts or not one_to_one or q_matrix["exer_id"].nunique() != q_matrix["cpt_seq"].nunique():
        raise RuntimeError("Junyi must retain its verified one-exercise/one-concept mapping.")
    return {
        "source": str((data_root / "junyi").resolve()),
        "split_seed": args.split_seed,
        "exact_duplicates_removed": removed + holdout_removed,
        "q_mapping_conflicts": conflicts,
        "exercise_concept_one_to_one": one_to_one,
        "target_semantics": "exact-zero is simultaneously student-unseen item and concept",
        "standard": write_dataset_variant(
            standard_dir,
            full_frame=standard_full,
            splits=standard_splits,
            q_matrix=q_matrix,
        ),
        "holdout": write_dataset_variant(
            holdout_dir,
            full_frame=holdout_full,
            splits=holdout_splits,
            q_matrix=q_matrix,
        ),
    }


def prepare_ednet(args: argparse.Namespace) -> dict[str, Any]:
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    standard_dir = output_root / "ednet_clean_v2"
    holdout_dir = output_root / "ednet_clean_chold_v2"
    _replace_directory(standard_dir, overwrite=args.overwrite)
    _replace_directory(holdout_dir, overwrite=args.overwrite)

    raw = pd.concat(
        [pd.read_csv(data_root / "ednet_icdm" / f"{split}.csv") for split in SPLIT_NAMES],
        ignore_index=True,
    )
    clean, removed = canonicalize_interactions(raw)
    q_matrix, conflicts = derive_q_matrix(clean)
    if conflicts:
        raise RuntimeError(f"EdNet has {conflicts} conflicting exercise-to-Q mappings.")
    standard_splits = assign_atomic_standard_split(clean, seed=args.split_seed)
    holdout_splits, assignments = assign_atomic_holdout_split(clean, seed=args.split_seed)
    standard = write_dataset_variant(
        standard_dir,
        full_frame=clean,
        splits=standard_splits,
        q_matrix=q_matrix,
    )
    holdout = write_dataset_variant(
        holdout_dir,
        full_frame=clean,
        splits=holdout_splits,
        q_matrix=q_matrix,
        assignments=assignments,
    )
    for report in (standard, holdout):
        if any(report["student_exercise_group_overlap"].values()):
            raise RuntimeError("EdNet atomic student-exercise split invariant failed.")
    return {
        "source": str((data_root / "ednet_icdm").resolve()),
        "split_seed": args.split_seed,
        "raw_rows": len(raw),
        "clean_rows": len(clean),
        "exact_duplicates_removed": removed,
        "conflicting_repeated_attempts_retained": int(
            clean.groupby(["stu_id", "exer_id"])["label"].nunique().gt(1).sum()
        ),
        "q_mapping_conflicts": conflicts,
        "standard": standard,
        "holdout": holdout,
    }


def main() -> None:
    args = parse_args()
    if args.split_seed != 2024:
        raise ValueError("r29 data protocols are fixed to split_seed=2024.")
    payload: dict[str, Any] = {"schema_version": 1, "split_seed": 2024, "datasets": {}}
    if args.dataset in {"junyi", "all"}:
        payload["datasets"]["junyi"] = prepare_junyi(args)
    if args.dataset in {"ednet", "all"}:
        payload["datasets"]["ednet"] = prepare_ednet(args)
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
