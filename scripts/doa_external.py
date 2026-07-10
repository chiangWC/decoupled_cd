from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.q_matrix import normalize_concept_sequence
from scripts.plugin_campaign import sha256_file
from utils import compute_doa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DOA for an externally extracted mastery matrix (mastery.npy + id_maps.json "
            "written by the extract_mastery_* scripts, using the external model's own id order)."
        )
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--mastery-dir", action="append", required=True, help="Dir with mastery.npy + id_maps.json.")
    parser.add_argument("--model-name", action="append", required=True)
    parser.add_argument("--holdout-assignments", default=None)
    parser.add_argument("--min-responses", type=int, default=1)
    parser.add_argument("--max-pairs-per-concept", type=int, default=100_000)
    parser.add_argument("--doa-seed", type=int, default=2024)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_frame = pd.read_csv(Path(args.split_dir) / f"{args.split}.csv")

    rows = []
    for mastery_dir, model_name in zip(args.mastery_dir, args.model_name, strict=True):
        mastery_path = Path(mastery_dir) / "mastery.npy"
        id_maps_path = Path(mastery_dir) / "id_maps.json"
        mastery = np.load(mastery_path)
        id_maps = json.loads(id_maps_path.read_text(encoding="utf-8"))
        stu2row = {token: index for index, token in enumerate(id_maps["stu_ids"])}
        cpt2col = {token: index for index, token in enumerate(id_maps["cpt_ids"])}

        student_ids, concept_lists, labels = [], [], []
        for row in evaluation_frame.itertuples(index=False):
            student_index = stu2row.get(str(row.stu_id))
            if student_index is None:
                continue
            concepts = [
                cpt2col[token] for token in normalize_concept_sequence(row.cpt_seq) if token in cpt2col
            ]
            student_ids.append(student_index)
            concept_lists.append(concepts)
            labels.append(float(row.label))

        result = {
            "dataset": args.dataset_name,
            "model": model_name,
            "split": args.split,
            "doa_seed": args.doa_seed,
            "min_responses": args.min_responses,
            "max_pairs_per_concept": args.max_pairs_per_concept,
            "mastery_sha256": sha256_file(mastery_path),
            "id_maps_sha256": sha256_file(id_maps_path),
        }
        result.update(
            compute_doa(
                mastery=mastery,
                student_ids=student_ids,
                concept_lists=concept_lists,
                labels=labels,
                min_responses=args.min_responses,
                max_pairs_per_concept=args.max_pairs_per_concept,
                seed=args.doa_seed,
            )
        )

        if args.holdout_assignments:
            assignments = pd.read_csv(args.holdout_assignments)
            holdout_map: dict[int, set[int]] = {}
            for arow in assignments.itertuples(index=False):
                raw = getattr(arow, "holdout_concepts", "")
                if pd.isna(raw) or not str(raw).strip():
                    continue
                student_index = stu2row.get(str(arow.stu_id))
                if student_index is None:
                    continue
                held = {cpt2col[t.strip()] for t in str(raw).split(",") if t.strip() in cpt2col}
                if held:
                    holdout_map[student_index] = held
            h_students, h_concepts, h_labels = [], [], []
            for student_index, concepts, label in zip(student_ids, concept_lists, labels, strict=True):
                held = holdout_map.get(student_index)
                if not held:
                    continue
                kept = [c for c in concepts if c in held]
                if kept:
                    h_students.append(student_index)
                    h_concepts.append(kept)
                    h_labels.append(label)
            holdout_result = compute_doa(
                mastery=mastery,
                student_ids=h_students,
                concept_lists=h_concepts,
                labels=h_labels,
                min_responses=args.min_responses,
                max_pairs_per_concept=args.max_pairs_per_concept,
                seed=args.doa_seed,
            )
            result.update({f"holdout_{k}": v for k, v in holdout_result.items()})
        rows.append(result)
        print(result)

    pd.DataFrame(rows).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
