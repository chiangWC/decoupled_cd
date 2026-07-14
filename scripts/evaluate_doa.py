from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.q_matrix import normalize_concept_sequence
from models import CountPriorBaseline
from scripts.analyze_prediction_slices import load_model
from scripts.evaluate_coverage_slice import prepare_bundles_from_paths, resolve_split_paths
from scripts.evaluate_history_hiding_stress import (
    infer_model_name,
    load_summary,
    normalize_summary_for_current_loader,
)
from trainers.engine import _bundle_tensors
from utils import compute_doa, resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Degree-of-Agreement evaluation over diagnosed per-concept mastery. "
            "Supports v2 models with a mastery head and the b0 count baseline "
            "(smoothed per-concept accuracy as its mastery estimate). "
            "With --holdout-assignments, also reports DOA restricted to "
            "(student, concept) pairs that were strictly held out."
        )
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--summary", action="append", required=True, help="Training summary JSON. Repeat per model.")
    parser.add_argument("--model-name", action="append", default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--train-interactions", default=None)
    parser.add_argument("--valid-interactions", default=None)
    parser.add_argument("--test-interactions", default=None)
    parser.add_argument("--q-matrix", default=None)
    parser.add_argument("--concept-graph", default=None)
    parser.add_argument(
        "--holdout-assignments",
        default=None,
        help="student_concept_holdout_assignments.csv from split_student_concept_holdout.py.",
    )
    parser.add_argument("--min-responses", type=int, default=1)
    parser.add_argument("--max-pairs-per-concept", type=int, default=100_000)
    parser.add_argument("--doa-seed", type=int, default=2024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-csv", default=None)
    return parser.parse_args()


def extract_mastery(*, model: Any, bundle: Any, device: str) -> torch.Tensor | None:
    """Return a (num_students, num_concepts) mastery matrix in pipeline index space."""
    if isinstance(model, CountPriorBaseline):
        evidence = bundle.student_concept_evidence_tensor
        mask = bundle.student_exercise_mask
        response = bundle.response_matrix_tensor
        global_attempts = mask.sum()
        global_correct = (mask * response).sum()
        global_rate = (
            (global_correct / global_attempts.clamp_min(1.0)) if float(global_attempts) > 0 else torch.tensor(0.5)
        ).clamp(min=1e-6, max=1.0 - 1e-6)
        prior_weight = float(model.prior_weight)
        return (evidence[..., 1] + global_rate * prior_weight) / (evidence[..., 0] + prior_weight)

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()
    tensors = _bundle_tensors(bundle, torch_device)
    # DOA consumes the complete per-student mastery matrix, not interaction
    # predictions.  A single valid target satisfies model forward contracts
    # without materializing a redundant state tensor for every test row.
    target_student_ids = tensors["interaction_student_ids"][:1]
    target_exercise_ids = tensors["interaction_exercise_ids"][:1]
    if target_student_ids.numel() == 0:
        raise ValueError("Cannot extract mastery from an empty split.")
    with torch.no_grad():
        output = model(
            q_matrix=tensors["q_matrix"],
            concept_graph=tensors["concept_graph"],
            prerequisite_graph=tensors["prerequisite_graph"],
            similarity_graph=tensors["similarity_graph"],
            student_exercise_mask=tensors["student_exercise_mask"],
            response_matrix=tensors["response_matrix"],
            student_tkc_mask=tensors["student_tkc_mask"],
            student_ukc_mask=tensors["student_ukc_mask"],
            student_concept_evidence=tensors["student_concept_evidence"],
            exercise_evidence=tensors["exercise_evidence"],
            target_student_ids=target_student_ids,
            target_exercise_ids=target_exercise_ids,
        )
    mastery = getattr(output, "mastery", None)
    return mastery.detach().cpu() if mastery is not None else None


def build_response_rows(
    *,
    bundle: Any,
) -> tuple[list[int], list[list[int]], list[float]]:
    student_id_map = bundle.student_id_map
    concept_id_map = bundle.concept_id_map
    student_ids: list[int] = []
    concept_lists: list[list[int]] = []
    labels: list[float] = []
    for row in bundle.interactions.itertuples(index=False):
        student_index = student_id_map.get(str(row.stu_id))
        if student_index is None:
            continue
        concepts = [
            concept_id_map[token]
            for token in normalize_concept_sequence(row.cpt_seq)
            if token in concept_id_map
        ]
        student_ids.append(student_index)
        concept_lists.append(concepts)
        labels.append(float(row.label))
    return student_ids, concept_lists, labels


def load_holdout_map(path: str, *, student_id_map: dict, concept_id_map: dict) -> dict[int, set[int]]:
    frame = pd.read_csv(path)
    holdout_map: dict[int, set[int]] = {}
    for row in frame.itertuples(index=False):
        raw_concepts = getattr(row, "holdout_concepts", "")
        if pd.isna(raw_concepts) or not str(raw_concepts).strip():
            continue
        student_index = student_id_map.get(str(row.stu_id))
        if student_index is None:
            continue
        concepts = {
            concept_id_map[token.strip()]
            for token in str(raw_concepts).split(",")
            if token.strip() in concept_id_map
        }
        if concepts:
            holdout_map[student_index] = concepts
    return holdout_map


def restrict_to_holdout(
    student_ids: list[int],
    concept_lists: list[list[int]],
    labels: list[float],
    holdout_map: dict[int, set[int]],
) -> tuple[list[int], list[list[int]], list[float]]:
    filtered_students: list[int] = []
    filtered_concepts: list[list[int]] = []
    filtered_labels: list[float] = []
    for student, concepts, label in zip(student_ids, concept_lists, labels, strict=True):
        held = holdout_map.get(student)
        if not held:
            continue
        kept = [concept for concept in concepts if concept in held]
        if not kept:
            continue
        filtered_students.append(student)
        filtered_concepts.append(kept)
        filtered_labels.append(label)
    return filtered_students, filtered_concepts, filtered_labels


def main() -> None:
    args = parse_args()
    device = str(resolve_device(args.device, args.gpus))
    model_names = args.model_name or [infer_model_name(path) for path in args.summary]

    rows: list[dict[str, Any]] = []
    for summary_path, model_name in zip(args.summary, model_names, strict=True):
        summary = normalize_summary_for_current_loader(load_summary(summary_path))
        paths = resolve_split_paths(summary, args)
        bundles = prepare_bundles_from_paths(paths)
        bundle = bundles[args.split]
        model = load_model(
            summary=summary,
            checkpoint_path=summary["best_checkpoint_path"],
            bundles=bundles,
            concept_dim=int(summary["concept_dim"]),
            device=device,
        )
        mastery = extract_mastery(model=model, bundle=bundle, device=device)
        if mastery is None:
            print(f"[skip] {model_name}: model exposes no per-concept mastery (train with a mastery head).")
            continue
        mastery_np = mastery.numpy()

        student_ids, concept_lists, labels = build_response_rows(bundle=bundle)
        row: dict[str, Any] = {"dataset": args.dataset_name, "model": model_name, "split": args.split}
        standard = compute_doa(
            mastery=mastery_np,
            student_ids=student_ids,
            concept_lists=concept_lists,
            labels=labels,
            min_responses=args.min_responses,
            max_pairs_per_concept=args.max_pairs_per_concept,
            seed=args.doa_seed,
        )
        row.update({f"doa_{k}" if not k.startswith("doa") else k: v for k, v in standard.items()})

        if args.holdout_assignments:
            holdout_map = load_holdout_map(
                args.holdout_assignments,
                student_id_map=bundle.student_id_map,
                concept_id_map=bundle.concept_id_map,
            )
            held_students, held_concepts, held_labels = restrict_to_holdout(
                student_ids, concept_lists, labels, holdout_map
            )
            holdout = compute_doa(
                mastery=mastery_np,
                student_ids=held_students,
                concept_lists=held_concepts,
                labels=held_labels,
                min_responses=args.min_responses,
                max_pairs_per_concept=args.max_pairs_per_concept,
                seed=args.doa_seed,
            )
            row.update({f"holdout_{k}": v for k, v in holdout.items()})
            row["holdout_rows"] = len(held_labels)
        rows.append(row)
        print(row)

    write_json({"dataset": args.dataset_name, "split": args.split, "rows": rows}, args.output)
    if args.summary_csv:
        pd.DataFrame(rows).to_csv(args.summary_csv, index=False)


if __name__ == "__main__":
    main()
