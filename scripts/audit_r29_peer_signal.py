from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.r29_protocol import canonical_concepts, stable_fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-only response-conditioned peer activation audit.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cells", type=int, default=20000)
    parser.add_argument("--retrieval-pool", type=int, default=256)
    parser.add_argument("--peer-k", type=int, default=32)
    parser.add_argument("--min-overlap", type=int, default=3)
    return parser.parse_args()


def _concept_rows(frame: pd.DataFrame) -> pd.DataFrame:
    expanded = frame.copy()
    expanded["cpt_seq"] = expanded["cpt_seq"].map(canonical_concepts).str.split(",")
    return expanded.explode("cpt_seq", ignore_index=True).rename(columns={"cpt_seq": "concept"})


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    frame = pd.read_csv(args.train, dtype={"stu_id": str, "exer_id": str, "cpt_seq": str})
    expanded = _concept_rows(frame)
    cells = (
        expanded.groupby(["stu_id", "concept"], as_index=False)
        .agg(correct=("label", "sum"), attempts=("label", "size"))
    )
    cells["accuracy"] = cells["correct"] / cells["attempts"]
    student_vectors: dict[str, dict[str, float]] = defaultdict(dict)
    concept_students: dict[str, list[str]] = defaultdict(list)
    cell_accuracy: dict[tuple[str, str], float] = {}
    for row in cells.itertuples(index=False):
        student = str(row.stu_id)
        concept = str(row.concept)
        value = 2.0 * float(row.accuracy) - 1.0
        student_vectors[student][concept] = value
        concept_students[concept].append(student)
        cell_accuracy[(student, concept)] = float(row.accuracy)

    concept_pools = {
        concept: sorted(
            students,
            key=lambda peer: stable_fraction(args.seed, "peer-pool", concept, peer),
        )[: args.retrieval_pool + 1]
        for concept, students in concept_students.items()
    }

    eligible = cells[cells["concept"].map(lambda value: len(concept_students[str(value)]) >= 2)].copy()
    eligible["sample_key"] = [
        stable_fraction(args.seed, "peer-audit", row.stu_id, row.concept)
        for row in eligible.itertuples(index=False)
    ]
    eligible = eligible.nsmallest(min(args.max_cells, len(eligible)), "sample_key")
    full_by_cell: dict[tuple[str, str], float] = {}
    control_by_cell: dict[tuple[str, str], float] = {}
    fallback = 0
    overlap_counts: list[int] = []
    for row in eligible.itertuples(index=False):
        student = str(row.stu_id)
        concept = str(row.concept)
        query = student_vectors[student]
        candidates = [peer for peer in concept_pools[concept] if peer != student][
            : args.retrieval_pool
        ]
        scored: list[tuple[float, int, str]] = []
        for peer in candidates:
            common = (set(query) & set(student_vectors[peer])) - {concept}
            overlap = len(common)
            if overlap < args.min_overlap:
                continue
            query_values = np.asarray([query[key] for key in common], dtype=np.float64)
            peer_values = np.asarray([student_vectors[peer][key] for key in common], dtype=np.float64)
            denom = float(np.linalg.norm(query_values) * np.linalg.norm(peer_values))
            similarity = float(query_values @ peer_values / denom) if denom > 0 else 0.0
            scored.append((similarity, overlap, peer))
        if not scored:
            fallback += 1
            peer_values = [cell_accuracy[(peer, concept)] for peer in candidates]
            prediction = float(np.mean(peer_values)) if peer_values else 0.5
            full_by_cell[(student, concept)] = prediction
            control_by_cell[(student, concept)] = prediction
            overlap_counts.append(0)
            continue
        scored.sort(key=lambda value: (value[0], value[1]), reverse=True)
        selected = scored[: args.peer_k]
        full_weights = np.asarray(
            [math.exp(2.0 * similarity) * math.log1p(overlap) for similarity, overlap, _ in selected]
        )
        control_weights = np.asarray([math.log1p(overlap) for _, overlap, _ in selected])
        targets = np.asarray([cell_accuracy[(peer, concept)] for _, _, peer in selected])
        full_by_cell[(student, concept)] = float(np.average(targets, weights=full_weights))
        control_by_cell[(student, concept)] = float(np.average(targets, weights=control_weights))
        overlap_counts.append(max(value[1] for value in selected))

    selected_keys = set(full_by_cell)
    selected_mask = [
        (student, concept) in selected_keys
        for student, concept in zip(
            expanded["stu_id"].astype(str), expanded["concept"].astype(str), strict=True
        )
    ]
    selected_rows = expanded.loc[selected_mask].copy()
    keys = list(zip(selected_rows["stu_id"].astype(str), selected_rows["concept"].astype(str), strict=True))
    labels = selected_rows["label"].astype(int).to_numpy()
    full = np.asarray([full_by_cell[key] for key in keys]).clip(1e-6, 1 - 1e-6)
    control = np.asarray([control_by_cell[key] for key in keys]).clip(1e-6, 1 - 1e-6)
    full_auc = float(roc_auc_score(labels, full))
    control_auc = float(roc_auc_score(labels, control))
    full_brier = float(brier_score_loss(labels, full))
    control_brier = float(brier_score_loss(labels, control))
    delta_auc = full_auc - control_auc
    delta_brier = full_brier - control_brier
    passed = delta_auc >= 0.002 and delta_brier <= -0.001
    return {
        "schema_version": 1,
        "dataset": args.dataset,
        "seed": args.seed,
        "history_source": "train_only",
        "pseudo_holdout_unit": "student_concept_cell",
        "cells": len(full_by_cell),
        "interaction_rows": len(labels),
        "fallback_cells": fallback,
        "mean_max_overlap": float(np.mean(overlap_counts)) if overlap_counts else 0.0,
        "full": {"auc": full_auc, "brier": full_brier},
        "coverage_control": {"auc": control_auc, "brier": control_brier},
        "delta_auc": delta_auc,
        "delta_brier": delta_brier,
        "activation_threshold": {"min_delta_auc": 0.002, "max_delta_brier": -0.001},
        "passed": passed,
        "fixed_peer_config": {
            "retrieval_pool": args.retrieval_pool,
            "peer_k": args.peer_k,
            "min_overlap": args.min_overlap,
        },
    }


def main() -> None:
    args = parse_args()
    if args.seed != 42:
        raise ValueError("The r29 peer audit is fixed to seed=42.")
    payload = run_audit(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
