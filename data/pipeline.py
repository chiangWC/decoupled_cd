from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch

from .datasets import (
    StepDataBundle,
    build_interaction_tensors,
    build_interaction_tensors_from_frame,
    build_student_exercise_mask,
    build_student_tkc_mask,
)
from .concept_graph import load_concept_graph_csv
from .mappings import build_unified_id_mappings
from .q_matrix import build_concept_graph_from_q, build_q_matrix_tensor, normalize_concept_sequence
from .readers import read_interactions, read_q_matrix


def build_history_tensors(
    *,
    history_interactions: pd.DataFrame,
    student_id_map: Dict[str, int],
    exercise_id_map: Dict[str, int],
    concept_id_map: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    student_exercise_mask = build_student_exercise_mask(
        interactions=history_interactions,
        student_id_map=student_id_map,
        exercise_id_map=exercise_id_map,
    )
    student_tkc_mask = build_student_tkc_mask(
        interactions=history_interactions,
        student_id_map=student_id_map,
        concept_id_map=concept_id_map,
    )
    student_ukc_mask = (1.0 - student_tkc_mask).clamp(min=0.0, max=1.0)
    response_matrix_tensor = build_response_matrix(
        interactions=history_interactions,
        student_id_map=student_id_map,
        exercise_id_map=exercise_id_map,
    )
    student_concept_evidence_tensor = build_student_concept_evidence_tensor(
        interactions=history_interactions,
        student_id_map=student_id_map,
        concept_id_map=concept_id_map,
    )
    return {
        "student_exercise_mask": student_exercise_mask,
        "student_tkc_mask": student_tkc_mask,
        "student_ukc_mask": student_ukc_mask,
        "response_matrix_tensor": response_matrix_tensor,
        "student_concept_evidence_tensor": student_concept_evidence_tensor,
    }


def build_student_concept_evidence_tensor(
    *,
    interactions: pd.DataFrame,
    student_id_map: Dict[str, int],
    concept_id_map: Dict[str, int],
) -> torch.Tensor:
    attempt_counts = torch.zeros(len(student_id_map), len(concept_id_map), dtype=torch.float32)
    correct_counts = torch.zeros_like(attempt_counts)

    for row in interactions.itertuples(index=False):
        student_index = student_id_map[str(row.stu_id)]
        for concept_id in normalize_concept_sequence(row.cpt_seq):
            concept_index = concept_id_map.get(concept_id)
            if concept_index is None:
                continue
            attempt_counts[student_index, concept_index] += 1.0
            correct_counts[student_index, concept_index] += float(row.label)

    incorrect_counts = (attempt_counts - correct_counts).clamp_min(0.0)
    accuracy = correct_counts / attempt_counts.clamp_min(1.0)
    log_attempts = torch.log1p(attempt_counts)
    seen = (attempt_counts > 0.0).to(dtype=torch.float32)
    return torch.stack(
        [attempt_counts, correct_counts, incorrect_counts, accuracy, log_attempts, seen],
        dim=-1,
    )


def prepare_data_bundle(
    *,
    interactions_path: str | Path,
    q_matrix_path: str | Path,
) -> Dict[str, Any]:
    """
    Load the minimum data artifacts required by the new project:
    interactions, Q metadata, and unified ID mappings.
    """
    interactions = read_interactions(interactions_path)
    q_matrix = read_q_matrix(q_matrix_path)
    mappings = build_unified_id_mappings(interactions, q_matrix)

    return {
        "interactions": interactions,
        "q_matrix": q_matrix,
        "student_id_map": mappings["student_id_map"],
        "exercise_id_map": mappings["exercise_id_map"],
        "concept_id_map": mappings["concept_id_map"],
        "num_students": len(mappings["student_id_map"]),
        "num_exercises": len(mappings["exercise_id_map"]),
        "num_concepts": len(mappings["concept_id_map"]),
    }


def prepare_step_data_bundle(
    *,
    interactions_path: str | Path,
    q_matrix_path: str | Path,
    concept_graph_path: str | Path | None = None,
    prerequisite_graph_path: str | Path | None = None,
    similarity_graph_path: str | Path | None = None,
) -> StepDataBundle:
    base = prepare_data_bundle(interactions_path=interactions_path, q_matrix_path=q_matrix_path)

    q_matrix_tensor = build_q_matrix_tensor(
        q_matrix=base["q_matrix"],
        exercise_id_map=base["exercise_id_map"],
        concept_id_map=base["concept_id_map"],
    )
    concept_graph = (
        load_concept_graph_csv(concept_graph_path)
        if concept_graph_path is not None
        else build_concept_graph_from_q(q_matrix_tensor=q_matrix_tensor)
    )
    prerequisite_graph = (
        load_concept_graph_csv(prerequisite_graph_path) if prerequisite_graph_path is not None else None
    )
    similarity_graph = load_concept_graph_csv(similarity_graph_path) if similarity_graph_path is not None else None
    history_tensors = build_history_tensors(
        history_interactions=base["interactions"],
        student_id_map=base["student_id_map"],
        exercise_id_map=base["exercise_id_map"],
        concept_id_map=base["concept_id_map"],
    )
    interaction_student_ids, interaction_exercise_ids, interaction_labels = build_interaction_tensors(
        interactions=base["interactions"],
        student_id_map=base["student_id_map"],
        exercise_id_map=base["exercise_id_map"],
    )

    return StepDataBundle(
        interactions=base["interactions"],
        history_interactions=base["interactions"],
        split_name="full",
        allow_target_in_history=True,
        q_matrix=base["q_matrix"],
        student_id_map=base["student_id_map"],
        exercise_id_map=base["exercise_id_map"],
        concept_id_map=base["concept_id_map"],
        q_matrix_tensor=q_matrix_tensor,
        concept_graph=concept_graph,
        student_exercise_mask=history_tensors["student_exercise_mask"],
        student_tkc_mask=history_tensors["student_tkc_mask"],
        student_ukc_mask=history_tensors["student_ukc_mask"],
        response_matrix_tensor=history_tensors["response_matrix_tensor"],
        interaction_student_ids=interaction_student_ids,
        interaction_exercise_ids=interaction_exercise_ids,
        interaction_labels=interaction_labels,
        student_concept_evidence_tensor=history_tensors["student_concept_evidence_tensor"],
        prerequisite_graph=prerequisite_graph,
        similarity_graph=similarity_graph,
    )


def build_response_matrix(
    interactions: pd.DataFrame,
    student_id_map: Dict[str, int],
    exercise_id_map: Dict[str, int],
) -> torch.Tensor:
    response_matrix = torch.zeros(len(student_id_map), len(exercise_id_map), dtype=torch.float32)
    counts = torch.zeros(len(student_id_map), len(exercise_id_map), dtype=torch.float32)

    for row in interactions.itertuples(index=False):
        student_index = student_id_map[str(row.stu_id)]
        exercise_index = exercise_id_map[str(row.exer_id)]
        response_matrix[student_index, exercise_index] += float(row.label)
        counts[student_index, exercise_index] += 1.0

    return response_matrix / counts.clamp(min=1.0)


def prepare_experiment_split_bundles(
    *,
    train_interactions_path: str | Path,
    valid_interactions_path: str | Path,
    test_interactions_path: str | Path,
    q_matrix_path: str | Path,
    concept_graph_path: str | Path | None = None,
    prerequisite_graph_path: str | Path | None = None,
    similarity_graph_path: str | Path | None = None,
) -> Dict[str, Any]:
    train_df = read_interactions(train_interactions_path)
    valid_df = read_interactions(valid_interactions_path)
    test_df = read_interactions(test_interactions_path)
    q_matrix = read_q_matrix(q_matrix_path)

    all_interactions = pd.concat([train_df, valid_df, test_df], ignore_index=True)
    mappings = build_unified_id_mappings(all_interactions, q_matrix)

    q_matrix_tensor = build_q_matrix_tensor(
        q_matrix=q_matrix,
        exercise_id_map=mappings["exercise_id_map"],
        concept_id_map=mappings["concept_id_map"],
    )
    concept_graph = (
        load_concept_graph_csv(concept_graph_path)
        if concept_graph_path is not None
        else build_concept_graph_from_q(q_matrix_tensor=q_matrix_tensor)
    )
    prerequisite_graph = (
        load_concept_graph_csv(prerequisite_graph_path) if prerequisite_graph_path is not None else None
    )
    similarity_graph = load_concept_graph_csv(similarity_graph_path) if similarity_graph_path is not None else None

    history_tensors = build_history_tensors(
        history_interactions=train_df,
        student_id_map=mappings["student_id_map"],
        exercise_id_map=mappings["exercise_id_map"],
        concept_id_map=mappings["concept_id_map"],
    )

    train_ids = build_interaction_tensors_from_frame(
        train_df, mappings["student_id_map"], mappings["exercise_id_map"]
    )
    valid_ids = build_interaction_tensors_from_frame(
        valid_df, mappings["student_id_map"], mappings["exercise_id_map"]
    )
    test_ids = build_interaction_tensors_from_frame(
        test_df, mappings["student_id_map"], mappings["exercise_id_map"]
    )

    shared = {
        "q_matrix": q_matrix,
        "q_matrix_tensor": q_matrix_tensor,
        "concept_graph": concept_graph,
        "prerequisite_graph": prerequisite_graph,
        "similarity_graph": similarity_graph,
        "student_id_map": mappings["student_id_map"],
        "exercise_id_map": mappings["exercise_id_map"],
        "concept_id_map": mappings["concept_id_map"],
        "student_exercise_mask": history_tensors["student_exercise_mask"],
        "student_tkc_mask": history_tensors["student_tkc_mask"],
        "student_ukc_mask": history_tensors["student_ukc_mask"],
        "student_concept_evidence_tensor": history_tensors["student_concept_evidence_tensor"],
    }

    def _bundle(
        frame: pd.DataFrame,
        ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        split_name: str,
        allow_target_in_history: bool,
    ) -> StepDataBundle:
        student_ids, exercise_ids, labels = ids
        return StepDataBundle(
            interactions=frame,
            history_interactions=train_df,
            split_name=split_name,
            allow_target_in_history=allow_target_in_history,
            q_matrix=q_matrix,
            student_id_map=mappings["student_id_map"],
            exercise_id_map=mappings["exercise_id_map"],
            concept_id_map=mappings["concept_id_map"],
            q_matrix_tensor=q_matrix_tensor,
            concept_graph=concept_graph,
            student_exercise_mask=history_tensors["student_exercise_mask"],
            student_tkc_mask=history_tensors["student_tkc_mask"],
            student_ukc_mask=history_tensors["student_ukc_mask"],
            response_matrix_tensor=history_tensors["response_matrix_tensor"],
            interaction_student_ids=student_ids,
            interaction_exercise_ids=exercise_ids,
            interaction_labels=labels,
            student_concept_evidence_tensor=history_tensors["student_concept_evidence_tensor"],
            prerequisite_graph=prerequisite_graph,
            similarity_graph=similarity_graph,
        )

    return {
        "train": _bundle(train_df, train_ids, split_name="train", allow_target_in_history=True),
        "valid": _bundle(valid_df, valid_ids, split_name="valid", allow_target_in_history=False),
        "test": _bundle(test_df, test_ids, split_name="test", allow_target_in_history=False),
        "shared": shared,
    }
