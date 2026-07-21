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
from .static_relations import load_static_relation_graph


def _replace_history_concepts_from_q(
    *,
    history_interactions: pd.DataFrame,
    q_matrix: pd.DataFrame,
) -> pd.DataFrame:
    exercise_concepts: dict[str, set[str]] = {}
    for row in q_matrix.itertuples(index=False):
        exercise_concepts.setdefault(str(row.exer_id), set()).update(
            normalize_concept_sequence(row.cpt_seq)
        )
    history_exercises = set(history_interactions["exer_id"].astype(str))
    missing = sorted(history_exercises - set(exercise_concepts))
    if missing:
        raise ValueError(
            "History contains exercises absent from Q-matrix: "
            f"{missing[:10]}"
        )
    output = history_interactions.copy()
    output["cpt_seq"] = [
        ",".join(sorted(exercise_concepts[str(exercise)]))
        for exercise in output["exer_id"]
    ]
    return output


def build_history_tensors(
    *,
    history_interactions: pd.DataFrame,
    student_id_map: Dict[str, int],
    exercise_id_map: Dict[str, int],
    concept_id_map: Dict[str, int],
    q_matrix: pd.DataFrame | None = None,
) -> Dict[str, torch.Tensor]:
    concept_interactions = (
        _replace_history_concepts_from_q(
            history_interactions=history_interactions,
            q_matrix=q_matrix,
        )
        if q_matrix is not None
        else history_interactions
    )
    student_exercise_mask = build_student_exercise_mask(
        interactions=history_interactions,
        student_id_map=student_id_map,
        exercise_id_map=exercise_id_map,
    )
    student_tkc_mask = build_student_tkc_mask(
        interactions=concept_interactions,
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
        interactions=concept_interactions,
        student_id_map=student_id_map,
        concept_id_map=concept_id_map,
    )
    exercise_evidence_tensor = build_exercise_evidence_tensor(
        interactions=history_interactions,
        exercise_id_map=exercise_id_map,
    )
    return {
        "student_exercise_mask": student_exercise_mask,
        "student_tkc_mask": student_tkc_mask,
        "student_ukc_mask": student_ukc_mask,
        "response_matrix_tensor": response_matrix_tensor,
        "student_concept_evidence_tensor": student_concept_evidence_tensor,
        "exercise_evidence_tensor": exercise_evidence_tensor,
    }


def _student_keys(frame: pd.DataFrame) -> set[str]:
    return set(frame["stu_id"].astype(str))


def _student_exercise_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            frame["stu_id"].astype(str),
            frame["exer_id"].astype(str),
            strict=True,
        )
    )


def _validate_separate_history_protocol(
    *,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    valid_history: pd.DataFrame,
    test_history: pd.DataFrame,
) -> None:
    students = {
        "train": _student_keys(train),
        "valid": _student_keys(valid),
        "test": _student_keys(test),
    }
    overlaps = {
        "train_valid": students["train"] & students["valid"],
        "train_test": students["train"] & students["test"],
        "valid_test": students["valid"] & students["test"],
    }
    nonempty_overlaps = {
        name: sorted(values)[:10]
        for name, values in overlaps.items()
        if values
    }
    if nonempty_overlaps:
        raise ValueError(
            "Student-disjoint protocol has student overlap: "
            f"{nonempty_overlaps}"
        )

    for split_name, query, history in (
        ("valid", valid, valid_history),
        ("test", test, test_history),
    ):
        query_students = _student_keys(query)
        history_students = _student_keys(history)
        if query_students != history_students:
            raise ValueError(
                f"{split_name} query/support student sets differ: "
                f"query_only={sorted(query_students - history_students)[:10]}, "
                f"support_only={sorted(history_students - query_students)[:10]}"
            )
        group_overlap = (
            _student_exercise_keys(query)
            & _student_exercise_keys(history)
        )
        if group_overlap:
            raise ValueError(
                f"{split_name} query reuses support student-exercise groups: "
                f"{sorted(group_overlap)[:10]}"
            )

    known_exercises = set(train["exer_id"].astype(str))
    for split_name, frame in (
        ("valid", valid),
        ("test", test),
        ("valid_support", valid_history),
        ("test_support", test_history),
    ):
        unknown = sorted(set(frame["exer_id"].astype(str)) - known_exercises)
        if unknown:
            raise ValueError(
                f"{split_name} contains optimizer-train-unknown exercises: "
                f"{unknown[:10]}"
            )


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


def build_exercise_evidence_tensor(
    *,
    interactions: pd.DataFrame,
    exercise_id_map: Dict[str, int],
) -> torch.Tensor:
    attempt_counts = torch.zeros(len(exercise_id_map), dtype=torch.float32)
    correct_counts = torch.zeros_like(attempt_counts)

    for row in interactions.itertuples(index=False):
        exercise_index = exercise_id_map[str(row.exer_id)]
        attempt_counts[exercise_index] += 1.0
        correct_counts[exercise_index] += float(row.label)

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
    static_relation_path: str | Path | None = None,
    static_relation_mode: str = "full",
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
    static_relation_graph = (
        load_static_relation_graph(
            static_relation_path,
            mode=static_relation_mode,
            q_matrix_tensor=q_matrix_tensor,
            exercise_id_map=base["exercise_id_map"],
            concept_id_map=base["concept_id_map"],
        )
        if static_relation_path is not None
        else None
    )
    history_tensors = build_history_tensors(
        history_interactions=base["interactions"],
        student_id_map=base["student_id_map"],
        exercise_id_map=base["exercise_id_map"],
        concept_id_map=base["concept_id_map"],
        q_matrix=base["q_matrix"],
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
        exercise_evidence_tensor=history_tensors["exercise_evidence_tensor"],
        prerequisite_graph=prerequisite_graph,
        similarity_graph=similarity_graph,
        static_relation_graph=static_relation_graph,
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
    valid_history_interactions_path: str | Path | None = None,
    test_history_interactions_path: str | Path | None = None,
    concept_graph_path: str | Path | None = None,
    prerequisite_graph_path: str | Path | None = None,
    similarity_graph_path: str | Path | None = None,
    static_relation_path: str | Path | None = None,
    static_relation_mode: str = "full",
) -> Dict[str, Any]:
    if (valid_history_interactions_path is None) != (
        test_history_interactions_path is None
    ):
        raise ValueError(
            "valid/test history interaction paths must be provided together."
        )
    train_df = read_interactions(train_interactions_path)
    valid_df = read_interactions(valid_interactions_path)
    test_df = read_interactions(test_interactions_path)
    valid_history_df = (
        read_interactions(valid_history_interactions_path)
        if valid_history_interactions_path is not None
        else train_df
    )
    test_history_df = (
        read_interactions(test_history_interactions_path)
        if test_history_interactions_path is not None
        else train_df
    )
    q_matrix = read_q_matrix(q_matrix_path)
    if valid_history_interactions_path is not None:
        _validate_separate_history_protocol(
            train=train_df,
            valid=valid_df,
            test=test_df,
            valid_history=valid_history_df,
            test_history=test_history_df,
        )

    all_interactions = pd.concat(
        [train_df, valid_df, test_df, valid_history_df, test_history_df],
        ignore_index=True,
    )
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
    static_relation_graph = (
        load_static_relation_graph(
            static_relation_path,
            mode=static_relation_mode,
            q_matrix_tensor=q_matrix_tensor,
            exercise_id_map=mappings["exercise_id_map"],
            concept_id_map=mappings["concept_id_map"],
        )
        if static_relation_path is not None
        else None
    )

    history_frames = {
        "train": train_df,
        "valid": valid_history_df,
        "test": test_history_df,
    }
    history_tensors_by_split = {
        split_name: build_history_tensors(
            history_interactions=history_frame,
            student_id_map=mappings["student_id_map"],
            exercise_id_map=mappings["exercise_id_map"],
            concept_id_map=mappings["concept_id_map"],
            q_matrix=q_matrix,
        )
        for split_name, history_frame in history_frames.items()
    }
    global_exercise_evidence = build_exercise_evidence_tensor(
        interactions=train_df,
        exercise_id_map=mappings["exercise_id_map"],
    )

    interaction_ids = {
        "train": build_interaction_tensors_from_frame(
            train_df,
            mappings["student_id_map"],
            mappings["exercise_id_map"],
        ),
        "valid": build_interaction_tensors_from_frame(
            valid_df,
            mappings["student_id_map"],
            mappings["exercise_id_map"],
        ),
        "test": build_interaction_tensors_from_frame(
            test_df,
            mappings["student_id_map"],
            mappings["exercise_id_map"],
        ),
    }
    train_history_tensors = history_tensors_by_split["train"]

    shared = {
        "q_matrix": q_matrix,
        "q_matrix_tensor": q_matrix_tensor,
        "concept_graph": concept_graph,
        "prerequisite_graph": prerequisite_graph,
        "similarity_graph": similarity_graph,
        "student_id_map": mappings["student_id_map"],
        "exercise_id_map": mappings["exercise_id_map"],
        "concept_id_map": mappings["concept_id_map"],
        "student_exercise_mask": train_history_tensors["student_exercise_mask"],
        "student_tkc_mask": train_history_tensors["student_tkc_mask"],
        "student_ukc_mask": train_history_tensors["student_ukc_mask"],
        "student_concept_evidence_tensor": train_history_tensors[
            "student_concept_evidence_tensor"
        ],
        "exercise_evidence_tensor": global_exercise_evidence,
        "static_relation_graph": static_relation_graph,
    }

    def _bundle(
        frame: pd.DataFrame,
        history_frame: pd.DataFrame,
        ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        history_tensors: Dict[str, torch.Tensor],
        *,
        split_name: str,
        allow_target_in_history: bool,
    ) -> StepDataBundle:
        student_ids, exercise_ids, labels = ids
        return StepDataBundle(
            interactions=frame,
            history_interactions=history_frame,
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
            exercise_evidence_tensor=global_exercise_evidence,
            prerequisite_graph=prerequisite_graph,
            similarity_graph=similarity_graph,
            static_relation_graph=static_relation_graph,
        )

    return {
        "train": _bundle(
            train_df,
            history_frames["train"],
            interaction_ids["train"],
            history_tensors_by_split["train"],
            split_name="train",
            allow_target_in_history=True,
        ),
        "valid": _bundle(
            valid_df,
            history_frames["valid"],
            interaction_ids["valid"],
            history_tensors_by_split["valid"],
            split_name="valid",
            allow_target_in_history=False,
        ),
        "test": _bundle(
            test_df,
            history_frames["test"],
            interaction_ids["test"],
            history_tensors_by_split["test"],
            split_name="test",
            allow_target_in_history=False,
        ),
        "shared": shared,
    }
