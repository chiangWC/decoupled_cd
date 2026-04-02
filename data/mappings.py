from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd


def _normalize_concept_sequence(raw_value: object) -> list[str]:
    if pd.isna(raw_value):
        return []
    if isinstance(raw_value, str):
        return [token.strip() for token in raw_value.split(",") if token.strip()]
    return [str(raw_value).strip()]


def build_student_id_map(interactions: pd.DataFrame) -> Dict[str, int]:
    student_ids = sorted({str(student_id) for student_id in interactions["stu_id"].tolist()})
    return {student_id: index for index, student_id in enumerate(student_ids)}


def build_exercise_id_map(
    interactions: pd.DataFrame,
    q_matrix: pd.DataFrame | None = None,
) -> Dict[str, int]:
    exercise_ids = {str(exercise_id) for exercise_id in interactions["exer_id"].tolist()}
    if q_matrix is not None and "exer_id" in q_matrix.columns:
        exercise_ids.update(str(exercise_id) for exercise_id in q_matrix["exer_id"].tolist())
    ordered_ids = sorted(exercise_ids)
    return {exercise_id: index for index, exercise_id in enumerate(ordered_ids)}


def build_concept_id_map(q_matrix: pd.DataFrame) -> Dict[str, int]:
    concept_ids = set()
    for raw_value in q_matrix["cpt_seq"].tolist():
        concept_ids.update(_normalize_concept_sequence(raw_value))
    ordered_ids = sorted(concept_id for concept_id in concept_ids if concept_id)
    return {concept_id: index for index, concept_id in enumerate(ordered_ids)}


def build_unified_id_mappings(
    interactions: pd.DataFrame,
    q_matrix: pd.DataFrame,
) -> Dict[str, Dict[str, int]]:
    return {
        "student_id_map": build_student_id_map(interactions),
        "exercise_id_map": build_exercise_id_map(interactions, q_matrix=q_matrix),
        "concept_id_map": build_concept_id_map(q_matrix),
    }
