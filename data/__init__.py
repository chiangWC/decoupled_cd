"""Data processing entrypoints for the decoupled_cd project."""

from .concept_graph import build_transition_matrices, load_concept_graph_csv, save_concept_graph_csv
from .datasets import InteractionDataset, StepDataBundle
from .mappings import (
    build_concept_id_map,
    build_exercise_id_map,
    build_student_id_map,
    build_unified_id_mappings,
)
from .pipeline import (
    build_exercise_evidence_tensor,
    build_response_matrix,
    build_student_concept_evidence_tensor,
    prepare_data_bundle,
    prepare_step_data_bundle,
)
from .pipeline import prepare_experiment_split_bundles
from .q_matrix import build_concept_graph_from_q, build_q_matrix_tensor, normalize_concept_sequence
from .readers import read_interactions, read_q_matrix

__all__ = [
    "InteractionDataset",
    "StepDataBundle",
    "build_concept_id_map",
    "build_concept_graph_from_q",
    "build_transition_matrices",
    "build_exercise_id_map",
    "build_exercise_evidence_tensor",
    "build_q_matrix_tensor",
    "build_response_matrix",
    "build_student_concept_evidence_tensor",
    "build_student_id_map",
    "build_unified_id_mappings",
    "load_concept_graph_csv",
    "normalize_concept_sequence",
    "prepare_data_bundle",
    "prepare_experiment_split_bundles",
    "prepare_step_data_bundle",
    "read_interactions",
    "read_q_matrix",
    "save_concept_graph_csv",
]
