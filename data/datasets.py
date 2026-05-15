from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset

from .q_matrix import normalize_concept_sequence


@dataclass
class StepDataBundle:
    interactions: pd.DataFrame
    history_interactions: pd.DataFrame
    q_matrix: pd.DataFrame
    student_id_map: Dict[str, int]
    exercise_id_map: Dict[str, int]
    concept_id_map: Dict[str, int]
    q_matrix_tensor: torch.Tensor
    concept_graph: torch.Tensor
    student_exercise_mask: torch.Tensor
    student_tkc_mask: torch.Tensor
    student_ukc_mask: torch.Tensor
    response_matrix_tensor: torch.Tensor
    interaction_student_ids: torch.Tensor
    interaction_exercise_ids: torch.Tensor
    interaction_labels: torch.Tensor
    student_concept_evidence_tensor: torch.Tensor | None = None
    split_name: str = "train"
    allow_target_in_history: bool = True
    prerequisite_graph: torch.Tensor | None = None
    similarity_graph: torch.Tensor | None = None

    @property
    def num_students(self) -> int:
        return len(self.student_id_map)

    @property
    def num_exercises(self) -> int:
        return len(self.exercise_id_map)

    @property
    def num_concepts(self) -> int:
        return len(self.concept_id_map)


class InteractionDataset(Dataset):
    def __init__(self, bundle: StepDataBundle):
        self.bundle = bundle

    def __len__(self) -> int:
        return len(self.bundle.interactions)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        student_id = self.bundle.interaction_student_ids[index]
        exercise_id = self.bundle.interaction_exercise_ids[index]
        return {
            "student_id": student_id,
            "exercise_id": exercise_id,
            "label": self.bundle.interaction_labels[index],
            "exercise_concepts": self.bundle.q_matrix_tensor[exercise_id],
            "student_tkc_mask": self.bundle.student_tkc_mask[student_id],
            "student_ukc_mask": self.bundle.student_ukc_mask[student_id],
        }


def build_student_exercise_mask(
    interactions: pd.DataFrame,
    student_id_map: Dict[str, int],
    exercise_id_map: Dict[str, int],
) -> torch.Tensor:
    mask = torch.zeros(len(student_id_map), len(exercise_id_map), dtype=torch.float32)
    for row in interactions.itertuples(index=False):
        student_index = student_id_map[str(row.stu_id)]
        exercise_index = exercise_id_map[str(row.exer_id)]
        mask[student_index, exercise_index] = 1.0
    return mask


def build_student_tkc_mask(
    interactions: pd.DataFrame,
    student_id_map: Dict[str, int],
    concept_id_map: Dict[str, int],
) -> torch.Tensor:
    mask = torch.zeros(len(student_id_map), len(concept_id_map), dtype=torch.float32)
    for row in interactions.itertuples(index=False):
        student_index = student_id_map[str(row.stu_id)]
        for concept_id in normalize_concept_sequence(row.cpt_seq):
            concept_index = concept_id_map.get(concept_id)
            if concept_index is not None:
                mask[student_index, concept_index] = 1.0
    return mask


def build_interaction_tensors(
    interactions: pd.DataFrame,
    student_id_map: Dict[str, int],
    exercise_id_map: Dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student_ids = torch.tensor(
        [student_id_map[str(value)] for value in interactions["stu_id"].tolist()],
        dtype=torch.long,
    )
    exercise_ids = torch.tensor(
        [exercise_id_map[str(value)] for value in interactions["exer_id"].tolist()],
        dtype=torch.long,
    )
    labels = torch.tensor(interactions["label"].tolist(), dtype=torch.float32)
    return student_ids, exercise_ids, labels


def build_interaction_tensors_from_frame(
    interactions: pd.DataFrame,
    student_id_map: Dict[str, int],
    exercise_id_map: Dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return build_interaction_tensors(
        interactions=interactions,
        student_id_map=student_id_map,
        exercise_id_map=exercise_id_map,
    )
