from __future__ import annotations

from typing import Dict

import pandas as pd
import torch


def normalize_concept_sequence(raw_value: object) -> list[str]:
    if pd.isna(raw_value):
        return []
    return [token.strip() for token in str(raw_value).split(",") if token.strip()]


def build_q_matrix_tensor(
    q_matrix: pd.DataFrame,
    exercise_id_map: Dict[str, int],
    concept_id_map: Dict[str, int],
) -> torch.Tensor:
    num_exercises = len(exercise_id_map)
    num_concepts = len(concept_id_map)
    tensor = torch.zeros(num_exercises, num_concepts, dtype=torch.float32)

    for row in q_matrix.itertuples(index=False):
        exercise_id = str(row.exer_id)
        if exercise_id not in exercise_id_map:
            continue
        exercise_index = exercise_id_map[exercise_id]
        for concept_id in normalize_concept_sequence(row.cpt_seq):
            concept_index = concept_id_map.get(concept_id)
            if concept_index is not None:
                tensor[exercise_index, concept_index] = 1.0
    return tensor


def build_concept_graph_from_q(q_matrix_tensor: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    """
    Build a stronger concept graph from the Q-matrix:
    - normalize by concept frequency to avoid raw co-occurrence bias
    - keep only top-k neighbors for each concept
    - row-normalize to get a stable propagation matrix
    """
    concept_cooccurrence = (q_matrix_tensor.transpose(0, 1) @ q_matrix_tensor).to(dtype=torch.float32)
    concept_frequency = q_matrix_tensor.sum(dim=0).clamp(min=1.0)
    normalized = concept_cooccurrence / torch.sqrt(concept_frequency[:, None] * concept_frequency[None, :])
    normalized.fill_diagonal_(0.0)

    k = min(8, normalized.size(1))
    if k > 0:
        values, indices = torch.topk(normalized, k=k, dim=1)
        sparse = torch.zeros_like(normalized)
        sparse.scatter_(1, indices, values)
    else:
        sparse = normalized

    if add_self_loops:
        sparse.fill_diagonal_(1.0)
    else:
        sparse.fill_diagonal_(0.0)

    row_sums = sparse.sum(dim=1, keepdim=True).clamp(min=1.0)
    return sparse / row_sums
