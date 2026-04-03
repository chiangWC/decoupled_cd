from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PropagationOutput:
    tkc_states: torch.Tensor
    ukc_states: torch.Tensor
    student_state: torch.Tensor


class HeterogeneousGraphPropagation(nn.Module):
    """
    Minimal Step 2 implementation:
    - TKC branch uses exercise-response aggregation plus concept-graph propagation.
    - UKC branch uses only concept-graph propagation.
    """

    def __init__(self, concept_dim: int, alpha: float = 1.0, beta: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha), dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(float(beta), dtype=torch.float32))
        self.exercise_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.tkc_concept_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.ukc_concept_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.tkc_prerequisite_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.tkc_similarity_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.ukc_prerequisite_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.ukc_similarity_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.graph_fusion_gate = nn.Linear(concept_dim * 2, 1, bias=True)
        self.tkc_fusion_gate = nn.Linear(concept_dim * 2, 1, bias=True)

    def forward(
        self,
        *,
        concept_embeddings: torch.Tensor,
        exercise_embeddings: torch.Tensor,
        q_matrix: torch.Tensor,
        concept_graph: torch.Tensor,
        prerequisite_graph: torch.Tensor | None,
        similarity_graph: torch.Tensor | None,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        student_tkc_mask: torch.Tensor,
        student_ukc_mask: torch.Tensor,
    ) -> PropagationOutput:
        exercise_messages = self.exercise_to_concept(exercise_embeddings)
        if prerequisite_graph is not None and similarity_graph is not None:
            tkc_neighbor_messages = self._fuse_dual_graph_messages(
                concept_embeddings=concept_embeddings,
                prerequisite_graph=prerequisite_graph,
                similarity_graph=similarity_graph,
                prerequisite_transform=self.tkc_prerequisite_to_concept,
                similarity_transform=self.tkc_similarity_to_concept,
            )
            ukc_neighbor_messages = self._fuse_dual_graph_messages(
                concept_embeddings=concept_embeddings,
                prerequisite_graph=prerequisite_graph,
                similarity_graph=similarity_graph,
                prerequisite_transform=self.ukc_prerequisite_to_concept,
                similarity_transform=self.ukc_similarity_to_concept,
            )
        else:
            tkc_neighbor_messages = concept_graph @ self.tkc_concept_to_concept(concept_embeddings)
            ukc_neighbor_messages = concept_graph @ self.ukc_concept_to_concept(concept_embeddings)

        weighted_exercises = student_exercise_mask * response_matrix
        exercise_concept_weights = torch.einsum("se,ek->sk", weighted_exercises, q_matrix)
        exercise_concept_weights = exercise_concept_weights.clamp(min=0.0)
        normalized_exercise_weights = weighted_exercises / weighted_exercises.sum(dim=1, keepdim=True).clamp(min=1.0)
        tkc_exercise_component = _aggregate_exercise_messages_by_concept(
            normalized_exercise_weights=normalized_exercise_weights,
            q_matrix=q_matrix,
            exercise_messages=exercise_messages,
            concept_weights=exercise_concept_weights,
        )

        tkc_neighbor_component = tkc_neighbor_messages.unsqueeze(0).expand(student_tkc_mask.size(0), -1, -1)
        ukc_neighbor_component = ukc_neighbor_messages.unsqueeze(0).expand(student_tkc_mask.size(0), -1, -1)
        fusion_inputs = torch.cat([tkc_exercise_component, tkc_neighbor_component], dim=-1)
        fusion_gate = torch.sigmoid(self.tkc_fusion_gate(fusion_inputs))
        tkc_states = student_tkc_mask.unsqueeze(-1) * (
            fusion_gate * tkc_exercise_component + (1.0 - fusion_gate) * tkc_neighbor_component
        )
        ukc_states = student_ukc_mask.unsqueeze(-1) * ukc_neighbor_component

        tkc_mean = _masked_average(tkc_states, student_tkc_mask)
        ukc_mean = _masked_average(ukc_states, student_ukc_mask)
        student_state = self.alpha * tkc_mean + self.beta * ukc_mean
        return PropagationOutput(tkc_states=tkc_states, ukc_states=ukc_states, student_state=student_state)

    def _fuse_dual_graph_messages(
        self,
        *,
        concept_embeddings: torch.Tensor,
        prerequisite_graph: torch.Tensor,
        similarity_graph: torch.Tensor,
        prerequisite_transform: nn.Linear,
        similarity_transform: nn.Linear,
    ) -> torch.Tensor:
        prerequisite_messages = prerequisite_graph @ prerequisite_transform(concept_embeddings)
        similarity_messages = similarity_graph @ similarity_transform(concept_embeddings)
        graph_fusion_inputs = torch.cat([prerequisite_messages, similarity_messages], dim=-1)
        graph_fusion_gate = torch.sigmoid(self.graph_fusion_gate(graph_fusion_inputs))
        return graph_fusion_gate * prerequisite_messages + (1.0 - graph_fusion_gate) * similarity_messages


def _masked_average(node_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=node_states.dtype).unsqueeze(-1)
    total = (node_states * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return total / denom


def _aggregate_exercise_messages_by_concept(
    *,
    normalized_exercise_weights: torch.Tensor,
    q_matrix: torch.Tensor,
    exercise_messages: torch.Tensor,
    concept_weights: torch.Tensor,
) -> torch.Tensor:
    num_students = normalized_exercise_weights.size(0)
    num_concepts = q_matrix.size(1)
    dim = exercise_messages.size(1)
    output = exercise_messages.new_zeros((num_students, num_concepts, dim))

    for concept_index in range(num_concepts):
        concept_mask = q_matrix[:, concept_index].unsqueeze(0)
        concept_specific_weights = normalized_exercise_weights * concept_mask
        concept_message = concept_specific_weights @ exercise_messages
        denom = concept_weights[:, concept_index].unsqueeze(-1).clamp(min=1.0)
        output[:, concept_index, :] = concept_message / denom

    return output
