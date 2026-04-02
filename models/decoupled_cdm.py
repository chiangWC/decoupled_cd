from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from .hetero_propagation import HeterogeneousGraphPropagation, PropagationOutput


@dataclass
class DecoupledForwardOutput:
    student_state: torch.Tensor
    tkc_states: torch.Tensor
    ukc_states: torch.Tensor
    concept_embeddings: torch.Tensor
    exercise_embeddings: torch.Tensor
    cognitive_probs: torch.Tensor
    probs: torch.Tensor
    guess_probs: torch.Tensor
    slip_probs: torch.Tensor
    difficulty: torch.Tensor


class DecoupledCDM(nn.Module):
    """
    Minimal runnable model for Step 1 + Step 2.
    Step 1 tensors are expected to be prepared by the data pipeline.
    Step 2 is implemented by a separated TKC/UKC propagation module.
    """

    def __init__(
        self,
        *,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        concept_dim: int = 32,
        alpha: float = 1.0,
        beta: float = 1.0,
    ):
        super().__init__()
        self.exercise_embedding = nn.Embedding(num_exercises, concept_dim)
        self.exercise_difficulty = nn.Embedding(num_exercises, 1)
        self.concept_embedding = nn.Embedding(num_concepts, concept_dim)
        self.q_pool_gate = nn.Linear(concept_dim, 1, bias=False)
        self.q_pool_mlp = nn.Sequential(
            nn.Linear(concept_dim, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, concept_dim),
        )
        self.guess_logit = nn.Embedding(num_students, 1)
        self.slip_logit = nn.Embedding(num_students, 1)
        self.propagation = HeterogeneousGraphPropagation(concept_dim=concept_dim, alpha=alpha, beta=beta)

    def forward(
        self,
        *,
        q_matrix: torch.Tensor,
        concept_graph: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        student_tkc_mask: torch.Tensor,
        student_ukc_mask: torch.Tensor,
        target_student_ids: torch.Tensor | None = None,
        target_exercise_ids: torch.Tensor | None = None,
    ) -> DecoupledForwardOutput:
        concept_embeddings = self.concept_embedding.weight
        exercise_embeddings = self.exercise_embedding.weight
        propagated: PropagationOutput = self.propagation(
            concept_embeddings=concept_embeddings,
            exercise_embeddings=exercise_embeddings,
            q_matrix=q_matrix,
            concept_graph=concept_graph,
            student_exercise_mask=student_exercise_mask,
            response_matrix=response_matrix,
            student_tkc_mask=student_tkc_mask,
            student_ukc_mask=student_ukc_mask,
        )

        if target_student_ids is None or target_exercise_ids is None:
            target_student_ids = torch.arange(student_exercise_mask.size(0), device=student_exercise_mask.device)
            target_exercise_ids = torch.zeros_like(target_student_ids)

        student_state = propagated.student_state[target_student_ids]
        q_vectors = q_matrix[target_exercise_ids]
        q_repr = self._build_exercise_q_representation(q_vectors=q_vectors, concept_embeddings=concept_embeddings)
        difficulty = self.exercise_difficulty(target_exercise_ids).squeeze(-1)
        cognitive_logits = (student_state * q_repr).sum(dim=-1) - difficulty
        cognitive_probs = torch.sigmoid(cognitive_logits)

        guess_probs = torch.sigmoid(self.guess_logit(target_student_ids)).squeeze(-1)
        slip_probs = torch.sigmoid(self.slip_logit(target_student_ids)).squeeze(-1)
        probs = (1.0 - slip_probs) * cognitive_probs + guess_probs * (1.0 - cognitive_probs)

        return DecoupledForwardOutput(
            student_state=propagated.student_state,
            tkc_states=propagated.tkc_states,
            ukc_states=propagated.ukc_states,
            concept_embeddings=concept_embeddings,
            exercise_embeddings=exercise_embeddings,
            cognitive_probs=cognitive_probs,
            probs=probs,
            guess_probs=guess_probs,
            slip_probs=slip_probs,
            difficulty=difficulty,
        )

    def _build_exercise_q_representation(
        self,
        *,
        q_vectors: torch.Tensor,
        concept_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        raw_scores = self.q_pool_gate(concept_embeddings).squeeze(-1)
        masked_scores = raw_scores.unsqueeze(0).expand(q_vectors.size(0), -1).masked_fill(q_vectors <= 0, -1e9)
        attn = torch.softmax(masked_scores, dim=1)
        pooled = attn @ concept_embeddings
        return self.q_pool_mlp(pooled)
