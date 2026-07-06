from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoupled_cdm import DecoupledForwardOutput


class PositiveLinear(nn.Module):
    """Linear layer with softplus-reparameterized non-negative weights (NCD monotonicity)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.raw_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, F.softplus(self.raw_weight), self.bias)


class KaNCDBaseline(nn.Module):
    """
    Faithful reimplementation of KaNCD (Wang et al., TKDE 2022: NeuralCD - A
    General Framework for Cognitive Diagnosis) for protocol-identical
    comparison inside this repo's training harness. KaNCD addresses the low
    knowledge coverage problem by low-rank factorization: per-concept mastery
    is sigma(<student_vec, concept_vec>) rather than a free (S, K) matrix, so
    untested concepts receive extrapolated, student-specific estimates.

    The interaction function is NCD's monotone positive-weight MLP over the
    Q-masked (mastery - difficulty) * discrimination vector.

    Numbers should be cross-checked against the authors' EduCDM implementation
    before publication. Graph/history tensors in the shared forward signature
    are accepted and ignored.
    """

    def __init__(
        self,
        *,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        latent_dim: int = 64,
        hidden_dims: tuple[int, int] = (256, 128),
        **_unused_kwargs,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.student_embedding = nn.Embedding(num_students, latent_dim)
        self.exercise_embedding = nn.Embedding(num_exercises, latent_dim)
        self.concept_embedding = nn.Parameter(torch.empty(num_concepts, latent_dim))
        nn.init.xavier_uniform_(self.concept_embedding)
        self.discrimination = nn.Embedding(num_exercises, 1)
        self.interaction = nn.Sequential(
            PositiveLinear(num_concepts, hidden_dims[0]),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            PositiveLinear(hidden_dims[0], hidden_dims[1]),
            nn.Sigmoid(),
            nn.Dropout(0.5),
            PositiveLinear(hidden_dims[1], 1),
        )
        # Present only for the training engine's difficulty-regularization touchpoint.
        self.exercise_difficulty = nn.Embedding(num_exercises, 1)
        nn.init.zeros_(self.exercise_difficulty.weight)

    def mastery_matrix(self) -> torch.Tensor:
        return torch.sigmoid(self.student_embedding.weight @ self.concept_embedding.t())

    def forward(
        self,
        *,
        q_matrix: torch.Tensor,
        concept_graph: torch.Tensor | None = None,
        prerequisite_graph: torch.Tensor | None = None,
        similarity_graph: torch.Tensor | None = None,
        student_exercise_mask: torch.Tensor | None = None,
        response_matrix: torch.Tensor | None = None,
        student_tkc_mask: torch.Tensor | None = None,
        student_ukc_mask: torch.Tensor | None = None,
        student_concept_evidence: torch.Tensor | None = None,
        exercise_evidence: torch.Tensor | None = None,
        target_student_ids: torch.Tensor | None = None,
        target_exercise_ids: torch.Tensor | None = None,
        use_student_subset: bool = False,
    ) -> DecoupledForwardOutput:
        if target_student_ids is None or target_exercise_ids is None:
            raise ValueError("target_student_ids and target_exercise_ids are required for prediction.")

        student_vectors = self.student_embedding(target_student_ids)
        exercise_vectors = self.exercise_embedding(target_exercise_ids)
        mastery_rows = torch.sigmoid(student_vectors @ self.concept_embedding.t())
        difficulty_rows = torch.sigmoid(exercise_vectors @ self.concept_embedding.t())
        discrimination = torch.sigmoid(self.discrimination(target_exercise_ids))

        q_vectors = (q_matrix[target_exercise_ids] > 0).to(mastery_rows.dtype)
        interaction_inputs = q_vectors * discrimination * (mastery_rows - difficulty_rows)
        logits = self.interaction(interaction_inputs).squeeze(-1)
        probs = torch.sigmoid(logits)

        zeros_like_probs = torch.zeros_like(probs)
        placeholder = probs.new_zeros((1, 1))
        return DecoupledForwardOutput(
            student_state=placeholder,
            tkc_states=probs.new_zeros((1, 1, 1)),
            ukc_states=probs.new_zeros((1, 1, 1)),
            tkc_weight=placeholder,
            concept_embeddings=self.concept_embedding,
            exercise_embeddings=self.exercise_embedding.weight,
            cognitive_probs=probs,
            probs=probs,
            guess_probs=zeros_like_probs,
            slip_probs=zeros_like_probs,
            difficulty=self.exercise_difficulty(target_exercise_ids).squeeze(-1),
            mastery=self.mastery_matrix(),
        )
