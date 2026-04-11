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
        graph_mode: str = "single",
        student_gate_prior_alpha: float | None = None,
        student_gate_prior_beta: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gs_mode: str = "conditional",
        high_concept_logit_adapter: bool = False,
        high_concept_logit_min_count: int = 3,
        gs_difficulty_adapter: bool = False,
    ):
        super().__init__()
        if gs_mode not in {"constant", "conditional"}:
            raise ValueError(f"Unsupported gs_mode: {gs_mode}")
        if high_concept_logit_min_count < 2:
            raise ValueError("high_concept_logit_min_count must be at least 2.")
        self.gs_mode = gs_mode
        self.high_concept_logit_adapter = high_concept_logit_adapter
        self.high_concept_logit_min_count = high_concept_logit_min_count
        self.gs_difficulty_adapter = gs_difficulty_adapter
        self.exercise_embedding = nn.Embedding(num_exercises, concept_dim)
        self.exercise_difficulty = nn.Embedding(num_exercises, 1)
        self.concept_embedding = nn.Embedding(num_concepts, concept_dim)
        self.q_pool_gate = nn.Linear(concept_dim, 1, bias=False)
        self.exercise_q_fusion = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, concept_dim),
        )
        self.q_pool_mlp = nn.Sequential(
            nn.Linear(concept_dim, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, concept_dim),
        )
        self.cognitive_match_mlp = nn.Sequential(
            nn.Linear(concept_dim * 4, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 1),
        )
        self.guess_logit = nn.Embedding(num_students, 1)
        self.slip_logit = nn.Embedding(num_students, 1)
        self.guess_mlp = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 1),
        )
        self.slip_mlp = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 1),
        )
        self.gs_difficulty_residual = nn.Sequential(
            nn.Linear(concept_dim * 2 + 1, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 2),
        )
        self.propagation = HeterogeneousGraphPropagation(
            concept_dim=concept_dim,
            graph_mode=graph_mode,
            student_gate_prior_alpha=student_gate_prior_alpha,
            student_gate_prior_beta=student_gate_prior_beta,
            alpha=alpha,
            beta=beta,
        )
        self.cognitive_difficulty_adapter = nn.Sequential(
            nn.Linear(concept_dim * 4 + 1, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 1),
        )
        self.high_concept_logit_residual = nn.Sequential(
            nn.Linear(concept_dim * 6 + 2, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 1),
        )
        nn.init.zeros_(self.cognitive_difficulty_adapter[-1].weight)
        nn.init.zeros_(self.cognitive_difficulty_adapter[-1].bias)
        nn.init.zeros_(self.high_concept_logit_residual[-1].weight)
        nn.init.zeros_(self.high_concept_logit_residual[-1].bias)
        nn.init.zeros_(self.gs_difficulty_residual[-1].weight)
        nn.init.zeros_(self.gs_difficulty_residual[-1].bias)

    def forward(
        self,
        *,
        q_matrix: torch.Tensor,
        concept_graph: torch.Tensor,
        prerequisite_graph: torch.Tensor | None = None,
        similarity_graph: torch.Tensor | None = None,
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
            prerequisite_graph=prerequisite_graph,
            similarity_graph=similarity_graph,
            student_exercise_mask=student_exercise_mask,
            response_matrix=response_matrix,
            student_tkc_mask=student_tkc_mask,
            student_ukc_mask=student_ukc_mask,
        )

        if (target_student_ids is None) != (target_exercise_ids is None):
            raise ValueError("target_student_ids and target_exercise_ids must be provided together.")
        if target_student_ids is None or target_exercise_ids is None:
            raise ValueError("target_student_ids and target_exercise_ids are required for prediction.")

        student_state = propagated.student_state[target_student_ids]
        q_vectors = q_matrix[target_exercise_ids]
        target_exercise_embeddings = exercise_embeddings[target_exercise_ids]
        concept_summary = self._summarize_exercise_concepts(
            q_vectors=q_vectors,
            concept_embeddings=concept_embeddings,
        )
        q_repr = self._build_exercise_q_representation(
            q_vectors=q_vectors,
            concept_embeddings=concept_embeddings,
            exercise_embeddings=target_exercise_embeddings,
        )
        difficulty = self.exercise_difficulty(target_exercise_ids).squeeze(-1)
        match_inputs = torch.cat(
            [student_state, q_repr, student_state * q_repr, torch.abs(student_state - q_repr)],
            dim=-1,
        )
        adapter_inputs = torch.cat([match_inputs.detach(), difficulty.detach().unsqueeze(-1)], dim=-1)
        cognitive_logits = (
            self.cognitive_match_mlp(match_inputs).squeeze(-1)
            - difficulty
            + self.cognitive_difficulty_adapter(adapter_inputs).squeeze(-1)
        )
        if self.high_concept_logit_adapter:
            concept_counts, mean_pooled, dispersion = concept_summary
            high_concept_inputs = torch.cat(
                [
                    student_state.detach(),
                    q_repr.detach(),
                    mean_pooled.detach(),
                    dispersion.detach(),
                    (student_state * q_repr).detach(),
                    torch.abs(student_state - q_repr).detach(),
                    difficulty.detach().unsqueeze(-1),
                    (concept_counts - 1.0).detach(),
                ],
                dim=-1,
            )
            high_concept_mask = (concept_counts >= float(self.high_concept_logit_min_count)).to(cognitive_logits.dtype)
            cognitive_logits = cognitive_logits + (
                self.high_concept_logit_residual(high_concept_inputs).squeeze(-1) * high_concept_mask.squeeze(-1)
            )
        cognitive_probs = torch.sigmoid(cognitive_logits)

        guess_logits = self.guess_logit(target_student_ids).squeeze(-1)
        slip_logits = self.slip_logit(target_student_ids).squeeze(-1)
        if self.gs_mode == "conditional":
            non_cognitive_inputs = torch.cat([student_state, q_repr], dim=-1)
            guess_logits = guess_logits + self.guess_mlp(non_cognitive_inputs).squeeze(-1)
            slip_logits = slip_logits + self.slip_mlp(non_cognitive_inputs).squeeze(-1)
            if self.gs_difficulty_adapter:
                gs_adapter_inputs = torch.cat([non_cognitive_inputs, difficulty.detach().unsqueeze(-1)], dim=-1)
                gs_residual = self.gs_difficulty_residual(gs_adapter_inputs)
                guess_logits = guess_logits + gs_residual[:, 0]
                slip_logits = slip_logits + gs_residual[:, 1]
        guess_probs = torch.sigmoid(guess_logits)
        slip_probs = torch.sigmoid(slip_logits)
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

    def _summarize_exercise_concepts(
        self,
        *,
        q_vectors: torch.Tensor,
        concept_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_mask = q_vectors > 0
        q_mask_float = q_mask.to(concept_embeddings.dtype)
        concept_counts = q_mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_pooled = q_mask_float @ concept_embeddings / concept_counts
        mean_square = q_mask_float @ concept_embeddings.square() / concept_counts
        dispersion = (mean_square - mean_pooled.square()).clamp_min(0.0).sqrt()
        return concept_counts, mean_pooled, dispersion

    def _build_exercise_q_representation(
        self,
        *,
        q_vectors: torch.Tensor,
        concept_embeddings: torch.Tensor,
        exercise_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        raw_scores = self.q_pool_gate(concept_embeddings).squeeze(-1)
        masked_scores = raw_scores.unsqueeze(0).expand(q_vectors.size(0), -1).masked_fill(q_vectors <= 0, -1e9)
        attn = torch.softmax(masked_scores, dim=1)
        pooled = attn @ concept_embeddings
        fused = self.exercise_q_fusion(torch.cat([pooled, exercise_embeddings], dim=-1))
        return self.q_pool_mlp(fused)
