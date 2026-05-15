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
        pairwise_history_interaction_adapter: bool = False,
        pairwise_history_interaction_min_count: int = 2,
        gs_difficulty_adapter: bool = False,
        interpretable_readout_expert_adapter: bool = False,
        interpretable_readout_expert_count: int = 3,
        student_conditioned_ukc_readout_residual: bool = False,
        evidence_calibrated_behavior_gate: bool = False,
        evidence_behavior_gate_max_logit: float = 0.5,
        evidence_behavior_gate_trigger: str = "all",
        evidence_behavior_gate_low_attempt_threshold: float = 3.0,
        concept_evidence_readout_residual: bool = False,
        concept_evidence_readout_min_count: int = 2,
        concept_evidence_readout_min_seen_ratio: float = 1.0,
        concept_evidence_readout_max_logit: float = 0.5,
        concept_evidence_prior_residual: bool = False,
        concept_evidence_prior_min_count: int = 2,
        concept_evidence_prior_min_seen_ratio: float = 1.0,
        concept_evidence_prior_max_logit: float = 0.5,
        concept_evidence_prior_strength: float = 2.0,
        concept_evidence_prior_confidence_cap: float = 20.0,
    ):
        super().__init__()
        if gs_mode not in {"constant", "conditional"}:
            raise ValueError(f"Unsupported gs_mode: {gs_mode}")
        if high_concept_logit_min_count < 2:
            raise ValueError("high_concept_logit_min_count must be at least 2.")
        if pairwise_history_interaction_min_count < 2:
            raise ValueError("pairwise_history_interaction_min_count must be at least 2.")
        if interpretable_readout_expert_count < 2:
            raise ValueError("interpretable_readout_expert_count must be at least 2.")
        if evidence_behavior_gate_max_logit <= 0.0:
            raise ValueError("evidence_behavior_gate_max_logit must be positive.")
        if evidence_behavior_gate_trigger not in {"all", "low_evidence"}:
            raise ValueError(f"Unsupported evidence_behavior_gate_trigger: {evidence_behavior_gate_trigger}")
        if evidence_behavior_gate_low_attempt_threshold < 0.0:
            raise ValueError("evidence_behavior_gate_low_attempt_threshold must be non-negative.")
        if concept_evidence_readout_min_count < 1:
            raise ValueError("concept_evidence_readout_min_count must be positive.")
        if concept_evidence_readout_min_seen_ratio < 0.0 or concept_evidence_readout_min_seen_ratio > 1.0:
            raise ValueError("concept_evidence_readout_min_seen_ratio must be in [0, 1].")
        if concept_evidence_readout_max_logit <= 0.0:
            raise ValueError("concept_evidence_readout_max_logit must be positive.")
        if concept_evidence_prior_min_count < 1:
            raise ValueError("concept_evidence_prior_min_count must be positive.")
        if concept_evidence_prior_min_seen_ratio < 0.0 or concept_evidence_prior_min_seen_ratio > 1.0:
            raise ValueError("concept_evidence_prior_min_seen_ratio must be in [0, 1].")
        if concept_evidence_prior_max_logit <= 0.0:
            raise ValueError("concept_evidence_prior_max_logit must be positive.")
        if concept_evidence_prior_strength <= 0.0:
            raise ValueError("concept_evidence_prior_strength must be positive.")
        if concept_evidence_prior_confidence_cap <= 0.0:
            raise ValueError("concept_evidence_prior_confidence_cap must be positive.")
        self.gs_mode = gs_mode
        self.high_concept_logit_adapter = high_concept_logit_adapter
        self.high_concept_logit_min_count = high_concept_logit_min_count
        self.pairwise_history_interaction_adapter = pairwise_history_interaction_adapter
        self.pairwise_history_interaction_min_count = pairwise_history_interaction_min_count
        self.gs_difficulty_adapter = gs_difficulty_adapter
        self.interpretable_readout_expert_adapter = interpretable_readout_expert_adapter
        self.interpretable_readout_expert_count = interpretable_readout_expert_count
        self.student_conditioned_ukc_readout_residual = student_conditioned_ukc_readout_residual
        self.evidence_calibrated_behavior_gate = evidence_calibrated_behavior_gate
        self.evidence_behavior_gate_max_logit = float(evidence_behavior_gate_max_logit)
        self.evidence_behavior_gate_trigger = evidence_behavior_gate_trigger
        self.evidence_behavior_gate_low_attempt_threshold = float(evidence_behavior_gate_low_attempt_threshold)
        self.concept_evidence_readout_residual = concept_evidence_readout_residual
        self.concept_evidence_readout_min_count = int(concept_evidence_readout_min_count)
        self.concept_evidence_readout_min_seen_ratio = float(concept_evidence_readout_min_seen_ratio)
        self.concept_evidence_readout_max_logit = float(concept_evidence_readout_max_logit)
        self.concept_evidence_prior_residual = concept_evidence_prior_residual
        self.concept_evidence_prior_min_count = int(concept_evidence_prior_min_count)
        self.concept_evidence_prior_min_seen_ratio = float(concept_evidence_prior_min_seen_ratio)
        self.concept_evidence_prior_max_logit = float(concept_evidence_prior_max_logit)
        self.concept_evidence_prior_strength = float(concept_evidence_prior_strength)
        self.concept_evidence_prior_confidence_cap = float(concept_evidence_prior_confidence_cap)
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
            evidence_calibrated_behavior_gate=evidence_calibrated_behavior_gate,
            evidence_behavior_gate_max_logit=evidence_behavior_gate_max_logit,
            evidence_behavior_gate_trigger=evidence_behavior_gate_trigger,
            evidence_behavior_gate_low_attempt_threshold=evidence_behavior_gate_low_attempt_threshold,
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
        self.pairwise_history_interaction_scorer = nn.Sequential(
            nn.Linear(concept_dim * 3 + 6, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 1),
        )
        self.interpretable_readout_expert_gate = nn.Linear(4, interpretable_readout_expert_count)
        self.interpretable_readout_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(concept_dim * 6 + 3, concept_dim),
                    nn.ReLU(),
                    nn.Linear(concept_dim, 1),
                )
                for _ in range(interpretable_readout_expert_count)
            ]
        )
        self.student_conditioned_ukc_readout_residual_head = (
            nn.Sequential(
                nn.Linear(concept_dim * 5 + 6, concept_dim),
                nn.ReLU(),
                nn.Linear(concept_dim, 1),
            )
            if student_conditioned_ukc_readout_residual
            else None
        )
        self.concept_evidence_readout_residual_head = (
            nn.Sequential(
                nn.Linear(7, concept_dim),
                nn.ReLU(),
                nn.Linear(concept_dim, 1),
            )
            if concept_evidence_readout_residual
            else None
        )
        nn.init.zeros_(self.cognitive_difficulty_adapter[-1].weight)
        nn.init.zeros_(self.cognitive_difficulty_adapter[-1].bias)
        nn.init.zeros_(self.high_concept_logit_residual[-1].weight)
        nn.init.zeros_(self.high_concept_logit_residual[-1].bias)
        nn.init.zeros_(self.pairwise_history_interaction_scorer[-1].weight)
        nn.init.zeros_(self.pairwise_history_interaction_scorer[-1].bias)
        nn.init.zeros_(self.gs_difficulty_residual[-1].weight)
        nn.init.zeros_(self.gs_difficulty_residual[-1].bias)
        for expert in self.interpretable_readout_experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)
        if self.student_conditioned_ukc_readout_residual_head is not None:
            nn.init.zeros_(self.student_conditioned_ukc_readout_residual_head[-1].weight)
            nn.init.zeros_(self.student_conditioned_ukc_readout_residual_head[-1].bias)
        if self.concept_evidence_readout_residual_head is not None:
            nn.init.zeros_(self.concept_evidence_readout_residual_head[-1].weight)
            nn.init.zeros_(self.concept_evidence_readout_residual_head[-1].bias)

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
        student_concept_evidence: torch.Tensor | None = None,
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
            student_concept_evidence=student_concept_evidence,
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
        if self.pairwise_history_interaction_adapter:
            cognitive_logits = cognitive_logits + self._build_pairwise_history_interaction_residual(
                q_matrix=q_matrix,
                q_vectors=q_vectors,
                student_exercise_mask=student_exercise_mask,
                response_matrix=response_matrix,
                target_student_ids=target_student_ids,
                concept_embeddings=concept_embeddings.detach(),
                target_exercise_embeddings=target_exercise_embeddings.detach(),
                concept_summary=concept_summary,
            )
        if self.interpretable_readout_expert_adapter:
            cognitive_logits = cognitive_logits + self._build_interpretable_readout_expert_residual(
                q_vectors=q_vectors,
                student_tkc_mask=student_tkc_mask,
                target_student_ids=target_student_ids,
                student_state=student_state,
                q_repr=q_repr,
                concept_summary=concept_summary,
                difficulty=difficulty,
            )
        if self.student_conditioned_ukc_readout_residual:
            cognitive_logits = cognitive_logits + self._build_student_conditioned_ukc_readout_residual(
                q_vectors=q_vectors,
                concept_graph=concept_graph,
                q_matrix=q_matrix,
                student_exercise_mask=student_exercise_mask,
                student_tkc_mask=student_tkc_mask,
                target_student_ids=target_student_ids,
                tkc_states=propagated.tkc_states,
                ukc_states=propagated.ukc_states,
                student_state=student_state,
                q_repr=q_repr,
                concept_summary=concept_summary,
                difficulty=difficulty,
            )
        if self.concept_evidence_readout_residual:
            cognitive_logits = cognitive_logits + self._build_concept_evidence_readout_residual(
                q_vectors=q_vectors,
                target_student_ids=target_student_ids,
                student_concept_evidence=student_concept_evidence,
                concept_summary=concept_summary,
                difficulty=difficulty,
            )
        if self.concept_evidence_prior_residual:
            cognitive_logits = cognitive_logits + self._build_concept_evidence_prior_residual(
                q_vectors=q_vectors,
                target_student_ids=target_student_ids,
                student_concept_evidence=student_concept_evidence,
                concept_summary=concept_summary,
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

    def _build_pairwise_history_interaction_residual(
        self,
        *,
        q_matrix: torch.Tensor,
        q_vectors: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        target_student_ids: torch.Tensor,
        concept_embeddings: torch.Tensor,
        target_exercise_embeddings: torch.Tensor,
        concept_summary: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        concept_counts, _, _ = concept_summary
        concept_count_mask = concept_counts >= float(self.pairwise_history_interaction_min_count)
        if torch.count_nonzero(concept_count_mask).item() == 0:
            return target_exercise_embeddings.new_zeros(target_exercise_embeddings.size(0))

        student_concept_attempt_counts = student_exercise_mask @ q_matrix
        student_concept_correct_sums = (student_exercise_mask * response_matrix) @ q_matrix
        student_concept_accuracy = student_concept_correct_sums / student_concept_attempt_counts.clamp_min(1.0)

        target_attempt_counts = student_concept_attempt_counts[target_student_ids]
        target_accuracy = student_concept_accuracy[target_student_ids]
        target_seen = (target_attempt_counts > 0).to(target_accuracy.dtype)
        target_log_attempts = torch.log1p(target_attempt_counts)

        q_mask = q_vectors > 0
        max_concept_count = int(q_mask.sum(dim=1).max().item())
        if max_concept_count < 2:
            return target_exercise_embeddings.new_zeros(target_exercise_embeddings.size(0))

        selected_scores, selected_indices = q_vectors.topk(k=max_concept_count, dim=1)
        selected_valid = selected_scores > 0
        invalid_index = concept_embeddings.size(0)
        ordered_indices = torch.where(
            selected_valid,
            selected_indices,
            selected_indices.new_full(selected_indices.shape, invalid_index),
        )
        ordered_indices, _ = ordered_indices.sort(dim=1)
        ordered_valid = ordered_indices != invalid_index
        safe_indices = ordered_indices.clamp_max(concept_embeddings.size(0) - 1)

        expanded_indices = safe_indices.unsqueeze(-1).expand(-1, -1, concept_embeddings.size(1))
        selected_concept_embeddings = concept_embeddings[safe_indices]
        selected_accuracy = target_accuracy.gather(1, safe_indices)
        selected_seen = target_seen.gather(1, safe_indices)
        selected_log_attempts = target_log_attempts.gather(1, safe_indices)

        aggregated_scores = target_exercise_embeddings.new_zeros(target_exercise_embeddings.size(0))
        pair_count = target_exercise_embeddings.new_zeros(target_exercise_embeddings.size(0))
        for left_index in range(max_concept_count - 1):
            for right_index in range(left_index + 1, max_concept_count):
                pair_mask = (ordered_valid[:, left_index] & ordered_valid[:, right_index]).to(
                    target_exercise_embeddings.dtype
                )
                pair_inputs = torch.cat(
                    [
                        selected_concept_embeddings[:, left_index],
                        selected_concept_embeddings[:, right_index],
                        target_exercise_embeddings,
                        selected_accuracy[:, left_index].unsqueeze(-1),
                        selected_seen[:, left_index].unsqueeze(-1),
                        selected_log_attempts[:, left_index].unsqueeze(-1),
                        selected_accuracy[:, right_index].unsqueeze(-1),
                        selected_seen[:, right_index].unsqueeze(-1),
                        selected_log_attempts[:, right_index].unsqueeze(-1),
                    ],
                    dim=-1,
                )
                pair_scores = self.pairwise_history_interaction_scorer(pair_inputs).squeeze(-1)
                aggregated_scores = aggregated_scores + pair_scores * pair_mask
                pair_count = pair_count + pair_mask

        aggregated_scores = aggregated_scores / pair_count.clamp_min(1.0)
        return aggregated_scores * concept_count_mask.squeeze(-1).to(aggregated_scores.dtype)

    def _build_interpretable_readout_expert_residual(
        self,
        *,
        q_vectors: torch.Tensor,
        student_tkc_mask: torch.Tensor,
        target_student_ids: torch.Tensor,
        student_state: torch.Tensor,
        q_repr: torch.Tensor,
        concept_summary: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        difficulty: torch.Tensor,
    ) -> torch.Tensor:
        concept_counts, mean_pooled, dispersion = concept_summary
        q_mask = (q_vectors > 0).to(student_state.dtype)
        target_concept_seen = student_tkc_mask[target_student_ids]
        seen_concept_count = (q_mask * target_concept_seen).sum(dim=1, keepdim=True)
        coverage = seen_concept_count / concept_counts.clamp_min(1.0)
        dispersion_score = dispersion.mean(dim=1, keepdim=True)
        gate_inputs = torch.cat(
            [
                (concept_counts - 1.0).detach(),
                difficulty.detach().unsqueeze(-1),
                dispersion_score.detach(),
                coverage.detach(),
            ],
            dim=-1,
        )
        gate_probs = torch.softmax(self.interpretable_readout_expert_gate(gate_inputs), dim=-1)
        expert_inputs = torch.cat(
            [
                student_state.detach(),
                q_repr.detach(),
                mean_pooled.detach(),
                dispersion.detach(),
                (student_state * q_repr).detach(),
                torch.abs(student_state - q_repr).detach(),
                difficulty.detach().unsqueeze(-1),
                (concept_counts - 1.0).detach(),
                coverage.detach(),
            ],
            dim=-1,
        )
        expert_scores = torch.stack(
            [expert(expert_inputs).squeeze(-1) for expert in self.interpretable_readout_experts],
            dim=-1,
        )
        return (expert_scores * gate_probs).sum(dim=-1)

    def _build_student_conditioned_ukc_readout_residual(
        self,
        *,
        q_vectors: torch.Tensor,
        concept_graph: torch.Tensor,
        q_matrix: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        student_tkc_mask: torch.Tensor,
        target_student_ids: torch.Tensor,
        tkc_states: torch.Tensor,
        ukc_states: torch.Tensor,
        student_state: torch.Tensor,
        q_repr: torch.Tensor,
        concept_summary: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        difficulty: torch.Tensor,
        student_concept_attempt_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.student_conditioned_ukc_readout_residual_head is None:
            return q_repr.new_zeros(q_repr.size(0))
        if student_concept_attempt_counts is None:
            student_concept_attempt_counts = student_exercise_mask.to(dtype=q_repr.dtype) @ q_matrix.to(dtype=q_repr.dtype)
        chunk_size = 8192
        if q_repr.size(0) > chunk_size:
            outputs = []
            for start in range(0, q_repr.size(0), chunk_size):
                stop = min(start + chunk_size, q_repr.size(0))
                outputs.append(
                    self._build_student_conditioned_ukc_readout_residual(
                        q_vectors=q_vectors[start:stop],
                        concept_graph=concept_graph,
                        q_matrix=q_matrix,
                        student_exercise_mask=student_exercise_mask,
                        student_tkc_mask=student_tkc_mask,
                        target_student_ids=target_student_ids[start:stop],
                        tkc_states=tkc_states,
                        ukc_states=ukc_states,
                        student_state=student_state[start:stop],
                        q_repr=q_repr[start:stop],
                        concept_summary=tuple(value[start:stop] for value in concept_summary),
                        difficulty=difficulty[start:stop],
                        student_concept_attempt_counts=student_concept_attempt_counts,
                    )
                )
            return torch.cat(outputs, dim=0)

        concept_counts, _, _ = concept_summary
        q_mask = q_vectors > 0
        max_concept_count = int(q_mask.sum(dim=1).max().item())
        if max_concept_count <= 0:
            return q_repr.new_zeros(q_repr.size(0))

        selected_scores, selected_indices = q_vectors.topk(k=max_concept_count, dim=1)
        selected_valid = selected_scores > 0
        safe_indices = selected_indices.masked_fill(~selected_valid, 0)

        if self.training:
            target_concept_attempt_counts = (student_concept_attempt_counts[target_student_ids] - q_vectors).clamp_min(
                0.0
            )
            target_tkc_mask = (target_concept_attempt_counts > 0).to(dtype=q_repr.dtype)
        else:
            target_concept_attempt_counts = student_concept_attempt_counts[target_student_ids]
            target_tkc_mask = student_tkc_mask[target_student_ids].to(dtype=q_repr.dtype)
        target_tkc_states = tkc_states[target_student_ids].detach()
        target_ukc_states = ukc_states[target_student_ids].detach()

        selected_graph_rows = concept_graph.to(dtype=q_repr.dtype)[safe_indices]
        graph_weights = selected_graph_rows * target_tkc_mask.unsqueeze(1)
        graph_weights = graph_weights * selected_valid.to(dtype=q_repr.dtype).unsqueeze(-1)
        neighbor_weight_mass = graph_weights.sum(dim=-1)
        neighbor_count = (graph_weights > 0).to(dtype=q_repr.dtype).sum(dim=-1)
        normalized_graph_weights = graph_weights / neighbor_weight_mass.unsqueeze(-1).clamp_min(1e-6)
        student_conditioned_concepts = torch.einsum("bcj,bjd->bcd", normalized_graph_weights, target_tkc_states)

        gather_indices = safe_indices.unsqueeze(-1).expand(-1, -1, target_ukc_states.size(-1))
        static_ukc_concepts = target_ukc_states.gather(dim=1, index=gather_indices)
        selected_valid_float = selected_valid.to(dtype=q_repr.dtype).unsqueeze(-1)
        pooled_student_conditioned = (student_conditioned_concepts * selected_valid_float).sum(dim=1) / concept_counts
        pooled_static_ukc = (static_ukc_concepts * selected_valid_float).sum(dim=1) / concept_counts

        target_log_attempts = torch.log1p(target_concept_attempt_counts)
        neighbor_log_attempts = torch.einsum("bcj,bj->bc", normalized_graph_weights, target_log_attempts)
        selected_valid_scalar = selected_valid.to(dtype=q_repr.dtype)
        mean_neighbor_count = (neighbor_count * selected_valid_scalar).sum(dim=1, keepdim=True) / concept_counts
        mean_neighbor_mass = (neighbor_weight_mass * selected_valid_scalar).sum(dim=1, keepdim=True) / concept_counts
        mean_neighbor_log_attempts = (neighbor_log_attempts * selected_valid_scalar).sum(dim=1, keepdim=True) / concept_counts

        seen_concept_count = (q_mask.to(dtype=q_repr.dtype) * target_tkc_mask).sum(dim=1, keepdim=True)
        coverage = seen_concept_count / concept_counts
        num_concepts = max(1, student_tkc_mask.size(1))
        residual_inputs = torch.cat(
            [
                student_state.detach(),
                q_repr.detach(),
                pooled_student_conditioned.detach(),
                pooled_static_ukc.detach(),
                torch.abs(pooled_student_conditioned - pooled_static_ukc).detach(),
                difficulty.detach().unsqueeze(-1),
                (concept_counts - 1.0).detach(),
                coverage.detach(),
                (mean_neighbor_count / float(num_concepts)).detach(),
                mean_neighbor_mass.detach(),
                mean_neighbor_log_attempts.detach(),
            ],
            dim=-1,
        )
        residual = self.student_conditioned_ukc_readout_residual_head(residual_inputs).squeeze(-1)
        target_mask = ((seen_concept_count.squeeze(-1) <= 0) & (mean_neighbor_count.squeeze(-1) > 0)).to(
            dtype=residual.dtype
        )
        return residual * target_mask

    def _build_concept_evidence_readout_residual(
        self,
        *,
        q_vectors: torch.Tensor,
        target_student_ids: torch.Tensor,
        student_concept_evidence: torch.Tensor | None,
        concept_summary: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        difficulty: torch.Tensor,
    ) -> torch.Tensor:
        if self.concept_evidence_readout_residual_head is None:
            return difficulty.new_zeros(difficulty.size(0))
        if student_concept_evidence is None:
            raise ValueError(
                "student_concept_evidence is required when concept_evidence_readout_residual is enabled."
            )

        concept_counts, _, _ = concept_summary
        q_mask = (q_vectors > 0).to(dtype=difficulty.dtype)
        target_evidence = student_concept_evidence[target_student_ids].to(dtype=difficulty.dtype)
        target_seen = target_evidence[..., 5] * q_mask
        seen_count = target_seen.sum(dim=1, keepdim=True)
        seen_ratio = seen_count / concept_counts.clamp_min(1.0)
        seen_denom = seen_count.clamp_min(1.0)

        target_accuracy = target_evidence[..., 3]
        target_log_attempts = target_evidence[..., 4]
        mean_accuracy = (target_accuracy * target_seen).sum(dim=1, keepdim=True) / seen_denom
        mean_log_attempts = (target_log_attempts * target_seen).sum(dim=1, keepdim=True) / seen_denom
        min_accuracy = target_accuracy.masked_fill(target_seen <= 0.0, 1.0).min(dim=1, keepdim=True).values
        max_accuracy = (target_accuracy * target_seen).max(dim=1, keepdim=True).values

        residual_inputs = torch.cat(
            [
                (concept_counts - 1.0).detach(),
                seen_ratio.detach(),
                mean_accuracy.detach(),
                min_accuracy.detach(),
                max_accuracy.detach(),
                mean_log_attempts.detach(),
                difficulty.detach().unsqueeze(-1),
            ],
            dim=-1,
        )
        residual = torch.tanh(self.concept_evidence_readout_residual_head(residual_inputs).squeeze(-1))
        residual = residual * self.concept_evidence_readout_max_logit
        trigger_mask = (
            (concept_counts.squeeze(-1) >= float(self.concept_evidence_readout_min_count))
            & (seen_ratio.squeeze(-1) >= self.concept_evidence_readout_min_seen_ratio)
        ).to(dtype=residual.dtype)
        return residual * trigger_mask

    def _build_concept_evidence_prior_residual(
        self,
        *,
        q_vectors: torch.Tensor,
        target_student_ids: torch.Tensor,
        student_concept_evidence: torch.Tensor | None,
        concept_summary: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        if not self.concept_evidence_prior_residual:
            return q_vectors.new_zeros(q_vectors.size(0))
        if student_concept_evidence is None:
            raise ValueError("student_concept_evidence is required when concept_evidence_prior_residual is enabled.")

        concept_counts, _, _ = concept_summary
        q_mask = (q_vectors > 0).to(dtype=q_vectors.dtype)
        target_evidence = student_concept_evidence[target_student_ids].to(dtype=q_vectors.dtype)
        target_attempts = target_evidence[..., 0] * q_mask
        target_correct = target_evidence[..., 1] * q_mask
        target_seen = target_evidence[..., 5] * q_mask

        attempt_count = target_attempts.sum(dim=1, keepdim=True)
        correct_count = target_correct.sum(dim=1, keepdim=True)
        seen_count = target_seen.sum(dim=1, keepdim=True)
        seen_ratio = seen_count / concept_counts.clamp_min(1.0)

        prior = self.concept_evidence_prior_strength
        smoothed_accuracy = (correct_count + 0.5 * prior) / (attempt_count + prior)
        signed_mastery = (smoothed_accuracy - 0.5) * 2.0
        confidence = torch.log1p(attempt_count) / torch.log1p(
            attempt_count.new_tensor(self.concept_evidence_prior_confidence_cap)
        )
        confidence = confidence.clamp(min=0.0, max=1.0)
        residual = signed_mastery.squeeze(-1) * confidence.squeeze(-1) * seen_ratio.squeeze(-1)
        residual = residual.clamp(min=-1.0, max=1.0) * self.concept_evidence_prior_max_logit
        trigger_mask = (
            (concept_counts.squeeze(-1) >= float(self.concept_evidence_prior_min_count))
            & (seen_ratio.squeeze(-1) >= self.concept_evidence_prior_min_seen_ratio)
        ).to(dtype=residual.dtype)
        return residual * trigger_mask
