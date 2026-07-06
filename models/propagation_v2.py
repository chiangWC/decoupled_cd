from __future__ import annotations

import torch
import torch.nn as nn

from .hetero_propagation import (
    PropagationOutput,
    _build_exercise_component,
    _logit,
    _masked_average,
    _resolve_tkc_prior,
)


class DecoupledPropagationV2(nn.Module):
    """
    V2 propagation. Single-graph only.

    Base behaviour reproduces the v1 semantics: TKC states mix behaviour-gated
    exercise messages with static concept-graph neighbor messages; UKC states
    are the static neighbor messages under the student's UKC mask.

    With ``ukc_personalization`` enabled (module 1), UKC states additionally
    receive a zero-initialized residual propagated from the student's own TKC
    states through the concept graph, weighted by evidence confidence. UKC
    nodes still receive no direct response supervision; behaviour signal flows
    in only through the TKC states. At initialization the residual is exactly
    zero, so the module starts equivalent to the base model.
    """

    def __init__(
        self,
        concept_dim: int,
        *,
        student_fusion_mode: str = "adaptive",
        student_gate_prior_alpha: float = 1.0,
        student_gate_prior_beta: float = 1.0,
        ukc_personalization: bool = False,
        ukc_layers: int = 1,
        ukc_evidence_cap: float = 20.0,
        ukc_hop_decay: float = 0.5,
    ):
        super().__init__()
        if student_fusion_mode not in {"adaptive", "tkc_only", "ukc_only", "mean"}:
            raise ValueError(f"Unsupported student_fusion_mode: {student_fusion_mode}")
        if ukc_layers < 1:
            raise ValueError("ukc_layers must be positive.")
        if ukc_evidence_cap <= 0.0:
            raise ValueError("ukc_evidence_cap must be positive.")
        self.student_fusion_mode = student_fusion_mode
        self.ukc_personalization = ukc_personalization
        self.ukc_layers = int(ukc_layers)
        self.ukc_evidence_cap = float(ukc_evidence_cap)
        self.ukc_hop_decay = float(ukc_hop_decay)

        self.correct_exercise_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.incorrect_exercise_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.tkc_concept_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.ukc_concept_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.exercise_behavior_gate = nn.Linear(concept_dim * 2, 1, bias=True)
        self.tkc_fusion_gate = nn.Linear(concept_dim * 2, 1, bias=True)
        self.student_fusion_gate = nn.Sequential(
            nn.Linear(concept_dim * 2 + 1, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 1),
        )
        fusion_prior = _resolve_tkc_prior(
            student_gate_prior_alpha=float(student_gate_prior_alpha),
            student_gate_prior_beta=float(student_gate_prior_beta),
        )
        nn.init.zeros_(self.student_fusion_gate[-1].weight)
        nn.init.constant_(self.student_fusion_gate[-1].bias, _logit(fusion_prior))

        if ukc_personalization:
            self.ukc_layer_transforms = nn.ModuleList(
                [nn.Linear(concept_dim, concept_dim, bias=False) for _ in range(self.ukc_layers)]
            )
            self.ukc_residual_out = nn.Linear(concept_dim, concept_dim, bias=False)
            nn.init.zeros_(self.ukc_residual_out.weight)
        else:
            self.ukc_layer_transforms = None
            self.ukc_residual_out = None

    def forward(
        self,
        *,
        concept_embeddings: torch.Tensor,
        exercise_embeddings: torch.Tensor,
        q_matrix: torch.Tensor,
        concept_graph: torch.Tensor,
        prerequisite_graph: torch.Tensor | None = None,
        similarity_graph: torch.Tensor | None = None,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        student_tkc_mask: torch.Tensor,
        student_ukc_mask: torch.Tensor,
        student_concept_evidence: torch.Tensor | None = None,
        student_indices: torch.Tensor | None = None,
    ) -> PropagationOutput:
        if prerequisite_graph is not None or similarity_graph is not None:
            raise ValueError("DecoupledPropagationV2 supports single-graph mode only.")
        if student_indices is not None:
            if student_indices.dim() != 1:
                raise ValueError("student_indices must be a 1D tensor.")
            student_exercise_mask = student_exercise_mask.index_select(0, student_indices)
            response_matrix = response_matrix.index_select(0, student_indices)
            student_tkc_mask = student_tkc_mask.index_select(0, student_indices)
            student_ukc_mask = student_ukc_mask.index_select(0, student_indices)
            if student_concept_evidence is not None:
                student_concept_evidence = student_concept_evidence.index_select(0, student_indices)

        correct_messages = self.correct_exercise_to_concept(exercise_embeddings)
        incorrect_messages = self.incorrect_exercise_to_concept(exercise_embeddings)
        tkc_neighbor_messages = concept_graph @ self.tkc_concept_to_concept(concept_embeddings)
        ukc_neighbor_messages = concept_graph @ self.ukc_concept_to_concept(concept_embeddings)

        correct_component = _build_exercise_component(
            weighted_exercises=student_exercise_mask * response_matrix,
            q_matrix=q_matrix,
            exercise_messages=correct_messages,
        )
        incorrect_component = _build_exercise_component(
            weighted_exercises=student_exercise_mask * (1.0 - response_matrix),
            q_matrix=q_matrix,
            exercise_messages=incorrect_messages,
        )
        behavior_inputs = torch.cat([correct_component, incorrect_component], dim=-1)
        behavior_gate = torch.sigmoid(self.exercise_behavior_gate(behavior_inputs))
        tkc_exercise_component = (
            behavior_gate * correct_component + (1.0 - behavior_gate) * incorrect_component
        )

        num_students = student_tkc_mask.size(0)
        tkc_neighbor_component = tkc_neighbor_messages.unsqueeze(0).expand(num_students, -1, -1)
        fusion_inputs = torch.cat([tkc_exercise_component, tkc_neighbor_component], dim=-1)
        fusion_gate = torch.sigmoid(self.tkc_fusion_gate(fusion_inputs))
        tkc_states = student_tkc_mask.unsqueeze(-1) * (
            fusion_gate * tkc_exercise_component + (1.0 - fusion_gate) * tkc_neighbor_component
        )

        static_ukc = ukc_neighbor_messages.unsqueeze(0).expand(num_students, -1, -1)
        if self.ukc_personalization:
            residual = self._personalized_ukc_residual(
                tkc_states=tkc_states,
                student_tkc_mask=student_tkc_mask,
                concept_graph=concept_graph,
                student_concept_evidence=student_concept_evidence,
            )
            ukc_states = student_ukc_mask.unsqueeze(-1) * (static_ukc + residual)
        else:
            ukc_states = student_ukc_mask.unsqueeze(-1) * static_ukc

        tkc_mean = _masked_average(tkc_states, student_tkc_mask)
        ukc_mean = _masked_average(ukc_states, student_ukc_mask)
        coverage = student_tkc_mask.to(dtype=tkc_mean.dtype).mean(dim=1, keepdim=True)
        tkc_weight = self._build_student_tkc_weight(
            coverage=coverage,
            tkc_mean=tkc_mean,
            ukc_mean=ukc_mean,
        )
        student_state = tkc_weight * tkc_mean + (1.0 - tkc_weight) * ukc_mean
        return PropagationOutput(
            tkc_states=tkc_states,
            ukc_states=ukc_states,
            student_state=student_state,
            tkc_weight=tkc_weight,
        )

    def _personalized_ukc_residual(
        self,
        *,
        tkc_states: torch.Tensor,
        student_tkc_mask: torch.Tensor,
        concept_graph: torch.Tensor,
        student_concept_evidence: torch.Tensor | None,
    ) -> torch.Tensor:
        if student_concept_evidence is not None:
            attempts = student_concept_evidence[..., 0].to(dtype=tkc_states.dtype)
            confidence = (
                torch.log1p(attempts)
                / torch.log1p(torch.tensor(self.ukc_evidence_cap, dtype=tkc_states.dtype, device=tkc_states.device))
            ).clamp(min=0.0, max=1.0)
        else:
            confidence = student_tkc_mask.to(dtype=tkc_states.dtype)

        source_weight = student_tkc_mask.to(dtype=tkc_states.dtype) * confidence
        # edge[s, k, k'] = A[k, k'] * source_weight[s, k']: UKC node k receives
        # from graph neighbors k' proportionally to the student's evidence there.
        edge = concept_graph.unsqueeze(0) * source_weight.unsqueeze(1)
        edge = edge / edge.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        static_norm = concept_graph / concept_graph.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        states = tkc_states
        aggregated = torch.zeros_like(tkc_states)
        for layer_index, transform in enumerate(self.ukc_layer_transforms):
            if layer_index == 0:
                states = torch.einsum("skj,sjd->skd", edge, transform(states))
            else:
                states = torch.einsum("kj,sjd->skd", static_norm, transform(states))
            aggregated = aggregated + (self.ukc_hop_decay**layer_index) * states
        return self.ukc_residual_out(aggregated)

    def _build_student_tkc_weight(
        self,
        *,
        coverage: torch.Tensor,
        tkc_mean: torch.Tensor,
        ukc_mean: torch.Tensor,
    ) -> torch.Tensor:
        if self.student_fusion_mode == "tkc_only":
            return torch.ones_like(coverage)
        if self.student_fusion_mode == "ukc_only":
            return torch.zeros_like(coverage)
        if self.student_fusion_mode == "mean":
            return torch.full_like(coverage, 0.5)
        fusion_inputs = torch.cat([coverage, tkc_mean, ukc_mean], dim=-1)
        return torch.sigmoid(self.student_fusion_gate(fusion_inputs))
