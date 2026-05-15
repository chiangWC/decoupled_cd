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

    def __init__(
        self,
        concept_dim: int,
        graph_mode: str = "single",
        student_gate_prior_alpha: float | None = None,
        student_gate_prior_beta: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        evidence_calibrated_behavior_gate: bool = False,
        evidence_behavior_gate_max_logit: float = 0.5,
        evidence_behavior_gate_trigger: str = "all",
        evidence_behavior_gate_low_attempt_threshold: float = 3.0,
    ):
        super().__init__()
        if graph_mode not in {"single", "dual"}:
            raise ValueError(f"Unsupported graph_mode: {graph_mode}")
        if evidence_behavior_gate_max_logit <= 0.0:
            raise ValueError("evidence_behavior_gate_max_logit must be positive.")
        if evidence_behavior_gate_trigger not in {"all", "low_evidence"}:
            raise ValueError(f"Unsupported evidence_behavior_gate_trigger: {evidence_behavior_gate_trigger}")
        if evidence_behavior_gate_low_attempt_threshold < 0.0:
            raise ValueError("evidence_behavior_gate_low_attempt_threshold must be non-negative.")
        self.graph_mode = graph_mode
        self.evidence_calibrated_behavior_gate = evidence_calibrated_behavior_gate
        self.evidence_behavior_gate_max_logit = float(evidence_behavior_gate_max_logit)
        self.evidence_behavior_gate_trigger = evidence_behavior_gate_trigger
        self.evidence_behavior_gate_low_attempt_threshold = float(evidence_behavior_gate_low_attempt_threshold)
        self.correct_exercise_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.incorrect_exercise_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.tkc_concept_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.ukc_concept_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.tkc_prerequisite_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.tkc_similarity_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.ukc_prerequisite_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.ukc_similarity_to_concept = nn.Linear(concept_dim, concept_dim, bias=False)
        self.graph_fusion_gate = nn.Linear(concept_dim * 2, 1, bias=True)
        self.exercise_behavior_gate = nn.Linear(concept_dim * 2, 1, bias=True)
        self.evidence_behavior_gate_residual = (
            nn.Linear(6, 1, bias=True) if evidence_calibrated_behavior_gate else None
        )
        self.tkc_fusion_gate = nn.Linear(concept_dim * 2, 1, bias=True)
        self.student_fusion_gate = nn.Sequential(
            nn.Linear(concept_dim * 2 + 1, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 1),
        )
        resolved_prior_alpha = _resolve_student_gate_prior_arg(
            explicit=student_gate_prior_alpha,
            legacy=alpha,
            default=1.0,
            new_name="student_gate_prior_alpha",
            legacy_name="alpha",
        )
        resolved_prior_beta = _resolve_student_gate_prior_arg(
            explicit=student_gate_prior_beta,
            legacy=beta,
            default=1.0,
            new_name="student_gate_prior_beta",
            legacy_name="beta",
        )
        fusion_prior = _resolve_tkc_prior(
            student_gate_prior_alpha=resolved_prior_alpha,
            student_gate_prior_beta=resolved_prior_beta,
        )
        nn.init.zeros_(self.student_fusion_gate[-1].weight)
        nn.init.constant_(self.student_fusion_gate[-1].bias, _logit(fusion_prior))
        if self.evidence_behavior_gate_residual is not None:
            nn.init.zeros_(self.evidence_behavior_gate_residual.weight)
            nn.init.zeros_(self.evidence_behavior_gate_residual.bias)

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
        student_concept_evidence: torch.Tensor | None = None,
    ) -> PropagationOutput:
        correct_exercise_messages = self.correct_exercise_to_concept(exercise_embeddings)
        incorrect_exercise_messages = self.incorrect_exercise_to_concept(exercise_embeddings)
        if self.graph_mode == "dual":
            if prerequisite_graph is None or similarity_graph is None:
                raise ValueError("Dual graph mode requires both prerequisite_graph and similarity_graph tensors.")
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
            if prerequisite_graph is not None or similarity_graph is not None:
                raise ValueError(
                    "Single graph mode does not accept prerequisite_graph or similarity_graph tensors."
                )
            tkc_neighbor_messages = concept_graph @ self.tkc_concept_to_concept(concept_embeddings)
            ukc_neighbor_messages = concept_graph @ self.ukc_concept_to_concept(concept_embeddings)

        correct_tkc_component = _build_exercise_component(
            weighted_exercises=student_exercise_mask * response_matrix,
            q_matrix=q_matrix,
            exercise_messages=correct_exercise_messages,
        )

        incorrect_tkc_component = _build_exercise_component(
            weighted_exercises=student_exercise_mask * (1.0 - response_matrix),
            q_matrix=q_matrix,
            exercise_messages=incorrect_exercise_messages,
        )

        exercise_behavior_inputs = torch.cat([correct_tkc_component, incorrect_tkc_component], dim=-1)
        exercise_behavior_gate_logits = self.exercise_behavior_gate(exercise_behavior_inputs)
        if self.evidence_calibrated_behavior_gate:
            if student_concept_evidence is None:
                raise ValueError(
                    "student_concept_evidence is required when evidence_calibrated_behavior_gate is enabled."
                )
            if student_concept_evidence.shape[:2] != exercise_behavior_gate_logits.shape[:2]:
                raise ValueError("student_concept_evidence must have shape [students, concepts, features].")
            if student_concept_evidence.size(-1) != 6:
                raise ValueError("student_concept_evidence must contain 6 interpretable features.")
            assert self.evidence_behavior_gate_residual is not None
            evidence_residual = torch.tanh(
                self.evidence_behavior_gate_residual(student_concept_evidence.to(exercise_behavior_inputs.dtype))
            ) * self.evidence_behavior_gate_max_logit
            trigger_mask = _evidence_trigger_mask(
                student_concept_evidence=student_concept_evidence,
                trigger=self.evidence_behavior_gate_trigger,
                low_attempt_threshold=self.evidence_behavior_gate_low_attempt_threshold,
            ).to(dtype=exercise_behavior_inputs.dtype)
            exercise_behavior_gate_logits = exercise_behavior_gate_logits + evidence_residual * trigger_mask
        exercise_behavior_gate = torch.sigmoid(exercise_behavior_gate_logits)
        tkc_exercise_component = (
            exercise_behavior_gate * correct_tkc_component
            + (1.0 - exercise_behavior_gate) * incorrect_tkc_component
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
        coverage = student_tkc_mask.to(dtype=tkc_mean.dtype).mean(dim=1, keepdim=True)
        fusion_inputs = torch.cat([coverage, tkc_mean, ukc_mean], dim=-1)
        tkc_weight = torch.sigmoid(self.student_fusion_gate(fusion_inputs))
        student_state = tkc_weight * tkc_mean + (1.0 - tkc_weight) * ukc_mean
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


def _evidence_trigger_mask(
    *,
    student_concept_evidence: torch.Tensor,
    trigger: str,
    low_attempt_threshold: float,
) -> torch.Tensor:
    if trigger == "all":
        return torch.ones_like(student_concept_evidence[..., :1])
    if trigger == "low_evidence":
        attempt_counts = student_concept_evidence[..., :1]
        seen = student_concept_evidence[..., 5:6]
        return ((attempt_counts <= float(low_attempt_threshold)) & (seen > 0.0)).to(dtype=student_concept_evidence.dtype)
    raise ValueError(f"Unsupported evidence behavior gate trigger: {trigger}")


def _resolve_student_gate_prior_arg(
    *,
    explicit: float | None,
    legacy: float | None,
    default: float,
    new_name: str,
    legacy_name: str,
) -> float:
    if explicit is not None and legacy is not None and float(explicit) != float(legacy):
        raise ValueError(f"Received conflicting values for {new_name} and deprecated {legacy_name}.")
    if explicit is not None:
        return float(explicit)
    if legacy is not None:
        return float(legacy)
    return float(default)


def _resolve_tkc_prior(*, student_gate_prior_alpha: float, student_gate_prior_beta: float) -> float:
    total = float(student_gate_prior_alpha) + float(student_gate_prior_beta)
    if total <= 0.0:
        return 0.5
    return min(max(float(student_gate_prior_alpha) / total, 1e-4), 1.0 - 1e-4)


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-6), 1.0 - 1e-6)
    return float(torch.logit(torch.tensor(clipped, dtype=torch.float32)).item())


def _aggregate_exercise_messages_by_concept(
    *,
    exercise_weights: torch.Tensor,
    q_matrix: torch.Tensor,
    exercise_messages: torch.Tensor,
    concept_weights: torch.Tensor,
) -> torch.Tensor:
    num_students = exercise_weights.size(0)
    num_concepts = q_matrix.size(1)
    dim = exercise_messages.size(1)
    output = exercise_messages.new_zeros((num_students, num_concepts, dim))

    for concept_index in range(num_concepts):
        exercise_indices = torch.nonzero(q_matrix[:, concept_index] > 0, as_tuple=False).squeeze(-1)
        if exercise_indices.numel() == 0:
            continue
        concept_message = exercise_weights[:, exercise_indices] @ exercise_messages[exercise_indices]
        denom = concept_weights[:, concept_index].unsqueeze(-1).clamp(min=1.0)
        output[:, concept_index, :] = concept_message / denom

    return output


def _build_exercise_component(
    *,
    weighted_exercises: torch.Tensor,
    q_matrix: torch.Tensor,
    exercise_messages: torch.Tensor,
) -> torch.Tensor:
    concept_weights = torch.einsum("se,ek->sk", weighted_exercises, q_matrix).clamp(min=1.0)
    return _aggregate_exercise_messages_by_concept(
        exercise_weights=weighted_exercises,
        q_matrix=q_matrix,
        exercise_messages=exercise_messages,
        concept_weights=concept_weights,
    )
