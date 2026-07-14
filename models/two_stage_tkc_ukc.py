from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch
from torch import nn
import torch.nn.functional as F


VALID_DIAGNOSIS_MODES = {
    "item_hypernetwork",
    "target_conditioned",
    "monotonic_control",
}


@dataclass
class SemanticNodeOutput:
    concept_nodes: torch.Tensor
    exercise_nodes: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


@dataclass
class EvidenceRepresentationOutput:
    student_evidence: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


@dataclass
class ConceptPriorOutput:
    concept_prior: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


@dataclass
class StateCompletionOutput:
    framework_state: torch.Tensor
    reliability: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


@dataclass
class DiagnosisOutput:
    probs: torch.Tensor
    cognitive_probs: torch.Tensor
    guess_probs: torch.Tensor
    slip_probs: torch.Tensor


@dataclass
class TwoStageForwardOutput:
    probs: torch.Tensor
    cognitive_probs: torch.Tensor
    framework_state: torch.Tensor
    mastery: torch.Tensor
    state_reliability: torch.Tensor
    student_state: torch.Tensor
    concept_embeddings: torch.Tensor
    exercise_embeddings: torch.Tensor
    guess_probs: torch.Tensor
    slip_probs: torch.Tensor
    difficulty: torch.Tensor
    module_diagnostics: dict[str, torch.Tensor]
    architecture_fingerprint: str
    mastery_aux_logits: torch.Tensor | None = None


class QSemanticNodeAlignment(nn.Module):
    """Align concept and exercise semantics through the Q incidence map.

    The component has no private trainable parameters: all variants receive
    the same learned identity embeddings and differ only in whether context is
    routed through the exercise-concept structure.  The ablation therefore
    tests Q-specific semantic alignment instead of capacity.
    """

    VALID_MODES = {
        "bidirectional_q",
        "raw_identity_control",
        "global_context_control",
    }

    def forward(
        self,
        *,
        concept_embeddings: torch.Tensor,
        exercise_embeddings: torch.Tensor,
        q_matrix: torch.Tensor,
        mode: str,
    ) -> SemanticNodeOutput:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported semantic node mode: {mode}")
        q_binary = (q_matrix > 0.0).to(
            dtype=concept_embeddings.dtype,
            device=concept_embeddings.device,
        )
        if mode == "bidirectional_q":
            concept_degree = q_binary.sum(dim=0).clamp_min(1.0)
            exercise_degree = q_binary.sum(dim=1, keepdim=True).clamp_min(1.0)
            concept_context = (
                q_binary.transpose(0, 1) @ exercise_embeddings
            ) / concept_degree.unsqueeze(-1)
            exercise_context = (
                q_binary @ concept_embeddings
            ) / exercise_degree
        elif mode == "global_context_control":
            concept_context = exercise_embeddings.mean(
                dim=0, keepdim=True
            ).expand_as(concept_embeddings)
            exercise_context = concept_embeddings.mean(
                dim=0, keepdim=True
            ).expand_as(exercise_embeddings)
        else:
            concept_context = torch.zeros_like(concept_embeddings)
            exercise_context = torch.zeros_like(exercise_embeddings)
        concept_nodes = concept_embeddings + concept_context
        exercise_nodes = exercise_embeddings + exercise_context
        return SemanticNodeOutput(
            concept_nodes=concept_nodes,
            exercise_nodes=exercise_nodes,
            diagnostics={
                "semantic_q_edge_count": q_binary.sum(),
                "semantic_concept_context_norm": concept_context.norm(
                    dim=-1
                ).mean(),
                "semantic_exercise_context_norm": exercise_context.norm(
                    dim=-1
                ).mean(),
            },
        )


class CalibratedEvidenceRepresentation(nn.Module):
    """Module 1: train-only responses to a student evidence representation.

    The full path conditions the student summary on the identities and
    population difficulty of attempted exercises.  Its clean control receives
    only raw response marginals.  Both paths have identical trainable capacity
    and return the same tensor contract.
    """

    VALID_MODES = {
        "calibrated_history",
        "identity_raw_control",
        "calibrated_summary_control",
        "raw_summary_control",
    }

    def __init__(self, *, dim: int, evidence_cap: float) -> None:
        super().__init__()
        if evidence_cap <= 0.0:
            raise ValueError("evidence_cap must be positive.")
        self.dim = int(dim)
        self.evidence_cap = float(evidence_cap)
        self.calibrated_encoder = nn.Sequential(
            nn.Linear(dim + 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.raw_control_encoder = nn.Sequential(
            nn.Linear(dim + 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        full_rng_state = torch.random.get_rng_state()
        self.identity_raw_encoder = nn.Sequential(
            nn.Linear(dim + 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.calibrated_summary_encoder = nn.Sequential(
            nn.Linear(dim + 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        torch.random.set_rng_state(full_rng_state)

    def active_parameter_counts(self) -> dict[str, int]:
        return {
            "calibrated_history": sum(
                parameter.numel()
                for parameter in self.calibrated_encoder.parameters()
            ),
            "identity_raw_control": sum(
                parameter.numel()
                for parameter in self.identity_raw_encoder.parameters()
            ),
            "calibrated_summary_control": sum(
                parameter.numel()
                for parameter in self.calibrated_summary_encoder.parameters()
            ),
            "raw_summary_control": sum(
                parameter.numel()
                for parameter in self.raw_control_encoder.parameters()
            ),
        }

    def forward(
        self,
        *,
        exercise_nodes: torch.Tensor,
        exercise_evidence: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        mode: str,
    ) -> EvidenceRepresentationOutput:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported evidence representation mode: {mode}")
        dtype = exercise_nodes.dtype
        mask = student_exercise_mask.to(dtype=dtype)
        responses = response_matrix.to(dtype=dtype)
        item_accuracy = exercise_evidence[:, 3].to(dtype=dtype)

        observed_count = mask.sum(dim=1, keepdim=True)
        safe_count = observed_count.clamp_min(1.0)
        correct_count = (mask * responses).sum(dim=1, keepdim=True)
        success = torch.where(
            observed_count > 0.0,
            correct_count / safe_count,
            correct_count.new_full(correct_count.shape, 0.5),
        )
        observed_difficulty = (
            mask * item_accuracy.unsqueeze(0)
        ).sum(dim=1, keepdim=True) / safe_count
        history_confidence = (
            torch.log1p(observed_count)
            / math.log1p(float(max(mask.size(1), 1)) + 1.0)
        ).clamp(0.0, 1.0)
        coverage = observed_count / float(max(mask.size(1), 1))
        calibrated_features = torch.cat(
            [
                success.mul(2.0).sub(1.0),
                success - observed_difficulty,
                history_confidence,
                coverage,
            ],
            dim=-1,
        )
        raw_features = torch.cat(
            [
                success.mul(2.0).sub(1.0),
                torch.zeros_like(success),
                history_confidence,
                coverage,
            ],
            dim=-1,
        )
        pooled_exercise = mask @ exercise_nodes / safe_count
        inputs_by_mode = {
            "calibrated_history": (pooled_exercise, calibrated_features),
            "identity_raw_control": (pooled_exercise, raw_features),
            "calibrated_summary_control": (
                torch.zeros_like(pooled_exercise),
                calibrated_features,
            ),
            "raw_summary_control": (
                torch.zeros_like(pooled_exercise), raw_features
            ),
        }
        encoders_by_mode = {
            "calibrated_history": self.calibrated_encoder,
            "identity_raw_control": self.identity_raw_encoder,
            "calibrated_summary_control": self.calibrated_summary_encoder,
            "raw_summary_control": self.raw_control_encoder,
        }
        semantic_input, statistic_input = inputs_by_mode[mode]
        student_evidence = encoders_by_mode[mode](
            torch.cat([semantic_input, statistic_input], dim=-1)
        )

        return EvidenceRepresentationOutput(
            student_evidence=student_evidence,
            diagnostics={
                "history_count": observed_count.squeeze(-1),
                "history_success": success.squeeze(-1),
                "history_difficulty_residual": (
                    success - observed_difficulty
                ).squeeze(-1),
            },
        )


class PopulationCalibratedConceptPrior(nn.Module):
    """Train-population item evidence to one concept-side prior state.

    Full uses item correctness and confidence with Q-specific aggregation.
    Semantic control keeps the Q path but removes population statistics.
    Global control keeps population statistics but removes concept-specific
    Q aggregation. All paths share the same tensor contract and capacity.
    """

    VALID_MODES = {
        "population_q",
        "semantic_q_control",
        "global_population_control",
    }

    def __init__(self, *, dim: int, evidence_cap: float) -> None:
        super().__init__()
        self.evidence_cap = float(evidence_cap)
        self.population_q_encoder = nn.Sequential(
            nn.Linear(dim + 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        full_rng_state = torch.random.get_rng_state()
        self.semantic_q_encoder = nn.Sequential(
            nn.Linear(dim + 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.global_population_encoder = nn.Sequential(
            nn.Linear(dim + 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        torch.random.set_rng_state(full_rng_state)

    def active_parameter_counts(self) -> dict[str, int]:
        return {
            "population_q": sum(
                parameter.numel()
                for parameter in self.population_q_encoder.parameters()
            ),
            "semantic_q_control": sum(
                parameter.numel()
                for parameter in self.semantic_q_encoder.parameters()
            ),
            "global_population_control": sum(
                parameter.numel()
                for parameter in self.global_population_encoder.parameters()
            ),
        }

    def forward(
        self,
        *,
        exercise_nodes: torch.Tensor,
        exercise_evidence: torch.Tensor,
        q_matrix: torch.Tensor,
        mode: str,
    ) -> ConceptPriorOutput:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported concept prior mode: {mode}")
        dtype = exercise_nodes.dtype
        item_attempts = exercise_evidence[:, 0].to(dtype=dtype)
        item_accuracy = exercise_evidence[:, 3].to(dtype=dtype)
        item_confidence = (
            exercise_evidence[:, 4].to(dtype=dtype)
            / math.log1p(self.evidence_cap * 100.0)
        ).clamp(0.0, 1.0)
        population_features = torch.stack(
            [item_accuracy.mul(2.0).sub(1.0), item_confidence],
            dim=-1,
        )
        q_binary = (q_matrix > 0.0).to(dtype=dtype)
        concept_item_count = q_binary.sum(dim=0).clamp_min(1.0)

        if mode == "population_q":
            item_tokens = self.population_q_encoder(
                torch.cat([exercise_nodes, population_features], dim=-1)
            )
            concept_prior = q_binary.transpose(0, 1) @ item_tokens
            concept_prior = concept_prior / concept_item_count.unsqueeze(-1)
        elif mode == "semantic_q_control":
            item_tokens = self.semantic_q_encoder(
                torch.cat(
                    [exercise_nodes, torch.zeros_like(population_features)],
                    dim=-1,
                )
            )
            concept_prior = q_binary.transpose(0, 1) @ item_tokens
            concept_prior = concept_prior / concept_item_count.unsqueeze(-1)
        else:
            item_tokens = self.global_population_encoder(
                torch.cat([exercise_nodes, population_features], dim=-1)
            )
            global_prior = item_tokens.mean(dim=0, keepdim=True)
            concept_prior = global_prior.expand(q_matrix.size(1), -1)

        return ConceptPriorOutput(
            concept_prior=concept_prior,
            diagnostics={
                "population_attempts": item_attempts,
                "concept_prior_norm": concept_prior.norm(dim=-1),
            },
        )


class PersonalizedStateCompletion(nn.Module):
    """Module 2: evidence-conditioned completion of TKC and UKC states.

    Full completion generates every concept state from the student evidence and
    a population concept prior.  The control directly projects observed
    concept statistics and assigns a student-independent prior to UKC.
    Identical decoder shapes make the comparison capacity matched.
    """

    VALID_MODES = {
        "personalized_interaction",
        "additive_personalized_control",
        "direct_prior_control",
    }

    def __init__(self, *, dim: int, evidence_cap: float) -> None:
        super().__init__()
        self.evidence_cap = float(evidence_cap)
        self.raw_evidence_projection = nn.Linear(3, dim)
        self.personalized_decoder = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.direct_control_decoder = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(4, max(8, dim // 2)),
            nn.ReLU(),
            nn.Linear(max(8, dim // 2), 1),
        )
        # Instantiated after the already validated Full path so adding this
        # stronger control does not perturb any existing Full initialization.
        # It consumes the same student and concept inputs as Full but removes
        # the declared multiplicative student-concept interaction.
        full_rng_state = torch.random.get_rng_state()
        self.additive_control_decoder = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        torch.random.set_rng_state(full_rng_state)

    def active_parameter_counts(self) -> dict[str, int]:
        return {
            "personalized_interaction": sum(
                parameter.numel()
                for parameter in self.personalized_decoder.parameters()
            ),
            "additive_personalized_control": sum(
                parameter.numel()
                for parameter in self.additive_control_decoder.parameters()
            ),
            "direct_prior_control": sum(
                parameter.numel()
                for parameter in self.direct_control_decoder.parameters()
            ),
        }

    def _raw_features(
        self,
        evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attempts = evidence[..., 0].clamp_min(0.0)
        correct = torch.minimum(evidence[..., 1].clamp_min(0.0), attempts)
        seen = evidence[..., 5].clamp(0.0, 1.0)
        accuracy = correct / attempts.clamp_min(1.0)
        confidence = (
            evidence[..., 4] / math.log1p(self.evidence_cap)
        ).clamp(0.0, 1.0)
        features = torch.stack(
            [
                accuracy.mul(2.0).sub(1.0) * confidence,
                confidence,
                seen,
            ],
            dim=-1,
        )
        return features, seen, confidence

    def forward(
        self,
        *,
        student_evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        concept_prior: torch.Tensor,
        student_concept_evidence: torch.Tensor,
        mode: str,
    ) -> StateCompletionOutput:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported state completion mode: {mode}")
        dtype = concept_nodes.dtype
        raw_features, seen, confidence = self._raw_features(
            student_concept_evidence.to(dtype=dtype)
        )
        batch_size = student_evidence.size(0)
        num_concepts = concept_nodes.size(0)
        student_grid = student_evidence.unsqueeze(1).expand(
            -1, num_concepts, -1
        )
        concept_grid = concept_nodes.unsqueeze(0).expand(batch_size, -1, -1)
        prior_grid = concept_prior.unsqueeze(0).expand(batch_size, -1, -1)
        if mode == "personalized_interaction":
            framework_state = self.personalized_decoder(
                torch.cat(
                    [
                        student_grid,
                        concept_grid,
                        student_grid * concept_grid,
                        prior_grid,
                    ],
                    dim=-1,
                )
            )
        elif mode == "additive_personalized_control":
            framework_state = self.additive_control_decoder(
                torch.cat(
                    [
                        student_grid,
                        concept_grid,
                        torch.zeros_like(concept_grid),
                        prior_grid,
                    ],
                    dim=-1,
                )
            )
        else:
            observed_raw = concept_grid + self.raw_evidence_projection(
                raw_features
            )
            static_prior = concept_grid + prior_grid
            direct_base = torch.where(
                seen.unsqueeze(-1) > 0.0,
                observed_raw,
                static_prior,
            )
            framework_state = self.direct_control_decoder(
                torch.cat(
                    [
                        direct_base,
                        concept_grid,
                        torch.zeros_like(concept_grid),
                        prior_grid,
                    ],
                    dim=-1,
                )
            )
        history_count = student_concept_evidence[..., 0].sum(dim=1)
        student_success = (
            student_concept_evidence[..., 1].sum(dim=1)
            / history_count.clamp_min(1.0)
        )
        reliability_features = torch.stack(
            [
                seen,
                confidence,
                student_success.unsqueeze(1).expand(-1, num_concepts),
                (
                    history_count / (history_count + self.evidence_cap)
                ).unsqueeze(1).expand(-1, num_concepts),
            ],
            dim=-1,
        )
        reliability = torch.sigmoid(
            self.reliability_head(reliability_features)
        ).squeeze(-1)
        return StateCompletionOutput(
            framework_state=framework_state,
            reliability=reliability,
            diagnostics={
                "tkc_count": seen.sum(dim=1),
                "ukc_count": (1.0 - seen).sum(dim=1),
                "mean_state_reliability": reliability.mean(dim=1),
            },
        )


class _ObservedAnchorFieldBranch(nn.Module):
    """One capacity-matched branch of the observed-anchor state field."""

    def __init__(self, *, dim: int) -> None:
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(dim + 3, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
        )
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.direct_projection = nn.Linear(3, dim)
        self.state_decoder = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(3, max(8, dim // 2)),
            nn.ReLU(),
            nn.Linear(max(8, dim // 2), 1),
        )

    def forward(
        self,
        *,
        student_evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        concept_prior: torch.Tensor,
        raw_features: torch.Tensor,
        seen: torch.Tensor,
        confidence: torch.Tensor,
        query_specific: bool,
    ) -> StateCompletionOutput:
        batch_size = student_evidence.size(0)
        num_concepts = concept_nodes.size(0)
        concept_grid = concept_nodes.unsqueeze(0).expand(batch_size, -1, -1)
        prior_grid = concept_prior.unsqueeze(0).expand(batch_size, -1, -1)
        coordinate = self.coordinate_encoder(
            torch.cat([concept_grid, prior_grid], dim=-1)
        )
        context_tokens = self.context_encoder(
            torch.cat([concept_grid, raw_features], dim=-1)
        )
        keys = self.key_projection(context_tokens)
        values = self.value_projection(context_tokens)
        if query_specific:
            queries = self.query_projection(coordinate)
        else:
            queries = self.query_projection(student_evidence).unsqueeze(1)
            queries = queries.expand(-1, num_concepts, -1)
        attention_logits = torch.matmul(
            queries,
            keys.transpose(1, 2),
        ) / math.sqrt(float(queries.size(-1)))
        observed_keys = seen.unsqueeze(1) > 0.0
        attention_logits = attention_logits.masked_fill(
            ~observed_keys,
            -1.0e4,
        )
        attention = torch.softmax(attention_logits, dim=-1)
        attention = attention * observed_keys.to(dtype=attention.dtype)
        attention = attention / attention.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-8)
        attended_context = torch.matmul(attention, values)
        direct_context = self.direct_projection(raw_features)
        direct_context = direct_context * seen.unsqueeze(-1)
        student_grid = student_evidence.unsqueeze(1).expand(
            -1, num_concepts, -1
        )
        framework_state = coordinate + self.state_decoder(
            torch.cat(
                [
                    student_grid,
                    coordinate,
                    attended_context,
                    direct_context,
                ],
                dim=-1,
            )
        )
        max_attention = attention.max(dim=-1).values
        reliability = torch.sigmoid(
            self.reliability_head(
                torch.stack([seen, confidence, max_attention], dim=-1)
            )
        ).squeeze(-1)
        attention_entropy = -(
            attention.clamp_min(1.0e-8).log() * attention
        ).sum(dim=-1)
        return StateCompletionOutput(
            framework_state=framework_state,
            reliability=reliability,
            diagnostics={
                "observed_anchor_count": seen.sum(dim=1),
                "mean_anchor_attention_entropy": attention_entropy.mean(
                    dim=1
                ),
                "mean_anchor_attention_max": max_attention.mean(dim=1),
                "mean_state_reliability": reliability.mean(dim=1),
            },
        )


class ObservedAnchorStateField(nn.Module):
    """Generate all concept states by reading observed concept anchors.

    The Full path uses a concept-specific query, following the deterministic
    cross-attention idea of Attentive Neural Processes.  The capacity control
    uses an otherwise identical branch but one student-global query, so every
    concept receives the same compressed context.  Both produce the complete
    state tensor consumed by diagnosis.
    """

    VALID_MODES = {
        "query_attentive_field",
        "global_attentive_control",
    }

    def __init__(self, *, dim: int, evidence_cap: float) -> None:
        super().__init__()
        self.evidence_cap = float(evidence_cap)
        initial_rng_state = torch.random.get_rng_state()
        self.query_branch = _ObservedAnchorFieldBranch(dim=dim)
        post_branch_rng_state = torch.random.get_rng_state()
        torch.random.set_rng_state(initial_rng_state)
        self.global_control_branch = _ObservedAnchorFieldBranch(dim=dim)
        torch.random.set_rng_state(post_branch_rng_state)

    def active_parameter_counts(self) -> dict[str, int]:
        return {
            "query_attentive_field": sum(
                parameter.numel()
                for parameter in self.query_branch.parameters()
            ),
            "global_attentive_control": sum(
                parameter.numel()
                for parameter in self.global_control_branch.parameters()
            ),
        }

    def _raw_features(
        self,
        evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attempts = evidence[..., 0].clamp_min(0.0)
        correct = torch.minimum(evidence[..., 1].clamp_min(0.0), attempts)
        seen = evidence[..., 5].clamp(0.0, 1.0)
        accuracy = correct / attempts.clamp_min(1.0)
        confidence = (
            evidence[..., 4] / math.log1p(self.evidence_cap)
        ).clamp(0.0, 1.0)
        raw_features = torch.stack(
            [
                accuracy.mul(2.0).sub(1.0) * confidence,
                confidence,
                seen,
            ],
            dim=-1,
        )
        return raw_features, seen, confidence

    def forward(
        self,
        *,
        student_evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        concept_prior: torch.Tensor,
        student_concept_evidence: torch.Tensor,
        mode: str,
    ) -> StateCompletionOutput:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported observed-anchor field mode: {mode}")
        raw_features, seen, confidence = self._raw_features(
            student_concept_evidence.to(dtype=concept_nodes.dtype)
        )
        branch = (
            self.query_branch
            if mode == "query_attentive_field"
            else self.global_control_branch
        )
        return branch(
            student_evidence=student_evidence,
            concept_nodes=concept_nodes,
            concept_prior=concept_prior,
            raw_features=raw_features,
            seen=seen,
            confidence=confidence,
            query_specific=mode == "query_attentive_field",
        )


class OutcomePartitionedEvidenceRefinement(nn.Module):
    """Refine student evidence from correct and incorrect history sets.

    The Full path treats correct and incorrect interactions as two related
    sets and preserves their distinct semantic centroids.  Capacity and direct
    controls use the same inputs, output contract, and number of parameters,
    but remove outcome-specific set partitioning.
    """

    VALID_MODES = {
        "identity_passthrough",
        "outcome_multiset",
        "unconditioned_set_control",
        "base_capacity_control",
    }

    def __init__(self, *, dim: int) -> None:
        super().__init__()
        initial_rng_state = torch.random.get_rng_state()
        self.outcome_encoder = self._make_encoder(dim)
        post_full_rng_state = torch.random.get_rng_state()
        torch.random.set_rng_state(initial_rng_state)
        self.unconditioned_control_encoder = self._make_encoder(dim)
        torch.random.set_rng_state(initial_rng_state)
        self.base_control_encoder = self._make_encoder(dim)
        torch.random.set_rng_state(post_full_rng_state)

    @staticmethod
    def _make_encoder(dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(dim * 4 + 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def active_parameter_counts(self) -> dict[str, int]:
        return {
            "outcome_multiset": sum(
                parameter.numel()
                for parameter in self.outcome_encoder.parameters()
            ),
            "unconditioned_set_control": sum(
                parameter.numel()
                for parameter in (
                    self.unconditioned_control_encoder.parameters()
                )
            ),
            "base_capacity_control": sum(
                parameter.numel()
                for parameter in self.base_control_encoder.parameters()
            ),
        }

    def forward(
        self,
        *,
        student_evidence: torch.Tensor,
        exercise_nodes: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        mode: str,
    ) -> EvidenceRepresentationOutput:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported evidence refinement mode: {mode}")
        if mode == "identity_passthrough":
            return EvidenceRepresentationOutput(
                student_evidence=student_evidence,
                diagnostics={
                    "outcome_refinement_active": student_evidence.new_zeros(
                        student_evidence.size(0)
                    )
                },
            )

        dtype = exercise_nodes.dtype
        mask = student_exercise_mask.to(dtype=dtype)
        responses = response_matrix.to(dtype=dtype)
        correct_mask = mask * responses
        incorrect_mask = mask * (1.0 - responses)
        observed_count = mask.sum(dim=1, keepdim=True)
        correct_count = correct_mask.sum(dim=1, keepdim=True)
        incorrect_count = incorrect_mask.sum(dim=1, keepdim=True)
        exposure_pool = mask @ exercise_nodes / observed_count.clamp_min(1.0)
        correct_pool = (
            correct_mask @ exercise_nodes / correct_count.clamp_min(1.0)
        )
        incorrect_pool = (
            incorrect_mask @ exercise_nodes / incorrect_count.clamp_min(1.0)
        )
        response_contrast = correct_pool - incorrect_pool
        normalizer = math.log1p(float(max(mask.size(1), 1)))
        statistics = torch.cat(
            [
                correct_count / observed_count.clamp_min(1.0),
                torch.log1p(correct_count).div(normalizer).clamp(0.0, 1.0),
                torch.log1p(incorrect_count).div(normalizer).clamp(0.0, 1.0),
                observed_count / float(max(mask.size(1), 1)),
            ],
            dim=-1,
        )
        if mode == "outcome_multiset":
            semantic_inputs = [
                correct_pool,
                incorrect_pool,
                response_contrast,
            ]
            encoder = self.outcome_encoder
        elif mode == "unconditioned_set_control":
            semantic_inputs = [
                exposure_pool,
                exposure_pool,
                torch.zeros_like(exposure_pool),
            ]
            encoder = self.unconditioned_control_encoder
        else:
            semantic_inputs = [
                torch.zeros_like(student_evidence),
                torch.zeros_like(student_evidence),
                torch.zeros_like(student_evidence),
            ]
            encoder = self.base_control_encoder
        refined_evidence = encoder(
            torch.cat(
                [student_evidence, *semantic_inputs, statistics],
                dim=-1,
            )
        )
        return EvidenceRepresentationOutput(
            student_evidence=refined_evidence,
            diagnostics={
                "outcome_refinement_active": refined_evidence.new_ones(
                    refined_evidence.size(0)
                ),
                "outcome_correct_count": correct_count.squeeze(-1),
                "outcome_incorrect_count": incorrect_count.squeeze(-1),
                "outcome_contrast_norm": response_contrast.norm(dim=-1),
            },
        )


class ItemConditionedHyperDiagnosis(nn.Module):
    """Generate a low-rank diagnosis function from the target item.

    This module treats student state and target-item representation as a
    Cartesian-product input.  Instead of concatenating the two vectors into a
    fixed predictor, the item generates a low-rank linear map that acts on the
    student state.  It owns the complete cognitive/guess/slip response path.
    """

    def __init__(
        self,
        *,
        dim: int,
        rank: int = 3,
        dropout: float = 0.0,
        max_guess: float = 0.3,
        max_slip: float = 0.3,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("Hypernetwork rank must be positive.")
        self.rank = int(rank)
        self.max_guess = float(max_guess)
        self.max_slip = float(max_slip)
        self.state_to_rank = nn.Linear(dim, rank)
        self.item_to_weights = nn.Linear(dim, dim * rank)
        self.dynamic_norm = nn.LayerNorm(dim)
        hidden_dim = max(8, dim // 2)
        self.cognitive_head = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.guess_head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        self.slip_head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        *,
        q_state: torch.Tensor,
        q_repr: torch.Tensor,
        difficulty: torch.Tensor,
    ) -> DiagnosisOutput:
        batch_size, dim = q_state.shape
        rank_state = self.state_to_rank(q_state)
        item_weights = self.item_to_weights(q_repr).view(
            batch_size,
            dim,
            self.rank,
        )
        dynamic_state = torch.einsum(
            "ndr,nr->nd",
            item_weights,
            rank_state,
        ) / math.sqrt(float(self.rank))
        dynamic_state = self.dynamic_norm(dynamic_state)
        diagnosis_features = torch.cat([dynamic_state, q_repr], dim=-1)
        cognitive_probs = torch.sigmoid(
            self.cognitive_head(diagnosis_features).squeeze(-1) - difficulty
        )
        guess_probs = self.max_guess * torch.sigmoid(
            self.guess_head(diagnosis_features).squeeze(-1)
        )
        slip_probs = self.max_slip * torch.sigmoid(
            self.slip_head(diagnosis_features).squeeze(-1)
        )
        probs = (
            (1.0 - slip_probs) * cognitive_probs
            + guess_probs * (1.0 - cognitive_probs)
        )
        return DiagnosisOutput(
            probs=probs,
            cognitive_probs=cognitive_probs,
            guess_probs=guess_probs,
            slip_probs=slip_probs,
        )


class TwoStageTKCUKCCDM(nn.Module):
    """Two claimed modules followed by a fixed Q-conditioned diagnosis."""

    def __init__(
        self,
        *,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        concept_dim: int = 64,
        semantic_node_mode: str = "bidirectional_q",
        evidence_mode: str = "calibrated_history",
        evidence_refinement_mode: str = "identity_passthrough",
        concept_prior_mode: str = "population_q",
        completion_mode: str = "personalized_interaction",
        diagnosis_mode: str = "target_conditioned",
        evidence_cap: float = 20.0,
        readout_dropout: float = 0.0,
        max_guess: float = 0.3,
        max_slip: float = 0.3,
        **_unused_kwargs,
    ) -> None:
        super().__init__()
        del num_students
        if semantic_node_mode not in QSemanticNodeAlignment.VALID_MODES:
            raise ValueError(
                f"Unsupported semantic node mode: {semantic_node_mode}"
            )
        if evidence_mode not in CalibratedEvidenceRepresentation.VALID_MODES:
            raise ValueError(f"Unsupported evidence mode: {evidence_mode}")
        if (
            evidence_refinement_mode
            not in OutcomePartitionedEvidenceRefinement.VALID_MODES
        ):
            raise ValueError(
                "Unsupported evidence refinement mode: "
                f"{evidence_refinement_mode}"
            )
        if concept_prior_mode not in PopulationCalibratedConceptPrior.VALID_MODES:
            raise ValueError(
                f"Unsupported concept prior mode: {concept_prior_mode}"
            )
        valid_completion_modes = (
            PersonalizedStateCompletion.VALID_MODES
            | ObservedAnchorStateField.VALID_MODES
        )
        if completion_mode not in valid_completion_modes:
            raise ValueError(f"Unsupported completion mode: {completion_mode}")
        if diagnosis_mode not in VALID_DIAGNOSIS_MODES:
            raise ValueError(f"Unsupported diagnosis mode: {diagnosis_mode}")
        self.concept_dim = int(concept_dim)
        self.semantic_node_mode = semantic_node_mode
        self.evidence_mode = evidence_mode
        self.evidence_refinement_mode = evidence_refinement_mode
        self.concept_prior_mode = concept_prior_mode
        self.completion_mode = completion_mode
        self.diagnosis_mode = diagnosis_mode
        self.evidence_cap = float(evidence_cap)
        self.max_guess = float(max_guess)
        self.max_slip = float(max_slip)

        self.concept_embedding = nn.Embedding(num_concepts, concept_dim)
        self.exercise_embedding = nn.Embedding(num_exercises, concept_dim)
        self.semantic_node_alignment = QSemanticNodeAlignment()
        self.evidence_representation = CalibratedEvidenceRepresentation(
            dim=concept_dim,
            evidence_cap=evidence_cap,
        )
        self.concept_prior = PopulationCalibratedConceptPrior(
            dim=concept_dim,
            evidence_cap=evidence_cap,
        )
        self.state_completion = PersonalizedStateCompletion(
            dim=concept_dim,
            evidence_cap=evidence_cap,
        )
        self.q_projection = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU(),
            nn.LayerNorm(concept_dim),
        )
        self.cognitive_match = nn.Sequential(
            nn.Linear(concept_dim * 4, concept_dim),
            nn.ReLU(),
            nn.Dropout(readout_dropout),
            nn.Linear(concept_dim, 1),
        )
        self.guess_head = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 1),
        )
        self.slip_head = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 1),
        )
        self.exercise_difficulty = nn.Embedding(num_exercises, 1)
        self.mastery_head = nn.Linear(concept_dim, 1)
        nn.init.xavier_uniform_(self.concept_embedding.weight)
        nn.init.xavier_uniform_(self.exercise_embedding.weight)
        nn.init.zeros_(self.exercise_difficulty.weight)

        # The capacity-matched monotonic diagnosis is instantiated only after
        # all Full parameters are initialized, preserving the validated Full
        # initialization while keeping one common state dict for new variants.
        self.control_cognitive_match = nn.Sequential(
            nn.Linear(concept_dim * 4, concept_dim),
            nn.ReLU(),
            nn.Dropout(readout_dropout),
            nn.Linear(concept_dim, 1),
        )
        self.control_guess_head = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 1),
        )
        self.control_slip_head = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 1),
        )
        self.control_discrimination_raw = nn.Parameter(
            torch.tensor(0.54132485)
        )
        self.observed_anchor_state_field = ObservedAnchorStateField(
            dim=concept_dim,
            evidence_cap=evidence_cap,
        )
        self.item_conditioned_hyper_diagnosis = (
            ItemConditionedHyperDiagnosis(
                dim=concept_dim,
                rank=3,
                dropout=readout_dropout,
                max_guess=max_guess,
                max_slip=max_slip,
            )
        )
        self.outcome_evidence_refinement = (
            OutcomePartitionedEvidenceRefinement(dim=concept_dim)
        )

    @property
    def architecture_fingerprint(self) -> str:
        payload = {
            "family": "two_stage_tkc_ukc_v8",
            "concept_dim": self.concept_dim,
            "student_id_embedding": False,
            "student_specific_bypass": False,
            "evidence_refinement": (
                "outcome_multiset_or_capacity_controls_or_passthrough"
            ),
            "diagnosis": (
                "item_hypernetwork_or_target_conditioned_or_"
                "monotonic_control"
            ),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    def initialization_hash(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def active_module_parameter_counts(self) -> dict[str, dict[str, int]]:
        full_diagnosis = sum(
            parameter.numel()
            for module in [
                self.cognitive_match,
                self.guess_head,
                self.slip_head,
            ]
            for parameter in module.parameters()
        )
        control_diagnosis = sum(
            parameter.numel()
            for module in [
                self.control_cognitive_match,
                self.control_guess_head,
                self.control_slip_head,
            ]
            for parameter in module.parameters()
        ) + self.control_discrimination_raw.numel()
        hyper_diagnosis = sum(
            parameter.numel()
            for parameter in (
                self.item_conditioned_hyper_diagnosis.parameters()
            )
        )
        shared_semantic_capacity = (
            self.concept_embedding.weight.numel()
            + self.exercise_embedding.weight.numel()
        )
        return {
            "semantic_node_alignment": {
                "bidirectional_q": shared_semantic_capacity,
                "raw_identity_control": shared_semantic_capacity,
                "global_context_control": shared_semantic_capacity,
            },
            "evidence_representation": (
                self.evidence_representation.active_parameter_counts()
            ),
            "outcome_evidence_refinement": (
                self.outcome_evidence_refinement.active_parameter_counts()
            ),
            "concept_prior": self.concept_prior.active_parameter_counts(),
            "state_completion": self.state_completion.active_parameter_counts(),
            "observed_anchor_state_field": (
                self.observed_anchor_state_field.active_parameter_counts()
            ),
            "diagnosis": {
                "item_hypernetwork": hyper_diagnosis,
                "target_conditioned": full_diagnosis,
                "monotonic_control": control_diagnosis,
            },
        }

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
        exercise_evidence: torch.Tensor | None = None,
        target_student_ids: torch.Tensor | None = None,
        target_exercise_ids: torch.Tensor | None = None,
        use_student_subset: bool = False,
    ) -> TwoStageForwardOutput:
        del concept_graph, prerequisite_graph, similarity_graph
        del student_tkc_mask, student_ukc_mask
        if target_student_ids is None or target_exercise_ids is None:
            raise ValueError("target student and exercise IDs are required.")
        if student_concept_evidence is None or exercise_evidence is None:
            raise ValueError("train-only evidence tensors are required.")
        if use_student_subset:
            student_indices, target_state_rows = torch.unique(
                target_student_ids,
                sorted=True,
                return_inverse=True,
            )
            mask = student_exercise_mask.index_select(0, student_indices)
            responses = response_matrix.index_select(0, student_indices)
            concept_evidence = student_concept_evidence.index_select(
                0, student_indices
            )
        else:
            target_state_rows = target_student_ids
            mask = student_exercise_mask
            responses = response_matrix
            concept_evidence = student_concept_evidence

        semantic_output = self.semantic_node_alignment(
            concept_embeddings=self.concept_embedding.weight,
            exercise_embeddings=self.exercise_embedding.weight,
            q_matrix=q_matrix,
            mode=self.semantic_node_mode,
        )
        concept_nodes = semantic_output.concept_nodes
        exercise_nodes = semantic_output.exercise_nodes
        evidence_output = self.evidence_representation(
            exercise_nodes=exercise_nodes,
            exercise_evidence=exercise_evidence,
            student_exercise_mask=mask,
            response_matrix=responses,
            mode=self.evidence_mode,
        )
        refinement_output = self.outcome_evidence_refinement(
            student_evidence=evidence_output.student_evidence,
            exercise_nodes=exercise_nodes,
            student_exercise_mask=mask,
            response_matrix=responses,
            mode=self.evidence_refinement_mode,
        )
        prior_output = self.concept_prior(
            exercise_nodes=exercise_nodes,
            exercise_evidence=exercise_evidence,
            q_matrix=q_matrix,
            mode=self.concept_prior_mode,
        )
        if self.completion_mode in ObservedAnchorStateField.VALID_MODES:
            completion_output = self.observed_anchor_state_field(
                student_evidence=refinement_output.student_evidence,
                concept_nodes=concept_nodes,
                concept_prior=prior_output.concept_prior,
                student_concept_evidence=concept_evidence,
                mode=self.completion_mode,
            )
        else:
            completion_output = self.state_completion(
                student_evidence=refinement_output.student_evidence,
                concept_nodes=concept_nodes,
                concept_prior=prior_output.concept_prior,
                student_concept_evidence=concept_evidence,
                mode=self.completion_mode,
            )
        framework_state = completion_output.framework_state
        mastery = torch.sigmoid(
            self.mastery_head(framework_state).squeeze(-1)
        )

        q_vectors = q_matrix.index_select(0, target_exercise_ids).to(
            dtype=framework_state.dtype
        )
        q_count = q_vectors.sum(dim=1, keepdim=True).clamp_min(1.0)
        target_states = framework_state.index_select(0, target_state_rows)
        q_state = (
            target_states * q_vectors.unsqueeze(-1)
        ).sum(dim=1) / q_count
        q_concept = q_vectors @ concept_nodes / q_count
        target_exercise = exercise_nodes.index_select(0, target_exercise_ids)
        q_repr = self.q_projection(
            torch.cat([q_concept, target_exercise], dim=-1)
        )
        match_features = torch.cat(
            [
                q_state,
                q_repr,
                q_state * q_repr,
                torch.abs(q_state - q_repr),
            ],
            dim=-1,
        )
        difficulty = self.exercise_difficulty(
            target_exercise_ids
        ).squeeze(-1)
        diagnosis_output: DiagnosisOutput | None = None
        if self.diagnosis_mode == "item_hypernetwork":
            diagnosis_output = self.item_conditioned_hyper_diagnosis(
                q_state=q_state,
                q_repr=q_repr,
                difficulty=difficulty,
            )
            cognitive_probs = diagnosis_output.cognitive_probs
            guess_probs = diagnosis_output.guess_probs
            slip_probs = diagnosis_output.slip_probs
        elif self.diagnosis_mode == "target_conditioned":
            cognitive_probs = torch.sigmoid(
                self.cognitive_match(match_features).squeeze(-1) - difficulty
            )
            state_condition = torch.cat([q_state, q_repr], dim=-1)
            guess_probs = self.max_guess * torch.sigmoid(
                self.guess_head(state_condition).squeeze(-1)
            )
            slip_probs = self.max_slip * torch.sigmoid(
                self.slip_head(state_condition).squeeze(-1)
            )
        else:
            target_mastery = (
                mastery.index_select(0, target_state_rows) * q_vectors
            ).sum(dim=1) / q_count.squeeze(-1)
            target_mastery = target_mastery.clamp(
                1.0e-5, 1.0 - 1.0e-5
            )
            item_only_features = torch.cat(
                [
                    q_repr,
                    q_repr,
                    q_repr * q_repr,
                    torch.zeros_like(q_repr),
                ],
                dim=-1,
            )
            item_condition = torch.cat([q_repr, q_repr], dim=-1)
            item_bias = self.control_cognitive_match(
                item_only_features
            ).squeeze(-1)
            discrimination = F.softplus(
                self.control_discrimination_raw
            )
            cognitive_probs = torch.sigmoid(
                discrimination * torch.logit(target_mastery)
                + item_bias
                - difficulty
            )
            guess_probs = self.max_guess * torch.sigmoid(
                self.control_guess_head(item_condition).squeeze(-1)
            )
            slip_probs = self.max_slip * torch.sigmoid(
                self.control_slip_head(item_condition).squeeze(-1)
            )
        if diagnosis_output is None:
            probs = (
                (1.0 - slip_probs) * cognitive_probs
                + guess_probs * (1.0 - cognitive_probs)
            )
        else:
            probs = diagnosis_output.probs
        return TwoStageForwardOutput(
            probs=probs,
            cognitive_probs=cognitive_probs,
            framework_state=framework_state,
            mastery=mastery,
            state_reliability=completion_output.reliability,
            student_state=refinement_output.student_evidence,
            concept_embeddings=concept_nodes,
            exercise_embeddings=exercise_nodes,
            guess_probs=guess_probs,
            slip_probs=slip_probs,
            difficulty=difficulty,
            module_diagnostics={
                **semantic_output.diagnostics,
                **evidence_output.diagnostics,
                **refinement_output.diagnostics,
                **prior_output.diagnostics,
                **completion_output.diagnostics,
            },
            architecture_fingerprint=self.architecture_fingerprint,
        )
