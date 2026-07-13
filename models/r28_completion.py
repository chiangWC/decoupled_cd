from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Final

import torch
from torch import nn
import torch.nn.functional as F


COMPLETER_MODES: Final[tuple[str, ...]] = (
    "relational",
    "direct_prior",
    "capacity_mlp",
    "exposure_dr",
    "partial_vae",
    "difficulty_set",
    "difficulty_capacity",
    "poe_ability",
    "poe_capacity",
    "hierarchical_bayes",
    "hierarchical_capacity",
    "cohort_conditioned",
    "cohort_capacity",
    "bipolar_prototype",
    "bipolar_capacity",
)
COMPLETION_OBJECTIVES: Final[tuple[str, ...]] = ("none", "masked_reconstruction")


@dataclass
class CompletionState:
    framework_state: torch.Tensor
    reliability: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    variational_kl: torch.Tensor | None = None


@dataclass
class R28ForwardOutput:
    probs: torch.Tensor
    cognitive_probs: torch.Tensor
    framework_state: torch.Tensor
    mastery: torch.Tensor
    state_reliability: torch.Tensor
    module_diagnostics: dict[str, torch.Tensor]
    architecture_fingerprint: str
    completion_reconstruction_loss: torch.Tensor | None
    student_state: torch.Tensor
    guess_probs: torch.Tensor
    slip_probs: torch.Tensor
    difficulty: torch.Tensor
    mastery_aux_logits: torch.Tensor | None = None


def _evidence_features(
    evidence: torch.Tensor,
    *,
    evidence_cap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return permutation-invariant train-history evidence features."""
    if evidence.ndim != 3 or evidence.size(-1) < 2:
        raise ValueError("student_concept_evidence must have shape [student, concept, >=2].")
    attempts = evidence[..., 0].to(dtype=torch.float32).clamp_min(0.0)
    correct = evidence[..., 1].to(dtype=torch.float32).clamp_min(0.0)
    correct = torch.minimum(correct, attempts)
    incorrect = attempts - correct
    seen = (attempts > 0).to(dtype=attempts.dtype)
    balance = (correct - incorrect) / attempts.clamp_min(1.0)
    confidence = (
        torch.log1p(attempts.clamp_max(float(evidence_cap)))
        / math.log1p(float(evidence_cap))
    ).clamp(0.0, 1.0)
    features = torch.stack([balance * confidence, confidence, seen], dim=-1)
    return features, attempts, correct, seen


def _global_summary_features(
    *,
    attempts: torch.Tensor,
    correct: torch.Tensor,
    seen: torch.Tensor,
    evidence_cap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    student_attempts = attempts.sum(dim=1)
    student_success = correct.sum(dim=1) / student_attempts.clamp_min(1.0)
    student_coverage = seen.mean(dim=1)
    student_confidence = (
        torch.log1p(student_attempts.clamp_max(float(evidence_cap) * attempts.size(1)))
        / math.log1p(float(evidence_cap) * attempts.size(1))
    ).clamp(0.0, 1.0)
    student_features = torch.stack(
        [2.0 * student_success - 1.0, student_coverage, student_confidence],
        dim=-1,
    )

    concept_attempts = attempts.sum(dim=0)
    concept_success = correct.sum(dim=0) / concept_attempts.clamp_min(1.0)
    concept_coverage = seen.mean(dim=0)
    concept_confidence = (
        torch.log1p(concept_attempts.clamp_max(float(evidence_cap) * attempts.size(0)))
        / math.log1p(float(evidence_cap) * attempts.size(0))
    ).clamp(0.0, 1.0)
    concept_features = torch.stack(
        [2.0 * concept_success - 1.0, concept_coverage, concept_confidence],
        dim=-1,
    )
    return student_features, concept_features


class DirectPriorCompleter(nn.Module):
    """Clean w/o-module control: raw observed evidence and a static concept prior."""

    def __init__(self, dim: int, evidence_cap: float) -> None:
        super().__init__()
        self.evidence_cap = float(evidence_cap)
        self.evidence_projection = nn.Linear(3, dim)
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        *,
        evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        student_indices: torch.Tensor,
    ) -> CompletionState:
        selected = evidence.index_select(0, student_indices)
        features, attempts, _, seen = _evidence_features(
            selected,
            evidence_cap=self.evidence_cap,
        )
        observed = concept_nodes.unsqueeze(0) + self.evidence_projection(features)
        prior = concept_nodes.unsqueeze(0).expand(selected.size(0), -1, -1)
        state = torch.where(seen.unsqueeze(-1).bool(), observed, prior)
        state = self.output_norm(state)
        confidence = features[..., 1] * seen
        return CompletionState(
            framework_state=state,
            reliability=confidence,
            diagnostics={
                "mean_reliability": confidence.mean().detach(),
                "observed_ratio": seen.mean().detach(),
                "mean_attempts": attempts.mean().detach(),
            },
        )


class EvidenceRelationalCompleter(nn.Module):
    """Student-concept relation completion or its parameter-matched MLP control.

    Both modes instantiate and consume the exact same trainable blocks. The only
    difference is whether student/concept contexts are constructed by relational
    aggregation over observed student-concept cells or from marginal summaries.
    """

    def __init__(
        self,
        *,
        dim: int,
        evidence_cap: float,
        relational: bool,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.evidence_cap = float(evidence_cap)
        self.relational = bool(relational)

        self.evidence_projection = nn.Linear(3, dim)
        self.positive_transform = nn.Linear(dim, dim, bias=False)
        self.negative_transform = nn.Linear(dim, dim, bias=False)
        self.student_update = nn.Sequential(
            nn.Linear(dim * 2 + 3, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
        )
        self.concept_update = nn.Sequential(
            nn.Linear(dim * 3 + 3, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
        )
        self.pair_decoder = nn.Sequential(
            nn.Linear(dim * 3 + 3, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        reliability_width = max(8, dim // 2)
        self.reliability_head = nn.Sequential(
            nn.Linear(6, reliability_width),
            nn.ReLU(),
            nn.Linear(reliability_width, 1),
        )

    def _relational_contexts(
        self,
        *,
        concept_nodes: torch.Tensor,
        attempts: torch.Tensor,
        correct: torch.Tensor,
        student_features: torch.Tensor,
        concept_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        incorrect = (attempts - correct).clamp_min(0.0)
        positive_weights = correct / correct.sum(dim=1, keepdim=True).clamp_min(1.0)
        negative_weights = incorrect / incorrect.sum(dim=1, keepdim=True).clamp_min(1.0)
        positive_student = positive_weights @ self.positive_transform(concept_nodes)
        negative_student = negative_weights @ self.negative_transform(concept_nodes)
        student_context = self.student_update(
            torch.cat([positive_student, negative_student, student_features], dim=-1)
        )

        positive_by_concept = correct / correct.sum(dim=0, keepdim=True).clamp_min(1.0)
        negative_by_concept = incorrect / incorrect.sum(dim=0, keepdim=True).clamp_min(1.0)
        positive_concept = positive_by_concept.transpose(0, 1) @ self.positive_transform(student_context)
        negative_concept = negative_by_concept.transpose(0, 1) @ self.negative_transform(student_context)
        concept_context = self.concept_update(
            torch.cat(
                [concept_nodes, positive_concept, negative_concept, concept_features],
                dim=-1,
            )
        )
        return student_context, concept_context

    def _capacity_contexts(
        self,
        *,
        concept_nodes: torch.Tensor,
        student_features: torch.Tensor,
        concept_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        student_base = self.evidence_projection(student_features)
        student_context = self.student_update(
            torch.cat(
                [
                    self.positive_transform(student_base),
                    self.negative_transform(student_base),
                    student_features,
                ],
                dim=-1,
            )
        )
        concept_base = concept_nodes + self.evidence_projection(concept_features)
        concept_context = self.concept_update(
            torch.cat(
                [
                    concept_nodes,
                    self.positive_transform(concept_base),
                    self.negative_transform(concept_base),
                    concept_features,
                ],
                dim=-1,
            )
        )
        return student_context, concept_context

    def forward(
        self,
        *,
        evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        student_indices: torch.Tensor,
    ) -> CompletionState:
        features, attempts, correct, seen = _evidence_features(
            evidence,
            evidence_cap=self.evidence_cap,
        )
        student_features, concept_features = _global_summary_features(
            attempts=attempts,
            correct=correct,
            seen=seen,
            evidence_cap=self.evidence_cap,
        )
        if self.relational:
            student_context, concept_context = self._relational_contexts(
                concept_nodes=concept_nodes,
                attempts=attempts,
                correct=correct,
                student_features=student_features,
                concept_features=concept_features,
            )
        else:
            student_context, concept_context = self._capacity_contexts(
                concept_nodes=concept_nodes,
                student_features=student_features,
                concept_features=concept_features,
            )

        local_features = features.index_select(0, student_indices)
        selected_seen = seen.index_select(0, student_indices)
        selected_student_features = student_features.index_select(0, student_indices)
        selected_student = student_context.index_select(0, student_indices)
        batch_size = int(student_indices.numel())
        num_concepts = int(concept_nodes.size(0))
        student_grid = selected_student[:, None, :].expand(-1, num_concepts, -1)
        concept_grid = concept_context[None, :, :].expand(batch_size, -1, -1)
        observed_state = concept_grid + self.evidence_projection(local_features)
        completed_state = self.pair_decoder(
            torch.cat(
                [student_grid, concept_grid, student_grid * concept_grid, local_features],
                dim=-1,
            )
        )

        reliability_inputs = torch.cat(
            [
                local_features,
                selected_student_features[:, None, 1:].expand(-1, num_concepts, -1),
                concept_features[None, :, 1:2].expand(batch_size, -1, -1),
            ],
            dim=-1,
        )
        reliability = torch.sigmoid(self.reliability_head(reliability_inputs)).squeeze(-1)
        observed_reliability = reliability * selected_seen
        state = (
            observed_reliability.unsqueeze(-1) * observed_state
            + (1.0 - observed_reliability).unsqueeze(-1) * completed_state
        )
        return CompletionState(
            framework_state=state,
            reliability=reliability,
            diagnostics={
                "mean_reliability": reliability.mean().detach(),
                "observed_reliability": (
                    (reliability * selected_seen).sum() / selected_seen.sum().clamp_min(1.0)
                ).detach(),
                "missing_reliability": (
                    (reliability * (1.0 - selected_seen)).sum()
                    / (1.0 - selected_seen).sum().clamp_min(1.0)
                ).detach(),
                "observed_ratio": selected_seen.mean().detach(),
                "state_variance": state.var(unbiased=False).detach(),
            },
        )


class PartialVAECompleter(nn.Module):
    """Graph-free amortized completion from a variable-size evidence set.

    This is a formula-level CD adaptation of p-VAE: each observed
    student-concept cell is encoded independently, the factors are aggregated
    by a permutation-invariant sum, and a shared Gaussian student posterior is
    decoded with Q-derived concept features into a complete state tensor.
    """

    def __init__(self, *, dim: int, evidence_cap: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.evidence_cap = float(evidence_cap)
        self.observation_encoder = nn.Sequential(
            nn.Linear(dim + 3, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.set_norm = nn.LayerNorm(dim)
        self.posterior = nn.Sequential(
            nn.Linear(dim + 3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(
        self,
        *,
        evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        student_indices: torch.Tensor,
    ) -> CompletionState:
        features, attempts, correct, seen = _evidence_features(
            evidence,
            evidence_cap=self.evidence_cap,
        )
        student_features, _ = _global_summary_features(
            attempts=attempts,
            correct=correct,
            seen=seen,
            evidence_cap=self.evidence_cap,
        )
        selected_features = features.index_select(0, student_indices)
        selected_seen = seen.index_select(0, student_indices)
        selected_student_features = student_features.index_select(0, student_indices)
        batch_size = int(student_indices.numel())
        num_concepts = int(concept_nodes.size(0))

        concept_grid = concept_nodes.unsqueeze(0).expand(batch_size, -1, -1)
        factors = self.observation_encoder(
            torch.cat([concept_grid, selected_features], dim=-1)
        )
        factor_sum = (factors * selected_seen.unsqueeze(-1)).sum(dim=1)
        # Sum is the p-VAE set operator.  Scaling by sqrt(|O|) keeps its range
        # stable across the very different CD coverage regimes without losing
        # the observed-set cardinality signal included below.
        observed_count = selected_seen.sum(dim=1, keepdim=True).clamp_min(1.0)
        set_code = self.set_norm(factor_sum / observed_count.sqrt())
        posterior_stats = self.posterior(
            torch.cat([set_code, selected_student_features], dim=-1)
        )
        posterior_mean, posterior_logvar = posterior_stats.chunk(2, dim=-1)
        posterior_logvar = posterior_logvar.clamp(min=-8.0, max=6.0)
        if self.training:
            noise = torch.randn_like(posterior_mean)
            latent = posterior_mean + noise * torch.exp(0.5 * posterior_logvar)
        else:
            latent = posterior_mean

        latent_grid = latent[:, None, :].expand(-1, num_concepts, -1)
        state = self.decoder(
            torch.cat(
                [latent_grid, concept_grid, latent_grid * concept_grid],
                dim=-1,
            )
        )
        kl = -0.5 * (
            1.0 + posterior_logvar - posterior_mean.square() - posterior_logvar.exp()
        ).sum(dim=-1).mean()
        posterior_confidence = torch.exp(-0.5 * posterior_logvar).mean(dim=-1)
        posterior_confidence = posterior_confidence / (1.0 + posterior_confidence)
        evidence_confidence = selected_features[..., 1]
        reliability = (
            0.5 * posterior_confidence.unsqueeze(1)
            + 0.5 * evidence_confidence * selected_seen
        ).clamp(0.0, 1.0)
        return CompletionState(
            framework_state=state,
            reliability=reliability,
            diagnostics={
                "mean_reliability": reliability.mean().detach(),
                "posterior_variance": posterior_logvar.exp().mean().detach(),
                "posterior_kl": kl.detach(),
                "observed_ratio": selected_seen.mean().detach(),
                "state_variance": state.var(unbiased=False).detach(),
            },
            variational_kl=kl,
        )


class DifficultyCalibratedResponseCompleter(nn.Module):
    """Complete concept states from item-difficulty-calibrated response sets.

    The full path preserves each train-only exercise/response pairing.  The
    capacity control instantiates the same blocks but collapses those pairings
    to student marginals before encoding, isolating the item-calibration and
    response-set mechanism rather than generic parameter count.
    """

    def __init__(self, *, dim: int, evidence_cap: float, calibrated: bool) -> None:
        super().__init__()
        self.dim = int(dim)
        self.evidence_cap = float(evidence_cap)
        self.calibrated = bool(calibrated)
        self.response_encoder = nn.Sequential(
            nn.Linear(dim + 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.set_norm = nn.LayerNorm(dim)
        self.state_decoder = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        reliability_width = max(8, dim // 2)
        self.reliability_head = nn.Sequential(
            nn.Linear(4, reliability_width),
            nn.ReLU(),
            nn.Linear(reliability_width, 1),
        )

    def forward(
        self,
        *,
        evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        student_indices: torch.Tensor,
        response_matrix: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        q_matrix: torch.Tensor,
        exercise_nodes: torch.Tensor,
        exercise_evidence: torch.Tensor,
    ) -> CompletionState:
        del evidence
        selected_mask = student_exercise_mask.index_select(0, student_indices).to(
            dtype=exercise_nodes.dtype
        )
        selected_response = response_matrix.index_select(0, student_indices).to(
            dtype=exercise_nodes.dtype
        )
        q_binary = (q_matrix > 0).to(dtype=exercise_nodes.dtype)
        item_attempts = exercise_evidence[:, 0].to(dtype=exercise_nodes.dtype)
        item_accuracy = exercise_evidence[:, 3].to(dtype=exercise_nodes.dtype)
        item_confidence = (
            torch.log1p(item_attempts.clamp_max(self.evidence_cap * 100.0))
            / math.log1p(self.evidence_cap * 100.0)
        ).clamp(0.0, 1.0)
        observed_count = selected_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        student_success = (selected_response * selected_mask).sum(dim=1, keepdim=True) / observed_count
        observed_item_accuracy = (selected_mask * item_accuracy.unsqueeze(0)).sum(
            dim=1, keepdim=True
        ) / observed_count
        student_confidence = (
            torch.log1p(observed_count)
            / math.log1p(float(response_matrix.size(1)) + 1.0)
        ).clamp(0.0, 1.0)

        batch_size = int(student_indices.numel())
        num_exercises = int(exercise_nodes.size(0))
        exercise_grid = exercise_nodes.unsqueeze(0).expand(batch_size, -1, -1)
        if self.calibrated:
            correctness = 2.0 * selected_response - 1.0
            calibrated_residual = selected_response - item_accuracy.unsqueeze(0)
            token_features = torch.stack(
                [
                    correctness,
                    calibrated_residual,
                    item_confidence.unsqueeze(0).expand(batch_size, -1),
                    selected_mask,
                ],
                dim=-1,
            )
            response_tokens = self.response_encoder(
                torch.cat([exercise_grid, token_features], dim=-1)
            ) * selected_mask.unsqueeze(-1)
            student_context = self.set_norm(
                response_tokens.sum(dim=1) / observed_count.sqrt()
            )
            concept_counts = selected_mask @ q_binary
            concept_evidence = torch.einsum("bed,ek->bkd", response_tokens, q_binary)
            concept_evidence = concept_evidence / concept_counts.clamp_min(1.0).unsqueeze(-1)
        else:
            pooled_exercise = selected_mask @ exercise_nodes / observed_count
            marginal_features = torch.cat(
                [
                    2.0 * student_success - 1.0,
                    student_success - observed_item_accuracy,
                    student_confidence,
                    torch.ones_like(student_success),
                ],
                dim=-1,
            )
            student_context = self.set_norm(
                self.response_encoder(torch.cat([pooled_exercise, marginal_features], dim=-1))
            )
            prior_features = torch.stack(
                [
                    torch.zeros(num_exercises, device=exercise_nodes.device, dtype=exercise_nodes.dtype),
                    torch.zeros(num_exercises, device=exercise_nodes.device, dtype=exercise_nodes.dtype),
                    item_confidence,
                    torch.ones_like(item_confidence),
                ],
                dim=-1,
            )
            prior_tokens = self.response_encoder(
                torch.cat([exercise_nodes, prior_features], dim=-1)
            )
            concept_item_count = q_binary.sum(dim=0).clamp_min(1.0)
            concept_prior = q_binary.transpose(0, 1) @ prior_tokens
            concept_prior = concept_prior / concept_item_count.unsqueeze(-1)
            concept_evidence = concept_prior.unsqueeze(0).expand(batch_size, -1, -1)
            concept_counts = selected_mask @ q_binary

        num_concepts = int(concept_nodes.size(0))
        student_grid = student_context[:, None, :].expand(-1, num_concepts, -1)
        concept_grid = concept_nodes.unsqueeze(0).expand(batch_size, -1, -1)
        state = self.state_decoder(
            torch.cat(
                [
                    student_grid,
                    concept_grid,
                    student_grid * concept_grid,
                    concept_evidence,
                ],
                dim=-1,
            )
        )
        concept_confidence = (
            torch.log1p(concept_counts.clamp_max(self.evidence_cap))
            / math.log1p(self.evidence_cap)
        ).clamp(0.0, 1.0)
        reliability_features = torch.stack(
            [
                (concept_counts > 0).to(dtype=state.dtype),
                concept_confidence,
                student_success.expand(-1, num_concepts),
                student_confidence.expand(-1, num_concepts),
            ],
            dim=-1,
        )
        reliability = torch.sigmoid(self.reliability_head(reliability_features)).squeeze(-1)
        return CompletionState(
            framework_state=state,
            reliability=reliability,
            diagnostics={
                "mean_reliability": reliability.mean().detach(),
                "observed_ratio": (concept_counts > 0).to(state.dtype).mean().detach(),
                "mean_item_calibrated_residual": (
                    ((selected_response - item_accuracy.unsqueeze(0)) * selected_mask).sum()
                    / selected_mask.sum().clamp_min(1.0)
                ).detach(),
                "state_variance": state.var(unbiased=False).detach(),
            },
        )


class BipolarPrototypeCompleter(nn.Module):
    """Complete states from separate correct and incorrect response prototypes.

    Unlike the rejected item-token set encoder, this component uses two fixed
    sufficient-statistic centroids to constrain the student evidence path.  The
    capacity control duplicates the response-agnostic exercise centroid into
    both channels while retaining the exact same trainable blocks.
    """

    def __init__(self, *, dim: int, evidence_cap: float, bipolar: bool) -> None:
        super().__init__()
        self.dim = int(dim)
        self.evidence_cap = float(evidence_cap)
        self.bipolar = bool(bipolar)
        self.response_encoder = nn.Sequential(
            nn.Linear(dim * 2 + 4, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.set_norm = nn.LayerNorm(dim)
        self.prior_projection = nn.Linear(dim + 2, dim)
        self.state_decoder = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        reliability_width = max(8, dim // 2)
        self.reliability_head = nn.Sequential(
            nn.Linear(4, reliability_width),
            nn.ReLU(),
            nn.Linear(reliability_width, 1),
        )

    def forward(
        self,
        *,
        evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        student_indices: torch.Tensor,
        response_matrix: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        q_matrix: torch.Tensor,
        exercise_nodes: torch.Tensor,
        exercise_evidence: torch.Tensor,
    ) -> CompletionState:
        selected_mask = student_exercise_mask.index_select(0, student_indices).to(
            dtype=exercise_nodes.dtype
        )
        selected_response = response_matrix.index_select(0, student_indices).to(
            dtype=exercise_nodes.dtype
        )
        observed_count = selected_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled_exercise = selected_mask @ exercise_nodes / observed_count
        student_success = (selected_response * selected_mask).sum(
            dim=1, keepdim=True
        ) / observed_count
        student_confidence = (
            torch.log1p(observed_count)
            / math.log1p(float(response_matrix.size(1)) + 1.0)
        ).clamp(0.0, 1.0)

        item_attempts = exercise_evidence[:, 0].to(dtype=exercise_nodes.dtype)
        item_accuracy = exercise_evidence[:, 3].to(dtype=exercise_nodes.dtype)
        item_confidence = (
            torch.log1p(item_attempts.clamp_max(self.evidence_cap * 100.0))
            / math.log1p(self.evidence_cap * 100.0)
        ).clamp(0.0, 1.0)
        observed_item_accuracy = (selected_mask * item_accuracy.unsqueeze(0)).sum(
            dim=1, keepdim=True
        ) / observed_count

        if self.bipolar:
            positive_mask = selected_mask * selected_response
            negative_mask = selected_mask * (1.0 - selected_response)
            positive_count = positive_mask.sum(dim=1, keepdim=True)
            negative_count = negative_mask.sum(dim=1, keepdim=True)
            positive_centroid = positive_mask @ exercise_nodes / positive_count.clamp_min(1.0)
            negative_centroid = negative_mask @ exercise_nodes / negative_count.clamp_min(1.0)
            positive_centroid = torch.where(
                (positive_count > 0), positive_centroid, pooled_exercise
            )
            negative_centroid = torch.where(
                (negative_count > 0), negative_centroid, pooled_exercise
            )
            positive_difficulty = (
                positive_mask * item_accuracy.unsqueeze(0)
            ).sum(dim=1, keepdim=True) / positive_count.clamp_min(1.0)
            negative_difficulty = (
                negative_mask * item_accuracy.unsqueeze(0)
            ).sum(dim=1, keepdim=True) / negative_count.clamp_min(1.0)
            positive_difficulty = torch.where(
                positive_count > 0, positive_difficulty, observed_item_accuracy
            )
            negative_difficulty = torch.where(
                negative_count > 0, negative_difficulty, observed_item_accuracy
            )
            evidence_features = torch.cat(
                [
                    2.0 * student_success - 1.0,
                    positive_difficulty - negative_difficulty,
                    (
                        torch.log1p(positive_count)
                        / math.log1p(float(response_matrix.size(1)) + 1.0)
                    ).clamp(0.0, 1.0),
                    (
                        torch.log1p(negative_count)
                        / math.log1p(float(response_matrix.size(1)) + 1.0)
                    ).clamp(0.0, 1.0),
                ],
                dim=-1,
            )
        else:
            positive_centroid = pooled_exercise
            negative_centroid = pooled_exercise
            evidence_features = torch.cat(
                [
                    2.0 * student_success - 1.0,
                    student_success - observed_item_accuracy,
                    student_confidence,
                    torch.ones_like(student_success),
                ],
                dim=-1,
            )
        student_context = self.set_norm(
            self.response_encoder(
                torch.cat(
                    [positive_centroid, negative_centroid, evidence_features], dim=-1
                )
            )
        )

        q_binary = (q_matrix > 0).to(dtype=exercise_nodes.dtype)
        prior_tokens = self.prior_projection(
            torch.cat(
                [
                    exercise_nodes,
                    item_confidence.unsqueeze(-1),
                    torch.ones_like(item_confidence).unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        concept_item_count = q_binary.sum(dim=0).clamp_min(1.0)
        concept_prior = q_binary.transpose(0, 1) @ prior_tokens
        concept_prior = concept_prior / concept_item_count.unsqueeze(-1)

        batch_size = int(student_indices.numel())
        num_concepts = int(concept_nodes.size(0))
        student_grid = student_context[:, None, :].expand(-1, num_concepts, -1)
        concept_grid = concept_nodes.unsqueeze(0).expand(batch_size, -1, -1)
        state = self.state_decoder(
            torch.cat(
                [
                    student_grid,
                    concept_grid,
                    student_grid * concept_grid,
                    concept_prior.unsqueeze(0).expand(batch_size, -1, -1),
                ],
                dim=-1,
            )
        )
        selected_evidence = evidence.index_select(0, student_indices).to(dtype=state.dtype)
        concept_attempts = selected_evidence[..., 0].clamp_min(0.0)
        concept_confidence = (
            torch.log1p(concept_attempts.clamp_max(self.evidence_cap))
            / math.log1p(self.evidence_cap)
        ).clamp(0.0, 1.0)
        reliability_features = torch.stack(
            [
                (concept_attempts > 0).to(dtype=state.dtype),
                concept_confidence,
                student_success.expand(-1, num_concepts),
                student_confidence.expand(-1, num_concepts),
            ],
            dim=-1,
        )
        reliability = torch.sigmoid(self.reliability_head(reliability_features)).squeeze(-1)
        return CompletionState(
            framework_state=state,
            reliability=reliability,
            diagnostics={
                "mean_reliability": reliability.mean().detach(),
                "observed_ratio": (concept_attempts > 0).to(state.dtype).mean().detach(),
                "student_success": student_success.mean().detach(),
                "prototype_separation": (
                    positive_centroid - negative_centroid
                ).norm(dim=-1).mean().detach(),
                "state_variance": state.var(unbiased=False).detach(),
            },
        )


class HierarchicalBayesianCompleter(nn.Module):
    """Complete concept states with a train-history empirical-Bayes posterior.

    The full path combines a student's global response posterior, the
    population difficulty of each concept, and the student's local
    student-concept counts.  The capacity control instantiates the exact same
    trainable blocks but uses the deterministic marginal-summary/static-prior
    path of the strongest r28 performance anchor.  This isolates posterior
    completion from parameter count and generic decoder capacity.
    """

    def __init__(self, *, dim: int, evidence_cap: float, bayesian: bool) -> None:
        super().__init__()
        self.dim = int(dim)
        self.evidence_cap = float(evidence_cap)
        self.bayesian = bool(bayesian)
        self.response_encoder = nn.Sequential(
            nn.Linear(dim + 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.set_norm = nn.LayerNorm(dim)
        self.state_decoder = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        reliability_width = max(8, dim // 2)
        self.reliability_head = nn.Sequential(
            nn.Linear(4, reliability_width),
            nn.ReLU(),
            nn.Linear(reliability_width, 1),
        )

    @staticmethod
    def _safe_logit(probability: torch.Tensor) -> torch.Tensor:
        probability = probability.clamp(1e-4, 1.0 - 1e-4)
        return torch.log(probability) - torch.log1p(-probability)

    def forward(
        self,
        *,
        evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        student_indices: torch.Tensor,
        response_matrix: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        q_matrix: torch.Tensor,
        exercise_nodes: torch.Tensor,
        exercise_evidence: torch.Tensor,
    ) -> CompletionState:
        selected_mask = student_exercise_mask.index_select(0, student_indices).to(
            dtype=exercise_nodes.dtype
        )
        selected_response = response_matrix.index_select(0, student_indices).to(
            dtype=exercise_nodes.dtype
        )
        item_attempts = exercise_evidence[:, 0].to(dtype=exercise_nodes.dtype)
        item_accuracy = exercise_evidence[:, 3].to(dtype=exercise_nodes.dtype)
        item_confidence = (
            torch.log1p(item_attempts.clamp_max(self.evidence_cap * 100.0))
            / math.log1p(self.evidence_cap * 100.0)
        ).clamp(0.0, 1.0)
        observed_count = selected_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        student_correct = (selected_response * selected_mask).sum(dim=1, keepdim=True)
        student_success = student_correct / observed_count
        observed_item_accuracy = (selected_mask * item_accuracy.unsqueeze(0)).sum(
            dim=1, keepdim=True
        ) / observed_count
        student_confidence = (
            torch.log1p(observed_count)
            / math.log1p(float(response_matrix.size(1)) + 1.0)
        ).clamp(0.0, 1.0)

        pooled_exercise = selected_mask @ exercise_nodes / observed_count
        marginal_features = torch.cat(
            [
                2.0 * student_success - 1.0,
                student_success - observed_item_accuracy,
                student_confidence,
                torch.ones_like(student_success),
            ],
            dim=-1,
        )
        student_context = self.set_norm(
            self.response_encoder(torch.cat([pooled_exercise, marginal_features], dim=-1))
        )

        q_binary = (q_matrix > 0).to(dtype=exercise_nodes.dtype)
        batch_size = int(student_indices.numel())
        num_concepts = int(concept_nodes.size(0))
        if self.bayesian:
            selected_evidence = evidence.index_select(0, student_indices).to(
                dtype=exercise_nodes.dtype
            )
            local_attempts = selected_evidence[..., 0].clamp_min(0.0)
            local_correct = torch.minimum(
                selected_evidence[..., 1].clamp_min(0.0), local_attempts
            )
            local_incorrect = local_attempts - local_correct

            population_attempts = evidence[..., 0].to(dtype=exercise_nodes.dtype).sum(dim=0)
            population_correct = evidence[..., 1].to(dtype=exercise_nodes.dtype).sum(dim=0)
            concept_mean = (population_correct + 1.0) / (population_attempts + 2.0)
            global_mean = (population_correct.sum() + 1.0) / (
                population_attempts.sum() + 2.0
            )
            smoothed_student_mean = (student_correct + 1.0) / (observed_count + 2.0)
            prior_logit = (
                self._safe_logit(smoothed_student_mean)
                + self._safe_logit(concept_mean).unsqueeze(0)
                - self._safe_logit(global_mean)
            )
            prior_mean = torch.sigmoid(prior_logit)
            prior_strength = 2.0
            posterior_alpha = prior_strength * prior_mean + local_correct
            posterior_beta = prior_strength * (1.0 - prior_mean) + local_incorrect
            posterior_total = posterior_alpha + posterior_beta
            posterior_mean = posterior_alpha / posterior_total.clamp_min(1e-6)
            posterior_variance = (
                posterior_alpha
                * posterior_beta
                / (
                    posterior_total.square()
                    * (posterior_total + 1.0)
                ).clamp_min(1e-6)
            )
            posterior_confidence = (
                1.0 - torch.sqrt((posterior_variance / (1.0 / 12.0)).clamp_min(0.0))
            ).clamp(0.0, 1.0)
            local_seen = (local_attempts > 0).to(dtype=exercise_nodes.dtype)
            posterior_features = torch.stack(
                [
                    2.0 * posterior_mean - 1.0,
                    posterior_mean - prior_mean,
                    posterior_confidence,
                    local_seen,
                ],
                dim=-1,
            )
            concept_grid = concept_nodes.unsqueeze(0).expand(batch_size, -1, -1)
            concept_evidence = self.response_encoder(
                torch.cat([concept_grid, posterior_features], dim=-1)
            )
            concept_counts = local_attempts
        else:
            num_exercises = int(exercise_nodes.size(0))
            prior_features = torch.stack(
                [
                    torch.zeros(
                        num_exercises,
                        device=exercise_nodes.device,
                        dtype=exercise_nodes.dtype,
                    ),
                    torch.zeros(
                        num_exercises,
                        device=exercise_nodes.device,
                        dtype=exercise_nodes.dtype,
                    ),
                    item_confidence,
                    torch.ones_like(item_confidence),
                ],
                dim=-1,
            )
            prior_tokens = self.response_encoder(
                torch.cat([exercise_nodes, prior_features], dim=-1)
            )
            concept_item_count = q_binary.sum(dim=0).clamp_min(1.0)
            concept_prior = q_binary.transpose(0, 1) @ prior_tokens
            concept_prior = concept_prior / concept_item_count.unsqueeze(-1)
            concept_evidence = concept_prior.unsqueeze(0).expand(batch_size, -1, -1)
            concept_counts = selected_mask @ q_binary
            posterior_mean = student_success.expand(-1, num_concepts)
            posterior_confidence = student_confidence.expand(-1, num_concepts)
            prior_mean = posterior_mean

        student_grid = student_context[:, None, :].expand(-1, num_concepts, -1)
        concept_grid = concept_nodes.unsqueeze(0).expand(batch_size, -1, -1)
        state = self.state_decoder(
            torch.cat(
                [student_grid, concept_grid, student_grid * concept_grid, concept_evidence],
                dim=-1,
            )
        )
        concept_confidence = (
            torch.log1p(concept_counts.clamp_max(self.evidence_cap))
            / math.log1p(self.evidence_cap)
        ).clamp(0.0, 1.0)
        reliability_features = torch.stack(
            [
                (concept_counts > 0).to(dtype=state.dtype),
                concept_confidence,
                posterior_confidence,
                student_confidence.expand(-1, num_concepts),
            ],
            dim=-1,
        )
        reliability = torch.sigmoid(self.reliability_head(reliability_features)).squeeze(-1)
        return CompletionState(
            framework_state=state,
            reliability=reliability,
            diagnostics={
                "mean_reliability": reliability.mean().detach(),
                "observed_ratio": (concept_counts > 0).to(state.dtype).mean().detach(),
                "posterior_mean": posterior_mean.mean().detach(),
                "prior_mean": prior_mean.mean().detach(),
                "posterior_confidence": posterior_confidence.mean().detach(),
                "state_variance": state.var(unbiased=False).detach(),
            },
        )


class CohortConditionedCompleter(nn.Module):
    """Complete missing concept states from train-only ability cohorts.

    Students are assigned softly to fixed ability levels using only their
    aggregate training responses.  Each level contributes a non-parametric
    concept response profile, which becomes the prior for missing concepts and
    is updated by local evidence when present.  The capacity control keeps the
    exact same blocks but collapses the levels to a population concept profile.
    """

    def __init__(
        self,
        *,
        dim: int,
        evidence_cap: float,
        conditioned: bool,
        num_levels: int = 5,
    ) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError("num_levels must be at least 2.")
        self.dim = int(dim)
        self.evidence_cap = float(evidence_cap)
        self.conditioned = bool(conditioned)
        self.num_levels = int(num_levels)
        self.response_encoder = nn.Sequential(
            nn.Linear(dim + 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        self.set_norm = nn.LayerNorm(dim)
        self.state_decoder = nn.Sequential(
            nn.Linear(dim * 5, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        reliability_width = max(8, dim // 2)
        self.reliability_head = nn.Sequential(
            nn.Linear(4, reliability_width),
            nn.ReLU(),
            nn.Linear(reliability_width, 1),
        )

    def _cohort_profiles(
        self,
        *,
        evidence: torch.Tensor,
        student_success: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centers = torch.linspace(
            0.1,
            0.9,
            self.num_levels,
            device=evidence.device,
            dtype=evidence.dtype,
        )
        # The fixed bandwidth covers adjacent levels without introducing a
        # dataset-tuned threshold or trainable routing gate.
        squared_distance = (student_success - centers.unsqueeze(0)).square()
        membership = torch.softmax(-squared_distance / 0.04, dim=-1)
        attempts = evidence[..., 0].clamp_min(0.0)
        correct = torch.minimum(evidence[..., 1].clamp_min(0.0), attempts)
        population_attempts = attempts.sum(dim=0)
        population_correct = correct.sum(dim=0)
        population_mean = (population_correct + 1.0) / (population_attempts + 2.0)
        cohort_attempts = membership.transpose(0, 1) @ attempts
        cohort_correct = membership.transpose(0, 1) @ correct
        cohort_mean = (
            cohort_correct + 2.0 * population_mean.unsqueeze(0)
        ) / (cohort_attempts + 2.0)
        return membership, cohort_mean, population_mean

    def forward(
        self,
        *,
        evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        student_indices: torch.Tensor,
        response_matrix: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        q_matrix: torch.Tensor,
        exercise_nodes: torch.Tensor,
        exercise_evidence: torch.Tensor,
    ) -> CompletionState:
        selected_mask = student_exercise_mask.index_select(0, student_indices).to(
            dtype=exercise_nodes.dtype
        )
        selected_response = response_matrix.index_select(0, student_indices).to(
            dtype=exercise_nodes.dtype
        )
        all_mask = student_exercise_mask.to(dtype=exercise_nodes.dtype)
        all_response = response_matrix.to(dtype=exercise_nodes.dtype)
        all_count = all_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        all_success = (all_response * all_mask).sum(dim=1, keepdim=True) / all_count

        item_attempts = exercise_evidence[:, 0].to(dtype=exercise_nodes.dtype)
        item_accuracy = exercise_evidence[:, 3].to(dtype=exercise_nodes.dtype)
        item_confidence = (
            torch.log1p(item_attempts.clamp_max(self.evidence_cap * 100.0))
            / math.log1p(self.evidence_cap * 100.0)
        ).clamp(0.0, 1.0)
        observed_count = selected_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        student_success = (selected_response * selected_mask).sum(
            dim=1, keepdim=True
        ) / observed_count
        observed_item_accuracy = (selected_mask * item_accuracy.unsqueeze(0)).sum(
            dim=1, keepdim=True
        ) / observed_count
        student_confidence = (
            torch.log1p(observed_count)
            / math.log1p(float(response_matrix.size(1)) + 1.0)
        ).clamp(0.0, 1.0)

        pooled_exercise = selected_mask @ exercise_nodes / observed_count
        marginal_features = torch.cat(
            [
                2.0 * student_success - 1.0,
                student_success - observed_item_accuracy,
                student_confidence,
                torch.ones_like(student_success),
            ],
            dim=-1,
        )
        student_context = self.set_norm(
            self.response_encoder(torch.cat([pooled_exercise, marginal_features], dim=-1))
        )

        q_binary = (q_matrix > 0).to(dtype=exercise_nodes.dtype)
        prior_features = torch.stack(
            [
                torch.zeros_like(item_confidence),
                torch.zeros_like(item_confidence),
                item_confidence,
                torch.ones_like(item_confidence),
            ],
            dim=-1,
        )
        prior_tokens = self.response_encoder(
            torch.cat([exercise_nodes, prior_features], dim=-1)
        )
        concept_item_count = q_binary.sum(dim=0).clamp_min(1.0)
        static_concept_prior = q_binary.transpose(0, 1) @ prior_tokens
        static_concept_prior = static_concept_prior / concept_item_count.unsqueeze(-1)

        evidence_float = evidence.to(dtype=exercise_nodes.dtype)
        membership, cohort_mean, population_mean = self._cohort_profiles(
            evidence=evidence_float,
            student_success=all_success,
        )
        selected_evidence = evidence_float.index_select(0, student_indices)
        local_attempts = selected_evidence[..., 0].clamp_min(0.0)
        local_correct = torch.minimum(selected_evidence[..., 1].clamp_min(0.0), local_attempts)
        local_seen = (local_attempts > 0).to(dtype=exercise_nodes.dtype)
        selected_membership = membership.index_select(0, student_indices)
        if self.conditioned:
            cohort_prior = selected_membership @ cohort_mean
            cohort_certainty = selected_membership.max(dim=-1, keepdim=True).values
            posterior_mean = (
                2.0 * cohort_prior + local_correct
            ) / (2.0 + local_attempts)
            posterior_confidence = (
                0.5 * cohort_certainty.expand_as(posterior_mean)
                + 0.5
                * (
                    torch.log1p(local_attempts.clamp_max(self.evidence_cap))
                    / math.log1p(self.evidence_cap)
                ).clamp(0.0, 1.0)
            ).clamp(0.0, 1.0)
        else:
            cohort_prior = population_mean.unsqueeze(0).expand_as(local_attempts)
            posterior_mean = cohort_prior
            population_attempts = evidence_float[..., 0].sum(dim=0)
            population_confidence = (
                torch.log1p(
                    population_attempts.clamp_max(
                        self.evidence_cap * evidence_float.size(0)
                    )
                )
                / math.log1p(self.evidence_cap * evidence_float.size(0))
            ).clamp(0.0, 1.0)
            posterior_confidence = population_confidence.unsqueeze(0).expand_as(
                posterior_mean
            )

        cohort_features = torch.stack(
            [
                2.0 * posterior_mean - 1.0,
                posterior_mean - population_mean.unsqueeze(0),
                posterior_confidence,
                local_seen if self.conditioned else torch.zeros_like(local_seen),
            ],
            dim=-1,
        )
        batch_size = int(student_indices.numel())
        num_concepts = int(concept_nodes.size(0))
        concept_grid = concept_nodes.unsqueeze(0).expand(batch_size, -1, -1)
        cohort_evidence = self.response_encoder(
            torch.cat([concept_grid, cohort_features], dim=-1)
        )
        student_grid = student_context[:, None, :].expand(-1, num_concepts, -1)
        state = self.state_decoder(
            torch.cat(
                [
                    student_grid,
                    concept_grid,
                    student_grid * concept_grid,
                    static_concept_prior.unsqueeze(0).expand(batch_size, -1, -1),
                    cohort_evidence,
                ],
                dim=-1,
            )
        )
        local_confidence = (
            torch.log1p(local_attempts.clamp_max(self.evidence_cap))
            / math.log1p(self.evidence_cap)
        ).clamp(0.0, 1.0)
        reliability_features = torch.stack(
            [
                local_seen,
                local_confidence,
                posterior_confidence,
                student_confidence.expand(-1, num_concepts),
            ],
            dim=-1,
        )
        reliability = torch.sigmoid(self.reliability_head(reliability_features)).squeeze(-1)
        return CompletionState(
            framework_state=state,
            reliability=reliability,
            diagnostics={
                "mean_reliability": reliability.mean().detach(),
                "observed_ratio": local_seen.mean().detach(),
                "cohort_posterior_mean": posterior_mean.mean().detach(),
                "cohort_prior_shift": (
                    cohort_prior - population_mean.unsqueeze(0)
                ).abs().mean().detach(),
                "membership_certainty": selected_membership.max(dim=-1).values.mean().detach(),
                "state_variance": state.var(unbiased=False).detach(),
            },
        )


class ProductOfExpertsAbilityCompleter(nn.Module):
    """Amortized ability posterior from a product of observed item experts.

    Missing exercises contribute only the unit Gaussian prior.  The capacity
    control uses the same networks and produces one posterior factor from
    student marginals, removing the item-conditioned product mechanism.
    """

    def __init__(self, *, dim: int, evidence_cap: float, product_experts: bool) -> None:
        super().__init__()
        self.dim = int(dim)
        self.evidence_cap = float(evidence_cap)
        self.product_experts = bool(product_experts)
        self.expert_encoder = nn.Sequential(
            nn.Linear(dim + 3, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim * 2),
        )
        self.state_decoder = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        reliability_width = max(8, dim // 2)
        self.reliability_head = nn.Sequential(
            nn.Linear(4, reliability_width),
            nn.ReLU(),
            nn.Linear(reliability_width, 1),
        )

    def forward(
        self,
        *,
        evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        student_indices: torch.Tensor,
        response_matrix: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        q_matrix: torch.Tensor,
        exercise_nodes: torch.Tensor,
        exercise_evidence: torch.Tensor,
    ) -> CompletionState:
        del evidence
        selected_mask = student_exercise_mask.index_select(0, student_indices).to(
            dtype=exercise_nodes.dtype
        )
        selected_response = response_matrix.index_select(0, student_indices).to(
            dtype=exercise_nodes.dtype
        )
        item_attempts = exercise_evidence[:, 0].to(dtype=exercise_nodes.dtype)
        item_accuracy = exercise_evidence[:, 3].to(dtype=exercise_nodes.dtype)
        item_confidence = (
            torch.log1p(item_attempts.clamp_max(self.evidence_cap * 100.0))
            / math.log1p(self.evidence_cap * 100.0)
        ).clamp(0.0, 1.0)
        observed_count = selected_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        student_success = (selected_response * selected_mask).sum(dim=1, keepdim=True) / observed_count
        observed_item_accuracy = (selected_mask * item_accuracy.unsqueeze(0)).sum(
            dim=1, keepdim=True
        ) / observed_count
        student_confidence = (
            torch.log1p(observed_count)
            / math.log1p(float(response_matrix.size(1)) + 1.0)
        ).clamp(0.0, 1.0)

        if self.product_experts:
            batch_size = int(student_indices.numel())
            exercise_grid = exercise_nodes.unsqueeze(0).expand(batch_size, -1, -1)
            expert_features = torch.stack(
                [
                    2.0 * selected_response - 1.0,
                    item_accuracy.unsqueeze(0).expand(batch_size, -1),
                    item_confidence.unsqueeze(0).expand(batch_size, -1),
                ],
                dim=-1,
            )
            expert_mean, raw_precision = self.expert_encoder(
                torch.cat([exercise_grid, expert_features], dim=-1)
            ).chunk(2, dim=-1)
            expert_precision = F.softplus(raw_precision).clamp_max(20.0)
            mask = selected_mask.unsqueeze(-1)
            posterior_precision = 1.0 + (expert_precision * mask).sum(dim=1)
            posterior_mean = (
                expert_mean * expert_precision * mask
            ).sum(dim=1) / posterior_precision
        else:
            pooled_exercise = selected_mask @ exercise_nodes / observed_count
            marginal_features = torch.cat(
                [
                    2.0 * student_success - 1.0,
                    observed_item_accuracy,
                    student_confidence,
                ],
                dim=-1,
            )
            expert_mean, raw_precision = self.expert_encoder(
                torch.cat([pooled_exercise, marginal_features], dim=-1)
            ).chunk(2, dim=-1)
            posterior_precision = 1.0 + F.softplus(raw_precision).clamp_max(20.0) * observed_count.sqrt()
            posterior_mean = (
                expert_mean * (posterior_precision - 1.0) / posterior_precision
            )

        posterior_logvar = -torch.log(posterior_precision.clamp_min(1e-6))
        if self.training:
            ability = posterior_mean + torch.randn_like(posterior_mean) * torch.exp(
                0.5 * posterior_logvar
            )
        else:
            ability = posterior_mean
        num_concepts = int(concept_nodes.size(0))
        ability_grid = ability[:, None, :].expand(-1, num_concepts, -1)
        concept_grid = concept_nodes.unsqueeze(0).expand(ability.size(0), -1, -1)
        state = self.state_decoder(
            torch.cat(
                [ability_grid, concept_grid, ability_grid * concept_grid],
                dim=-1,
            )
        )

        q_binary = (q_matrix > 0).to(dtype=state.dtype)
        concept_counts = selected_mask @ q_binary
        concept_confidence = (
            torch.log1p(concept_counts.clamp_max(self.evidence_cap))
            / math.log1p(self.evidence_cap)
        ).clamp(0.0, 1.0)
        posterior_confidence = (
            posterior_precision / (1.0 + posterior_precision)
        ).mean(dim=-1, keepdim=True)
        reliability_features = torch.stack(
            [
                (concept_counts > 0).to(dtype=state.dtype),
                concept_confidence,
                posterior_confidence.expand(-1, num_concepts),
                student_confidence.expand(-1, num_concepts),
            ],
            dim=-1,
        )
        reliability = torch.sigmoid(self.reliability_head(reliability_features)).squeeze(-1)
        kl = -0.5 * (
            1.0 + posterior_logvar - posterior_mean.square() - posterior_logvar.exp()
        ).sum(dim=-1).mean()
        return CompletionState(
            framework_state=state,
            reliability=reliability,
            diagnostics={
                "mean_reliability": reliability.mean().detach(),
                "posterior_variance": posterior_logvar.exp().mean().detach(),
                "posterior_kl": kl.detach(),
                "observed_ratio": (concept_counts > 0).to(state.dtype).mean().detach(),
                "state_variance": state.var(unbiased=False).detach(),
            },
            variational_kl=kl,
        )


class R28CompletionCDM(nn.Module):
    """One replaceable state-completion component plus a fixed pooled NCF diagnosis."""

    def __init__(
        self,
        *,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        concept_dim: int = 64,
        state_completer: str = "relational",
        completion_objective: str = "none",
        completion_mask_frac: float = 0.15,
        evidence_cap: float = 20.0,
        readout_dropout: float = 0.0,
        max_guess: float = 0.3,
        max_slip: float = 0.3,
        partial_vae_kl_weight: float = 0.001,
    ) -> None:
        super().__init__()
        if state_completer not in COMPLETER_MODES:
            raise ValueError(f"Unsupported state completer: {state_completer}")
        if completion_objective not in COMPLETION_OBJECTIVES:
            raise ValueError(f"Unsupported completion objective: {completion_objective}")
        if state_completer == "exposure_dr":
            raise NotImplementedError(
                f"{state_completer} is a gated follow-up replacement and is not activated "
                "until the relational candidate is rejected by validation evidence."
            )
        if concept_dim < 4:
            raise ValueError("concept_dim must be at least 4.")
        if not 0.0 < completion_mask_frac < 1.0:
            raise ValueError("completion_mask_frac must be in (0, 1).")
        if evidence_cap <= 0.0:
            raise ValueError("evidence_cap must be positive.")
        if not 0.0 <= readout_dropout < 1.0:
            raise ValueError("readout_dropout must be in [0, 1).")
        if not 0.0 <= max_guess < 1.0 or not 0.0 <= max_slip < 1.0:
            raise ValueError("max_guess and max_slip must be in [0, 1).")
        if partial_vae_kl_weight < 0.0:
            raise ValueError("partial_vae_kl_weight must be non-negative.")
        if (
            state_completer in {
                "difficulty_set",
                "difficulty_capacity",
                "hierarchical_bayes",
                "hierarchical_capacity",
                "cohort_conditioned",
                "cohort_capacity",
                "bipolar_prototype",
                "bipolar_capacity",
            }
            and completion_objective != "none"
        ):
            raise ValueError(
                "difficulty-calibrated completion is preregistered with response BCE only."
            )

        self.num_students = int(num_students)
        self.num_exercises = int(num_exercises)
        self.num_concepts = int(num_concepts)
        self.concept_dim = int(concept_dim)
        self.state_completer = state_completer
        self.completion_objective = completion_objective
        self.completion_mask_frac = float(completion_mask_frac)
        self.evidence_cap = float(evidence_cap)
        self.max_guess = float(max_guess)
        self.max_slip = float(max_slip)
        self.partial_vae_kl_weight = float(partial_vae_kl_weight)

        # Common diagnosis parameters are instantiated first so their seed-42
        # initialization is identical across all completer variants.
        self.concept_embedding = nn.Embedding(num_concepts, concept_dim)
        self.exercise_embedding = nn.Embedding(num_exercises, concept_dim)
        self.exercise_difficulty = nn.Embedding(num_exercises, 1)
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
        self.mastery_head = nn.Linear(concept_dim, 1)
        self.reconstruction_head = nn.Linear(concept_dim, 1)

        nn.init.xavier_uniform_(self.concept_embedding.weight)
        nn.init.xavier_uniform_(self.exercise_embedding.weight)
        nn.init.zeros_(self.exercise_difficulty.weight)

        if state_completer == "direct_prior":
            self.completer: nn.Module = DirectPriorCompleter(concept_dim, evidence_cap)
        elif state_completer == "partial_vae":
            self.completer = PartialVAECompleter(
                dim=concept_dim,
                evidence_cap=evidence_cap,
            )
        elif state_completer in {"difficulty_set", "difficulty_capacity"}:
            self.completer = DifficultyCalibratedResponseCompleter(
                dim=concept_dim,
                evidence_cap=evidence_cap,
                calibrated=state_completer == "difficulty_set",
            )
        elif state_completer in {"poe_ability", "poe_capacity"}:
            self.completer = ProductOfExpertsAbilityCompleter(
                dim=concept_dim,
                evidence_cap=evidence_cap,
                product_experts=state_completer == "poe_ability",
            )
        elif state_completer in {"hierarchical_bayes", "hierarchical_capacity"}:
            self.completer = HierarchicalBayesianCompleter(
                dim=concept_dim,
                evidence_cap=evidence_cap,
                bayesian=state_completer == "hierarchical_bayes",
            )
        elif state_completer in {"cohort_conditioned", "cohort_capacity"}:
            self.completer = CohortConditionedCompleter(
                dim=concept_dim,
                evidence_cap=evidence_cap,
                conditioned=state_completer == "cohort_conditioned",
            )
        elif state_completer in {"bipolar_prototype", "bipolar_capacity"}:
            self.completer = BipolarPrototypeCompleter(
                dim=concept_dim,
                evidence_cap=evidence_cap,
                bipolar=state_completer == "bipolar_prototype",
            )
        else:
            self.completer = EvidenceRelationalCompleter(
                dim=concept_dim,
                evidence_cap=evidence_cap,
                relational=state_completer == "relational",
            )

    @property
    def architecture_fingerprint(self) -> str:
        payload = {
            "family": "r28_completion_v1",
            "state_completer": self.state_completer,
            "diagnosis": "q_conditioned_pooled_ncf_state_only",
            "student_id_bypass": False,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def common_initialization_hash(self) -> str:
        digest = hashlib.sha256()
        common_prefixes = (
            "concept_embedding.",
            "exercise_embedding.",
            "exercise_difficulty.",
            "q_projection.",
            "cognitive_match.",
            "guess_head.",
            "slip_head.",
            "mastery_head.",
            "reconstruction_head.",
        )
        for name, tensor in sorted(self.state_dict().items()):
            if name.startswith(common_prefixes):
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def active_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _concept_nodes(self, q_matrix: torch.Tensor) -> torch.Tensor:
        q_binary = (q_matrix > 0).to(dtype=self.exercise_embedding.weight.dtype)
        concept_exercise_count = q_binary.sum(dim=0).clamp_min(1.0)
        exercise_side = q_binary.transpose(0, 1) @ self.exercise_embedding.weight
        exercise_side = exercise_side / concept_exercise_count.unsqueeze(-1)
        return self.concept_embedding.weight + exercise_side

    def _exercise_nodes(self, q_matrix: torch.Tensor) -> torch.Tensor:
        q_binary = (q_matrix > 0).to(dtype=self.concept_embedding.weight.dtype)
        concept_count = q_binary.sum(dim=1, keepdim=True).clamp_min(1.0)
        concept_side = q_binary @ self.concept_embedding.weight / concept_count
        return self.exercise_embedding.weight + concept_side

    def _masked_reconstruction_loss(
        self,
        *,
        evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        student_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        selected_attempts = evidence.index_select(0, student_indices)[..., 0]
        observed = selected_attempts > 0
        sampled = torch.rand_like(selected_attempts, dtype=torch.float32) < self.completion_mask_frac
        hidden = observed & sampled
        if not bool(hidden.any()):
            return None
        masked_evidence = evidence.clone()
        selected_masked = masked_evidence.index_select(0, student_indices)
        selected_masked[hidden] = 0
        masked_evidence.index_copy_(0, student_indices, selected_masked)
        reconstructed = self.completer(
            evidence=masked_evidence,
            concept_nodes=concept_nodes,
            student_indices=student_indices,
        ).framework_state
        logits = self.reconstruction_head(reconstructed).squeeze(-1)
        selected_original = evidence.index_select(0, student_indices)
        target = selected_original[..., 1] / selected_original[..., 0].clamp_min(1.0)
        return F.binary_cross_entropy_with_logits(logits[hidden], target[hidden])

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
    ) -> R28ForwardOutput:
        del concept_graph, prerequisite_graph, similarity_graph
        del student_tkc_mask, student_ukc_mask
        if student_concept_evidence is None:
            raise ValueError("r28_completion requires train-history student_concept_evidence.")
        if target_student_ids is None or target_exercise_ids is None:
            raise ValueError("target_student_ids and target_exercise_ids are required.")
        if target_student_ids.numel() != target_exercise_ids.numel():
            raise ValueError("target student and exercise tensors must have equal length.")

        if use_student_subset:
            student_indices, target_state_rows = torch.unique(
                target_student_ids,
                sorted=True,
                return_inverse=True,
            )
        else:
            student_indices = torch.arange(
                student_concept_evidence.size(0),
                device=student_concept_evidence.device,
            )
            target_state_rows = target_student_ids

        concept_nodes = self._concept_nodes(q_matrix)
        if self.state_completer in {
            "difficulty_set",
            "difficulty_capacity",
            "poe_ability",
            "poe_capacity",
            "hierarchical_bayes",
            "hierarchical_capacity",
            "cohort_conditioned",
            "cohort_capacity",
            "bipolar_prototype",
            "bipolar_capacity",
        }:
            if exercise_evidence is None:
                raise ValueError(
                    "difficulty-calibrated completion requires train-only exercise_evidence."
                )
            completion = self.completer(
                evidence=student_concept_evidence,
                concept_nodes=concept_nodes,
                student_indices=student_indices,
                response_matrix=response_matrix,
                student_exercise_mask=student_exercise_mask,
                q_matrix=q_matrix,
                exercise_nodes=self._exercise_nodes(q_matrix),
                exercise_evidence=exercise_evidence,
            )
        else:
            completion = self.completer(
                evidence=student_concept_evidence,
                concept_nodes=concept_nodes,
                student_indices=student_indices,
            )
        framework_state = completion.framework_state
        mastery = torch.sigmoid(self.mastery_head(framework_state).squeeze(-1))

        q_vectors = (q_matrix.index_select(0, target_exercise_ids) > 0).to(framework_state.dtype)
        q_count = q_vectors.sum(dim=1, keepdim=True).clamp_min(1.0)
        target_states = framework_state.index_select(0, target_state_rows)
        q_state = (target_states * q_vectors.unsqueeze(-1)).sum(dim=1) / q_count
        q_concept = q_vectors @ self.concept_embedding.weight / q_count
        target_exercise = self.exercise_embedding(target_exercise_ids)
        q_repr = self.q_projection(torch.cat([q_concept, target_exercise], dim=-1))
        match_features = torch.cat(
            [q_state, q_repr, q_state * q_repr, torch.abs(q_state - q_repr)],
            dim=-1,
        )
        difficulty = self.exercise_difficulty(target_exercise_ids).squeeze(-1)
        cognitive_logits = self.cognitive_match(match_features).squeeze(-1) - difficulty
        cognitive_probs = torch.sigmoid(cognitive_logits)

        state_condition = torch.cat([q_state, q_repr], dim=-1)
        guess_probs = self.max_guess * torch.sigmoid(self.guess_head(state_condition).squeeze(-1))
        slip_probs = self.max_slip * torch.sigmoid(self.slip_head(state_condition).squeeze(-1))
        probs = (1.0 - slip_probs) * cognitive_probs + guess_probs * (1.0 - cognitive_probs)

        reconstruction_loss = None
        if self.training and self.completion_objective == "masked_reconstruction":
            reconstruction_loss = self._masked_reconstruction_loss(
                evidence=student_concept_evidence,
                concept_nodes=concept_nodes,
                student_indices=student_indices,
            )
            if completion.variational_kl is not None:
                kl_term = self.partial_vae_kl_weight * completion.variational_kl
                reconstruction_loss = (
                    kl_term
                    if reconstruction_loss is None
                    else reconstruction_loss + kl_term
                )

        student_state = framework_state.mean(dim=1)
        return R28ForwardOutput(
            probs=probs,
            cognitive_probs=cognitive_probs,
            framework_state=framework_state,
            mastery=mastery,
            state_reliability=completion.reliability,
            module_diagnostics=completion.diagnostics,
            architecture_fingerprint=self.architecture_fingerprint,
            completion_reconstruction_loss=reconstruction_loss,
            student_state=student_state,
            guess_probs=guess_probs,
            slip_probs=slip_probs,
            difficulty=difficulty,
        )
