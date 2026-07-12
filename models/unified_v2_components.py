from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ObservedMasteryState:
    mastery: torch.Tensor
    reliability: torch.Tensor
    observed_mask: torch.Tensor


def smoothed_evidence_logits(
    evidence: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    if evidence.ndim != 3 or evidence.shape[-1] != 2:
        raise ValueError(
            "evidence must have shape [students, concepts, 2]"
        )
    attempts, correct = evidence.unbind(dim=-1)
    rate = (correct + alpha) / (attempts + 2.0 * alpha)
    return torch.logit(rate.clamp(1e-6, 1.0 - 1e-6))


class ObservedMasteryEstimator(nn.Module):
    def __init__(
        self,
        num_students: int,
        num_concepts: int,
        initial_logits: torch.Tensor | None = None,
        evidence_cap: float = 20.0,
    ) -> None:
        super().__init__()
        if evidence_cap <= 0.0:
            raise ValueError("evidence_cap must be positive")
        shape = (num_students, num_concepts)
        values = (
            torch.zeros(shape)
            if initial_logits is None
            else initial_logits.detach().clone()
        )
        if tuple(values.shape) != shape:
            raise ValueError(f"initial_logits must have shape {shape}")
        self.logits = nn.Parameter(values)
        self.evidence_cap = float(evidence_cap)

    def forward(
        self,
        evidence: torch.Tensor,
        student_ids: torch.Tensor | None = None,
    ) -> ObservedMasteryState:
        selected = evidence if student_ids is None else evidence[student_ids]
        logits = self.logits if student_ids is None else self.logits[student_ids]
        attempts = selected[..., 0]
        return ObservedMasteryState(
            mastery=logits.sigmoid(),
            reliability=(
                attempts.clamp(max=self.evidence_cap) / self.evidence_cap
            ),
            observed_mask=attempts > 0,
        )


class GlobalConceptPriorCompleter(nn.Module):
    def __init__(
        self,
        num_concepts: int,
        initial_prior: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        prior = (
            torch.zeros(num_concepts)
            if initial_prior is None
            else torch.logit(
                initial_prior.detach().clone().clamp(1e-6, 1.0 - 1e-6)
            )
        )
        self.logits = nn.Parameter(prior)

    def forward(self, num_students: int) -> torch.Tensor:
        return self.logits.sigmoid().unsqueeze(0).expand(num_students, -1)


class LowRankMasteryCompleter(nn.Module):
    def __init__(
        self,
        num_students: int,
        num_concepts: int,
        rank: int,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.student_factors = nn.Parameter(torch.empty(num_students, rank))
        self.concept_factors = nn.Parameter(torch.empty(num_concepts, rank))
        self.student_bias = nn.Parameter(torch.zeros(num_students, 1))
        self.concept_bias = nn.Parameter(torch.zeros(1, num_concepts))
        nn.init.normal_(self.student_factors, std=rank ** -0.5)
        nn.init.normal_(self.concept_factors, std=rank ** -0.5)

    def forward(
        self,
        student_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        users = (
            self.student_factors
            if student_ids is None
            else self.student_factors[student_ids]
        )
        bias = (
            self.student_bias
            if student_ids is None
            else self.student_bias[student_ids]
        )
        return (
            users @ self.concept_factors.T + bias + self.concept_bias
        ).sigmoid()


def assemble_mastery(
    observed: torch.Tensor,
    missing: torch.Tensor,
    observed_mask: torch.Tensor,
) -> torch.Tensor:
    if observed.shape != missing.shape or observed.shape != observed_mask.shape:
        raise ValueError("observed, missing, and observed_mask shapes must match")
    return torch.where(observed_mask, observed, missing)


@dataclass(frozen=True)
class TestedKnowledgeState:
    tkc_states: torch.Tensor
    direct_reliability: torch.Tensor


@dataclass(frozen=True)
class InferredKnowledgeState:
    ukc_states: torch.Tensor
    inferred_reliability: torch.Tensor
    reachable_mask: torch.Tensor


class TestedKnowledgeEvidenceEncoder(nn.Module):
    def __init__(self, *, dim: int, evidence_cap: float) -> None:
        super().__init__()
        self.evidence_cap = float(evidence_cap)
        self.encoder = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        q_matrix: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        student_tkc_mask: torch.Tensor,
    ) -> TestedKnowledgeState:
        correct = (student_exercise_mask * response_matrix) @ q_matrix
        incorrect = (student_exercise_mask * (1.0 - response_matrix)) @ q_matrix
        attempts = correct + incorrect
        features = torch.stack(
            [
                torch.log1p(correct),
                torch.log1p(incorrect),
                correct / attempts.clamp_min(1.0),
            ],
            dim=-1,
        )
        tkc_states = self.encoder(features) * student_tkc_mask.unsqueeze(-1)
        direct_reliability = (
            torch.log1p(attempts) / math.log1p(self.evidence_cap)
        ).clamp(0.0, 1.0)
        return TestedKnowledgeState(
            tkc_states=tkc_states,
            direct_reliability=direct_reliability,
        )


class UntestedKnowledgeInferenceNetwork(nn.Module):
    RELIABILITY_DECAY = 0.9

    def __init__(
        self,
        *,
        num_concepts: int,
        dim: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.concept_prior = nn.Parameter(torch.zeros(num_concepts, dim))
        self.layers = nn.ModuleList(
            nn.Linear(dim, dim, bias=False) for _ in range(layers)
        )

    def forward(
        self,
        tkc_states: torch.Tensor,
        tkc_mask: torch.Tensor,
        ukc_mask: torch.Tensor,
        concept_graph: torch.Tensor,
        direct_reliability: torch.Tensor,
    ) -> InferredKnowledgeState:
        topology = concept_graph.ne(0).to(tkc_states.dtype)
        topology = topology - torch.diag_embed(torch.diagonal(topology))
        incoming_topology = topology.transpose(0, 1)
        eligible_source = direct_reliability * tkc_mask
        source = incoming_topology * eligible_source.unsqueeze(1)
        first_transition = source / source.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)
        later_transition = incoming_topology / incoming_topology.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        hidden = tkc_states
        structurally_reachable = tkc_mask.bool() & direct_reliability.gt(0)
        propagated_reliability = eligible_source
        for index, layer in enumerate(self.layers):
            transition = (
                first_transition
                if index == 0
                else later_transition.unsqueeze(0)
            )
            propagated = torch.einsum(
                "skj,sjd->skd",
                transition,
                layer(hidden),
            )
            hidden = torch.where(
                tkc_mask.bool().unsqueeze(-1),
                tkc_states,
                torch.tanh(propagated),
            )
            propagated_reachability = torch.einsum(
                "kj,sj->sk",
                incoming_topology,
                structurally_reachable.to(tkc_states.dtype),
            ).gt(0)
            structurally_reachable = (
                structurally_reachable | propagated_reachability
            )
            propagated_reliability = torch.where(
                tkc_mask.bool(),
                eligible_source,
                self.RELIABILITY_DECAY
                * torch.einsum(
                    "kj,sj->sk",
                    later_transition,
                    propagated_reliability,
                ),
            )

        reachable = structurally_reachable & ukc_mask.bool()
        prior = self.concept_prior.unsqueeze(0).expand_as(hidden)
        ukc_states = torch.where(
            reachable.unsqueeze(-1),
            hidden,
            prior,
        ) * ukc_mask.unsqueeze(-1)
        inferred_reliability = torch.where(
            reachable,
            propagated_reliability,
            torch.zeros_like(propagated_reliability),
        )
        return InferredKnowledgeState(
            ukc_states=ukc_states,
            inferred_reliability=inferred_reliability,
            reachable_mask=reachable,
        )


class CoverageAwareStateComposer(nn.Module):
    def __init__(self, *, dim: int) -> None:
        super().__init__()
        del dim
        self.direct_log_slope = nn.Parameter(torch.zeros(()))
        self.direct_intercept = nn.Parameter(torch.zeros(()))
        self.inferred_log_slope = nn.Parameter(torch.zeros(()))
        self.inferred_intercept = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        tkc_states: torch.Tensor,
        ukc_states: torch.Tensor,
        concept_prior: torch.Tensor,
        tkc_mask: torch.Tensor,
        direct_reliability: torch.Tensor,
        inferred_reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        confidence = torch.stack(
            [
                direct_reliability,
                inferred_reliability,
                1.0
                - torch.maximum(direct_reliability, inferred_reliability),
            ],
            dim=-1,
        )
        log_odds = torch.logit(confidence.clamp(1e-6, 1.0 - 1e-6))
        direct_logit = (
            self.direct_log_slope.exp()
            * (log_odds[..., 0] - log_odds[..., 2])
            + self.direct_intercept
        )
        inferred_logit = (
            self.inferred_log_slope.exp()
            * (log_odds[..., 1] - log_odds[..., 2])
            + self.inferred_intercept
        )
        logits = torch.stack(
            [
                direct_logit,
                inferred_logit,
                torch.zeros_like(direct_logit),
            ],
            dim=-1,
        )
        tkc_valid = tkc_mask.bool()
        valid = torch.stack(
            [
                tkc_valid,
                ~tkc_valid & inferred_reliability.gt(0),
                torch.ones_like(tkc_valid),
            ],
            dim=-1,
        )
        weights = torch.softmax(
            logits.masked_fill(~valid, -1e9),
            dim=-1,
        )
        candidates = torch.stack(
            [
                tkc_states,
                ukc_states,
                concept_prior.unsqueeze(0).expand_as(tkc_states),
            ],
            dim=-2,
        )
        state_map = (weights.unsqueeze(-1) * candidates).sum(dim=-2)
        return state_map, weights


def _inverse_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError("effective positive weight must be greater than zero")
    return math.log(math.expm1(value))


@dataclass(frozen=True)
class BehaviorState:
    probs: torch.Tensor
    guess_probs: torch.Tensor
    slip_probs: torch.Tensor
    cognitive_weight: torch.Tensor


class ConditionalSimplexBehaviorModel(nn.Module):
    COGNITIVE_FLOOR = 2.0 ** -20

    def __init__(
        self,
        num_students: int,
        num_exercises: int,
        dim: int,
    ) -> None:
        super().__init__()
        self.student_embedding = nn.Embedding(num_students, dim)
        self.exercise_embedding = nn.Embedding(num_exercises, dim)
        self.out = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 3),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.constant_(self.out[-1].bias, -2.0)
        with torch.no_grad():
            self.out[-1].bias[2] = 2.0

    def forward(
        self,
        cognitive_probs: torch.Tensor,
        student_ids: torch.Tensor,
        exercise_ids: torch.Tensor,
    ) -> BehaviorState:
        features = torch.cat(
            (
                self.student_embedding(student_ids),
                self.exercise_embedding(exercise_ids),
            ),
            dim=-1,
        )
        logits = self.out(features)
        raw_cognitive = logits.softmax(dim=-1)[..., 2]
        cognitive_weight = self.COGNITIVE_FLOOR + (
            1.0 - self.COGNITIVE_FLOOR
        ) * raw_cognitive
        behavior_mass = 1.0 - cognitive_weight
        guess = behavior_mass * torch.sigmoid(logits[..., 0] - logits[..., 1])
        slip = behavior_mass - guess
        probs = guess + cognitive_weight * cognitive_probs
        return BehaviorState(
            probs=probs,
            guess_probs=guess,
            slip_probs=slip,
            cognitive_weight=cognitive_weight,
        )


class PositiveLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        effective_weight: float,
        bias: float,
        perturbation: float = 0.01,
    ) -> None:
        super().__init__()
        center = _inverse_softplus(effective_weight)
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.raw_weight, mean=center, std=perturbation)
        self.bias = nn.Parameter(torch.full((out_features,), bias))

    @property
    def effective_weight(self) -> torch.Tensor:
        return F.softplus(self.raw_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.effective_weight, self.bias)


class MonotonicDiagnosisDecoder(nn.Module):
    DISCRIMINATION_FLOOR = 1e-6

    def __init__(
        self,
        *,
        num_exercises: int,
        num_concepts: int,
        dim: int,
    ) -> None:
        super().__init__()
        del dim
        self.item_concept_difficulty = nn.Embedding(
            num_exercises, num_concepts
        )
        self.exercise_discrimination = nn.Embedding(num_exercises, 1)
        self.interaction_layers = nn.ModuleList(
            [
                PositiveLinear(
                    num_concepts,
                    512,
                    effective_weight=0.5,
                    bias=0.0,
                ),
                PositiveLinear(
                    512,
                    256,
                    effective_weight=1.0 / 512,
                    bias=-0.5,
                ),
                PositiveLinear(
                    256,
                    1,
                    effective_weight=1.0 / 256,
                    bias=-0.5,
                ),
            ]
        )
        nn.init.zeros_(self.item_concept_difficulty.weight)
        nn.init.constant_(
            self.exercise_discrimination.weight,
            _inverse_softplus(1.0 - self.DISCRIMINATION_FLOOR),
        )

    def decode_from_mastery(
        self,
        target_mastery: torch.Tensor,
        q_vectors: torch.Tensor,
        target_exercise_ids: torch.Tensor,
    ) -> torch.Tensor:
        beta = torch.sigmoid(
            self.item_concept_difficulty(target_exercise_ids)
        )
        discrimination = F.softplus(
            self.exercise_discrimination(target_exercise_ids)
        ) + self.DISCRIMINATION_FLOOR
        hidden = (
            q_vectors
            * (target_mastery - beta)
            * discrimination
        )
        hidden = torch.sigmoid(self.interaction_layers[0](hidden))
        hidden = torch.sigmoid(self.interaction_layers[1](hidden))
        logits = self.interaction_layers[2](hidden).squeeze(-1)
        return torch.sigmoid(logits)

    def forward(
        self,
        mastery: torch.Tensor,
        q_matrix: torch.Tensor,
        target_student_ids: torch.Tensor,
        target_exercise_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_mastery = mastery[target_student_ids]
        q_vectors = q_matrix[target_exercise_ids]
        cognitive_probs = self.decode_from_mastery(
            target_mastery,
            q_vectors,
            target_exercise_ids,
        )
        beta = torch.sigmoid(
            self.item_concept_difficulty(target_exercise_ids)
        )
        q_weights = q_vectors / q_vectors.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)
        difficulty = (q_weights * beta).sum(dim=1)
        return cognitive_probs, difficulty
