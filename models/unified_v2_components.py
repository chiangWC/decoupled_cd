from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.confidence_scale = nn.Parameter(torch.ones(3))
        self.confidence_bias = nn.Parameter(torch.zeros(3))

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
        logits = self.confidence_scale * log_odds + self.confidence_bias
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


class MonotonicDiagnosisDecoder(nn.Module):
    def __init__(
        self,
        *,
        num_exercises: int,
        num_concepts: int,
        dim: int,
    ) -> None:
        super().__init__()
        self.mastery_head = nn.Linear(dim, 1)
        self.concept_difficulty = nn.Embedding(num_concepts, 1)
        self.exercise_discrimination = nn.Embedding(num_exercises, 1)
        self.exercise_bias = nn.Embedding(num_exercises, 1)

    def decode_from_mastery(
        self,
        target_mastery: torch.Tensor,
        q_vectors: torch.Tensor,
        target_exercise_ids: torch.Tensor,
    ) -> torch.Tensor:
        weights = q_vectors / q_vectors.sum(dim=1, keepdim=True).clamp_min(1.0)
        difficulty = self.concept_difficulty.weight.squeeze(-1)
        margin = (target_mastery - difficulty.unsqueeze(0)) * weights
        discrimination = F.softplus(
            self.exercise_discrimination(target_exercise_ids)
        ).squeeze(-1)
        logits = discrimination * margin.sum(dim=1) + self.exercise_bias(
            target_exercise_ids
        ).squeeze(-1)
        return torch.sigmoid(logits)

    def forward(
        self,
        state_map: torch.Tensor,
        q_matrix: torch.Tensor,
        target_student_ids: torch.Tensor,
        target_exercise_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mastery_logits = self.mastery_head(state_map).squeeze(-1)
        mastery = torch.sigmoid(mastery_logits)
        target_mastery = mastery[target_student_ids]
        cognitive_probs = self.decode_from_mastery(
            target_mastery,
            q_matrix[target_exercise_ids],
            target_exercise_ids,
        )
        return cognitive_probs, cognitive_probs, mastery
