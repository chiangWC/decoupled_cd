from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RelationGraphBatch:
    positive_weight: torch.Tensor
    negative_weight: torch.Tensor
    reconstruction_mask: torch.Tensor
    target: torch.Tensor


@dataclass(frozen=True)
class GraphCompletionState:
    mastery: torch.Tensor
    reconstruction_mask: torch.Tensor
    target: torch.Tensor
    student_state: torch.Tensor
    concept_state: torch.Tensor
    full_target_count: int


class RelationMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.positive = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.negative = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        adjacency: tuple[torch.Tensor, torch.Tensor],
        source: torch.Tensor,
    ) -> torch.Tensor:
        positive, negative = adjacency
        degree = (
            (positive + negative).sum(dim=-1, keepdim=True).clamp_min(1.0)
        )
        message = (
            positive @ self.positive(source)
            + negative @ self.negative(source)
        ) / degree
        return self.norm(F.relu(message))


def _evidence_counts(evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if evidence.ndim != 3 or evidence.shape[-1] < 2:
        raise ValueError(
            "evidence must have shape [students, concepts, features>=2]"
        )
    attempts = evidence[..., 0]
    correct = evidence[..., 1]
    if not bool(torch.isfinite(attempts).all()):
        raise ValueError("attempts must be finite")
    if bool((attempts < 0).any()):
        raise ValueError("attempts must be nonnegative")
    if not bool(torch.isfinite(correct).all()):
        raise ValueError("correct must be finite")
    if bool((correct < 0).any()):
        raise ValueError("correct must be nonnegative")
    if bool((correct > attempts).any()):
        raise ValueError("correct cannot exceed attempts")

    float_dtype = (
        evidence.dtype
        if torch.is_floating_point(evidence)
        else torch.get_default_dtype()
    )
    return attempts.to(dtype=float_dtype), correct.to(dtype=float_dtype)


def _deterministic_mask(
    observed: torch.Tensor,
    epoch: int,
    fraction: float,
) -> torch.Tensor:
    flat_ids = torch.arange(
        observed.numel(), device=observed.device, dtype=torch.int64
    ).reshape_as(observed)
    hashed = (
        flat_ids * 1103515245 + (epoch + 42) * 12345
    ) & 0x7FFFFFFF
    rank = (
        hashed.masked_fill(~observed, torch.iinfo(torch.int64).max)
        .flatten()
        .argsort()
    )
    count = int(observed.sum().item() * fraction)
    result = torch.zeros_like(observed)
    if count:
        result.flatten()[rank[:count]] = True
    return result


def build_relation_graph(
    evidence: torch.Tensor,
    *,
    epoch: int | None,
    training: bool,
    mask_fraction: float = 0.2,
    reliability_cap: float = 20.0,
) -> RelationGraphBatch:
    if not 0.0 <= mask_fraction <= 1.0:
        raise ValueError("mask_fraction must be in [0, 1]")
    if not math.isfinite(reliability_cap) or reliability_cap <= 0.0:
        raise ValueError("reliability_cap must be finite and positive")
    if training and epoch is None:
        raise ValueError("epoch is required during training")
    if not training and epoch is not None:
        raise ValueError("epoch must be None during inference")

    attempts, correct = _evidence_counts(evidence)
    observed = attempts > 0
    target = (correct + 1.0) / (attempts + 2.0)
    reliability = attempts.clamp(max=reliability_cap) / reliability_cap
    positive_weight = reliability * observed * (target >= 0.5)
    negative_weight = reliability * observed * (target < 0.5)

    reconstruction_mask = (
        _deterministic_mask(observed, epoch, mask_fraction)
        if training
        else torch.zeros_like(observed)
    )
    positive_weight = positive_weight.masked_fill(reconstruction_mask, 0.0)
    negative_weight = negative_weight.masked_fill(reconstruction_mask, 0.0)
    return RelationGraphBatch(
        positive_weight=positive_weight,
        negative_weight=negative_weight,
        reconstruction_mask=reconstruction_mask,
        target=target,
    )


def node_summary_features(
    evidence: torch.Tensor,
    q_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    attempts, correct = _evidence_counts(evidence)
    if q_matrix.ndim != 2 or q_matrix.shape[1] != evidence.shape[1]:
        raise ValueError("q_matrix must have shape [items, concepts]")

    observed = attempts > 0
    student_attempts = attempts.sum(dim=1)
    concept_attempts = attempts.sum(dim=0)
    student_features = torch.stack(
        [
            observed.to(dtype=attempts.dtype).mean(dim=1),
            (correct.sum(dim=1) + 1.0) / (student_attempts + 2.0),
            torch.log1p(student_attempts),
        ],
        dim=-1,
    )
    concept_features = torch.stack(
        [
            observed.to(dtype=attempts.dtype).mean(dim=0),
            (correct.sum(dim=0) + 1.0) / (concept_attempts + 2.0),
            torch.log1p(concept_attempts),
            q_matrix.to(device=evidence.device, dtype=attempts.dtype).sum(dim=0),
        ],
        dim=-1,
    )
    return student_features, concept_features


class EvidenceRelationGraphCompleter(nn.Module):
    def __init__(
        self,
        num_students: int,
        num_concepts: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_students = num_students
        self.num_concepts = num_concepts
        self.student_encoder = nn.Linear(3, hidden_dim)
        self.concept_encoder = nn.Linear(4, hidden_dim)
        self.c2s = nn.ModuleList(
            RelationMessageLayer(hidden_dim) for _ in range(2)
        )
        self.s2c = nn.ModuleList(
            RelationMessageLayer(hidden_dim) for _ in range(2)
        )
        self.bilinear = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.student_bias = nn.Linear(3, 1, bias=False)
        self.concept_bias = nn.Linear(4, 1, bias=False)
        nn.init.xavier_uniform_(self.bilinear)

    def forward(
        self,
        evidence: torch.Tensor,
        q_matrix: torch.Tensor,
        student_ids: torch.Tensor | None = None,
        epoch: int | None = None,
        training: bool = False,
    ) -> GraphCompletionState:
        expected_shape = (self.num_students, self.num_concepts)
        if tuple(evidence.shape[:2]) != expected_shape:
            raise ValueError(
                "evidence shape does not match model configured for "
                f"{self.num_students} students and "
                f"{self.num_concepts} concepts"
            )
        graph = build_relation_graph(
            evidence,
            epoch=epoch,
            training=training,
        )
        summary_evidence = evidence.masked_fill(
            graph.reconstruction_mask.unsqueeze(-1),
            0,
        )
        student_features, concept_features = node_summary_features(
            summary_evidence,
            q_matrix,
        )
        students = self.student_encoder(student_features)
        concepts = self.concept_encoder(concept_features)
        for concept_to_student, student_to_concept in zip(
            self.c2s, self.s2c
        ):
            next_students = concept_to_student(
                (graph.positive_weight, graph.negative_weight),
                concepts,
            )
            next_concepts = student_to_concept(
                (
                    graph.positive_weight.T,
                    graph.negative_weight.T,
                ),
                students,
            )
            students, concepts = next_students, next_concepts

        logits = students @ self.bilinear @ concepts.T
        logits = (
            logits
            + self.student_bias(student_features)
            + self.concept_bias(concept_features).T
        )
        selected = slice(None) if student_ids is None else student_ids
        return GraphCompletionState(
            mastery=torch.sigmoid(logits[selected]),
            reconstruction_mask=graph.reconstruction_mask[selected],
            target=graph.target[selected],
            student_state=students[selected],
            concept_state=concepts,
            full_target_count=int(graph.reconstruction_mask.sum().item()),
        )
