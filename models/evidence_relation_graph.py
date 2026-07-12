from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RelationGraphBatch:
    positive_weight: torch.Tensor
    negative_weight: torch.Tensor
    reconstruction_mask: torch.Tensor
    target: torch.Tensor


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
