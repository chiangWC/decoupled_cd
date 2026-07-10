from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MasteryAuxiliaryObjective(nn.Module):
    """Monotonic response BCE driven by target-concept mastery probabilities."""

    def __init__(
        self,
        *,
        aux_weight: float,
        detach_item_difficulty: bool = False,
        warmup_fraction: float = 0.0,
    ) -> None:
        super().__init__()
        if aux_weight < 0.0:
            raise ValueError("aux_weight must be non-negative.")
        if not 0.0 <= warmup_fraction <= 1.0:
            raise ValueError("warmup_fraction must be in [0, 1].")
        self.aux_weight = float(aux_weight)
        self.detach_item_difficulty = bool(detach_item_difficulty)
        self.warmup_fraction = float(warmup_fraction)
        self.register_buffer("scale", F.softplus(torch.tensor(2.0)))

    def effective_weight(self, epoch: int, total_epochs: int) -> float:
        if epoch < 1:
            raise ValueError("epoch must be at least 1.")
        if total_epochs < 1:
            raise ValueError("total_epochs must be at least 1.")
        if self.aux_weight == 0.0 or self.warmup_fraction == 0.0:
            return self.aux_weight
        warmup_span = float(total_epochs) * self.warmup_fraction
        progress = min(1.0, float(epoch) / warmup_span)
        return self.aux_weight * progress

    def logits(
        self,
        mastery_logits: torch.Tensor,
        difficulty_logits: torch.Tensor,
        q_mask: torch.Tensor,
    ) -> torch.Tensor:
        difficulty_source = (
            difficulty_logits.detach()
            if self.detach_item_difficulty
            else difficulty_logits
        )
        mastery_probability = torch.sigmoid(mastery_logits)
        difficulty_probability = torch.sigmoid(difficulty_source)
        q_weights = q_mask.to(dtype=mastery_probability.dtype)
        target_signal = q_weights * (mastery_probability - difficulty_probability)
        scale = self.scale.to(
            device=mastery_probability.device,
            dtype=mastery_probability.dtype,
        )
        return scale * target_signal.sum(dim=1) / q_weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        mastery_logits: torch.Tensor,
        difficulty_logits: torch.Tensor,
        q_mask: torch.Tensor,
        labels: torch.Tensor,
        *,
        epoch: int,
        total_epochs: int,
    ) -> torch.Tensor:
        weight = self.effective_weight(epoch, total_epochs)
        if weight == 0.0:
            return mastery_logits.detach().new_zeros(())
        auxiliary_logits = self.logits(mastery_logits, difficulty_logits, q_mask)
        labels = labels.to(dtype=auxiliary_logits.dtype)
        return weight * F.binary_cross_entropy_with_logits(auxiliary_logits, labels)
