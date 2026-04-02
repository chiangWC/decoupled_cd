from __future__ import annotations

import torch


def build_ukc_mask_from_tkc(student_tkc_mask: torch.Tensor) -> torch.Tensor:
    return (1.0 - student_tkc_mask).clamp(min=0.0, max=1.0)


def masked_mean(node_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=node_states.dtype).unsqueeze(-1)
    total = (node_states * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return total / denom
