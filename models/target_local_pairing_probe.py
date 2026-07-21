from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn


PLACEMENTS: Final[tuple[str, ...]] = (
    "real_pair",
    "late_fusion",
    "perm_pair",
)
DEFAULT_RESPONSE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "group_accuracy",
    "residual",
    "confidence",
)
_INTEGER_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average a padded support set without letting padding affect the result."""
    if values.ndim != 3:
        raise ValueError("values must have shape [batch, support, dim].")
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.ndim != 2 or mask.shape != values.shape[:2]:
        raise ValueError("mask must have shape [batch, support].")

    valid = mask.to(dtype=torch.bool).unsqueeze(-1)
    zero = torch.zeros((), dtype=values.dtype, device=values.device)
    masked = torch.where(valid, values, zero)
    count = valid.sum(dim=1).clamp_min(1).to(dtype=values.dtype)
    return masked.sum(dim=1) / count


class SharedItemEncoder(nn.Module):
    """Encode support, true-target and donor items through one parameter path.

    Item ID zero is reserved for UNK/padding. Known items therefore use IDs
    1..num_items. Q is represented by the mean embedding of every active
    concept, and a zero-Q row contributes a zero concept vector. Numeric item
    descriptors are projected separately before the three views are fused.
    """

    def __init__(
        self,
        *,
        num_items: int,
        num_concepts: int,
        numeric_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        if min(num_items, num_concepts, numeric_dim, hidden_dim) < 1:
            raise ValueError(
                "num_items, num_concepts, numeric_dim and hidden_dim "
                "must be positive."
            )
        self.num_items = int(num_items)
        self.num_concepts = int(num_concepts)
        self.numeric_dim = int(numeric_dim)
        self.hidden_dim = int(hidden_dim)
        self.unk_item_id = 0

        self.item_id_embedding = nn.Embedding(
            self.num_items + 1,
            self.hidden_dim,
            padding_idx=self.unk_item_id,
        )
        self.concept_embedding = nn.Embedding(
            self.num_concepts,
            self.hidden_dim,
        )
        self.numeric_projection = nn.Sequential(
            nn.Linear(self.numeric_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.view_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(
        self,
        *,
        item_ids: torch.Tensor,
        q_multi_hot: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(
            item_ids=item_ids,
            q_multi_hot=q_multi_hot,
            numeric_features=numeric_features,
        )
        item_state = self.item_id_embedding(item_ids.to(dtype=torch.long))

        q_weights = q_multi_hot.to(
            dtype=self.concept_embedding.weight.dtype
        )
        q_count = q_weights.sum(dim=-1, keepdim=True)
        q_state = torch.matmul(q_weights, self.concept_embedding.weight)
        q_state = q_state / q_count.clamp_min(1.0)
        q_state = torch.where(
            q_count > 0,
            q_state,
            torch.zeros((), dtype=q_state.dtype, device=q_state.device),
        )

        numeric_state = self.numeric_projection(
            numeric_features.to(dtype=item_state.dtype)
        )
        return self.view_fusion(
            torch.cat([item_state, q_state, numeric_state], dim=-1)
        )

    def _validate_inputs(
        self,
        *,
        item_ids: torch.Tensor,
        q_multi_hot: torch.Tensor,
        numeric_features: torch.Tensor,
    ) -> None:
        if item_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("item_ids must use an integer dtype.")
        expected_q_shape = (*item_ids.shape, self.num_concepts)
        if tuple(q_multi_hot.shape) != expected_q_shape:
            raise ValueError(
                "q_multi_hot must have shape "
                "[*item_ids.shape, num_concepts]."
            )
        expected_numeric_shape = (*item_ids.shape, self.numeric_dim)
        if tuple(numeric_features.shape) != expected_numeric_shape:
            raise ValueError(
                "numeric_features must have shape "
                "[*item_ids.shape, numeric_dim]."
            )
        if bool(torch.any(item_ids < 0)) or bool(
            torch.any(item_ids > self.num_items)
        ):
            raise ValueError(
                "item_ids must be zero (UNK) or in 1..num_items."
            )


@dataclass(frozen=True)
class TargetLocalPairingOutput:
    """Observable states used by the probe's intervention and audit tests."""

    logits: torch.Tensor
    probs: torch.Tensor
    history_summary: torch.Tensor
    local_state: torch.Tensor
    target_state: torch.Tensor
    local_target_state: torch.Tensor
    opms_state: torch.Tensor


class TargetLocalPairingProbe(nn.Module):
    """Probe whether target/support pairing must happen before set aggregation.

    All placements register and execute the same trainable modules. They differ
    only in where the target state enters the local pairing computation:

    real_pair
        Pair every support interaction with the real target, then masked-mean.
    late_fusion
        Masked-mean the support interactions first, then pair with the target.
    perm_pair
        Pair every support interaction with a donor target, then masked-mean.

    The diagnosis path always receives the real target and the same external
    OPMS summary. Consequently a donor target can perturb only the local branch
    of perm_pair and cannot replace the real target in diagnosis.
    """

    def __init__(
        self,
        *,
        num_items: int,
        num_concepts: int,
        item_numeric_dim: int,
        opms_dim: int,
        response_feature_dim: int = 3,
        hidden_dim: int = 32,
        placement: str = "real_pair",
    ) -> None:
        super().__init__()
        if placement not in PLACEMENTS:
            raise ValueError(
                f"Unsupported placement {placement!r}; "
                f"expected one of {PLACEMENTS}."
            )
        if min(
            num_items,
            num_concepts,
            item_numeric_dim,
            opms_dim,
            response_feature_dim,
            hidden_dim,
        ) < 1:
            raise ValueError("All model dimensions must be positive.")

        self.placement = placement
        self.num_items = int(num_items)
        self.num_concepts = int(num_concepts)
        self.item_numeric_dim = int(item_numeric_dim)
        self.opms_dim = int(opms_dim)
        self.response_feature_dim = int(response_feature_dim)
        self.hidden_dim = int(hidden_dim)

        # This is the only item/target encoder in the probe. Support items, the
        # real target and a permuted donor target all traverse these parameters.
        self.item_encoder = SharedItemEncoder(
            num_items=self.num_items,
            num_concepts=self.num_concepts,
            numeric_dim=self.item_numeric_dim,
            hidden_dim=self.hidden_dim,
        )
        # E(d_i, a_ui, e_ui, c_ui): bind the three response descriptors to
        # the support item before any set aggregation in every placement.
        self.interaction_encoder = nn.Sequential(
            nn.Linear(
                self.hidden_dim + self.response_feature_dim,
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        # F(h_i, t_j), also used after aggregation by late_fusion.
        self.pair_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        # B(u, j) is prepared by the runner as a common Strong-OPMS summary.
        self.opms_encoder = nn.Sequential(
            nn.Linear(self.opms_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.diagnosis_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        *,
        support_item_ids: torch.Tensor,
        support_q_multi_hot: torch.Tensor,
        support_item_numeric: torch.Tensor,
        support_response_features: torch.Tensor,
        support_mask: torch.Tensor,
        target_item_ids: torch.Tensor,
        target_q_multi_hot: torch.Tensor,
        target_item_numeric: torch.Tensor,
        opms_summary: torch.Tensor,
        donor_target_item_ids: torch.Tensor | None = None,
        donor_target_q_multi_hot: torch.Tensor | None = None,
        donor_target_item_numeric: torch.Tensor | None = None,
    ) -> TargetLocalPairingOutput:
        self._validate_probe_inputs(
            support_item_ids=support_item_ids,
            support_q_multi_hot=support_q_multi_hot,
            support_item_numeric=support_item_numeric,
            support_response_features=support_response_features,
            support_mask=support_mask,
            target_item_ids=target_item_ids,
            target_q_multi_hot=target_q_multi_hot,
            target_item_numeric=target_item_numeric,
            opms_summary=opms_summary,
            donor_target_item_ids=donor_target_item_ids,
            donor_target_q_multi_hot=donor_target_q_multi_hot,
            donor_target_item_numeric=donor_target_item_numeric,
        )

        support_item_state = self.item_encoder(
            item_ids=support_item_ids,
            q_multi_hot=support_q_multi_hot,
            numeric_features=support_item_numeric,
        )
        encoded_support = self.interaction_encoder(
            torch.cat(
                [
                    support_item_state,
                    support_response_features.to(
                        dtype=support_item_state.dtype
                    ),
                ],
                dim=-1,
            )
        )
        history_summary = masked_mean(encoded_support, support_mask)

        # The real target is retained in the common diagnosis path.
        target_state = self.item_encoder(
            item_ids=target_item_ids,
            q_multi_hot=target_q_multi_hot,
            numeric_features=target_item_numeric,
        )
        if self.placement == "perm_pair":
            assert donor_target_item_ids is not None
            assert donor_target_q_multi_hot is not None
            assert donor_target_item_numeric is not None
            local_target_state = self.item_encoder(
                item_ids=donor_target_item_ids,
                q_multi_hot=donor_target_q_multi_hot,
                numeric_features=donor_target_item_numeric,
            )
        else:
            local_target_state = target_state

        if self.placement == "late_fusion":
            local_state = self.pair_encoder(
                torch.cat([history_summary, local_target_state], dim=-1)
            )
        else:
            expanded_target = local_target_state.unsqueeze(1).expand(
                -1,
                encoded_support.shape[1],
                -1,
            )
            paired_support = self.pair_encoder(
                torch.cat([encoded_support, expanded_target], dim=-1)
            )
            local_state = masked_mean(paired_support, support_mask)

        opms_state = self.opms_encoder(
            opms_summary.to(dtype=target_state.dtype)
        )
        diagnosis_input = torch.cat(
            [opms_state, local_state, target_state],
            dim=-1,
        )
        logits = self.diagnosis_head(diagnosis_input).squeeze(-1)
        return TargetLocalPairingOutput(
            logits=logits,
            probs=torch.sigmoid(logits),
            history_summary=history_summary,
            local_state=local_state,
            target_state=target_state,
            local_target_state=local_target_state,
            opms_state=opms_state,
        )

    def _validate_probe_inputs(
        self,
        *,
        support_item_ids: torch.Tensor,
        support_q_multi_hot: torch.Tensor,
        support_item_numeric: torch.Tensor,
        support_response_features: torch.Tensor,
        support_mask: torch.Tensor,
        target_item_ids: torch.Tensor,
        target_q_multi_hot: torch.Tensor,
        target_item_numeric: torch.Tensor,
        opms_summary: torch.Tensor,
        donor_target_item_ids: torch.Tensor | None,
        donor_target_q_multi_hot: torch.Tensor | None,
        donor_target_item_numeric: torch.Tensor | None,
    ) -> None:
        if support_item_ids.ndim != 2:
            raise ValueError(
                "support_item_ids must have shape [batch, support]."
            )
        batch, support = support_item_ids.shape
        if tuple(support_q_multi_hot.shape) != (
            batch,
            support,
            self.num_concepts,
        ):
            raise ValueError(
                "support_q_multi_hot must have shape "
                "[batch, support, num_concepts]."
            )
        if tuple(support_item_numeric.shape) != (
            batch,
            support,
            self.item_numeric_dim,
        ):
            raise ValueError(
                "support_item_numeric must have shape "
                "[batch, support, item_numeric_dim]."
            )
        if tuple(support_response_features.shape) != (
            batch,
            support,
            self.response_feature_dim,
        ):
            raise ValueError(
                "support_response_features must have shape "
                "[batch, support, response_feature_dim]."
            )
        mask_shape = tuple(support_mask.shape)
        if mask_shape not in {
            (batch, support),
            (batch, support, 1),
        }:
            raise ValueError(
                "support_mask must have shape [batch, support] or "
                "[batch, support, 1]."
            )
        if tuple(target_item_ids.shape) != (batch,):
            raise ValueError(
                "target_item_ids must have shape [batch]."
            )
        if tuple(target_q_multi_hot.shape) != (
            batch,
            self.num_concepts,
        ):
            raise ValueError(
                "target_q_multi_hot must have shape "
                "[batch, num_concepts]."
            )
        if tuple(target_item_numeric.shape) != (
            batch,
            self.item_numeric_dim,
        ):
            raise ValueError(
                "target_item_numeric must have shape "
                "[batch, item_numeric_dim]."
            )
        if tuple(opms_summary.shape) != (batch, self.opms_dim):
            raise ValueError(
                "opms_summary must have shape [batch, opms_dim]."
            )

        donor_values = (
            donor_target_item_ids,
            donor_target_q_multi_hot,
            donor_target_item_numeric,
        )
        any_donor = any(value is not None for value in donor_values)
        all_donor = all(value is not None for value in donor_values)
        if any_donor and not all_donor:
            raise ValueError(
                "All donor target tensors must be provided together."
            )
        if self.placement == "perm_pair" and not all_donor:
            raise ValueError(
                "perm_pair requires all donor target tensors."
            )
        if all_donor:
            assert donor_target_item_ids is not None
            assert donor_target_q_multi_hot is not None
            assert donor_target_item_numeric is not None
            if tuple(donor_target_item_ids.shape) != (batch,):
                raise ValueError(
                    "donor_target_item_ids must have shape [batch]."
                )
            if tuple(donor_target_q_multi_hot.shape) != (
                batch,
                self.num_concepts,
            ):
                raise ValueError(
                    "donor_target_q_multi_hot must have shape "
                    "[batch, num_concepts]."
                )
            if tuple(donor_target_item_numeric.shape) != (
                batch,
                self.item_numeric_dim,
            ):
                raise ValueError(
                    "donor_target_item_numeric must have shape "
                    "[batch, item_numeric_dim]."
                )
