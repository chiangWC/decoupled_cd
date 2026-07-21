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
RESPONSE_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "response",
    "response_minus_item_ease",
    "group_attempt_confidence",
)
SUPPORT_STATISTIC_NAMES: Final[tuple[str, ...]] = (
    "theta_logit",
    "raw_accuracy",
    "log1p_support_rows",
    "log1p_unique_support_items",
    "log1p_correct_rows",
    "log1p_incorrect_rows",
)
DEFAULT_EMBEDDING_DIM: Final[int] = 16
DEFAULT_STATE_DIM: Final[int] = 32
DEFAULT_MLP_HIDDEN_DIM: Final[int] = 64
_INTEGER_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average a padded support set without padding leakage."""
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


def build_response_features(
    *,
    support_responses: torch.Tensor,
    support_item_ease: torch.Tensor,
    support_group_attempt_confidence: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Construct the frozen three-channel response descriptor."""
    if support_responses.shape != support_item_ease.shape:
        raise ValueError(
            "support_responses and support_item_ease must share shape."
        )
    if (
        support_group_attempt_confidence.shape
        != support_responses.shape
    ):
        raise ValueError(
            "support_group_attempt_confidence must match "
            "support_responses."
        )
    response = support_responses.to(dtype=dtype)
    item_ease = support_item_ease.to(dtype=dtype)
    confidence = support_group_attempt_confidence.to(dtype=dtype)
    return torch.stack(
        [
            response,
            response - item_ease,
            confidence,
        ],
        dim=-1,
    )


class SharedItemEncoder(nn.Module):
    """Use one ID/Q/numeric encoder for support, true and donor items.

    Item ID zero is reserved for UNK/padding and known IDs are 1..num_items.
    The Q view is the mean embedding of active concepts. A zero-Q row produces
    an exact zero concept vector before view fusion.
    """

    def __init__(
        self,
        *,
        num_items: int,
        num_concepts: int,
        numeric_dim: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        state_dim: int = DEFAULT_STATE_DIM,
        mlp_hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if min(
            num_items,
            num_concepts,
            numeric_dim,
            embedding_dim,
            state_dim,
            mlp_hidden_dim,
        ) < 1:
            raise ValueError("All SharedItemEncoder dimensions must be positive.")
        self.num_items = int(num_items)
        self.num_concepts = int(num_concepts)
        self.numeric_dim = int(numeric_dim)
        self.embedding_dim = int(embedding_dim)
        self.state_dim = int(state_dim)
        self.mlp_hidden_dim = int(mlp_hidden_dim)
        self.unk_item_id = 0

        self.item_id_embedding = nn.Embedding(
            self.num_items + 1,
            self.embedding_dim,
            padding_idx=self.unk_item_id,
        )
        self.concept_embedding = nn.Embedding(
            self.num_concepts,
            self.embedding_dim,
        )
        self.numeric_projection = nn.Sequential(
            nn.Linear(self.numeric_dim, self.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mlp_hidden_dim, self.embedding_dim),
        )
        self.view_fusion = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, self.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mlp_hidden_dim, self.state_dim),
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
class OutcomePartitionedOutput:
    opms_state: torch.Tensor
    history_summary: torch.Tensor
    support_item_state: torch.Tensor
    response_features: torch.Tensor
    correct_pool: torch.Tensor
    incorrect_pool: torch.Tensor
    contrast: torch.Tensor
    correct_mass: torch.Tensor
    incorrect_mass: torch.Tensor


class OutcomePartitionedState(nn.Module):
    """Build the shared learned Strong-OPMS state from raw support evidence."""

    def __init__(
        self,
        *,
        state_dim: int = DEFAULT_STATE_DIM,
        mlp_hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        if state_dim < 1 or mlp_hidden_dim < 1:
            raise ValueError(
                "state_dim and mlp_hidden_dim must be positive."
            )
        self.state_dim = int(state_dim)
        self.mlp_hidden_dim = int(mlp_hidden_dim)
        self.response_feature_dim = len(RESPONSE_FEATURE_NAMES)
        self.support_statistic_dim = len(SUPPORT_STATISTIC_NAMES)

        partition_input_dim = (
            self.state_dim * 3
            + 2
            + self.support_statistic_dim
        )
        self.partition_encoder = nn.Sequential(
            nn.Linear(partition_input_dim, self.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mlp_hidden_dim, self.state_dim),
        )

    def forward(
        self,
        *,
        support_item_state: torch.Tensor,
        support_responses: torch.Tensor,
        support_item_ease: torch.Tensor,
        support_group_attempt_confidence: torch.Tensor,
        support_statistics: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> OutcomePartitionedOutput:
        if support_item_state.ndim != 3:
            raise ValueError(
                "support_item_state must have shape "
                "[batch, support, state_dim]."
            )
        batch, support, state_dim = support_item_state.shape
        if state_dim != self.state_dim:
            raise ValueError(
                f"Expected state dim {self.state_dim}, got {state_dim}."
            )
        expected_support_shape = (batch, support)
        for name, value in (
            ("support_responses", support_responses),
            ("support_item_ease", support_item_ease),
            (
                "support_group_attempt_confidence",
                support_group_attempt_confidence,
            ),
        ):
            if tuple(value.shape) != expected_support_shape:
                raise ValueError(
                    f"{name} must have shape [batch, support]."
                )
        if tuple(support_statistics.shape) != (
            batch,
            self.support_statistic_dim,
        ):
            raise ValueError(
                "support_statistics must have shape "
                "[batch, 6] in SUPPORT_STATISTIC_NAMES order."
            )
        if support_mask.ndim == 3 and support_mask.shape[-1] == 1:
            support_mask = support_mask.squeeze(-1)
        if tuple(support_mask.shape) != expected_support_shape:
            raise ValueError(
                "support_mask must have shape [batch, support]."
            )
        valid_mask = support_mask.to(dtype=torch.bool)
        valid_responses = support_responses.masked_select(valid_mask)
        if bool(
            torch.any(
                (valid_responses != 0)
                & (valid_responses != 1)
            )
        ):
            raise ValueError(
                "Valid support_responses must be exactly binary."
            )

        response_features = build_response_features(
            support_responses=support_responses,
            support_item_ease=support_item_ease,
            support_group_attempt_confidence=(
                support_group_attempt_confidence
            ),
            dtype=support_item_state.dtype,
        )
        correct_mask = valid_mask & (support_responses == 1)
        incorrect_mask = valid_mask & (support_responses == 0)
        correct_pool = masked_mean(support_item_state, correct_mask)
        incorrect_pool = masked_mean(support_item_state, incorrect_mask)
        history_summary = masked_mean(support_item_state, valid_mask)
        contrast = correct_pool - incorrect_pool

        valid_count = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        mass_dtype = support_item_state.dtype
        correct_mass = (
            correct_mask.sum(dim=1, keepdim=True).to(dtype=mass_dtype)
            / valid_count.to(dtype=mass_dtype)
        )
        incorrect_mass = (
            incorrect_mask.sum(dim=1, keepdim=True).to(dtype=mass_dtype)
            / valid_count.to(dtype=mass_dtype)
        )
        partition_input = torch.cat(
            [
                correct_pool,
                incorrect_pool,
                contrast,
                correct_mass,
                incorrect_mass,
                support_statistics.to(dtype=mass_dtype),
            ],
            dim=-1,
        )
        opms_state = self.partition_encoder(partition_input)
        return OutcomePartitionedOutput(
            opms_state=opms_state,
            history_summary=history_summary,
            support_item_state=support_item_state,
            response_features=response_features,
            correct_pool=correct_pool,
            incorrect_pool=incorrect_pool,
            contrast=contrast,
            correct_mass=correct_mass,
            incorrect_mass=incorrect_mass,
        )


@dataclass(frozen=True)
class StrongOPMSOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    opms_state: torch.Tensor
    target_state: torch.Tensor
    history_summary: torch.Tensor
    response_features: torch.Tensor
    correct_pool: torch.Tensor
    incorrect_pool: torch.Tensor
    contrast: torch.Tensor
    correct_mass: torch.Tensor
    incorrect_mass: torch.Tensor


@dataclass(frozen=True)
class TargetLocalPairingOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    opms_state: torch.Tensor
    history_summary: torch.Tensor
    local_state: torch.Tensor
    target_state: torch.Tensor
    local_target_state: torch.Tensor
    response_features: torch.Tensor
    correct_pool: torch.Tensor
    incorrect_pool: torch.Tensor
    contrast: torch.Tensor
    correct_mass: torch.Tensor
    incorrect_mass: torch.Tensor


class _SupportProbeBase(nn.Module):
    def __init__(
        self,
        *,
        num_items: int,
        num_concepts: int,
        item_numeric_dim: int,
        embedding_dim: int,
        state_dim: int,
        mlp_hidden_dim: int,
    ) -> None:
        super().__init__()
        if min(
            num_items,
            num_concepts,
            item_numeric_dim,
            embedding_dim,
            state_dim,
            mlp_hidden_dim,
        ) < 1:
            raise ValueError("All model dimensions must be positive.")
        self.num_items = int(num_items)
        self.num_concepts = int(num_concepts)
        self.item_numeric_dim = int(item_numeric_dim)
        self.embedding_dim = int(embedding_dim)
        self.state_dim = int(state_dim)
        self.mlp_hidden_dim = int(mlp_hidden_dim)

        self.item_encoder = SharedItemEncoder(
            num_items=self.num_items,
            num_concepts=self.num_concepts,
            numeric_dim=self.item_numeric_dim,
            embedding_dim=self.embedding_dim,
            state_dim=self.state_dim,
            mlp_hidden_dim=self.mlp_hidden_dim,
        )
        self.outcome_state = OutcomePartitionedState(
            state_dim=self.state_dim,
            mlp_hidden_dim=self.mlp_hidden_dim,
        )

    def _encode_common(
        self,
        *,
        support_item_ids: torch.Tensor,
        support_q_multi_hot: torch.Tensor,
        support_item_numeric: torch.Tensor,
        support_responses: torch.Tensor,
        support_item_ease: torch.Tensor,
        support_group_attempt_confidence: torch.Tensor,
        support_statistics: torch.Tensor,
        support_mask: torch.Tensor,
        target_item_ids: torch.Tensor,
        target_q_multi_hot: torch.Tensor,
        target_item_numeric: torch.Tensor,
    ) -> tuple[torch.Tensor, OutcomePartitionedOutput]:
        self._validate_common_inputs(
            support_item_ids=support_item_ids,
            support_q_multi_hot=support_q_multi_hot,
            support_item_numeric=support_item_numeric,
            support_responses=support_responses,
            support_item_ease=support_item_ease,
            support_group_attempt_confidence=(
                support_group_attempt_confidence
            ),
            support_statistics=support_statistics,
            support_mask=support_mask,
            target_item_ids=target_item_ids,
            target_q_multi_hot=target_q_multi_hot,
            target_item_numeric=target_item_numeric,
        )
        support_item_state = self.item_encoder(
            item_ids=support_item_ids,
            q_multi_hot=support_q_multi_hot,
            numeric_features=support_item_numeric,
        )
        target_state = self.item_encoder(
            item_ids=target_item_ids,
            q_multi_hot=target_q_multi_hot,
            numeric_features=target_item_numeric,
        )
        outcome = self.outcome_state(
            support_item_state=support_item_state,
            support_responses=support_responses,
            support_item_ease=support_item_ease,
            support_group_attempt_confidence=(
                support_group_attempt_confidence
            ),
            support_statistics=support_statistics,
            support_mask=support_mask,
        )
        return target_state, outcome

    def _validate_common_inputs(
        self,
        *,
        support_item_ids: torch.Tensor,
        support_q_multi_hot: torch.Tensor,
        support_item_numeric: torch.Tensor,
        support_responses: torch.Tensor,
        support_item_ease: torch.Tensor,
        support_group_attempt_confidence: torch.Tensor,
        support_statistics: torch.Tensor,
        support_mask: torch.Tensor,
        target_item_ids: torch.Tensor,
        target_q_multi_hot: torch.Tensor,
        target_item_numeric: torch.Tensor,
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
        for name, value in (
            ("support_responses", support_responses),
            ("support_item_ease", support_item_ease),
            (
                "support_group_attempt_confidence",
                support_group_attempt_confidence,
            ),
        ):
            if tuple(value.shape) != (batch, support):
                raise ValueError(
                    f"{name} must have shape [batch, support]."
                )
        if tuple(support_statistics.shape) != (
            batch,
            len(SUPPORT_STATISTIC_NAMES),
        ):
            raise ValueError(
                "support_statistics must have shape [batch, 6]."
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


class StrongOPMSProbe(_SupportProbeBase):
    """End-to-end Strong-OPMS control over the same raw probe inputs."""

    def __init__(
        self,
        *,
        num_items: int,
        num_concepts: int,
        item_numeric_dim: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        state_dim: int = DEFAULT_STATE_DIM,
        mlp_hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
    ) -> None:
        super().__init__(
            num_items=num_items,
            num_concepts=num_concepts,
            item_numeric_dim=item_numeric_dim,
            embedding_dim=embedding_dim,
            state_dim=state_dim,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self.diagnosis_head = nn.Sequential(
            nn.Linear(self.state_dim * 2, self.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mlp_hidden_dim, 1),
        )

    def forward(
        self,
        *,
        support_item_ids: torch.Tensor,
        support_q_multi_hot: torch.Tensor,
        support_item_numeric: torch.Tensor,
        support_responses: torch.Tensor,
        support_item_ease: torch.Tensor,
        support_group_attempt_confidence: torch.Tensor,
        support_statistics: torch.Tensor,
        support_mask: torch.Tensor,
        target_item_ids: torch.Tensor,
        target_q_multi_hot: torch.Tensor,
        target_item_numeric: torch.Tensor,
    ) -> StrongOPMSOutput:
        target_state, outcome = self._encode_common(
            support_item_ids=support_item_ids,
            support_q_multi_hot=support_q_multi_hot,
            support_item_numeric=support_item_numeric,
            support_responses=support_responses,
            support_item_ease=support_item_ease,
            support_group_attempt_confidence=(
                support_group_attempt_confidence
            ),
            support_statistics=support_statistics,
            support_mask=support_mask,
            target_item_ids=target_item_ids,
            target_q_multi_hot=target_q_multi_hot,
            target_item_numeric=target_item_numeric,
        )
        logits = self.diagnosis_head(
            torch.cat([outcome.opms_state, target_state], dim=-1)
        ).squeeze(-1)
        return StrongOPMSOutput(
            logits=logits,
            probs=torch.sigmoid(logits),
            opms_state=outcome.opms_state,
            target_state=target_state,
            history_summary=outcome.history_summary,
            response_features=outcome.response_features,
            correct_pool=outcome.correct_pool,
            incorrect_pool=outcome.incorrect_pool,
            contrast=outcome.contrast,
            correct_mass=outcome.correct_mass,
            incorrect_mass=outcome.incorrect_mass,
        )


class TargetLocalPairingProbe(_SupportProbeBase):
    """Test whether target/support pairing must precede set aggregation."""

    def __init__(
        self,
        *,
        num_items: int,
        num_concepts: int,
        item_numeric_dim: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        state_dim: int = DEFAULT_STATE_DIM,
        mlp_hidden_dim: int = DEFAULT_MLP_HIDDEN_DIM,
        placement: str = "real_pair",
    ) -> None:
        super().__init__(
            num_items=num_items,
            num_concepts=num_concepts,
            item_numeric_dim=item_numeric_dim,
            embedding_dim=embedding_dim,
            state_dim=state_dim,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        if placement not in PLACEMENTS:
            raise ValueError(
                f"Unsupported placement {placement!r}; "
                f"expected one of {PLACEMENTS}."
            )
        self.placement = placement
        self.interaction_encoder = nn.Sequential(
            nn.Linear(
                self.state_dim + len(RESPONSE_FEATURE_NAMES),
                self.mlp_hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(self.mlp_hidden_dim, self.state_dim),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(self.state_dim * 2, self.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mlp_hidden_dim, self.state_dim),
        )
        self.diagnosis_head = nn.Sequential(
            nn.Linear(self.state_dim * 3, self.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mlp_hidden_dim, 1),
        )

    def forward(
        self,
        *,
        support_item_ids: torch.Tensor,
        support_q_multi_hot: torch.Tensor,
        support_item_numeric: torch.Tensor,
        support_responses: torch.Tensor,
        support_item_ease: torch.Tensor,
        support_group_attempt_confidence: torch.Tensor,
        support_statistics: torch.Tensor,
        support_mask: torch.Tensor,
        target_item_ids: torch.Tensor,
        target_q_multi_hot: torch.Tensor,
        target_item_numeric: torch.Tensor,
        donor_target_item_ids: torch.Tensor | None = None,
        donor_target_q_multi_hot: torch.Tensor | None = None,
        donor_target_item_numeric: torch.Tensor | None = None,
    ) -> TargetLocalPairingOutput:
        target_state, outcome = self._encode_common(
            support_item_ids=support_item_ids,
            support_q_multi_hot=support_q_multi_hot,
            support_item_numeric=support_item_numeric,
            support_responses=support_responses,
            support_item_ease=support_item_ease,
            support_group_attempt_confidence=(
                support_group_attempt_confidence
            ),
            support_statistics=support_statistics,
            support_mask=support_mask,
            target_item_ids=target_item_ids,
            target_q_multi_hot=target_q_multi_hot,
            target_item_numeric=target_item_numeric,
        )
        encoded_support = self.interaction_encoder(
            torch.cat(
                [
                    outcome.support_item_state,
                    outcome.response_features,
                ],
                dim=-1,
            )
        )
        history_summary = masked_mean(encoded_support, support_mask)
        local_target_state = self._local_target_state(
            batch=target_item_ids.shape[0],
            target_state=target_state,
            donor_target_item_ids=donor_target_item_ids,
            donor_target_q_multi_hot=donor_target_q_multi_hot,
            donor_target_item_numeric=donor_target_item_numeric,
        )

        if self.placement == "late_fusion":
            local_state = self.pair_encoder(
                torch.cat(
                    [history_summary, local_target_state],
                    dim=-1,
                )
            )
        else:
            expanded_target = local_target_state.unsqueeze(1).expand(
                -1,
                encoded_support.shape[1],
                -1,
            )
            paired_support = self.pair_encoder(
                torch.cat(
                    [encoded_support, expanded_target],
                    dim=-1,
                )
            )
            local_state = masked_mean(paired_support, support_mask)

        logits = self.diagnosis_head(
            torch.cat(
                [outcome.opms_state, local_state, target_state],
                dim=-1,
            )
        ).squeeze(-1)
        return TargetLocalPairingOutput(
            logits=logits,
            probs=torch.sigmoid(logits),
            opms_state=outcome.opms_state,
            history_summary=history_summary,
            local_state=local_state,
            target_state=target_state,
            local_target_state=local_target_state,
            response_features=outcome.response_features,
            correct_pool=outcome.correct_pool,
            incorrect_pool=outcome.incorrect_pool,
            contrast=outcome.contrast,
            correct_mass=outcome.correct_mass,
            incorrect_mass=outcome.incorrect_mass,
        )

    def _local_target_state(
        self,
        *,
        batch: int,
        target_state: torch.Tensor,
        donor_target_item_ids: torch.Tensor | None,
        donor_target_q_multi_hot: torch.Tensor | None,
        donor_target_item_numeric: torch.Tensor | None,
    ) -> torch.Tensor:
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
        if not all_donor:
            return target_state

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
        if self.placement != "perm_pair":
            return target_state
        return self.item_encoder(
            item_ids=donor_target_item_ids,
            q_multi_hot=donor_target_q_multi_hot,
            numeric_features=donor_target_item_numeric,
        )
