from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn


VARIANTS: Final[tuple[str, ...]] = (
    "full",
    "direct",
    "capacity",
)
DEFAULT_EMBEDDING_DIM: Final[int] = 16
DEFAULT_STATE_DIM: Final[int] = 64
DEFAULT_HIDDEN_DIM: Final[int] = 64
DEFAULT_ROUTING_ITERATIONS: Final[int] = 3
_INTEGER_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average a padded set without allowing padding into the result."""
    valid = mask.to(dtype=torch.bool).unsqueeze(-1)
    masked = torch.where(
        valid,
        values,
        torch.zeros((), dtype=values.dtype, device=values.device),
    )
    count = valid.sum(dim=1).clamp_min(1).to(dtype=values.dtype)
    return masked.sum(dim=1) / count


def _masked_q_softmax(
    logits: torch.Tensor,
    q_mask: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """Softmax over each interaction's eligible Q entries.

    Padded support rows have an empty Q set.  A temporary safe entry avoids an
    all-masked softmax; multiplication by ``support_mask`` removes it exactly.
    """
    eligible = q_mask.to(dtype=torch.bool)
    valid_support = support_mask.to(dtype=torch.bool)
    safe_eligible = eligible.clone()
    safe_eligible[..., 0] |= ~valid_support
    masked_logits = logits.masked_fill(~safe_eligible, float("-inf"))
    responsibility = torch.softmax(masked_logits, dim=-1)
    responsibility = responsibility * eligible.to(dtype=logits.dtype)
    responsibility = responsibility * valid_support.unsqueeze(-1).to(
        dtype=logits.dtype
    )
    return responsibility


@dataclass(frozen=True)
class ResponseConceptCreditOutput:
    logits: torch.Tensor
    probs: torch.Tensor
    framework_state: torch.Tensor
    mastery: torch.Tensor
    routing_responsibility: torch.Tensor
    routing_credit: torch.Tensor
    routing_entropy: torch.Tensor
    routed_mass: torch.Tensor
    concept_evidence_mass: torch.Tensor
    observed_concept_mask: torch.Tensor
    module_diagnostics: dict[str, torch.Tensor]
    architecture_fingerprint: str


class ResponseConceptCreditProbe(nn.Module):
    """Route multi-concept responses, then generate a complete concept state.

    Item and concept IDs are one based; zero is reserved for padding.  The
    model has no student-ID parameter.  All student-specific information must
    pass through ``framework_state`` before the fixed Q-conditioned diagnosis.

    ``full`` learns response- and student-conditioned Q-constrained routing.
    ``capacity`` retains exactly the same parameters but neutralizes every
    response, confidence, and student-context channel before routing.
    ``direct`` fixes credit to one for every eligible Q entry.  Response
    features enter the shared evidence value only after credit is determined.
    """

    def __init__(
        self,
        *,
        num_items: int,
        num_concepts: int,
        max_q_cardinality: int,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        state_dim: int = DEFAULT_STATE_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        routing_iterations: int = DEFAULT_ROUTING_ITERATIONS,
    ) -> None:
        super().__init__()
        if min(
            num_items,
            num_concepts,
            max_q_cardinality,
            embedding_dim,
            state_dim,
            hidden_dim,
            routing_iterations,
        ) < 1:
            raise ValueError("All model dimensions must be positive.")
        self.num_items = int(num_items)
        self.num_concepts = int(num_concepts)
        self.max_q_cardinality = int(max_q_cardinality)
        self.embedding_dim = int(embedding_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.routing_iterations = int(routing_iterations)

        self.item_embedding = nn.Embedding(
            self.num_items + 1,
            self.embedding_dim,
            padding_idx=0,
        )
        self.concept_embedding = nn.Embedding(
            self.num_concepts + 1,
            self.embedding_dim,
            padding_idx=0,
        )
        self.item_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        self.concept_projection = nn.Linear(
            self.embedding_dim,
            self.state_dim,
        )

        # Four dynamic channels: response, response-minus-ease, item
        # confidence, and repeated-group attempt confidence.
        self.student_context_encoder = nn.Sequential(
            nn.Linear(self.state_dim + 4, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        self.routing_scorer = nn.Sequential(
            nn.Linear(self.state_dim * 3 + 5, self.hidden_dim),
            nn.GELU(),
        )
        self.routing_output = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.routing_output.weight)
        nn.init.zeros_(self.routing_output.bias)

        self.routing_value_encoder = nn.Sequential(
            nn.Linear(self.state_dim + 1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        self.routing_update = nn.GRUCell(
            self.state_dim,
            self.state_dim,
        )
        self.evidence_value_encoder = nn.Sequential(
            nn.Linear(self.state_dim + 4, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        self.completion_network = nn.Sequential(
            nn.Linear(self.state_dim * 3 + 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )
        # Fixed infrastructure: target Q selects state coordinates; no raw
        # support tensor or student identity is accepted here.
        self.diagnosis_head = nn.Sequential(
            nn.Linear(self.state_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        *,
        variant: str,
        support_item_ids: torch.Tensor,
        support_q_indices: torch.Tensor,
        support_q_mask: torch.Tensor,
        support_responses: torch.Tensor,
        support_item_ease: torch.Tensor,
        support_item_confidence: torch.Tensor,
        support_group_attempt_confidence: torch.Tensor,
        support_mask: torch.Tensor,
        query_context_indices: torch.Tensor,
        target_q_indices: torch.Tensor,
        target_q_mask: torch.Tensor,
    ) -> ResponseConceptCreditOutput:
        self._validate_forward_inputs(
            variant=variant,
            support_item_ids=support_item_ids,
            support_q_indices=support_q_indices,
            support_q_mask=support_q_mask,
            support_responses=support_responses,
            support_item_ease=support_item_ease,
            support_item_confidence=support_item_confidence,
            support_group_attempt_confidence=(
                support_group_attempt_confidence
            ),
            support_mask=support_mask,
            query_context_indices=query_context_indices,
            target_q_indices=target_q_indices,
            target_q_mask=target_q_mask,
        )
        valid_support = support_mask.to(dtype=torch.bool)
        q_mask = support_q_mask.to(dtype=torch.bool)

        support_item_state = self._encode_support_items(
            support_item_ids=support_item_ids,
            support_q_indices=support_q_indices,
            support_q_mask=q_mask,
        )
        scalar_dtype = support_item_state.dtype
        response = support_responses.to(dtype=scalar_dtype)
        ease = support_item_ease.to(dtype=scalar_dtype)
        item_confidence = support_item_confidence.to(dtype=scalar_dtype)
        attempt_confidence = support_group_attempt_confidence.to(
            dtype=scalar_dtype
        )
        response_features = torch.stack(
            [
                response,
                response - ease,
                item_confidence,
                attempt_confidence,
            ],
            dim=-1,
        )
        student_tokens = self.student_context_encoder(
            torch.cat([support_item_state, response_features], dim=-1)
        )
        student_context = _masked_mean(student_tokens, valid_support)

        base_concept_state = self.concept_projection(
            self.concept_embedding.weight[1:]
        ).unsqueeze(0).expand(support_item_ids.shape[0], -1, -1)
        # Each support interaction owns a local copy of only its eligible
        # concept candidates.  Consequently Capacity's routing decision for a
        # focal item cannot depend on the other items in the student's set.
        local_candidate_slots = self._gather_concept_state(
            base_concept_state,
            support_q_indices,
        )
        static_route_value = self.routing_value_encoder(
            torch.cat([support_item_state, ease.unsqueeze(-1)], dim=-1)
        )

        responsibility = torch.empty(0, device=support_item_ids.device)
        credit = torch.empty(0, device=support_item_ids.device)
        route_logits = torch.empty(0, device=support_item_ids.device)
        for _ in range(self.routing_iterations):
            expanded_item = support_item_state.unsqueeze(2).expand(
                -1,
                -1,
                support_q_indices.shape[-1],
                -1,
            )
            expanded_ease = ease[..., None, None].expand(
                -1,
                -1,
                support_q_indices.shape[-1],
                -1,
            )
            expanded_dynamic = response_features.unsqueeze(2).expand(
                -1,
                -1,
                support_q_indices.shape[-1],
                -1,
            )
            expanded_context = student_context[:, None, None, :].expand(
                -1,
                support_item_ids.shape[1],
                support_q_indices.shape[-1],
                -1,
            )
            if variant == "capacity":
                # Keep a zero-Jacobian connection to response inputs while
                # fixing the routing latent to static item/Q/concept/ease.
                expanded_dynamic = expanded_dynamic * 0.0
                expanded_context = expanded_context * 0.0
            route_hidden = self.routing_scorer(
                torch.cat(
                    [
                        expanded_item,
                        local_candidate_slots,
                        expanded_context,
                        expanded_ease,
                        expanded_dynamic,
                    ],
                    dim=-1,
                )
            )
            route_logits = self.routing_output(route_hidden).squeeze(-1)
            learned_responsibility = _masked_q_softmax(
                route_logits,
                q_mask,
                valid_support,
            )
            if variant == "direct":
                responsibility = (
                    q_mask & valid_support.unsqueeze(-1)
                ).to(dtype=route_logits.dtype)
                q_cardinality = q_mask.sum(dim=-1, keepdim=True).clamp_min(1)
                responsibility = responsibility / q_cardinality.to(
                    dtype=route_logits.dtype
                )
            else:
                responsibility = learned_responsibility
            q_cardinality = q_mask.sum(dim=-1).to(
                dtype=route_logits.dtype
            )
            credit = responsibility * q_cardinality.unsqueeze(-1)
            local_update = (
                static_route_value.unsqueeze(2) * credit.unsqueeze(-1)
            )
            local_candidate_slots = self.routing_update(
                local_update.reshape(-1, self.state_dim),
                local_candidate_slots.reshape(-1, self.state_dim),
            ).reshape_as(local_candidate_slots)

        evidence_value = self.evidence_value_encoder(
            torch.cat([support_item_state, response_features], dim=-1)
        )
        routed_evidence, concept_mass = self._scatter_to_concepts(
            values=evidence_value,
            credit=credit,
            q_indices=support_q_indices,
        )
        routed_evidence = routed_evidence / concept_mass.clamp_min(1.0)
        observed = concept_mass.squeeze(-1) > 0
        routed_student_summary = _masked_mean(
            routed_evidence,
            observed,
        )
        expanded_summary = routed_student_summary.unsqueeze(1).expand(
            -1,
            self.num_concepts,
            -1,
        )
        completion_input = torch.cat(
            [
                base_concept_state,
                routed_evidence,
                expanded_summary,
                observed.unsqueeze(-1).to(dtype=scalar_dtype),
                torch.log1p(concept_mass),
            ],
            dim=-1,
        )
        framework_state = self.completion_network(completion_input)
        # Parameter-free diagnostic view; response BCE does not leave an
        # otherwise unused mastery-only head in the architecture.
        mastery = torch.sigmoid(framework_state.mean(dim=-1))
        logits = self.diagnose(
            framework_state=framework_state,
            query_context_indices=query_context_indices,
            target_q_indices=target_q_indices,
            target_q_mask=target_q_mask,
        )

        safe_responsibility = responsibility.clamp_min(
            torch.finfo(responsibility.dtype).tiny
        )
        entropy = -(
            responsibility * safe_responsibility.log()
        ).sum(dim=-1)
        entropy = entropy * valid_support.to(dtype=entropy.dtype)
        routed_mass = credit.sum(dim=-1)
        q_cardinality = q_mask.sum(dim=-1).to(dtype=credit.dtype)
        diagnostics = {
            "routing_logits": route_logits,
            "q_cardinality": q_cardinality,
            "routing_entropy": entropy,
            "routed_mass": routed_mass,
            "concept_evidence_mass": concept_mass.squeeze(-1),
            "observed_concept_fraction": observed.to(
                dtype=scalar_dtype
            ).mean(dim=-1),
        }
        return ResponseConceptCreditOutput(
            logits=logits,
            probs=torch.sigmoid(logits),
            framework_state=framework_state,
            mastery=mastery,
            routing_responsibility=responsibility,
            routing_credit=credit,
            routing_entropy=entropy,
            routed_mass=routed_mass,
            concept_evidence_mass=concept_mass.squeeze(-1),
            observed_concept_mask=observed,
            module_diagnostics=diagnostics,
            architecture_fingerprint=self.architecture_fingerprint(variant),
        )

    def diagnose(
        self,
        *,
        framework_state: torch.Tensor,
        query_context_indices: torch.Tensor,
        target_q_indices: torch.Tensor,
        target_q_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the fixed Q-conditioned diagnosis from state alone."""
        if framework_state.ndim != 3 or tuple(framework_state.shape[1:]) != (
            self.num_concepts,
            self.state_dim,
        ):
            raise ValueError(
                "framework_state must have shape "
                "[batch_students, num_concepts, state_dim]."
            )
        self._validate_query_inputs(
            batch_students=framework_state.shape[0],
            query_context_indices=query_context_indices,
            target_q_indices=target_q_indices,
            target_q_mask=target_q_mask,
        )
        selected_state = framework_state[
            query_context_indices.to(dtype=torch.long)
        ]
        q_state = self._gather_concept_state(
            selected_state,
            target_q_indices.unsqueeze(1),
        ).squeeze(1)
        q_mask = target_q_mask.to(dtype=torch.bool)
        q_weight = q_mask.unsqueeze(-1).to(dtype=q_state.dtype)
        student_requirement = (q_state * q_weight).sum(dim=1)
        student_requirement = student_requirement / q_weight.sum(
            dim=1
        ).clamp_min(1.0)

        target_semantic = self.concept_embedding(
            target_q_indices.to(dtype=torch.long)
        )
        target_semantic = self.concept_projection(target_semantic)
        target_requirement = (target_semantic * q_weight).sum(dim=1)
        target_requirement = target_requirement / q_weight.sum(
            dim=1
        ).clamp_min(1.0)
        return self.diagnosis_head(
            torch.cat([student_requirement, target_requirement], dim=-1)
        ).squeeze(-1)

    def architecture_fingerprint(self, variant: str) -> str:
        if variant not in VARIANTS:
            raise ValueError(
                f"Unsupported variant {variant!r}; expected one of {VARIANTS}."
            )
        payload = {
            "model": self.__class__.__name__,
            "variant": variant,
            "num_items": self.num_items,
            "num_concepts": self.num_concepts,
            "max_q_cardinality": self.max_q_cardinality,
            "embedding_dim": self.embedding_dim,
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "routing_iterations": self.routing_iterations,
            "student_id_embedding": False,
            "diagnosis_student_input": "framework_state_only",
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    def _encode_support_items(
        self,
        *,
        support_item_ids: torch.Tensor,
        support_q_indices: torch.Tensor,
        support_q_mask: torch.Tensor,
    ) -> torch.Tensor:
        item_semantic = self.item_embedding(
            support_item_ids.to(dtype=torch.long)
        )
        concept_semantic = self.concept_embedding(
            support_q_indices.to(dtype=torch.long)
        )
        q_weight = support_q_mask.unsqueeze(-1).to(
            dtype=concept_semantic.dtype
        )
        q_semantic = (concept_semantic * q_weight).sum(dim=2)
        q_semantic = q_semantic / q_weight.sum(dim=2).clamp_min(1.0)
        return self.item_encoder(
            torch.cat([item_semantic, q_semantic], dim=-1)
        )

    def _gather_concept_state(
        self,
        concept_state: torch.Tensor,
        q_indices: torch.Tensor,
    ) -> torch.Tensor:
        if q_indices.ndim != 3:
            raise ValueError("q_indices must have shape [batch, rows, Q].")
        dense_indices = q_indices.to(dtype=torch.long).clamp_min(1) - 1
        expanded_state = concept_state.unsqueeze(1).expand(
            -1,
            q_indices.shape[1],
            -1,
            -1,
        )
        return torch.gather(
            expanded_state,
            dim=2,
            index=dense_indices.unsqueeze(-1).expand(
                -1,
                -1,
                -1,
                self.state_dim,
            ),
        )

    def _scatter_to_concepts(
        self,
        *,
        values: torch.Tensor,
        credit: torch.Tensor,
        q_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, support, value_dim = values.shape
        q_width = q_indices.shape[-1]
        dense_indices = q_indices.to(dtype=torch.long).clamp_min(1) - 1
        weighted_values = (
            values.unsqueeze(2).expand(-1, -1, q_width, -1)
            * credit.unsqueeze(-1)
        )
        flat_indices = dense_indices.reshape(batch, support * q_width)
        flat_values = weighted_values.reshape(
            batch,
            support * q_width,
            value_dim,
        )
        routed = torch.zeros(
            batch,
            self.num_concepts,
            value_dim,
            dtype=values.dtype,
            device=values.device,
        ).scatter_add(
            1,
            flat_indices.unsqueeze(-1).expand(-1, -1, value_dim),
            flat_values,
        )
        flat_credit = credit.reshape(batch, support * q_width, 1)
        mass = torch.zeros(
            batch,
            self.num_concepts,
            1,
            dtype=credit.dtype,
            device=credit.device,
        ).scatter_add(
            1,
            flat_indices.unsqueeze(-1),
            flat_credit,
        )
        return routed, mass

    def _validate_forward_inputs(
        self,
        *,
        variant: str,
        support_item_ids: torch.Tensor,
        support_q_indices: torch.Tensor,
        support_q_mask: torch.Tensor,
        support_responses: torch.Tensor,
        support_item_ease: torch.Tensor,
        support_item_confidence: torch.Tensor,
        support_group_attempt_confidence: torch.Tensor,
        support_mask: torch.Tensor,
        query_context_indices: torch.Tensor,
        target_q_indices: torch.Tensor,
        target_q_mask: torch.Tensor,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(
                f"Unsupported variant {variant!r}; expected one of {VARIANTS}."
            )
        if support_item_ids.ndim != 2:
            raise ValueError(
                "support_item_ids must have shape [batch, support]."
            )
        if support_item_ids.dtype not in _INTEGER_DTYPES:
            raise ValueError("support_item_ids must use an integer dtype.")
        batch, support = support_item_ids.shape
        if support_q_indices.ndim != 3 or tuple(
            support_q_indices.shape[:2]
        ) != (batch, support):
            raise ValueError(
                "support_q_indices must have shape [batch, support, Q]."
            )
        if support_q_indices.shape[-1] > self.max_q_cardinality:
            raise ValueError("Support Q width exceeds max_q_cardinality.")
        if tuple(support_q_mask.shape) != tuple(support_q_indices.shape):
            raise ValueError(
                "support_q_mask must match support_q_indices."
            )
        if support_q_indices.dtype not in _INTEGER_DTYPES:
            raise ValueError("support_q_indices must use an integer dtype.")
        if support_q_mask.dtype != torch.bool:
            raise ValueError("support_q_mask must use bool dtype.")
        expected_support = (batch, support)
        for name, value in (
            ("support_responses", support_responses),
            ("support_item_ease", support_item_ease),
            ("support_item_confidence", support_item_confidence),
            (
                "support_group_attempt_confidence",
                support_group_attempt_confidence,
            ),
            ("support_mask", support_mask),
        ):
            if tuple(value.shape) != expected_support:
                raise ValueError(f"{name} must have shape [batch, support].")
        if support_mask.dtype != torch.bool:
            raise ValueError("support_mask must use bool dtype.")
        if bool(torch.any(support_q_mask != (support_q_indices != 0))):
            raise ValueError(
                "support_q_mask must be exactly support_q_indices != 0."
            )
        if bool(torch.any(support_q_indices < 0)) or bool(
            torch.any(support_q_indices > self.num_concepts)
        ):
            raise ValueError("support_q_indices are outside 0..num_concepts.")
        valid_support = support_mask.to(dtype=torch.bool)
        if bool(torch.any(valid_support & (support_item_ids < 1))) or bool(
            torch.any(valid_support & (support_item_ids > self.num_items))
        ):
            raise ValueError("Valid support item IDs must be in 1..num_items.")
        if bool(torch.any(~valid_support & (support_item_ids != 0))):
            raise ValueError("Padded support item IDs must be zero.")
        if bool(torch.any(~valid_support.unsqueeze(-1) & support_q_mask)):
            raise ValueError("Padded support rows must have an empty Q set.")
        if bool(torch.any(valid_support & ~support_q_mask.any(dim=-1))):
            raise ValueError("Every valid support row needs a non-empty Q set.")
        valid_response = support_responses.masked_select(valid_support)
        if bool(
            torch.any((valid_response != 0) & (valid_response != 1))
        ):
            raise ValueError("Valid support responses must be binary.")
        for name, value in (
            ("support_responses", support_responses),
            ("support_item_ease", support_item_ease),
            ("support_item_confidence", support_item_confidence),
            (
                "support_group_attempt_confidence",
                support_group_attempt_confidence,
            ),
        ):
            if not bool(torch.isfinite(value.masked_select(valid_support)).all()):
                raise ValueError(f"Valid {name} values must be finite.")
        self._validate_query_inputs(
            batch_students=batch,
            query_context_indices=query_context_indices,
            target_q_indices=target_q_indices,
            target_q_mask=target_q_mask,
        )

    def _validate_query_inputs(
        self,
        *,
        batch_students: int,
        query_context_indices: torch.Tensor,
        target_q_indices: torch.Tensor,
        target_q_mask: torch.Tensor,
    ) -> None:
        if query_context_indices.ndim != 1:
            raise ValueError("query_context_indices must have shape [rows].")
        if query_context_indices.dtype not in _INTEGER_DTYPES:
            raise ValueError("query_context_indices must use integer dtype.")
        rows = query_context_indices.shape[0]
        if target_q_indices.ndim != 2 or target_q_indices.shape[0] != rows:
            raise ValueError("target_q_indices must have shape [rows, Q].")
        if target_q_indices.shape[-1] > self.max_q_cardinality:
            raise ValueError("Target Q width exceeds max_q_cardinality.")
        if tuple(target_q_mask.shape) != tuple(target_q_indices.shape):
            raise ValueError("target_q_mask must match target_q_indices.")
        if target_q_indices.dtype not in _INTEGER_DTYPES:
            raise ValueError("target_q_indices must use integer dtype.")
        if target_q_mask.dtype != torch.bool:
            raise ValueError("target_q_mask must use bool dtype.")
        if bool(torch.any(target_q_mask != (target_q_indices != 0))):
            raise ValueError(
                "target_q_mask must be exactly target_q_indices != 0."
            )
        if bool(torch.any(target_q_indices < 0)) or bool(
            torch.any(target_q_indices > self.num_concepts)
        ):
            raise ValueError("target_q_indices are outside 0..num_concepts.")
        if bool(torch.any(~target_q_mask.any(dim=-1))):
            raise ValueError("Every target row needs a non-empty Q set.")
        if bool(torch.any(query_context_indices < 0)) or bool(
            torch.any(query_context_indices >= batch_students)
        ):
            raise ValueError("query_context_indices are outside the batch.")
