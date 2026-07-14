from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch
from torch import nn
import torch.nn.functional as F


VALID_DIAGNOSIS_MODES = {"target_conditioned", "monotonic_control"}


@dataclass
class EvidenceRepresentationOutput:
    student_evidence: torch.Tensor
    concept_prior: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


@dataclass
class StateCompletionOutput:
    framework_state: torch.Tensor
    reliability: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


@dataclass
class TwoStageForwardOutput:
    probs: torch.Tensor
    cognitive_probs: torch.Tensor
    framework_state: torch.Tensor
    mastery: torch.Tensor
    state_reliability: torch.Tensor
    student_state: torch.Tensor
    concept_embeddings: torch.Tensor
    exercise_embeddings: torch.Tensor
    guess_probs: torch.Tensor
    slip_probs: torch.Tensor
    difficulty: torch.Tensor
    module_diagnostics: dict[str, torch.Tensor]
    architecture_fingerprint: str
    mastery_aux_logits: torch.Tensor | None = None


class CalibratedEvidenceRepresentation(nn.Module):
    """Module 1: train-only responses to a student evidence representation.

    The full path conditions the student summary on the identities and
    population difficulty of attempted exercises.  Its clean control receives
    only raw response marginals.  Both paths have identical trainable capacity
    and return the same tensor contract.
    """

    VALID_MODES = {
        "calibrated_history",
        "identity_raw_control",
        "calibrated_summary_control",
        "raw_summary_control",
    }

    def __init__(self, *, dim: int, evidence_cap: float) -> None:
        super().__init__()
        if evidence_cap <= 0.0:
            raise ValueError("evidence_cap must be positive.")
        self.dim = int(dim)
        self.evidence_cap = float(evidence_cap)
        self.calibrated_encoder = nn.Sequential(
            nn.Linear(dim + 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.raw_control_encoder = nn.Sequential(
            nn.Linear(dim + 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.population_token_encoder = nn.Sequential(
            nn.Linear(dim + 2, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
        )
        full_rng_state = torch.random.get_rng_state()
        self.identity_raw_encoder = nn.Sequential(
            nn.Linear(dim + 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.calibrated_summary_encoder = nn.Sequential(
            nn.Linear(dim + 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        torch.random.set_rng_state(full_rng_state)

    def active_parameter_counts(self) -> dict[str, int]:
        return {
            "calibrated_history": sum(
                parameter.numel()
                for parameter in self.calibrated_encoder.parameters()
            ),
            "identity_raw_control": sum(
                parameter.numel()
                for parameter in self.identity_raw_encoder.parameters()
            ),
            "calibrated_summary_control": sum(
                parameter.numel()
                for parameter in self.calibrated_summary_encoder.parameters()
            ),
            "raw_summary_control": sum(
                parameter.numel()
                for parameter in self.raw_control_encoder.parameters()
            ),
        }

    def forward(
        self,
        *,
        exercise_nodes: torch.Tensor,
        exercise_evidence: torch.Tensor,
        q_matrix: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        mode: str,
    ) -> EvidenceRepresentationOutput:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported evidence representation mode: {mode}")
        dtype = exercise_nodes.dtype
        mask = student_exercise_mask.to(dtype=dtype)
        responses = response_matrix.to(dtype=dtype)
        item_attempts = exercise_evidence[:, 0].to(dtype=dtype)
        item_accuracy = exercise_evidence[:, 3].to(dtype=dtype)
        item_confidence = (
            exercise_evidence[:, 4].to(dtype=dtype)
            / math.log1p(self.evidence_cap * 100.0)
        ).clamp(0.0, 1.0)

        observed_count = mask.sum(dim=1, keepdim=True)
        safe_count = observed_count.clamp_min(1.0)
        correct_count = (mask * responses).sum(dim=1, keepdim=True)
        success = torch.where(
            observed_count > 0.0,
            correct_count / safe_count,
            correct_count.new_full(correct_count.shape, 0.5),
        )
        observed_difficulty = (
            mask * item_accuracy.unsqueeze(0)
        ).sum(dim=1, keepdim=True) / safe_count
        history_confidence = (
            torch.log1p(observed_count)
            / math.log1p(float(max(mask.size(1), 1)) + 1.0)
        ).clamp(0.0, 1.0)
        coverage = observed_count / float(max(mask.size(1), 1))
        calibrated_features = torch.cat(
            [
                success.mul(2.0).sub(1.0),
                success - observed_difficulty,
                history_confidence,
                coverage,
            ],
            dim=-1,
        )
        raw_features = torch.cat(
            [
                success.mul(2.0).sub(1.0),
                torch.zeros_like(success),
                history_confidence,
                coverage,
            ],
            dim=-1,
        )
        pooled_exercise = mask @ exercise_nodes / safe_count
        inputs_by_mode = {
            "calibrated_history": (pooled_exercise, calibrated_features),
            "identity_raw_control": (pooled_exercise, raw_features),
            "calibrated_summary_control": (
                torch.zeros_like(pooled_exercise),
                calibrated_features,
            ),
            "raw_summary_control": (
                torch.zeros_like(pooled_exercise), raw_features
            ),
        }
        encoders_by_mode = {
            "calibrated_history": self.calibrated_encoder,
            "identity_raw_control": self.identity_raw_encoder,
            "calibrated_summary_control": self.calibrated_summary_encoder,
            "raw_summary_control": self.raw_control_encoder,
        }
        semantic_input, statistic_input = inputs_by_mode[mode]
        student_evidence = encoders_by_mode[mode](
            torch.cat([semantic_input, statistic_input], dim=-1)
        )

        population_features = torch.stack(
            [item_accuracy.mul(2.0).sub(1.0), item_confidence],
            dim=-1,
        )
        population_tokens = self.population_token_encoder(
            torch.cat([exercise_nodes, population_features], dim=-1)
        )
        q_binary = (q_matrix > 0.0).to(dtype=dtype)
        concept_item_count = q_binary.sum(dim=0).clamp_min(1.0)
        concept_prior = q_binary.transpose(0, 1) @ population_tokens
        concept_prior = concept_prior / concept_item_count.unsqueeze(-1)
        return EvidenceRepresentationOutput(
            student_evidence=student_evidence,
            concept_prior=concept_prior,
            diagnostics={
                "history_count": observed_count.squeeze(-1),
                "history_success": success.squeeze(-1),
                "history_difficulty_residual": (
                    success - observed_difficulty
                ).squeeze(-1),
                "population_attempts": item_attempts,
            },
        )


class PersonalizedStateCompletion(nn.Module):
    """Module 2: evidence-conditioned completion of TKC and UKC states.

    Full completion generates every concept state from the student evidence and
    a population concept prior.  The control directly projects observed
    concept statistics and assigns a student-independent prior to UKC.
    Identical decoder shapes make the comparison capacity matched.
    """

    VALID_MODES = {
        "personalized_interaction",
        "additive_personalized_control",
        "direct_prior_control",
    }

    def __init__(self, *, dim: int, evidence_cap: float) -> None:
        super().__init__()
        self.evidence_cap = float(evidence_cap)
        self.raw_evidence_projection = nn.Linear(3, dim)
        self.personalized_decoder = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.direct_control_decoder = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(4, max(8, dim // 2)),
            nn.ReLU(),
            nn.Linear(max(8, dim // 2), 1),
        )
        # Instantiated after the already validated Full path so adding this
        # stronger control does not perturb any existing Full initialization.
        # It consumes the same student and concept inputs as Full but removes
        # the declared multiplicative student-concept interaction.
        full_rng_state = torch.random.get_rng_state()
        self.additive_control_decoder = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        torch.random.set_rng_state(full_rng_state)

    def active_parameter_counts(self) -> dict[str, int]:
        return {
            "personalized_interaction": sum(
                parameter.numel()
                for parameter in self.personalized_decoder.parameters()
            ),
            "additive_personalized_control": sum(
                parameter.numel()
                for parameter in self.additive_control_decoder.parameters()
            ),
            "direct_prior_control": sum(
                parameter.numel()
                for parameter in self.direct_control_decoder.parameters()
            ),
        }

    def _raw_features(
        self,
        evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attempts = evidence[..., 0].clamp_min(0.0)
        correct = torch.minimum(evidence[..., 1].clamp_min(0.0), attempts)
        seen = evidence[..., 5].clamp(0.0, 1.0)
        accuracy = correct / attempts.clamp_min(1.0)
        confidence = (
            evidence[..., 4] / math.log1p(self.evidence_cap)
        ).clamp(0.0, 1.0)
        features = torch.stack(
            [
                accuracy.mul(2.0).sub(1.0) * confidence,
                confidence,
                seen,
            ],
            dim=-1,
        )
        return features, seen, confidence

    def forward(
        self,
        *,
        student_evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        concept_prior: torch.Tensor,
        student_concept_evidence: torch.Tensor,
        mode: str,
    ) -> StateCompletionOutput:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported state completion mode: {mode}")
        dtype = concept_nodes.dtype
        raw_features, seen, confidence = self._raw_features(
            student_concept_evidence.to(dtype=dtype)
        )
        batch_size = student_evidence.size(0)
        num_concepts = concept_nodes.size(0)
        student_grid = student_evidence.unsqueeze(1).expand(
            -1, num_concepts, -1
        )
        concept_grid = concept_nodes.unsqueeze(0).expand(batch_size, -1, -1)
        prior_grid = concept_prior.unsqueeze(0).expand(batch_size, -1, -1)
        if mode == "personalized_interaction":
            framework_state = self.personalized_decoder(
                torch.cat(
                    [
                        student_grid,
                        concept_grid,
                        student_grid * concept_grid,
                        prior_grid,
                    ],
                    dim=-1,
                )
            )
        elif mode == "additive_personalized_control":
            framework_state = self.additive_control_decoder(
                torch.cat(
                    [
                        student_grid,
                        concept_grid,
                        torch.zeros_like(concept_grid),
                        prior_grid,
                    ],
                    dim=-1,
                )
            )
        else:
            observed_raw = concept_grid + self.raw_evidence_projection(
                raw_features
            )
            static_prior = concept_grid + prior_grid
            direct_base = torch.where(
                seen.unsqueeze(-1) > 0.0,
                observed_raw,
                static_prior,
            )
            framework_state = self.direct_control_decoder(
                torch.cat(
                    [
                        direct_base,
                        concept_grid,
                        torch.zeros_like(concept_grid),
                        prior_grid,
                    ],
                    dim=-1,
                )
            )
        history_count = student_concept_evidence[..., 0].sum(dim=1)
        student_success = (
            student_concept_evidence[..., 1].sum(dim=1)
            / history_count.clamp_min(1.0)
        )
        reliability_features = torch.stack(
            [
                seen,
                confidence,
                student_success.unsqueeze(1).expand(-1, num_concepts),
                (
                    history_count / (history_count + self.evidence_cap)
                ).unsqueeze(1).expand(-1, num_concepts),
            ],
            dim=-1,
        )
        reliability = torch.sigmoid(
            self.reliability_head(reliability_features)
        ).squeeze(-1)
        return StateCompletionOutput(
            framework_state=framework_state,
            reliability=reliability,
            diagnostics={
                "tkc_count": seen.sum(dim=1),
                "ukc_count": (1.0 - seen).sum(dim=1),
                "mean_state_reliability": reliability.mean(dim=1),
            },
        )


class TwoStageTKCUKCCDM(nn.Module):
    """Two claimed modules followed by a fixed Q-conditioned diagnosis."""

    def __init__(
        self,
        *,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        concept_dim: int = 64,
        evidence_mode: str = "calibrated_history",
        completion_mode: str = "personalized_interaction",
        diagnosis_mode: str = "target_conditioned",
        evidence_cap: float = 20.0,
        readout_dropout: float = 0.0,
        max_guess: float = 0.3,
        max_slip: float = 0.3,
        **_unused_kwargs,
    ) -> None:
        super().__init__()
        del num_students
        if evidence_mode not in CalibratedEvidenceRepresentation.VALID_MODES:
            raise ValueError(f"Unsupported evidence mode: {evidence_mode}")
        if completion_mode not in PersonalizedStateCompletion.VALID_MODES:
            raise ValueError(f"Unsupported completion mode: {completion_mode}")
        if diagnosis_mode not in VALID_DIAGNOSIS_MODES:
            raise ValueError(f"Unsupported diagnosis mode: {diagnosis_mode}")
        self.concept_dim = int(concept_dim)
        self.evidence_mode = evidence_mode
        self.completion_mode = completion_mode
        self.diagnosis_mode = diagnosis_mode
        self.evidence_cap = float(evidence_cap)
        self.max_guess = float(max_guess)
        self.max_slip = float(max_slip)

        self.concept_embedding = nn.Embedding(num_concepts, concept_dim)
        self.exercise_embedding = nn.Embedding(num_exercises, concept_dim)
        self.evidence_representation = CalibratedEvidenceRepresentation(
            dim=concept_dim,
            evidence_cap=evidence_cap,
        )
        self.state_completion = PersonalizedStateCompletion(
            dim=concept_dim,
            evidence_cap=evidence_cap,
        )
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
        self.exercise_difficulty = nn.Embedding(num_exercises, 1)
        self.mastery_head = nn.Linear(concept_dim, 1)
        nn.init.xavier_uniform_(self.concept_embedding.weight)
        nn.init.xavier_uniform_(self.exercise_embedding.weight)
        nn.init.zeros_(self.exercise_difficulty.weight)

        # The capacity-matched monotonic diagnosis is instantiated only after
        # all Full parameters are initialized, preserving the validated Full
        # initialization while keeping one common state dict for new variants.
        self.control_cognitive_match = nn.Sequential(
            nn.Linear(concept_dim * 4, concept_dim),
            nn.ReLU(),
            nn.Dropout(readout_dropout),
            nn.Linear(concept_dim, 1),
        )
        self.control_guess_head = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 1),
        )
        self.control_slip_head = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 1),
        )
        self.control_discrimination_raw = nn.Parameter(
            torch.tensor(0.54132485)
        )

    @property
    def architecture_fingerprint(self) -> str:
        payload = {
            "family": "two_stage_tkc_ukc_v3",
            "concept_dim": self.concept_dim,
            "student_id_embedding": False,
            "student_specific_bypass": False,
            "diagnosis": "target_conditioned_or_monotonic_control",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]

    def initialization_hash(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def active_module_parameter_counts(self) -> dict[str, dict[str, int]]:
        full_diagnosis = sum(
            parameter.numel()
            for module in [
                self.cognitive_match,
                self.guess_head,
                self.slip_head,
            ]
            for parameter in module.parameters()
        )
        control_diagnosis = sum(
            parameter.numel()
            for module in [
                self.control_cognitive_match,
                self.control_guess_head,
                self.control_slip_head,
            ]
            for parameter in module.parameters()
        ) + self.control_discrimination_raw.numel()
        return {
            "evidence_representation": (
                self.evidence_representation.active_parameter_counts()
            ),
            "state_completion": self.state_completion.active_parameter_counts(),
            "diagnosis": {
                "target_conditioned": full_diagnosis,
                "monotonic_control": control_diagnosis,
            },
        }

    def _concept_nodes(self, q_matrix: torch.Tensor) -> torch.Tensor:
        q_binary = (q_matrix > 0.0).to(
            dtype=self.exercise_embedding.weight.dtype
        )
        count = q_binary.sum(dim=0).clamp_min(1.0)
        exercise_side = q_binary.transpose(0, 1) @ self.exercise_embedding.weight
        return self.concept_embedding.weight + exercise_side / count.unsqueeze(-1)

    def _exercise_nodes(self, q_matrix: torch.Tensor) -> torch.Tensor:
        q_binary = (q_matrix > 0.0).to(
            dtype=self.concept_embedding.weight.dtype
        )
        count = q_binary.sum(dim=1, keepdim=True).clamp_min(1.0)
        concept_side = q_binary @ self.concept_embedding.weight / count
        return self.exercise_embedding.weight + concept_side

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
    ) -> TwoStageForwardOutput:
        del concept_graph, prerequisite_graph, similarity_graph
        del student_tkc_mask, student_ukc_mask
        if target_student_ids is None or target_exercise_ids is None:
            raise ValueError("target student and exercise IDs are required.")
        if student_concept_evidence is None or exercise_evidence is None:
            raise ValueError("train-only evidence tensors are required.")
        if use_student_subset:
            student_indices, target_state_rows = torch.unique(
                target_student_ids,
                sorted=True,
                return_inverse=True,
            )
            mask = student_exercise_mask.index_select(0, student_indices)
            responses = response_matrix.index_select(0, student_indices)
            concept_evidence = student_concept_evidence.index_select(
                0, student_indices
            )
        else:
            target_state_rows = target_student_ids
            mask = student_exercise_mask
            responses = response_matrix
            concept_evidence = student_concept_evidence

        concept_nodes = self._concept_nodes(q_matrix)
        exercise_nodes = self._exercise_nodes(q_matrix)
        evidence_output = self.evidence_representation(
            exercise_nodes=exercise_nodes,
            exercise_evidence=exercise_evidence,
            q_matrix=q_matrix,
            student_exercise_mask=mask,
            response_matrix=responses,
            mode=self.evidence_mode,
        )
        completion_output = self.state_completion(
            student_evidence=evidence_output.student_evidence,
            concept_nodes=concept_nodes,
            concept_prior=evidence_output.concept_prior,
            student_concept_evidence=concept_evidence,
            mode=self.completion_mode,
        )
        framework_state = completion_output.framework_state
        mastery = torch.sigmoid(
            self.mastery_head(framework_state).squeeze(-1)
        )

        q_vectors = q_matrix.index_select(0, target_exercise_ids).to(
            dtype=framework_state.dtype
        )
        q_count = q_vectors.sum(dim=1, keepdim=True).clamp_min(1.0)
        target_states = framework_state.index_select(0, target_state_rows)
        q_state = (
            target_states * q_vectors.unsqueeze(-1)
        ).sum(dim=1) / q_count
        q_concept = q_vectors @ self.concept_embedding.weight / q_count
        target_exercise = self.exercise_embedding(target_exercise_ids)
        q_repr = self.q_projection(
            torch.cat([q_concept, target_exercise], dim=-1)
        )
        match_features = torch.cat(
            [
                q_state,
                q_repr,
                q_state * q_repr,
                torch.abs(q_state - q_repr),
            ],
            dim=-1,
        )
        difficulty = self.exercise_difficulty(
            target_exercise_ids
        ).squeeze(-1)
        if self.diagnosis_mode == "target_conditioned":
            cognitive_probs = torch.sigmoid(
                self.cognitive_match(match_features).squeeze(-1) - difficulty
            )
            state_condition = torch.cat([q_state, q_repr], dim=-1)
            guess_probs = self.max_guess * torch.sigmoid(
                self.guess_head(state_condition).squeeze(-1)
            )
            slip_probs = self.max_slip * torch.sigmoid(
                self.slip_head(state_condition).squeeze(-1)
            )
        else:
            target_mastery = (
                mastery.index_select(0, target_state_rows) * q_vectors
            ).sum(dim=1) / q_count.squeeze(-1)
            target_mastery = target_mastery.clamp(
                1.0e-5, 1.0 - 1.0e-5
            )
            item_only_features = torch.cat(
                [
                    q_repr,
                    q_repr,
                    q_repr * q_repr,
                    torch.zeros_like(q_repr),
                ],
                dim=-1,
            )
            item_condition = torch.cat([q_repr, q_repr], dim=-1)
            item_bias = self.control_cognitive_match(
                item_only_features
            ).squeeze(-1)
            discrimination = F.softplus(
                self.control_discrimination_raw
            )
            cognitive_probs = torch.sigmoid(
                discrimination * torch.logit(target_mastery)
                + item_bias
                - difficulty
            )
            guess_probs = self.max_guess * torch.sigmoid(
                self.control_guess_head(item_condition).squeeze(-1)
            )
            slip_probs = self.max_slip * torch.sigmoid(
                self.control_slip_head(item_condition).squeeze(-1)
            )
        probs = (
            (1.0 - slip_probs) * cognitive_probs
            + guess_probs * (1.0 - cognitive_probs)
        )
        return TwoStageForwardOutput(
            probs=probs,
            cognitive_probs=cognitive_probs,
            framework_state=framework_state,
            mastery=mastery,
            state_reliability=completion_output.reliability,
            student_state=evidence_output.student_evidence,
            concept_embeddings=self.concept_embedding.weight,
            exercise_embeddings=exercise_nodes,
            guess_probs=guess_probs,
            slip_probs=slip_probs,
            difficulty=difficulty,
            module_diagnostics={
                **evidence_output.diagnostics,
                **completion_output.diagnostics,
            },
            architecture_fingerprint=self.architecture_fingerprint,
        )
