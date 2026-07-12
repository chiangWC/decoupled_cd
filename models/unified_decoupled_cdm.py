from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from .decoupled_cdm import DecoupledForwardOutput
from .evidence_relation_graph import EvidenceRelationGraphCompleter
from .unified_v2_components import (
    ConditionalSimplexBehaviorModel,
    GlobalConceptPriorCompleter,
    MonotonicDiagnosisDecoder,
    ObservedMasteryEstimator,
    assemble_mastery,
)
from .unified_v2_spec import UnifiedArchitectureSpec


class UnifiedDecoupledCDM(nn.Module):
    @staticmethod
    def _text_tensor(value: str) -> torch.Tensor:
        return torch.tensor(
            list(value.encode("utf-8")),
            dtype=torch.uint8,
        )

    def __init__(
        self,
        *,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        dim: int,
        architecture: UnifiedArchitectureSpec,
        initial_mastery_logits: torch.Tensor | None = None,
        graph_hidden_dim: int = 32,
        evidence_cap: float = 20.0,
    ) -> None:
        super().__init__()
        if type(graph_hidden_dim) is not int or graph_hidden_dim <= 0:
            raise ValueError("graph_hidden_dim must be a positive integer")
        self.num_students = num_students
        self.architecture = architecture
        self.register_buffer(
            "_checkpoint_architecture_manifest",
            self._text_tensor(
                json.dumps(
                    architecture.manifest(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
        )
        self.register_buffer(
            "_checkpoint_architecture_fingerprint",
            self._text_tensor(architecture.fingerprint()),
        )
        self.register_buffer(
            "_checkpoint_unified_completion",
            self._text_tensor(architecture.completion),
        )
        self.register_buffer(
            "_checkpoint_unified_graph_hidden_dim",
            torch.tensor(graph_hidden_dim, dtype=torch.int64),
        )
        self.register_buffer(
            "_checkpoint_unified_evidence_loss_weight",
            torch.tensor(float("nan"), dtype=torch.float64),
        )
        self.register_buffer(
            "_checkpoint_unified_completion_loss_weight",
            torch.tensor(float("nan"), dtype=torch.float64),
        )
        self.mastery_estimator = ObservedMasteryEstimator(
            num_students=num_students,
            num_concepts=num_concepts,
            initial_logits=initial_mastery_logits,
            evidence_cap=evidence_cap,
        )
        if architecture.completion == "prior":
            self.completer = GlobalConceptPriorCompleter(num_concepts)
        else:
            self.completer = EvidenceRelationGraphCompleter(
                num_students=num_students,
                num_concepts=num_concepts,
                hidden_dim=graph_hidden_dim,
            )
        self.decoder = MonotonicDiagnosisDecoder(
            num_exercises=num_exercises,
            num_concepts=num_concepts,
            dim=dim,
        )
        self.behavior_model = ConditionalSimplexBehaviorModel(
            num_students=num_students,
            num_exercises=num_exercises,
            dim=dim,
        )

    @property
    def graph_hidden_dim(self) -> int:
        return int(self._checkpoint_unified_graph_hidden_dim.item())

    def set_checkpoint_loss_weights(
        self,
        *,
        evidence_loss_weight: float,
        completion_loss_weight: float,
    ) -> None:
        self._checkpoint_unified_evidence_loss_weight.fill_(
            evidence_loss_weight
        )
        self._checkpoint_unified_completion_loss_weight.fill_(
            completion_loss_weight
        )

    @staticmethod
    def _checkpoint_values_equal(
        expected: torch.Tensor,
        actual: torch.Tensor,
    ) -> bool:
        expected_cpu = expected.detach().cpu()
        actual_cpu = actual.detach().cpu()
        if expected_cpu.shape != actual_cpu.shape:
            return False
        if expected_cpu.is_floating_point() and expected_cpu.numel() == 1:
            expected_value = float(expected_cpu)
            actual_value = float(actual_cpu)
            return (
                math.isnan(expected_value) and math.isnan(actual_value)
            ) or expected_value == actual_value
        return torch.equal(expected_cpu, actual_cpu)

    def _validate_checkpoint_metadata(
        self,
        state_dict: Mapping[str, Any],
    ) -> None:
        fields = {
            "_checkpoint_architecture_manifest": "architecture manifest",
            "_checkpoint_architecture_fingerprint": (
                "architecture fingerprint"
            ),
            "_checkpoint_unified_completion": "unified completion",
            "_checkpoint_unified_graph_hidden_dim": (
                "unified graph_hidden_dim"
            ),
            "_checkpoint_unified_evidence_loss_weight": (
                "unified evidence_loss_weight"
            ),
            "_checkpoint_unified_completion_loss_weight": (
                "unified completion_loss_weight"
            ),
        }
        for key, label in fields.items():
            if key not in state_dict:
                raise ValueError(f"checkpoint is missing {label}")
            actual = state_dict[key]
            if not isinstance(actual, torch.Tensor):
                raise ValueError(f"checkpoint {label} must be a tensor")
            expected = getattr(self, key)
            if not self._checkpoint_values_equal(expected, actual):
                raise ValueError(f"checkpoint {label} mismatch")

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        self._validate_checkpoint_metadata(state_dict)
        return super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )

    @staticmethod
    def _select_students(
        target_student_ids: torch.Tensor,
        use_student_subset: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if not use_student_subset:
            return None, target_student_ids
        return torch.unique(
            target_student_ids,
            sorted=True,
            return_inverse=True,
        )

    def forward(
        self,
        q_matrix: torch.Tensor,
        concept_graph: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        student_tkc_mask: torch.Tensor,
        student_ukc_mask: torch.Tensor,
        student_concept_evidence: torch.Tensor | None,
        target_student_ids: torch.Tensor,
        target_exercise_ids: torch.Tensor,
        use_student_subset: bool = False,
        completion_epoch: int | None = None,
    ) -> DecoupledForwardOutput:
        del (
            concept_graph,
            student_exercise_mask,
            response_matrix,
            student_tkc_mask,
            student_ukc_mask,
        )
        if student_concept_evidence is None:
            raise ValueError(
                "unified v3 requires train-only student_concept_evidence"
            )
        student_ids, local_target_ids = self._select_students(
            target_student_ids,
            use_student_subset,
        )
        observed = self.mastery_estimator(
            student_concept_evidence,
            student_ids,
        )
        if self.architecture.completion == "prior":
            missing = self.completer(observed.mastery.shape[0])
            completion_target_mask = None
            completion_targets = None
            completion_student_state = None
            completion_concept_state = None
            completion_full_target_count = None
        else:
            completion = self.completer(
                student_concept_evidence,
                q_matrix,
                student_ids=student_ids,
                epoch=completion_epoch,
                training=self.training,
            )
            missing = completion.mastery
            completion_target_mask = completion.reconstruction_mask
            completion_targets = completion.target if self.training else None
            completion_student_state = completion.student_state
            completion_concept_state = completion.concept_state
            completion_full_target_count = completion.full_target_count
        mastery = assemble_mastery(
            observed.mastery,
            missing,
            observed.observed_mask,
        )
        cognitive_probs, difficulty = self.decoder(
            mastery,
            q_matrix,
            local_target_ids,
            target_exercise_ids,
        )
        behavior = self.behavior_model(
            cognitive_probs,
            target_student_ids,
            target_exercise_ids,
        )
        concept_basis = torch.eye(
            q_matrix.shape[1],
            device=mastery.device,
            dtype=mastery.dtype,
        )
        return DecoupledForwardOutput(
            student_state=mastery,
            tkc_states=(
                mastery.unsqueeze(-1)
                * observed.observed_mask.unsqueeze(-1)
            ),
            ukc_states=(
                mastery.unsqueeze(-1)
                * (~observed.observed_mask).unsqueeze(-1)
            ),
            tkc_weight=observed.reliability,
            concept_embeddings=concept_basis,
            exercise_embeddings=q_matrix.to(dtype=mastery.dtype),
            cognitive_probs=cognitive_probs,
            probs=behavior.probs,
            guess_probs=behavior.guess_probs,
            slip_probs=behavior.slip_probs,
            difficulty=difficulty,
            mastery=mastery,
            source_weights=None,
            observed_mastery=observed.mastery,
            completion_predictions=missing,
            mastery_observed_mask=observed.observed_mask,
            cognitive_weight=behavior.cognitive_weight,
            completion_target_mask=completion_target_mask,
            completion_targets=completion_targets,
            completion_student_state=completion_student_state,
            completion_concept_state=completion_concept_state,
            completion_full_target_count=completion_full_target_count,
        )
