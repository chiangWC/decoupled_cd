from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .decoupled_cdm import DecoupledCDM, DecoupledForwardOutput


class DecoupledCDMEnsemble(nn.Module):
    """Single-checkpoint probability average over two DecoupledCDM towers."""

    _RUNTIME_SYNC_ATTRS = (
        "history_evidence_logit_prior_residual",
        "history_evidence_logit_prior_location",
        "concept_evidence_prior_residual",
        "concept_evidence_prior_max_logit",
        "history_evidence_output_calibration",
        "history_evidence_output_calibration_apply_mode",
    )

    def __init__(
        self,
        *,
        concept_dim: int = 64,
        secondary_concept_dim: int = 80,
        secondary_weight: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if secondary_concept_dim < 1:
            raise ValueError("secondary_concept_dim must be positive.")
        if secondary_weight < 0.0 or secondary_weight > 1.0:
            raise ValueError("secondary_weight must be in [0, 1].")
        self.primary = DecoupledCDM(concept_dim=concept_dim, **kwargs)
        self.secondary = DecoupledCDM(concept_dim=secondary_concept_dim, **kwargs)
        self.concept_dim = int(concept_dim)
        self.secondary_concept_dim = int(secondary_concept_dim)
        self.secondary_weight = float(secondary_weight)
        for name in self._RUNTIME_SYNC_ATTRS:
            if hasattr(self.primary, name):
                setattr(self, name, getattr(self.primary, name))

    def forward(self, **kwargs: Any) -> DecoupledForwardOutput:
        self._sync_runtime_attrs()
        primary_output = self.primary(**kwargs)
        secondary_output = self.secondary(**kwargs)
        probs = self._blend(primary_output.probs, secondary_output.probs)
        cognitive_probs = self._blend(primary_output.cognitive_probs, secondary_output.cognitive_probs)
        guess_probs = self._blend(primary_output.guess_probs, secondary_output.guess_probs)
        slip_probs = self._blend(primary_output.slip_probs, secondary_output.slip_probs)
        return DecoupledForwardOutput(
            student_state=primary_output.student_state,
            tkc_states=primary_output.tkc_states,
            ukc_states=primary_output.ukc_states,
            concept_embeddings=primary_output.concept_embeddings,
            exercise_embeddings=primary_output.exercise_embeddings,
            cognitive_probs=cognitive_probs,
            probs=probs,
            guess_probs=guess_probs,
            slip_probs=slip_probs,
            difficulty=primary_output.difficulty,
            primary_probs=primary_output.probs,
            secondary_probs=secondary_output.probs,
            primary_cognitive_probs=primary_output.cognitive_probs,
            secondary_cognitive_probs=secondary_output.cognitive_probs,
        )

    def _sync_runtime_attrs(self) -> None:
        for name in self._RUNTIME_SYNC_ATTRS:
            if hasattr(self, name):
                value = getattr(self, name)
                if hasattr(self.primary, name):
                    setattr(self.primary, name, value)
                if hasattr(self.secondary, name):
                    setattr(self.secondary, name, value)

    def _blend(self, primary: torch.Tensor, secondary: torch.Tensor) -> torch.Tensor:
        return primary * (1.0 - self.secondary_weight) + secondary * self.secondary_weight

    def _build_history_evidence_logit_prior_residual(self, **kwargs: Any) -> torch.Tensor:
        return self.primary._build_history_evidence_logit_prior_residual(**kwargs)

    def build_exercise_difficulty_prior_target(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return self.primary.build_exercise_difficulty_prior_target(**kwargs)

    def initialize_exercise_difficulty_from_evidence(self, **kwargs: Any) -> int:
        primary_count = self.primary.initialize_exercise_difficulty_from_evidence(**kwargs)
        self.secondary.initialize_exercise_difficulty_from_evidence(**kwargs)
        return primary_count
