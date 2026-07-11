from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TestedKnowledgeState:
    tkc_states: torch.Tensor
    direct_reliability: torch.Tensor


class TestedKnowledgeEvidenceEncoder(nn.Module):
    def __init__(self, *, dim: int, evidence_cap: float) -> None:
        super().__init__()
        self.evidence_cap = float(evidence_cap)
        self.encoder = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(
        self,
        q_matrix: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        student_tkc_mask: torch.Tensor,
    ) -> TestedKnowledgeState:
        correct = (student_exercise_mask * response_matrix) @ q_matrix
        incorrect = (student_exercise_mask * (1.0 - response_matrix)) @ q_matrix
        attempts = correct + incorrect
        features = torch.stack(
            [
                torch.log1p(correct),
                torch.log1p(incorrect),
                correct / attempts.clamp_min(1.0),
            ],
            dim=-1,
        )
        tkc_states = self.encoder(features) * student_tkc_mask.unsqueeze(-1)
        direct_reliability = (
            torch.log1p(attempts) / math.log1p(self.evidence_cap)
        ).clamp(0.0, 1.0)
        return TestedKnowledgeState(
            tkc_states=tkc_states,
            direct_reliability=direct_reliability,
        )


class MonotonicDiagnosisDecoder(nn.Module):
    def __init__(
        self,
        *,
        num_exercises: int,
        num_concepts: int,
        dim: int,
    ) -> None:
        super().__init__()
        self.mastery_head = nn.Linear(dim, 1)
        self.concept_difficulty = nn.Embedding(num_concepts, 1)
        self.exercise_discrimination = nn.Embedding(num_exercises, 1)
        self.exercise_bias = nn.Embedding(num_exercises, 1)

    def decode_from_mastery(
        self,
        target_mastery: torch.Tensor,
        q_vectors: torch.Tensor,
        target_exercise_ids: torch.Tensor,
    ) -> torch.Tensor:
        weights = q_vectors / q_vectors.sum(dim=1, keepdim=True).clamp_min(1.0)
        difficulty = self.concept_difficulty.weight.squeeze(-1)
        margin = (target_mastery - difficulty.unsqueeze(0)) * weights
        discrimination = F.softplus(
            self.exercise_discrimination(target_exercise_ids)
        ).squeeze(-1)
        logits = discrimination * margin.sum(dim=1) + self.exercise_bias(
            target_exercise_ids
        ).squeeze(-1)
        return torch.sigmoid(logits)

    def forward(
        self,
        state_map: torch.Tensor,
        q_matrix: torch.Tensor,
        target_student_ids: torch.Tensor,
        target_exercise_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mastery_logits = self.mastery_head(state_map).squeeze(-1)
        mastery = torch.sigmoid(mastery_logits)
        target_mastery = mastery[target_student_ids]
        cognitive_probs = self.decode_from_mastery(
            target_mastery,
            q_matrix[target_exercise_ids],
            target_exercise_ids,
        )
        return cognitive_probs, cognitive_probs, mastery
