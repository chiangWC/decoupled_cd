from __future__ import annotations

import torch
import torch.nn as nn

from .decoupled_cdm import DecoupledForwardOutput
from .unified_v2_components import (
    CoverageAwareStateComposer,
    MonotonicDiagnosisDecoder,
    TestedKnowledgeEvidenceEncoder,
    UntestedKnowledgeInferenceNetwork,
)
from .unified_v2_spec import UnifiedArchitectureSpec


class UnifiedDecoupledCDM(nn.Module):
    def __init__(
        self,
        *,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        dim: int,
        architecture: UnifiedArchitectureSpec,
        evidence_cap: float = 20.0,
    ) -> None:
        super().__init__()
        self.num_students = num_students
        self.architecture = architecture
        self.evidence_encoder = TestedKnowledgeEvidenceEncoder(
            dim=dim,
            evidence_cap=evidence_cap,
        )
        self.concept_prior = (
            nn.Parameter(torch.zeros(num_concepts, dim))
            if architecture.inference == "prior"
            else None
        )
        self.inference_network = (
            UntestedKnowledgeInferenceNetwork(
                num_concepts=num_concepts,
                dim=dim,
                layers=2,
            )
            if architecture.inference == "graph"
            else None
        )
        self.state_composer = (
            CoverageAwareStateComposer(dim=dim)
            if architecture.composer == "coverage"
            else None
        )
        self.decoder = MonotonicDiagnosisDecoder(
            num_exercises=num_exercises,
            num_concepts=num_concepts,
            dim=dim,
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
    ) -> DecoupledForwardOutput:
        del student_concept_evidence
        state_target_student_ids = target_student_ids
        if use_student_subset:
            student_indices, state_target_student_ids = torch.unique(
                target_student_ids,
                sorted=True,
                return_inverse=True,
            )
            student_exercise_mask = student_exercise_mask[student_indices]
            response_matrix = response_matrix[student_indices]
            student_tkc_mask = student_tkc_mask[student_indices]
            student_ukc_mask = student_ukc_mask[student_indices]

        tested = self.evidence_encoder(
            q_matrix,
            student_exercise_mask,
            response_matrix,
            student_tkc_mask,
        )
        if self.architecture.inference == "prior":
            assert self.concept_prior is not None
            concept_prior = self.concept_prior
            ukc_states = (
                concept_prior.unsqueeze(0)
                * student_ukc_mask.unsqueeze(-1)
            )
            inferred_reliability = torch.zeros_like(
                tested.direct_reliability
            )
        else:
            assert self.inference_network is not None
            inferred = self.inference_network(
                tkc_states=tested.tkc_states,
                tkc_mask=student_tkc_mask,
                ukc_mask=student_ukc_mask,
                concept_graph=concept_graph,
                direct_reliability=tested.direct_reliability,
            )
            concept_prior = self.inference_network.concept_prior
            ukc_states = inferred.ukc_states
            inferred_reliability = inferred.inferred_reliability

        source_weights = None
        if self.architecture.composer == "coverage":
            assert self.state_composer is not None
            state_map, source_weights = self.state_composer(
                tkc_states=tested.tkc_states,
                ukc_states=ukc_states,
                concept_prior=concept_prior,
                tkc_mask=student_tkc_mask,
                direct_reliability=tested.direct_reliability,
                inferred_reliability=inferred_reliability,
            )
        else:
            state_map = tested.tkc_states + ukc_states
        cognitive_probs, probs, mastery = self.decoder(
            state_map,
            q_matrix,
            state_target_student_ids,
            target_exercise_ids,
        )

        q_vectors = q_matrix[target_exercise_ids]
        q_weights = q_vectors / q_vectors.sum(dim=1, keepdim=True).clamp_min(1.0)
        difficulty = q_weights @ self.decoder.concept_difficulty.weight.squeeze(-1)
        zeros = torch.zeros_like(probs)
        return DecoupledForwardOutput(
            student_state=state_map.mean(dim=1),
            tkc_states=tested.tkc_states,
            ukc_states=ukc_states,
            tkc_weight=tested.direct_reliability,
            concept_embeddings=concept_prior,
            exercise_embeddings=q_matrix @ concept_prior,
            cognitive_probs=cognitive_probs,
            probs=probs,
            guess_probs=zeros,
            slip_probs=zeros,
            difficulty=difficulty,
            mastery=mastery,
            source_weights=source_weights,
        )
