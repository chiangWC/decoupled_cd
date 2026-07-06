from __future__ import annotations

import torch
import torch.nn as nn

from .decoupled_cdm import DecoupledCDM, DecoupledForwardOutput


class CountPriorBaseline(nn.Module):
    """
    B0 attribution baseline: a logistic model over smoothed train-history count
    statistics only — no embeddings, no graph, no propagation. It answers the
    question "how much of the mainline's gain comes from history frequency
    statistics rather than the decoupled TKC/UKC mechanism".

    logit = logit(global_rate) + w_s * d_student + w_e * d_exercise
            + w_tc * d_target_concept + w_c * d_concept + bias

    where each d_* is the capped log-odds delta of a smoothed accuracy rate
    against the global rate, exactly matching the v1 history-evidence prior's
    feature construction, but with learnable component weights.
    """

    def __init__(
        self,
        *,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        prior_weight: float = 5.0,
        component_cap: float = 3.0,
        **_unused_kwargs,
    ):
        super().__init__()
        if prior_weight <= 0.0:
            raise ValueError("prior_weight must be positive.")
        if component_cap <= 0.0:
            raise ValueError("component_cap must be positive.")
        self.prior_weight = float(prior_weight)
        self.component_cap = float(component_cap)
        self.component_weights = nn.Parameter(torch.tensor([1.0, 1.0, 0.6, 0.35]))
        self.bias = nn.Parameter(torch.zeros(1))
        # Present only because the training engine touches this attribute when
        # building its (weight-zero) difficulty regularization term.
        self.exercise_difficulty = nn.Embedding(num_exercises, 1)
        nn.init.zeros_(self.exercise_difficulty.weight)

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
    ) -> DecoupledForwardOutput:
        if target_student_ids is None or target_exercise_ids is None:
            raise ValueError("target_student_ids and target_exercise_ids are required for prediction.")
        if student_concept_evidence is None or exercise_evidence is None:
            raise ValueError("CountPriorBaseline requires student_concept_evidence and exercise_evidence tensors.")

        dtype = torch.float32
        prior_weight = student_exercise_mask.new_tensor(self.prior_weight)

        global_attempts = student_exercise_mask.sum().to(dtype=dtype)
        global_correct = (student_exercise_mask * response_matrix).sum().to(dtype=dtype)
        global_rate = torch.where(
            global_attempts > 0.0,
            global_correct / global_attempts.clamp_min(1.0),
            global_attempts.new_tensor(0.5),
        ).clamp(min=1e-6, max=1.0 - 1e-6)

        student_attempts = student_exercise_mask.sum(dim=1).to(dtype=dtype)[target_student_ids]
        student_correct = (student_exercise_mask * response_matrix).sum(dim=1).to(dtype=dtype)[target_student_ids]
        student_rate = DecoupledCDM._smooth_rate_with_global(
            correct=student_correct,
            attempts=student_attempts,
            global_rate=global_rate,
            prior_weight=prior_weight,
        )

        target_exercise_evidence = exercise_evidence[target_exercise_ids].to(dtype=dtype)
        exercise_rate = DecoupledCDM._smooth_rate_with_global(
            correct=target_exercise_evidence[:, 1],
            attempts=target_exercise_evidence[:, 0],
            global_rate=global_rate,
            prior_weight=prior_weight,
        )

        q_vectors = q_matrix[target_exercise_ids]
        q_mask = (q_vectors > 0).to(dtype=dtype)
        concept_counts = q_mask.sum(dim=1).clamp_min(1.0)
        target_evidence = student_concept_evidence[target_student_ids].to(dtype=dtype)
        target_attempts = target_evidence[..., 0] * q_mask
        target_correct = target_evidence[..., 1] * q_mask
        target_concept_rate = DecoupledCDM._smooth_rate_with_global(
            correct=target_correct,
            attempts=target_attempts,
            global_rate=global_rate,
            prior_weight=prior_weight,
        )
        target_concept_rate_mean = (target_concept_rate * q_mask).sum(dim=1) / concept_counts

        all_concept_attempts = student_concept_evidence[..., 0].to(dtype=dtype).sum(dim=0)
        all_concept_correct = student_concept_evidence[..., 1].to(dtype=dtype).sum(dim=0)
        concept_rate = DecoupledCDM._smooth_rate_with_global(
            correct=all_concept_correct,
            attempts=all_concept_attempts,
            global_rate=global_rate,
            prior_weight=prior_weight,
        )
        concept_rate_mean = (concept_rate.unsqueeze(0) * q_mask).sum(dim=1) / concept_counts

        deltas = torch.stack(
            [
                DecoupledCDM._bounded_logit_delta(student_rate, global_rate, self.component_cap),
                DecoupledCDM._bounded_logit_delta(exercise_rate, global_rate, self.component_cap),
                DecoupledCDM._bounded_logit_delta(target_concept_rate_mean, global_rate, self.component_cap),
                DecoupledCDM._bounded_logit_delta(concept_rate_mean, global_rate, self.component_cap),
            ],
            dim=-1,
        )
        logits = torch.logit(global_rate) + deltas @ self.component_weights + self.bias
        probs = torch.sigmoid(logits)

        zeros_like_probs = torch.zeros_like(probs)
        placeholder = probs.new_zeros((1, 1))
        return DecoupledForwardOutput(
            student_state=placeholder,
            tkc_states=probs.new_zeros((1, 1, 1)),
            ukc_states=probs.new_zeros((1, 1, 1)),
            tkc_weight=placeholder,
            concept_embeddings=placeholder,
            exercise_embeddings=placeholder,
            cognitive_probs=probs,
            probs=probs,
            guess_probs=zeros_like_probs,
            slip_probs=zeros_like_probs,
            difficulty=self.exercise_difficulty(target_exercise_ids).squeeze(-1),
        )
