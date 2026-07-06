from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoupled_cdm import DecoupledForwardOutput
from .propagation_v2 import DecoupledPropagationV2


class DecoupledCDMV2(nn.Module):
    """
    V2 mainline model: the v1 core without any residual adapters (including the
    previously always-on cognitive difficulty adapter), plus three independently
    switchable modules for clean attribution experiments:

    - ``ukc_propagation`` (module 1): student-conditioned TKC->UKC propagation.
    - ``target_aware_readout`` (module 2): per-concept mastery head and a
      target-concept-aware readout replacing the global pooled NCF match.
    - ``monotonic_readout`` (module 3a, requires module 2): monotone-in-mastery
      interaction so that higher diagnosed mastery cannot lower the predicted
      probability.
    - ``bounded_gs`` (module 3b): bounded guess/slip conditioned only on the
      exercise representation, with no per-student parameters and no cognitive
      state input.

    The forward signature is a superset of DecoupledCDM's so the training
    engine drives both models unchanged. Dual-graph inputs are rejected.
    """

    def __init__(
        self,
        *,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        concept_dim: int = 64,
        student_fusion_mode: str = "adaptive",
        student_gate_prior_alpha: float = 1.0,
        student_gate_prior_beta: float = 1.0,
        gs_mode: str = "conditional",
        ukc_propagation: bool = False,
        ukc_propagation_layers: int = 1,
        ukc_evidence_cap: float = 20.0,
        target_aware_readout: bool = False,
        monotonic_readout: bool = False,
        bounded_gs: bool = False,
        gs_max_guess: float = 0.3,
        gs_max_slip: float = 0.3,
    ):
        super().__init__()
        if gs_mode not in {"constant", "conditional"}:
            raise ValueError(f"Unsupported gs_mode: {gs_mode}")
        if monotonic_readout and not target_aware_readout:
            raise ValueError("monotonic_readout requires target_aware_readout.")
        if not 0.0 < gs_max_guess < 1.0 or not 0.0 < gs_max_slip < 1.0:
            raise ValueError("gs_max_guess and gs_max_slip must be in (0, 1).")
        self.gs_mode = gs_mode
        self.ukc_propagation = ukc_propagation
        self.target_aware_readout = target_aware_readout
        self.monotonic_readout = monotonic_readout
        self.bounded_gs = bounded_gs
        self.gs_max_guess = float(gs_max_guess)
        self.gs_max_slip = float(gs_max_slip)
        self.ukc_evidence_cap = float(ukc_evidence_cap)

        self.exercise_embedding = nn.Embedding(num_exercises, concept_dim)
        self.exercise_difficulty = nn.Embedding(num_exercises, 1)
        self.concept_embedding = nn.Embedding(num_concepts, concept_dim)
        self.q_pool_gate = nn.Linear(concept_dim, 1, bias=False)
        self.exercise_q_fusion = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, concept_dim),
        )
        self.q_pool_mlp = nn.Sequential(
            nn.Linear(concept_dim, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, concept_dim),
        )
        self.propagation = DecoupledPropagationV2(
            concept_dim,
            student_fusion_mode=student_fusion_mode,
            student_gate_prior_alpha=student_gate_prior_alpha,
            student_gate_prior_beta=student_gate_prior_beta,
            ukc_personalization=ukc_propagation,
            ukc_layers=ukc_propagation_layers,
            ukc_evidence_cap=ukc_evidence_cap,
        )

        if target_aware_readout:
            self.mastery_head = nn.Linear(concept_dim, 1)
            self.concept_attention = nn.Linear(concept_dim + 2, 1)
            if monotonic_readout:
                self.concept_difficulty = nn.Embedding(num_concepts, 1)
                nn.init.zeros_(self.concept_difficulty.weight)
                self.exercise_discrimination = nn.Embedding(num_exercises, 1)
                nn.init.zeros_(self.exercise_discrimination.weight)
                self.mono_scale_raw = nn.Parameter(torch.tensor(1.5))
                self.concept_score_mlp = None
            else:
                self.concept_difficulty = None
                self.exercise_discrimination = None
                self.mono_scale_raw = None
                self.concept_score_mlp = nn.Sequential(
                    nn.Linear(concept_dim * 2 + 3, concept_dim),
                    nn.ReLU(),
                    nn.Linear(concept_dim, 1),
                )
            self.cognitive_match_mlp = None
        else:
            self.mastery_head = None
            self.concept_attention = None
            self.concept_difficulty = None
            self.exercise_discrimination = None
            self.mono_scale_raw = None
            self.concept_score_mlp = None
            self.cognitive_match_mlp = nn.Sequential(
                nn.Linear(concept_dim * 4, concept_dim),
                nn.ReLU(),
                nn.Linear(concept_dim, 1),
            )

        if bounded_gs:
            self.guess_logit = None
            self.slip_logit = None
            self.guess_mlp = nn.Sequential(
                nn.Linear(concept_dim, concept_dim),
                nn.ReLU(),
                nn.Linear(concept_dim, 1),
            )
            self.slip_mlp = nn.Sequential(
                nn.Linear(concept_dim, concept_dim),
                nn.ReLU(),
                nn.Linear(concept_dim, 1),
            )
        else:
            self.guess_logit = nn.Embedding(num_students, 1)
            self.slip_logit = nn.Embedding(num_students, 1)
            self.guess_mlp = nn.Sequential(
                nn.Linear(concept_dim * 2, concept_dim),
                nn.ReLU(),
                nn.Linear(concept_dim, 1),
            )
            self.slip_mlp = nn.Sequential(
                nn.Linear(concept_dim * 2, concept_dim),
                nn.ReLU(),
                nn.Linear(concept_dim, 1),
            )

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
        if prerequisite_graph is not None or similarity_graph is not None:
            raise ValueError("DecoupledCDMV2 supports single-graph mode only.")
        if target_student_ids is None or target_exercise_ids is None:
            raise ValueError("target_student_ids and target_exercise_ids are required for prediction.")

        concept_embeddings = self.concept_embedding.weight
        exercise_embeddings = self.exercise_embedding.weight
        student_indices = None
        state_target_student_ids = target_student_ids
        if use_student_subset:
            student_indices, state_target_student_ids = torch.unique(
                target_student_ids,
                sorted=True,
                return_inverse=True,
            )
        propagated = self.propagation(
            concept_embeddings=concept_embeddings,
            exercise_embeddings=exercise_embeddings,
            q_matrix=q_matrix,
            concept_graph=concept_graph,
            student_exercise_mask=student_exercise_mask,
            response_matrix=response_matrix,
            student_tkc_mask=student_tkc_mask,
            student_ukc_mask=student_ukc_mask,
            student_concept_evidence=student_concept_evidence,
            student_indices=student_indices,
        )

        student_state = propagated.student_state[state_target_student_ids]
        q_vectors = q_matrix[target_exercise_ids]
        target_exercise_embeddings = exercise_embeddings[target_exercise_ids]
        q_repr = self._build_exercise_q_representation(
            q_vectors=q_vectors,
            concept_embeddings=concept_embeddings,
            exercise_embeddings=target_exercise_embeddings,
        )
        difficulty = self.exercise_difficulty(target_exercise_ids).squeeze(-1)

        mastery = None
        if self.target_aware_readout:
            per_concept_states = propagated.tkc_states + propagated.ukc_states
            mastery = torch.sigmoid(self.mastery_head(per_concept_states).squeeze(-1))
            cognitive_logits = self._build_target_aware_logits(
                q_vectors=q_vectors,
                mastery=mastery,
                per_concept_states=per_concept_states,
                concept_embeddings=concept_embeddings,
                state_target_student_ids=state_target_student_ids,
                target_student_ids=target_student_ids,
                target_exercise_ids=target_exercise_ids,
                student_concept_evidence=student_concept_evidence,
                student_tkc_mask=student_tkc_mask,
                difficulty=difficulty,
            )
        else:
            match_inputs = torch.cat(
                [student_state, q_repr, student_state * q_repr, torch.abs(student_state - q_repr)],
                dim=-1,
            )
            cognitive_logits = self.cognitive_match_mlp(match_inputs).squeeze(-1) - difficulty
        cognitive_probs = torch.sigmoid(cognitive_logits)

        if self.bounded_gs:
            guess_probs = self.gs_max_guess * torch.sigmoid(self.guess_mlp(q_repr).squeeze(-1))
            slip_probs = self.gs_max_slip * torch.sigmoid(self.slip_mlp(q_repr).squeeze(-1))
        else:
            guess_logits = self.guess_logit(target_student_ids).squeeze(-1)
            slip_logits = self.slip_logit(target_student_ids).squeeze(-1)
            if self.gs_mode == "conditional":
                non_cognitive_inputs = torch.cat([student_state, q_repr], dim=-1)
                guess_logits = guess_logits + self.guess_mlp(non_cognitive_inputs).squeeze(-1)
                slip_logits = slip_logits + self.slip_mlp(non_cognitive_inputs).squeeze(-1)
            guess_probs = torch.sigmoid(guess_logits)
            slip_probs = torch.sigmoid(slip_logits)

        probs = (1.0 - slip_probs) * cognitive_probs + guess_probs * (1.0 - cognitive_probs)
        return DecoupledForwardOutput(
            student_state=propagated.student_state,
            tkc_states=propagated.tkc_states,
            ukc_states=propagated.ukc_states,
            tkc_weight=propagated.tkc_weight,
            concept_embeddings=concept_embeddings,
            exercise_embeddings=exercise_embeddings,
            cognitive_probs=cognitive_probs,
            probs=probs,
            guess_probs=guess_probs,
            slip_probs=slip_probs,
            difficulty=difficulty,
            mastery=mastery,
        )

    def _build_exercise_q_representation(
        self,
        *,
        q_vectors: torch.Tensor,
        concept_embeddings: torch.Tensor,
        exercise_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        raw_scores = self.q_pool_gate(concept_embeddings).squeeze(-1)
        masked_scores = raw_scores.unsqueeze(0).expand(q_vectors.size(0), -1).masked_fill(q_vectors <= 0, -1e9)
        attn = torch.softmax(masked_scores, dim=1)
        pooled = attn @ concept_embeddings
        fused = self.exercise_q_fusion(torch.cat([pooled, exercise_embeddings], dim=-1))
        return self.q_pool_mlp(fused)

    def _build_target_aware_logits(
        self,
        *,
        q_vectors: torch.Tensor,
        mastery: torch.Tensor,
        per_concept_states: torch.Tensor,
        concept_embeddings: torch.Tensor,
        state_target_student_ids: torch.Tensor,
        target_student_ids: torch.Tensor,
        target_exercise_ids: torch.Tensor,
        student_concept_evidence: torch.Tensor | None,
        student_tkc_mask: torch.Tensor,
        difficulty: torch.Tensor,
    ) -> torch.Tensor:
        q_mask = q_vectors > 0
        max_concepts = max(int(q_mask.sum(dim=1).max().item()), 1)
        scores, indices = q_vectors.topk(k=max_concepts, dim=1)
        valid = scores > 0
        safe_indices = torch.where(valid, indices, indices.new_zeros(indices.shape))

        batch_rows = state_target_student_ids.unsqueeze(1)
        gathered_states = per_concept_states[batch_rows, safe_indices]
        gathered_mastery = mastery[batch_rows, safe_indices]
        gathered_concept_emb = concept_embeddings[safe_indices]

        full_rows = target_student_ids.unsqueeze(1)
        if student_concept_evidence is not None:
            attempts = student_concept_evidence[full_rows, safe_indices, 0].to(mastery.dtype)
        else:
            attempts = student_tkc_mask[full_rows, safe_indices].to(mastery.dtype)
        seen = (attempts > 0).to(mastery.dtype)
        confidence = (
            torch.log1p(attempts)
            / torch.log1p(torch.tensor(self.ukc_evidence_cap, dtype=mastery.dtype, device=mastery.device))
        ).clamp(min=0.0, max=1.0)

        attention_inputs = torch.cat(
            [gathered_concept_emb, seen.unsqueeze(-1), confidence.unsqueeze(-1)],
            dim=-1,
        )
        attention_scores = self.concept_attention(attention_inputs).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~valid, -1e9)
        attention = torch.softmax(attention_scores, dim=1) * valid.to(mastery.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-6)

        if self.monotonic_readout:
            concept_difficulty = self.concept_difficulty(safe_indices).squeeze(-1)
            item_difficulty = torch.sigmoid(difficulty.unsqueeze(1) + concept_difficulty)
            per_concept_signal = gathered_mastery - item_difficulty
            discrimination = 1.0 + F.softplus(
                self.exercise_discrimination(target_exercise_ids).squeeze(-1)
            )
            scale = F.softplus(self.mono_scale_raw)
            return discrimination * scale * (attention * per_concept_signal).sum(dim=1)

        score_inputs = torch.cat(
            [
                gathered_states,
                gathered_concept_emb,
                gathered_mastery.unsqueeze(-1),
                seen.unsqueeze(-1),
                confidence.unsqueeze(-1),
            ],
            dim=-1,
        )
        per_concept_scores = self.concept_score_mlp(score_inputs).squeeze(-1)
        return (attention * per_concept_scores).sum(dim=1) - difficulty
