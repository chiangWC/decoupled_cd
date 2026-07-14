from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NeuralProcessForwardOutput:
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


class InducedEvidencePosterior(nn.Module):
    """
    Module 1: compress a variable-size train-only response set into an induced
    evidence memory and a student posterior.

    Both the claimed induced-attention path and the summary-only control are
    always instantiated. Switching evidence_mode changes only the data path, so
    every ablation has the same state dict, initialization order, and allocated
    parameter count.
    """

    VALID_MODES = {"induced_posterior", "summary_control"}

    def __init__(
        self,
        *,
        dim: int,
        memory_slots: int,
        attention_heads: int,
        evidence_cap: float,
    ) -> None:
        super().__init__()
        if dim < 4:
            raise ValueError("dim must be at least 4.")
        if memory_slots < 1:
            raise ValueError("memory_slots must be positive.")
        if dim % attention_heads != 0:
            raise ValueError("dim must be divisible by attention_heads.")
        if evidence_cap <= 0.0:
            raise ValueError("evidence_cap must be positive.")

        self.dim = int(dim)
        self.memory_slots = int(memory_slots)
        self.evidence_cap = float(evidence_cap)

        self.response_feature_projection = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.token_norm = nn.LayerNorm(dim)
        self.global_summary_projection = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        self.inducing_slots = nn.Parameter(torch.empty(memory_slots, dim))
        self.induced_attention = nn.MultiheadAttention(
            dim,
            attention_heads,
            batch_first=True,
        )
        self.induced_norm = nn.LayerNorm(dim)
        self.induced_ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.induced_ff_norm = nn.LayerNorm(dim)

        self.control_slot_offsets = nn.Parameter(torch.empty(memory_slots, dim))
        self.summary_control = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.control_norm = nn.LayerNorm(dim)

        self.posterior_head = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 2),
        )
        nn.init.normal_(self.inducing_slots, std=0.02)
        nn.init.normal_(self.control_slot_offsets, std=0.02)

    def _global_summary(
        self,
        *,
        mask: torch.Tensor,
        responses: torch.Tensor,
        item_accuracy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attempts = mask.sum(dim=1)
        correct = (mask * responses).sum(dim=1)
        accuracy = torch.where(
            attempts > 0.0,
            correct / attempts.clamp_min(1.0),
            attempts.new_full(attempts.shape, 0.5),
        )
        coverage = attempts / float(max(mask.size(1), 1))
        mean_residual = (
            mask * (responses - item_accuracy.unsqueeze(0))
        ).sum(dim=1) / attempts.clamp_min(1.0)
        log_count = (
            torch.log1p(attempts) / torch.log1p(attempts.new_tensor(self.evidence_cap))
        ).clamp(max=1.0)
        summary = torch.stack(
            [accuracy.mul(2.0).sub(1.0), log_count, coverage, mean_residual],
            dim=-1,
        )
        return summary, attempts

    def forward(
        self,
        *,
        item_base: torch.Tensor,
        item_accuracy: torch.Tensor,
        item_log_count: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        evidence_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if evidence_mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported evidence_mode: {evidence_mode}")

        mask = student_exercise_mask.to(dtype=item_base.dtype)
        responses = response_matrix.to(dtype=item_base.dtype)
        item_accuracy = item_accuracy.to(dtype=item_base.dtype)
        item_log_count = item_log_count.to(dtype=item_base.dtype)

        response_features = torch.stack(
            [
                responses - item_accuracy.unsqueeze(0),
                responses.mul(2.0).sub(1.0),
                item_accuracy.unsqueeze(0).expand_as(responses).mul(2.0).sub(1.0),
                item_log_count.unsqueeze(0).expand_as(responses),
            ],
            dim=-1,
        )
        tokens = self.token_norm(
            item_base.unsqueeze(0) + self.response_feature_projection(response_features)
        )

        summary_features, attempts = self._global_summary(
            mask=mask,
            responses=responses,
            item_accuracy=item_accuracy,
        )
        summary = self.global_summary_projection(summary_features)

        # MultiheadAttention cannot accept a row whose keys are all masked.
        # Such a row receives one zero-valued safe token; the student summary
        # still carries the explicit no-evidence condition.
        padding_mask = mask <= 0.0
        no_history = padding_mask.all(dim=1)
        if torch.any(no_history):
            padding_mask = padding_mask.clone()
            tokens = tokens.clone()
            padding_mask[no_history, 0] = False
            tokens[no_history, 0] = 0.0

        induced_queries = self.inducing_slots.unsqueeze(0) + summary.unsqueeze(1)
        induced_update, _ = self.induced_attention(
            induced_queries,
            tokens,
            tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        induced_memory = self.induced_norm(induced_queries + induced_update)
        induced_memory = self.induced_ff_norm(
            induced_memory + self.induced_ff(induced_memory)
        )

        masked_mean = (tokens * mask.unsqueeze(-1)).sum(dim=1) / attempts.clamp_min(1.0).unsqueeze(-1)
        control_base = self.summary_control(torch.cat([summary, masked_mean], dim=-1))
        control_memory = self.control_norm(
            control_base.unsqueeze(1) + self.control_slot_offsets.unsqueeze(0)
        )

        memory = induced_memory if evidence_mode == "induced_posterior" else control_memory
        posterior_params = self.posterior_head(
            torch.cat([memory.mean(dim=1), summary], dim=-1)
        )
        posterior_mean, posterior_logvar = posterior_params.chunk(2, dim=-1)
        posterior_logvar = posterior_logvar.clamp(min=-6.0, max=2.0)
        if self.training:
            posterior = posterior_mean + torch.randn_like(posterior_mean) * torch.exp(
                0.5 * posterior_logvar
            )
        else:
            posterior = posterior_mean

        diagnostics = {
            "evidence_count": attempts,
            "posterior_mean": posterior_mean,
            "posterior_std": torch.exp(0.5 * posterior_logvar),
            "no_history": no_history.to(dtype=item_base.dtype),
        }
        return memory, posterior, posterior_logvar, diagnostics


class AttentiveConceptQueryCompletion(nn.Module):
    """
    Module 2: complete every student-concept state by querying the evidence
    memory. The latent-control path has the same input/output contract but no
    cross-attention.
    """

    VALID_MODES = {"cross_attention", "latent_control"}

    def __init__(self, *, dim: int, attention_heads: int) -> None:
        super().__init__()
        if dim % attention_heads != 0:
            raise ValueError("dim must be divisible by attention_heads.")

        self.local_evidence_projection = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.posterior_projection = nn.Linear(dim, dim)

        self.query_attention = nn.MultiheadAttention(
            dim,
            attention_heads,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(dim)
        self.query_ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.query_ff_norm = nn.LayerNorm(dim)

        self.latent_control = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.control_norm = nn.LayerNorm(dim)
        self.mastery_head = nn.Linear(dim, 1)
        self.reliability_head = nn.Linear(dim, 1)

    @staticmethod
    def _normalize_local_evidence(
        student_concept_evidence: torch.Tensor,
        evidence_cap: float,
    ) -> torch.Tensor:
        attempts = student_concept_evidence[..., 0]
        correct = student_concept_evidence[..., 1]
        incorrect = student_concept_evidence[..., 2]
        accuracy = student_concept_evidence[..., 3]
        log_attempts = (
            student_concept_evidence[..., 4]
            / torch.log1p(attempts.new_tensor(evidence_cap))
        ).clamp(max=1.0)
        seen = student_concept_evidence[..., 5]
        balance = (correct - incorrect) / attempts.clamp_min(1.0)
        return torch.stack(
            [accuracy.mul(2.0).sub(1.0), balance, log_attempts, seen],
            dim=-1,
        )

    def forward(
        self,
        *,
        concept_embeddings: torch.Tensor,
        student_concept_evidence: torch.Tensor,
        evidence_memory: torch.Tensor,
        posterior: torch.Tensor,
        posterior_logvar: torch.Tensor,
        query_mode: str,
        evidence_cap: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if query_mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported query_mode: {query_mode}")

        local_features = self._normalize_local_evidence(
            student_concept_evidence,
            evidence_cap,
        )
        local_state = self.local_evidence_projection(local_features)
        query_base = (
            concept_embeddings.unsqueeze(0)
            + local_state
            + self.posterior_projection(posterior).unsqueeze(1)
        )

        attended, _ = self.query_attention(
            query_base,
            evidence_memory,
            evidence_memory,
            need_weights=False,
        )
        attentive_state = self.query_norm(query_base + attended)
        attentive_state = self.query_ff_norm(
            attentive_state + self.query_ff(attentive_state)
        )

        control_state = self.control_norm(
            query_base + self.latent_control(query_base)
        )
        framework_state = (
            attentive_state if query_mode == "cross_attention" else control_state
        )

        mastery = torch.sigmoid(self.mastery_head(framework_state).squeeze(-1))
        posterior_uncertainty = posterior_logvar.exp().mean(dim=-1, keepdim=True)
        reliability_logit = self.reliability_head(framework_state).squeeze(-1)
        reliability = torch.sigmoid(
            reliability_logit - posterior_uncertainty
        )
        diagnostics = {
            "local_seen": student_concept_evidence[..., 5],
            "mean_reliability": reliability.mean(dim=1),
        }
        return framework_state, mastery, reliability, diagnostics


class NeuralProcessCDM(nn.Module):
    """
    Two-module cognitive diagnosis candidate.

    Student-specific information can reach the fixed diagnosis head only via
    framework_state. There is no student-ID embedding, legacy state, residual
    prediction branch, dataset router, or second prediction head.
    """

    def __init__(
        self,
        *,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        concept_dim: int = 64,
        evidence_mode: str = "induced_posterior",
        query_mode: str = "cross_attention",
        memory_slots: int = 8,
        attention_heads: int = 4,
        evidence_cap: float = 20.0,
        **_unused_kwargs,
    ) -> None:
        super().__init__()
        del num_students
        if evidence_mode not in InducedEvidencePosterior.VALID_MODES:
            raise ValueError(f"Unsupported evidence_mode: {evidence_mode}")
        if query_mode not in AttentiveConceptQueryCompletion.VALID_MODES:
            raise ValueError(f"Unsupported query_mode: {query_mode}")

        self.num_exercises = int(num_exercises)
        self.num_concepts = int(num_concepts)
        self.concept_dim = int(concept_dim)
        self.memory_slots = int(memory_slots)
        self.attention_heads = int(attention_heads)
        self.evidence_cap = float(evidence_cap)
        self.evidence_mode = evidence_mode
        self.query_mode = query_mode

        self.concept_embedding = nn.Embedding(num_concepts, concept_dim)
        self.exercise_embedding = nn.Embedding(num_exercises, concept_dim)
        self.q_item_projection = nn.Linear(concept_dim, concept_dim)
        self.item_evidence_projection = nn.Sequential(
            nn.Linear(2, concept_dim),
            nn.GELU(),
            nn.Linear(concept_dim, concept_dim),
        )

        self.evidence_posterior = InducedEvidencePosterior(
            dim=concept_dim,
            memory_slots=memory_slots,
            attention_heads=attention_heads,
            evidence_cap=evidence_cap,
        )
        self.concept_completion = AttentiveConceptQueryCompletion(
            dim=concept_dim,
            attention_heads=attention_heads,
        )

        self.diagnosis_head = nn.Sequential(
            nn.Linear(concept_dim * 3, concept_dim * 2),
            nn.ReLU(),
            nn.Linear(concept_dim * 2, concept_dim),
            nn.ReLU(),
            nn.Linear(concept_dim, 1),
        )
        self.exercise_difficulty = nn.Embedding(num_exercises, 1)
        self.mastery_scale_raw = nn.Parameter(torch.tensor(0.54132485))

        nn.init.xavier_uniform_(self.concept_embedding.weight)
        nn.init.xavier_uniform_(self.exercise_embedding.weight)
        nn.init.zeros_(self.exercise_difficulty.weight)

    @property
    def architecture_fingerprint(self) -> str:
        return (
            f"np_completion:d{self.concept_dim}:m{self.memory_slots}:"
            f"h{self.attention_heads}:diagnosis=pooled_ncf"
        )

    def initialization_hash(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _q_average(
        *,
        q_matrix: torch.Tensor,
        concept_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        weights = q_matrix.to(dtype=concept_embeddings.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        return weights @ concept_embeddings

    def diagnose(
        self,
        *,
        framework_state_for_rows: torch.Tensor,
        mastery_for_rows: torch.Tensor,
        q_matrix: torch.Tensor,
        target_exercise_ids: torch.Tensor,
        exercise_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_vectors = q_matrix[target_exercise_ids].to(dtype=framework_state_for_rows.dtype)
        q_weights = q_vectors / q_vectors.sum(dim=1, keepdim=True).clamp_min(1.0)
        target_state = torch.einsum(
            "rk,rkd->rd",
            q_weights,
            framework_state_for_rows,
        )
        target_mastery = (q_weights * mastery_for_rows).sum(dim=1).clamp(
            min=1e-5,
            max=1.0 - 1e-5,
        )
        item_state = exercise_embeddings[target_exercise_ids]
        interaction_features = torch.cat(
            [target_state, item_state, target_state * item_state],
            dim=-1,
        )
        residual_logit = self.diagnosis_head(interaction_features).squeeze(-1)
        difficulty = self.exercise_difficulty(target_exercise_ids).squeeze(-1)
        mastery_scale = F.softplus(self.mastery_scale_raw)
        cognitive_logit = (
            residual_logit
            + mastery_scale * torch.logit(target_mastery)
            - difficulty
        )
        probs = torch.sigmoid(cognitive_logit)
        return probs, probs, difficulty

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
    ) -> NeuralProcessForwardOutput:
        del concept_graph, student_tkc_mask, student_ukc_mask
        if prerequisite_graph is not None or similarity_graph is not None:
            raise ValueError("NeuralProcessCDM supports the unified single-graph protocol only.")
        if target_student_ids is None or target_exercise_ids is None:
            raise ValueError("target student and exercise IDs are required.")
        if student_concept_evidence is None or exercise_evidence is None:
            raise ValueError("NeuralProcessCDM requires train-only evidence tensors.")

        concept_embeddings = self.concept_embedding.weight
        q_item_state = self._q_average(
            q_matrix=q_matrix,
            concept_embeddings=concept_embeddings,
        )
        exercise_features = torch.stack(
            [
                exercise_evidence[:, 3].mul(2.0).sub(1.0),
                (
                    exercise_evidence[:, 4]
                    / torch.log1p(exercise_evidence.new_tensor(self.evidence_cap))
                ).clamp(max=1.0),
            ],
            dim=-1,
        )
        exercise_embeddings = (
            self.exercise_embedding.weight
            + self.q_item_projection(q_item_state)
            + self.item_evidence_projection(exercise_features)
        )

        if use_student_subset:
            student_indices, row_student_ids = torch.unique(
                target_student_ids,
                sorted=True,
                return_inverse=True,
            )
            selected_mask = student_exercise_mask.index_select(0, student_indices)
            selected_responses = response_matrix.index_select(0, student_indices)
            selected_concept_evidence = student_concept_evidence.index_select(
                0,
                student_indices,
            )
        else:
            row_student_ids = target_student_ids
            selected_mask = student_exercise_mask
            selected_responses = response_matrix
            selected_concept_evidence = student_concept_evidence

        evidence_memory, posterior, posterior_logvar, evidence_diagnostics = (
            self.evidence_posterior(
                item_base=exercise_embeddings,
                item_accuracy=exercise_evidence[:, 3],
                item_log_count=exercise_features[:, 1],
                student_exercise_mask=selected_mask,
                response_matrix=selected_responses,
                evidence_mode=self.evidence_mode,
            )
        )
        framework_state, mastery, reliability, completion_diagnostics = (
            self.concept_completion(
                concept_embeddings=concept_embeddings,
                student_concept_evidence=selected_concept_evidence,
                evidence_memory=evidence_memory,
                posterior=posterior,
                posterior_logvar=posterior_logvar,
                query_mode=self.query_mode,
                evidence_cap=self.evidence_cap,
            )
        )

        framework_state_for_rows = framework_state[row_student_ids]
        mastery_for_rows = mastery[row_student_ids]
        probs, cognitive_probs, difficulty = self.diagnose(
            framework_state_for_rows=framework_state_for_rows,
            mastery_for_rows=mastery_for_rows,
            q_matrix=q_matrix,
            target_exercise_ids=target_exercise_ids,
            exercise_embeddings=exercise_embeddings,
        )
        zeros = torch.zeros_like(probs)
        module_diagnostics = {
            **evidence_diagnostics,
            **completion_diagnostics,
        }
        return NeuralProcessForwardOutput(
            probs=probs,
            cognitive_probs=cognitive_probs,
            framework_state=framework_state,
            mastery=mastery,
            state_reliability=reliability,
            student_state=framework_state.mean(dim=1),
            concept_embeddings=concept_embeddings,
            exercise_embeddings=exercise_embeddings,
            guess_probs=zeros,
            slip_probs=zeros,
            difficulty=difficulty,
            module_diagnostics=module_diagnostics,
            architecture_fingerprint=self.architecture_fingerprint,
        )
