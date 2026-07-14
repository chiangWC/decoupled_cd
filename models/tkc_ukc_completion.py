from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TKCUKCForwardOutput:
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


class RelationalTKCEvidence(nn.Module):
    """
    Module 1: turn correct/incorrect exercise relations into observed concept
    states. The raw-statistics path is the capacity-matched clean control.
    """

    VALID_MODES = {"relational", "raw_statistics"}

    def __init__(self, *, dim: int, evidence_cap: float) -> None:
        super().__init__()
        if evidence_cap <= 0.0:
            raise ValueError("evidence_cap must be positive.")
        self.evidence_cap = float(evidence_cap)

        self.correct_relation = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.incorrect_relation = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.raw_statistics_control = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.state_norm = nn.LayerNorm(dim)

    def active_parameter_counts(self) -> dict[str, int]:
        full = sum(parameter.numel() for parameter in self.correct_relation.parameters())
        full += sum(parameter.numel() for parameter in self.incorrect_relation.parameters())
        control = sum(
            parameter.numel() for parameter in self.raw_statistics_control.parameters()
        )
        return {"relational": full, "raw_statistics": control}

    def _raw_features(self, evidence: torch.Tensor) -> torch.Tensor:
        attempts = evidence[..., 0]
        correct = evidence[..., 1]
        incorrect = evidence[..., 2]
        accuracy = evidence[..., 3]
        confidence = (
            evidence[..., 4] / math.log1p(self.evidence_cap)
        ).clamp(max=1.0)
        balance = (correct - incorrect) / attempts.clamp_min(1.0)
        seen = evidence[..., 5]
        return torch.stack(
            [accuracy.mul(2.0).sub(1.0), balance, confidence, seen],
            dim=-1,
        )

    def forward(
        self,
        *,
        concept_embeddings: torch.Tensor,
        exercise_embeddings: torch.Tensor,
        q_matrix: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        student_concept_evidence: torch.Tensor,
        evidence_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if evidence_mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported evidence_mode: {evidence_mode}")

        dtype = concept_embeddings.dtype
        mask = student_exercise_mask.to(dtype=dtype)
        responses = response_matrix.to(dtype=dtype)
        q_binary = (q_matrix > 0.0).to(dtype=dtype)
        edges = torch.nonzero(q_binary > 0.0, as_tuple=False)
        if edges.numel() == 0:
            raise ValueError("Relational TKC evidence requires at least one Q edge.")
        edge_exercises = edges[:, 0]
        edge_concepts = edges[:, 1]
        edge_weights = q_binary[edge_exercises, edge_concepts]

        correct_item_state = self.correct_relation(exercise_embeddings)
        incorrect_item_state = self.incorrect_relation(exercise_embeddings)
        exercise_messages = (
            responses.unsqueeze(-1) * correct_item_state.unsqueeze(0)
            + (1.0 - responses).unsqueeze(-1) * incorrect_item_state.unsqueeze(0)
        )
        exercise_messages = exercise_messages * mask.unsqueeze(-1)
        edge_messages = exercise_messages.index_select(1, edge_exercises)
        edge_messages = edge_messages * edge_weights.view(1, -1, 1)

        batch_size = mask.size(0)
        num_concepts = concept_embeddings.size(0)
        dim = concept_embeddings.size(1)
        relational_sum = exercise_messages.new_zeros((batch_size, num_concepts, dim))
        scatter_index = edge_concepts.view(1, -1, 1).expand(batch_size, -1, dim)
        relational_sum.scatter_add_(1, scatter_index, edge_messages)

        edge_observed = mask.index_select(1, edge_exercises) * edge_weights.unsqueeze(0)
        relation_count = mask.new_zeros((batch_size, num_concepts))
        relation_count.scatter_add_(
            1,
            edge_concepts.view(1, -1).expand(batch_size, -1),
            edge_observed,
        )
        relational_mean = relational_sum / relation_count.clamp_min(1.0).unsqueeze(-1)
        relational_state = self.state_norm(
            concept_embeddings.unsqueeze(0) + relational_mean
        )

        raw_features = self._raw_features(student_concept_evidence.to(dtype=dtype))
        raw_state = self.state_norm(
            concept_embeddings.unsqueeze(0)
            + self.raw_statistics_control(raw_features)
        )
        tkc_state = relational_state if evidence_mode == "relational" else raw_state
        observed_mask = student_concept_evidence[..., 5].to(dtype=dtype)
        reliability = (
            student_concept_evidence[..., 4].to(dtype=dtype)
            / math.log1p(self.evidence_cap)
        ).clamp(min=0.0, max=1.0)
        diagnostics = {
            "observed_concept_count": observed_mask.sum(dim=1),
            "mean_tkc_reliability": (
                reliability.sum(dim=1) / observed_mask.sum(dim=1).clamp_min(1.0)
            ),
        }
        return tkc_state, observed_mask, reliability, diagnostics


class PersonalizedUKCCompletion(nn.Module):
    """
    Module 2: complete each unobserved concept by directly attending to that
    student's observed TKC states. The global-control path sees the same TKC
    input but removes query-specific attention.
    """

    VALID_MODES = {"personalized_attention", "global_control"}

    def __init__(
        self,
        *,
        dim: int,
        attention_heads: int,
        query_chunk_size: int,
    ) -> None:
        super().__init__()
        if attention_heads < 1 or dim % attention_heads != 0:
            raise ValueError("dim must be divisible by positive attention_heads.")
        if query_chunk_size < 1:
            raise ValueError("query_chunk_size must be positive.")
        self.dim = int(dim)
        self.attention_heads = int(attention_heads)
        self.head_dim = dim // attention_heads
        self.query_chunk_size = int(query_chunk_size)

        self.global_projection = nn.Linear(dim, dim)
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.attention_norm = nn.LayerNorm(dim)
        self.attention_ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.attention_ff_norm = nn.LayerNorm(dim)

        self.global_control = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.control_norm = nn.LayerNorm(dim)
        self.mastery_head = nn.Linear(dim, 1)
        self.missing_reliability_head = nn.Linear(dim, 1)

    def active_parameter_counts(self) -> dict[str, int]:
        full_modules = [
            self.query_projection,
            self.key_projection,
            self.value_projection,
            self.output_projection,
            self.attention_norm,
            self.attention_ff,
            self.attention_ff_norm,
        ]
        full = sum(
            parameter.numel()
            for module in full_modules
            for parameter in module.parameters()
        )
        control = sum(parameter.numel() for parameter in self.global_control.parameters())
        control += sum(parameter.numel() for parameter in self.control_norm.parameters())
        return {"personalized_attention": full, "global_control": control}

    def _personalized_attention(
        self,
        *,
        query_base: torch.Tensor,
        tkc_state: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_concepts, _ = tkc_state.shape
        heads = self.attention_heads
        head_dim = self.head_dim
        queries = self.query_projection(query_base).view(
            batch_size,
            num_concepts,
            heads,
            head_dim,
        )
        keys = self.key_projection(tkc_state).view(
            batch_size,
            num_concepts,
            heads,
            head_dim,
        )
        values = self.value_projection(tkc_state).view(
            batch_size,
            num_concepts,
            heads,
            head_dim,
        )
        valid_keys = observed_mask > 0.0
        contexts: list[torch.Tensor] = []
        scale = math.sqrt(float(head_dim))
        for start in range(0, num_concepts, self.query_chunk_size):
            query_chunk = queries[:, start : start + self.query_chunk_size]
            scores = torch.einsum("bqhd,bkhd->bhqk", query_chunk, keys) / scale
            scores = scores.masked_fill(
                ~valid_keys[:, None, None, :],
                -1.0e4,
            )
            weights = torch.softmax(scores, dim=-1)
            weights = weights * valid_keys[:, None, None, :].to(dtype=weights.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            context = torch.einsum("bhqk,bkhd->bqhd", weights, values)
            contexts.append(context.reshape(batch_size, -1, self.dim))
        return torch.cat(contexts, dim=1)

    def forward(
        self,
        *,
        concept_embeddings: torch.Tensor,
        tkc_state: torch.Tensor,
        observed_mask: torch.Tensor,
        tkc_reliability: torch.Tensor,
        completion_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if completion_mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported completion_mode: {completion_mode}")

        observed_weight = observed_mask.unsqueeze(-1)
        global_state = (tkc_state * observed_weight).sum(dim=1)
        global_state = global_state / observed_weight.sum(dim=1).clamp_min(1.0)
        query_base = (
            concept_embeddings.unsqueeze(0)
            + self.global_projection(global_state).unsqueeze(1)
        )

        attention_context = self._personalized_attention(
            query_base=query_base,
            tkc_state=tkc_state,
            observed_mask=observed_mask,
        )
        attentive_missing = self.attention_norm(
            query_base + self.output_projection(attention_context)
        )
        attentive_missing = self.attention_ff_norm(
            attentive_missing + self.attention_ff(attentive_missing)
        )

        expanded_global = global_state.unsqueeze(1).expand_as(query_base)
        controlled_missing = self.control_norm(
            query_base
            + self.global_control(torch.cat([query_base, expanded_global], dim=-1))
        )
        missing_state = (
            attentive_missing
            if completion_mode == "personalized_attention"
            else controlled_missing
        )
        framework_state = torch.where(
            observed_mask.unsqueeze(-1) > 0.0,
            tkc_state,
            missing_state,
        )
        mastery = torch.sigmoid(self.mastery_head(framework_state).squeeze(-1))
        missing_reliability = torch.sigmoid(
            self.missing_reliability_head(missing_state).squeeze(-1)
        )
        reliability = torch.where(
            observed_mask > 0.0,
            tkc_reliability,
            missing_reliability,
        )
        diagnostics = {
            "missing_concept_count": (1.0 - observed_mask).sum(dim=1),
            "mean_ukc_reliability": (
                (missing_reliability * (1.0 - observed_mask)).sum(dim=1)
                / (1.0 - observed_mask).sum(dim=1).clamp_min(1.0)
            ),
        }
        return framework_state, mastery, reliability, diagnostics


class TKCUKCCompletionCDM(nn.Module):
    """
    Two-module TKC/UKC candidate with a fixed Q-conditioned pooled NCF
    diagnosis. framework_state is the only student-specific prediction input.
    """

    def __init__(
        self,
        *,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        concept_dim: int = 64,
        evidence_mode: str = "relational",
        completion_mode: str = "personalized_attention",
        attention_heads: int = 4,
        query_chunk_size: int = 64,
        evidence_cap: float = 20.0,
        **_unused_kwargs,
    ) -> None:
        super().__init__()
        del num_students
        if evidence_mode not in RelationalTKCEvidence.VALID_MODES:
            raise ValueError(f"Unsupported evidence_mode: {evidence_mode}")
        if completion_mode not in PersonalizedUKCCompletion.VALID_MODES:
            raise ValueError(f"Unsupported completion_mode: {completion_mode}")

        self.concept_dim = int(concept_dim)
        self.attention_heads = int(attention_heads)
        self.query_chunk_size = int(query_chunk_size)
        self.evidence_cap = float(evidence_cap)
        self.evidence_mode = evidence_mode
        self.completion_mode = completion_mode

        self.concept_embedding = nn.Embedding(num_concepts, concept_dim)
        self.exercise_embedding = nn.Embedding(num_exercises, concept_dim)
        self.q_item_projection = nn.Linear(concept_dim, concept_dim)
        self.item_evidence_projection = nn.Sequential(
            nn.Linear(2, concept_dim),
            nn.GELU(),
            nn.Linear(concept_dim, concept_dim),
        )
        self.tkc_evidence = RelationalTKCEvidence(
            dim=concept_dim,
            evidence_cap=evidence_cap,
        )
        self.ukc_completion = PersonalizedUKCCompletion(
            dim=concept_dim,
            attention_heads=attention_heads,
            query_chunk_size=query_chunk_size,
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
            f"tkc_ukc_completion:d{self.concept_dim}:h{self.attention_heads}:"
            f"chunk{self.query_chunk_size}:diagnosis=pooled_ncf"
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
        target_state = torch.einsum("rk,rkd->rd", q_weights, framework_state_for_rows)
        target_mastery = (q_weights * mastery_for_rows).sum(dim=1).clamp(
            min=1.0e-5,
            max=1.0 - 1.0e-5,
        )
        item_state = exercise_embeddings[target_exercise_ids]
        residual = self.diagnosis_head(
            torch.cat([target_state, item_state, target_state * item_state], dim=-1)
        ).squeeze(-1)
        difficulty = self.exercise_difficulty(target_exercise_ids).squeeze(-1)
        logit = (
            residual
            + F.softplus(self.mastery_scale_raw) * torch.logit(target_mastery)
            - difficulty
        )
        probs = torch.sigmoid(logit)
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
    ) -> TKCUKCForwardOutput:
        del concept_graph, student_tkc_mask, student_ukc_mask
        if prerequisite_graph is not None or similarity_graph is not None:
            raise ValueError("TKCUKCCompletionCDM supports the single-graph protocol only.")
        if target_student_ids is None or target_exercise_ids is None:
            raise ValueError("target student and exercise IDs are required.")
        if student_concept_evidence is None or exercise_evidence is None:
            raise ValueError("TKCUKCCompletionCDM requires train-only evidence tensors.")

        concept_embeddings = self.concept_embedding.weight
        q_item_state = self._q_average(
            q_matrix=q_matrix,
            concept_embeddings=concept_embeddings,
        )
        exercise_features = torch.stack(
            [
                exercise_evidence[:, 3].mul(2.0).sub(1.0),
                (
                    exercise_evidence[:, 4] / math.log1p(self.evidence_cap)
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
            selected_evidence = student_concept_evidence.index_select(0, student_indices)
        else:
            row_student_ids = target_student_ids
            selected_mask = student_exercise_mask
            selected_responses = response_matrix
            selected_evidence = student_concept_evidence

        tkc_state, observed_mask, tkc_reliability, tkc_diagnostics = self.tkc_evidence(
            concept_embeddings=concept_embeddings,
            exercise_embeddings=exercise_embeddings,
            q_matrix=q_matrix,
            student_exercise_mask=selected_mask,
            response_matrix=selected_responses,
            student_concept_evidence=selected_evidence,
            evidence_mode=self.evidence_mode,
        )
        framework_state, mastery, reliability, ukc_diagnostics = self.ukc_completion(
            concept_embeddings=concept_embeddings,
            tkc_state=tkc_state,
            observed_mask=observed_mask,
            tkc_reliability=tkc_reliability,
            completion_mode=self.completion_mode,
        )
        probs, cognitive_probs, difficulty = self.diagnose(
            framework_state_for_rows=framework_state[row_student_ids],
            mastery_for_rows=mastery[row_student_ids],
            q_matrix=q_matrix,
            target_exercise_ids=target_exercise_ids,
            exercise_embeddings=exercise_embeddings,
        )
        zeros = torch.zeros_like(probs)
        return TKCUKCForwardOutput(
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
            module_diagnostics={**tkc_diagnostics, **ukc_diagnostics},
            architecture_fingerprint=self.architecture_fingerprint,
        )
