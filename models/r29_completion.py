from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Final

import torch
from torch import nn
import torch.nn.functional as F

from .r28_completion import DifficultyCalibratedResponseCompleter


R29_COMPLETER_MODES: Final[tuple[str, ...]] = (
    "marginal_anchor",
    "peer_completion",
    "wasserstein_flow",
    "meta_implicit",
    "direct_prior",
    "capacity_control",
)
R29_COMPLETION_OBJECTIVES: Final[tuple[str, ...]] = ("none", "masked_reconstruction")


@dataclass
class R29CompletionState:
    framework_state: torch.Tensor
    reliability: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


@dataclass
class R29ForwardOutput:
    probs: torch.Tensor
    framework_state: torch.Tensor
    mastery: torch.Tensor
    module_diagnostics: dict[str, torch.Tensor]
    architecture_fingerprint: str
    completion_reconstruction_loss: torch.Tensor | None
    cognitive_probs: torch.Tensor
    state_reliability: torch.Tensor
    student_state: torch.Tensor
    guess_probs: torch.Tensor
    slip_probs: torch.Tensor
    difficulty: torch.Tensor
    mastery_aux_logits: torch.Tensor | None = None


def evidence_features(
    evidence: torch.Tensor,
    *,
    evidence_cap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if evidence.ndim != 3 or evidence.size(-1) < 2:
        raise ValueError("student_concept_evidence must have shape [student, concept, >=2].")
    attempts = evidence[..., 0].float().clamp_min(0.0)
    correct = torch.minimum(evidence[..., 1].float().clamp_min(0.0), attempts)
    seen = (attempts > 0).float()
    accuracy = correct / attempts.clamp_min(1.0)
    balance = 2.0 * accuracy - 1.0
    confidence = (
        torch.log1p(attempts.clamp_max(evidence_cap)) / math.log1p(evidence_cap)
    ).clamp(0.0, 1.0)
    features = torch.stack([balance * confidence, confidence, seen], dim=-1)
    return features, attempts, accuracy, seen


def student_summary(
    attempts: torch.Tensor,
    accuracy: torch.Tensor,
    seen: torch.Tensor,
    *,
    evidence_cap: float,
) -> torch.Tensor:
    total_attempts = attempts.sum(dim=1)
    weighted_correct = (accuracy * attempts).sum(dim=1)
    success = weighted_correct / total_attempts.clamp_min(1.0)
    coverage = seen.mean(dim=1)
    confidence = (
        torch.log1p(total_attempts.clamp_max(evidence_cap * attempts.size(1)))
        / math.log1p(evidence_cap * attempts.size(1))
    ).clamp(0.0, 1.0)
    return torch.stack([2.0 * success - 1.0, coverage, confidence], dim=-1)


class DirectPriorCompleter(nn.Module):
    def __init__(self, *, dim: int, evidence_cap: float) -> None:
        super().__init__()
        self.evidence_cap = float(evidence_cap)
        self.evidence_projection = nn.Linear(3, dim)
        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        *,
        evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        student_indices: torch.Tensor,
    ) -> R29CompletionState:
        selected = evidence.index_select(0, student_indices)
        features, attempts, _, seen = evidence_features(selected, evidence_cap=self.evidence_cap)
        observed = concept_nodes.unsqueeze(0) + self.evidence_projection(features)
        prior = concept_nodes.unsqueeze(0).expand(selected.size(0), -1, -1)
        state = self.output_norm(torch.where(seen.unsqueeze(-1).bool(), observed, prior))
        reliability = features[..., 1] * seen
        return R29CompletionState(
            framework_state=state,
            reliability=reliability,
            diagnostics={
                "observed_ratio": seen.mean().detach(),
                "mean_reliability": reliability.mean().detach(),
                "mean_attempts": attempts.mean().detach(),
            },
        )


class MetaImplicitCompleter(nn.Module):
    """One implicit student-state function with either meta or amortized adaptation."""

    def __init__(
        self,
        *,
        dim: int,
        evidence_cap: float,
        meta_adaptation: bool,
        inner_steps: int = 3,
        inner_learning_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if inner_steps != 3:
            raise ValueError("r29 meta adaptation is fixed to three inner steps.")
        self.dim = int(dim)
        self.evidence_cap = float(evidence_cap)
        self.meta_adaptation = bool(meta_adaptation)
        self.inner_steps = int(inner_steps)
        self.inner_learning_rate = float(inner_learning_rate)

        self.context_initializer = nn.Sequential(
            nn.Linear(3, dim),
            nn.Tanh(),
            nn.LayerNorm(dim),
        )
        self.evidence_projection = nn.Linear(3, dim)
        self.implicit_function = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.Tanh(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )
        self.inner_readout = nn.Linear(dim, 1)
        self.context_update = nn.Sequential(
            nn.Linear(dim + 4, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.Tanh(),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(4, max(8, dim // 2)),
            nn.ReLU(),
            nn.Linear(max(8, dim // 2), 1),
        )

    def _states(
        self,
        *,
        context: torch.Tensor,
        concept_nodes: torch.Tensor,
        evidence_projection: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = context.size(0)
        num_concepts = concept_nodes.size(0)
        context_grid = context[:, None, :].expand(-1, num_concepts, -1)
        concept_grid = concept_nodes[None, :, :].expand(batch_size, -1, -1)
        return self.implicit_function(
            torch.cat(
                [
                    context_grid,
                    concept_grid,
                    context_grid * concept_grid,
                    evidence_projection,
                ],
                dim=-1,
            )
        )

    def forward(
        self,
        *,
        evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        student_indices: torch.Tensor,
    ) -> R29CompletionState:
        selected = evidence.index_select(0, student_indices)
        features, attempts, accuracy, seen = evidence_features(
            selected, evidence_cap=self.evidence_cap
        )
        summary = student_summary(
            attempts, accuracy, seen, evidence_cap=self.evidence_cap
        )
        evidence_projection = self.evidence_projection(features)
        context = self.context_initializer(summary)
        last_loss = context.new_zeros(())

        with torch.enable_grad():
            if not self.training:
                context = context.detach().requires_grad_(True)
                concept_nodes_inner = concept_nodes.detach()
                evidence_projection_inner = evidence_projection.detach()
            else:
                concept_nodes_inner = concept_nodes
                evidence_projection_inner = evidence_projection
            for _ in range(self.inner_steps):
                states = self._states(
                    context=context,
                    concept_nodes=concept_nodes_inner,
                    evidence_projection=evidence_projection_inner,
                )
                logits = self.inner_readout(states).squeeze(-1)
                weights = features[..., 1] * seen
                losses = F.binary_cross_entropy_with_logits(logits, accuracy, reduction="none")
                last_loss = (losses * weights).sum() / weights.sum().clamp_min(1.0)
                signed_error = ((torch.sigmoid(logits) - accuracy) * weights).sum(dim=1)
                signed_error = signed_error / weights.sum(dim=1).clamp_min(1.0)
                update_input = torch.cat([context, summary, signed_error.unsqueeze(-1)], dim=-1)
                learned_update = self.context_update(update_input)
                if self.meta_adaptation:
                    gradient = torch.autograd.grad(
                        last_loss,
                        context,
                        create_graph=self.training,
                        retain_graph=True,
                        allow_unused=False,
                    )[0]
                    preconditioner = 1.0 + 0.1 * learned_update
                    context = context - self.inner_learning_rate * preconditioner * gradient
                else:
                    context = context + self.inner_learning_rate * learned_update
            if not self.training:
                context = context.detach()

        state = self._states(
            context=context,
            concept_nodes=concept_nodes,
            evidence_projection=evidence_projection,
        )
        context_confidence = summary[:, 2:3].expand(-1, state.size(1))
        reliability_features = torch.stack(
            [seen, features[..., 1], context_confidence, seen * context_confidence], dim=-1
        )
        reliability = torch.sigmoid(self.reliability_head(reliability_features)).squeeze(-1)
        return R29CompletionState(
            framework_state=state,
            reliability=reliability,
            diagnostics={
                "observed_ratio": seen.mean().detach(),
                "mean_reliability": reliability.mean().detach(),
                "inner_reconstruction_loss": last_loss.detach(),
                "context_variance": context.var(unbiased=False).detach(),
                "state_variance": state.var(unbiased=False).detach(),
                "adaptation_steps": state.new_tensor(float(self.inner_steps)),
            },
        )


class R29CompletionCDM(nn.Module):
    """One replaceable state-completion module plus a fixed state-only diagnosis."""

    def __init__(
        self,
        *,
        num_students: int,
        num_exercises: int,
        num_concepts: int,
        concept_dim: int = 64,
        state_completer: str = "meta_implicit",
        completion_objective: str = "none",
        completion_mask_frac: float = 0.2,
        evidence_cap: float = 20.0,
        readout_dropout: float = 0.0,
        max_guess: float = 0.3,
        max_slip: float = 0.3,
    ) -> None:
        super().__init__()
        if state_completer not in R29_COMPLETER_MODES:
            raise ValueError(f"Unsupported r29 state completer: {state_completer}")
        if state_completer in {"peer_completion", "wasserstein_flow"}:
            raise ValueError(
                f"{state_completer} was rejected by its preregistered activation audit and cannot run."
            )
        if completion_objective not in R29_COMPLETION_OBJECTIVES:
            raise ValueError(f"Unsupported completion objective: {completion_objective}")
        if concept_dim < 4:
            raise ValueError("concept_dim must be at least 4.")
        if not 0.0 < completion_mask_frac < 1.0:
            raise ValueError("completion_mask_frac must be in (0, 1).")
        if evidence_cap <= 0:
            raise ValueError("evidence_cap must be positive.")
        if state_completer == "marginal_anchor" and completion_objective != "none":
            raise ValueError("The non-contribution marginal anchor uses response BCE only.")

        self.num_students = int(num_students)
        self.num_exercises = int(num_exercises)
        self.num_concepts = int(num_concepts)
        self.concept_dim = int(concept_dim)
        self.state_completer = state_completer
        self.completion_objective = completion_objective
        self.completion_mask_frac = float(completion_mask_frac)
        self.evidence_cap = float(evidence_cap)
        self.max_guess = float(max_guess)
        self.max_slip = float(max_slip)

        self.concept_embedding = nn.Embedding(num_concepts, concept_dim)
        self.exercise_embedding = nn.Embedding(num_exercises, concept_dim)
        self.exercise_difficulty = nn.Embedding(num_exercises, 1)
        self.q_projection = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim), nn.ReLU(), nn.LayerNorm(concept_dim)
        )
        self.cognitive_match = nn.Sequential(
            nn.Linear(concept_dim * 4, concept_dim),
            nn.ReLU(),
            nn.Dropout(readout_dropout),
            nn.Linear(concept_dim, 1),
        )
        self.guess_head = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim), nn.ReLU(), nn.Linear(concept_dim, 1)
        )
        self.slip_head = nn.Sequential(
            nn.Linear(concept_dim * 2, concept_dim), nn.ReLU(), nn.Linear(concept_dim, 1)
        )
        self.mastery_head = nn.Linear(concept_dim, 1)
        self.reconstruction_head = nn.Linear(concept_dim, 1)
        nn.init.xavier_uniform_(self.concept_embedding.weight)
        nn.init.xavier_uniform_(self.exercise_embedding.weight)
        nn.init.zeros_(self.exercise_difficulty.weight)

        if state_completer == "direct_prior":
            self.completer: nn.Module = DirectPriorCompleter(
                dim=concept_dim, evidence_cap=evidence_cap
            )
        elif state_completer == "marginal_anchor":
            self.completer = DifficultyCalibratedResponseCompleter(
                dim=concept_dim, evidence_cap=evidence_cap, calibrated=False
            )
        else:
            self.completer = MetaImplicitCompleter(
                dim=concept_dim,
                evidence_cap=evidence_cap,
                meta_adaptation=state_completer == "meta_implicit",
                inner_steps=3,
            )

    @property
    def architecture_fingerprint(self) -> str:
        payload = {
            "family": "r29_completion_v1",
            "state_completer": self.state_completer,
            "diagnosis": "q_conditioned_pooled_ncf_state_only",
            "student_id_embedding": False,
            "student_specific_bypass": False,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def common_initialization_hash(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def active_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _concept_nodes(self, q_matrix: torch.Tensor) -> torch.Tensor:
        q_binary = (q_matrix > 0).to(dtype=self.exercise_embedding.weight.dtype)
        counts = q_binary.sum(dim=0).clamp_min(1.0)
        exercise_side = q_binary.transpose(0, 1) @ self.exercise_embedding.weight / counts.unsqueeze(-1)
        return self.concept_embedding.weight + exercise_side

    def _exercise_nodes(self, q_matrix: torch.Tensor) -> torch.Tensor:
        q_binary = (q_matrix > 0).to(dtype=self.concept_embedding.weight.dtype)
        counts = q_binary.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.exercise_embedding.weight + q_binary @ self.concept_embedding.weight / counts

    def _complete(
        self,
        *,
        student_indices: torch.Tensor,
        student_concept_evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        response_matrix: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        q_matrix: torch.Tensor,
        exercise_evidence: torch.Tensor | None,
    ) -> R29CompletionState:
        if self.state_completer == "marginal_anchor":
            if exercise_evidence is None:
                raise ValueError("marginal_anchor requires train-only exercise evidence.")
            output = self.completer(
                evidence=student_concept_evidence,
                concept_nodes=concept_nodes,
                student_indices=student_indices,
                response_matrix=response_matrix,
                student_exercise_mask=student_exercise_mask,
                q_matrix=q_matrix,
                exercise_nodes=self._exercise_nodes(q_matrix),
                exercise_evidence=exercise_evidence,
            )
            return R29CompletionState(
                framework_state=output.framework_state,
                reliability=output.reliability,
                diagnostics=output.diagnostics,
            )
        return self.completer(
            evidence=student_concept_evidence,
            concept_nodes=concept_nodes,
            student_indices=student_indices,
        )

    def _masked_reconstruction_loss(
        self,
        *,
        evidence: torch.Tensor,
        concept_nodes: torch.Tensor,
        student_indices: torch.Tensor,
        response_matrix: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        q_matrix: torch.Tensor,
        exercise_evidence: torch.Tensor | None,
    ) -> torch.Tensor | None:
        selected = evidence.index_select(0, student_indices)
        observed = selected[..., 0] > 0
        hidden = observed & (torch.rand_like(selected[..., 0]) < self.completion_mask_frac)
        if not bool(hidden.any()):
            return None
        masked = evidence.clone()
        selected_masked = selected.clone()
        selected_masked[hidden] = 0
        masked.index_copy_(0, student_indices, selected_masked)
        reconstructed = self._complete(
            student_indices=student_indices,
            student_concept_evidence=masked,
            concept_nodes=concept_nodes,
            response_matrix=response_matrix,
            student_exercise_mask=student_exercise_mask,
            q_matrix=q_matrix,
            exercise_evidence=exercise_evidence,
        ).framework_state
        logits = self.reconstruction_head(reconstructed).squeeze(-1)
        target = selected[..., 1] / selected[..., 0].clamp_min(1.0)
        return F.binary_cross_entropy_with_logits(logits[hidden], target[hidden])

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
    ) -> R29ForwardOutput:
        del concept_graph, prerequisite_graph, similarity_graph
        del student_tkc_mask, student_ukc_mask
        if student_concept_evidence is None:
            raise ValueError("r29_completion requires train-only student_concept_evidence.")
        if target_student_ids is None or target_exercise_ids is None:
            raise ValueError("target student/exercise ids are required.")
        if target_student_ids.numel() != target_exercise_ids.numel():
            raise ValueError("target student/exercise tensors must have equal length.")
        if use_student_subset:
            student_indices, target_state_rows = torch.unique(
                target_student_ids, sorted=True, return_inverse=True
            )
        else:
            student_indices = torch.arange(
                student_concept_evidence.size(0), device=student_concept_evidence.device
            )
            target_state_rows = target_student_ids
        concept_nodes = self._concept_nodes(q_matrix)
        completion = self._complete(
            student_indices=student_indices,
            student_concept_evidence=student_concept_evidence,
            concept_nodes=concept_nodes,
            response_matrix=response_matrix,
            student_exercise_mask=student_exercise_mask,
            q_matrix=q_matrix,
            exercise_evidence=exercise_evidence,
        )
        framework_state = completion.framework_state
        mastery = torch.sigmoid(self.mastery_head(framework_state).squeeze(-1))

        q_vectors = (q_matrix.index_select(0, target_exercise_ids) > 0).to(framework_state.dtype)
        q_count = q_vectors.sum(dim=1, keepdim=True).clamp_min(1.0)
        target_states = framework_state.index_select(0, target_state_rows)
        q_state = (target_states * q_vectors.unsqueeze(-1)).sum(dim=1) / q_count
        q_concept = q_vectors @ self.concept_embedding.weight / q_count
        target_exercise = self.exercise_embedding(target_exercise_ids)
        q_repr = self.q_projection(torch.cat([q_concept, target_exercise], dim=-1))
        match = torch.cat(
            [q_state, q_repr, q_state * q_repr, torch.abs(q_state - q_repr)], dim=-1
        )
        difficulty = self.exercise_difficulty(target_exercise_ids).squeeze(-1)
        cognitive_probs = torch.sigmoid(self.cognitive_match(match).squeeze(-1) - difficulty)
        state_condition = torch.cat([q_state, q_repr], dim=-1)
        guess_probs = self.max_guess * torch.sigmoid(self.guess_head(state_condition).squeeze(-1))
        slip_probs = self.max_slip * torch.sigmoid(self.slip_head(state_condition).squeeze(-1))
        probs = (1.0 - slip_probs) * cognitive_probs + guess_probs * (1.0 - cognitive_probs)

        reconstruction_loss = None
        if self.training and self.completion_objective == "masked_reconstruction":
            reconstruction_loss = self._masked_reconstruction_loss(
                evidence=student_concept_evidence,
                concept_nodes=concept_nodes,
                student_indices=student_indices,
                response_matrix=response_matrix,
                student_exercise_mask=student_exercise_mask,
                q_matrix=q_matrix,
                exercise_evidence=exercise_evidence,
            )
        return R29ForwardOutput(
            probs=probs,
            framework_state=framework_state,
            mastery=mastery,
            module_diagnostics=completion.diagnostics,
            architecture_fingerprint=self.architecture_fingerprint,
            completion_reconstruction_loss=reconstruction_loss,
            cognitive_probs=cognitive_probs,
            state_reliability=completion.reliability,
            student_state=framework_state.mean(dim=1),
            guess_probs=guess_probs,
            slip_probs=slip_probs,
            difficulty=difficulty,
        )
