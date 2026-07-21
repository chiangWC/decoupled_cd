from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RelationQueryOutput:
    target_student_state: torch.Tensor
    reliability: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class StudentConditionedRelationQuery(nn.Module):
    """Query train-only response memory through typed curriculum relations.

    Graph tensors are fixed, non-persistent buffers.  Trainable parameters are
    therefore identical across real, Q-only, and rewired graph variants.
    """

    NUM_Q_RELATIONS = 2
    NUM_METADATA_RELATIONS = 7
    RESPONSE_CHANNELS = 4
    PATH_FEATURES = 5

    def __init__(
        self,
        *,
        dim: int,
        num_exercises: int,
        num_concepts: int,
        num_aux_nodes: int,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        hops: int = 4,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if hops <= 0:
            raise ValueError("hops must be positive.")
        if edge_index.ndim != 2 or tuple(edge_index.shape[:1]) != (2,):
            raise ValueError("edge_index must have shape [2, num_edges].")
        if edge_type.ndim != 1 or edge_type.numel() != edge_index.shape[1]:
            raise ValueError("edge_type must align with edge_index.")
        if edge_type.numel() == 0:
            raise ValueError("relation query requires Q edges.")
        if int(edge_type.min()) < 0 or int(edge_type.max()) >= (
            self.NUM_Q_RELATIONS + self.NUM_METADATA_RELATIONS
        ):
            raise ValueError("edge_type contains an unsupported relation slot.")

        self.dim = int(dim)
        self.num_exercises = int(num_exercises)
        self.num_concepts = int(num_concepts)
        self.num_aux_nodes = int(num_aux_nodes)
        self.num_nodes = (
            self.num_exercises + self.num_concepts + self.num_aux_nodes
        )
        self.hops = int(hops)
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= self.num_nodes:
            raise ValueError("edge_index refers to a node outside the graph.")

        transitions = self._build_transitions(
            edge_index=edge_index,
            edge_type=edge_type,
        )
        self.register_buffer(
            "q_transition_t",
            transitions["q"],
            persistent=False,
        )
        self.register_buffer(
            "all_transition_t",
            transitions["all"],
            persistent=False,
        )
        for relation in range(self.NUM_METADATA_RELATIONS):
            self.register_buffer(
                f"metadata_transition_t_{relation}",
                transitions[f"metadata_{relation}"],
                persistent=False,
            )
        self.register_buffer(
            "active_metadata_relations",
            transitions["active_metadata_relations"],
            persistent=False,
        )

        self.path_encoder = nn.Sequential(
            nn.Linear(self.PATH_FEATURES, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
        self.relation_embedding = nn.Parameter(
            torch.empty(self.NUM_METADATA_RELATIONS, dim)
        )
        self.hop_embedding = nn.Parameter(torch.empty(self.hops, dim))
        self.path_score = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(
                dim,
                self.NUM_METADATA_RELATIONS * self.hops,
            ),
        )
        self.query_composer = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        nn.init.xavier_uniform_(self.relation_embedding)
        nn.init.xavier_uniform_(self.hop_embedding)

    def _build_transitions(
        self,
        *,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        source = edge_index[0].to(dtype=torch.long, device="cpu")
        target = edge_index[1].to(dtype=torch.long, device="cpu")
        relation = edge_type.to(dtype=torch.long, device="cpu")
        weights = torch.ones(source.numel(), dtype=torch.float32)
        inverse = torch.isin(
            relation,
            torch.tensor([3, 6, 8], dtype=torch.long),
        )
        weights[inverse] = 0.5
        degree = torch.zeros(self.num_nodes, dtype=torch.float32)
        degree.scatter_add_(0, source, weights)
        normalized = weights / degree.index_select(0, source).clamp_min(1.0)

        def transition(mask: torch.Tensor) -> torch.Tensor:
            indices = torch.stack([target[mask], source[mask]], dim=0)
            values = normalized[mask]
            return torch.sparse_coo_tensor(
                indices,
                values,
                size=(self.num_nodes, self.num_nodes),
            ).coalesce()

        q_mask = relation < self.NUM_Q_RELATIONS
        metadata_mask = ~q_mask
        output: dict[str, torch.Tensor] = {
            "q": transition(q_mask),
            "all": transition(torch.ones_like(q_mask, dtype=torch.bool)),
        }
        active = torch.zeros(
            self.NUM_METADATA_RELATIONS,
            dtype=torch.bool,
        )
        for slot in range(self.NUM_METADATA_RELATIONS):
            relation_id = self.NUM_Q_RELATIONS + slot
            mask = relation == relation_id
            output[f"metadata_{slot}"] = transition(mask)
            active[slot] = bool(mask.any())
        output["active_metadata_relations"] = active
        if not bool(q_mask.any()):
            raise ValueError("relation query graph has no Q relations.")
        if bool(metadata_mask.any()) != bool(active.any()):
            raise RuntimeError("metadata relation activation is inconsistent.")
        return output

    @staticmethod
    def _propagate(
        transition_t: torch.Tensor,
        signals: torch.Tensor,
    ) -> torch.Tensor:
        if transition_t._nnz() == 0:
            return torch.zeros_like(signals)
        return torch.sparse.mm(
            transition_t,
            signals.transpose(0, 1),
        ).transpose(0, 1)

    def _response_memory(
        self,
        *,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        exercise_evidence: torch.Tensor,
    ) -> torch.Tensor:
        if student_exercise_mask.shape != response_matrix.shape:
            raise ValueError("history mask and response matrix must align.")
        if student_exercise_mask.shape[1] != self.num_exercises:
            raise ValueError("history exercise width does not match graph.")
        if exercise_evidence.shape[0] != self.num_exercises:
            raise ValueError("exercise evidence does not match graph.")
        attempts = exercise_evidence[:, 0].to(
            dtype=response_matrix.dtype,
            device=response_matrix.device,
        )
        correct = exercise_evidence[:, 1].to(
            dtype=response_matrix.dtype,
            device=response_matrix.device,
        )
        global_rate = (correct.sum() + 1.0) / (attempts.sum() + 2.0)
        ease = (correct + 20.0 * global_rate) / (attempts + 20.0)
        mask = student_exercise_mask.to(dtype=response_matrix.dtype)
        residual = (response_matrix - ease.unsqueeze(0)) * mask
        item_signals = torch.stack(
            [
                mask,
                response_matrix * mask,
                residual,
                residual.abs(),
            ],
            dim=0,
        )
        batch_students = mask.shape[0]
        flattened = item_signals.reshape(
            self.RESPONSE_CHANNELS * batch_students,
            self.num_exercises,
        )
        if self.num_nodes == self.num_exercises:
            return flattened
        padding = flattened.new_zeros(
            flattened.shape[0],
            self.num_nodes - self.num_exercises,
        )
        return torch.cat([flattened, padding], dim=1)

    def _transport_features(
        self,
        *,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        exercise_evidence: torch.Tensor,
        target_state_rows: torch.Tensor,
        target_exercise_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            initial = self._response_memory(
                student_exercise_mask=student_exercise_mask,
                response_matrix=response_matrix,
                exercise_evidence=exercise_evidence,
            )
            batch_students = student_exercise_mask.shape[0]
            target_rows = target_state_rows.to(dtype=torch.long)
            target_items = target_exercise_ids.to(dtype=torch.long)
            if target_rows.numel() != target_items.numel():
                raise ValueError("target row and exercise tensors must align.")
            if target_rows.numel() == 0:
                raise ValueError("relation query requires target interactions.")

            features = initial.new_zeros(
                target_rows.numel(),
                self.NUM_METADATA_RELATIONS,
                self.hops,
                self.PATH_FEATURES,
            )
            reachable = torch.zeros(
                target_rows.numel(),
                self.NUM_METADATA_RELATIONS,
                self.hops,
                dtype=torch.bool,
                device=initial.device,
            )
            no_metadata = initial
            used_by_relation: dict[int, torch.Tensor] = {
                relation: torch.zeros_like(initial)
                for relation in range(self.NUM_METADATA_RELATIONS)
                if bool(self.active_metadata_relations[relation])
            }
            for hop in range(self.hops):
                previous_no_metadata = no_metadata
                no_metadata = self._propagate(
                    self.q_transition_t,
                    previous_no_metadata,
                )
                for relation, used in tuple(used_by_relation.items()):
                    metadata_transition = getattr(
                        self,
                        f"metadata_transition_t_{relation}",
                    )
                    used = (
                        self._propagate(self.all_transition_t, used)
                        + self._propagate(
                            metadata_transition,
                            previous_no_metadata,
                        )
                    )
                    used_by_relation[relation] = used
                    channels = used.reshape(
                        self.RESPONSE_CHANNELS,
                        batch_students,
                        self.num_nodes,
                    )[:, target_rows, target_items].transpose(0, 1)
                    mass = channels[:, 0]
                    positive = mass > 0.0
                    denominator = mass.clamp_min(1.0)
                    features[:, relation, hop, 0] = (
                        channels[:, 1] / denominator
                    )
                    features[:, relation, hop, 1] = (
                        channels[:, 2] / denominator
                    )
                    features[:, relation, hop, 2] = (
                        channels[:, 3] / denominator
                    )
                    features[:, relation, hop, 3] = torch.log1p(mass)
                    features[:, relation, hop, 4] = mass / (mass + 1.0)
                    reachable[:, relation, hop] = positive
            return features, reachable

    def forward(
        self,
        *,
        upstream_target_state: torch.Tensor,
        target_requirement: torch.Tensor,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        exercise_evidence: torch.Tensor,
        target_state_rows: torch.Tensor,
        target_exercise_ids: torch.Tensor,
    ) -> RelationQueryOutput:
        if upstream_target_state.shape != target_requirement.shape:
            raise ValueError("upstream state and target requirement must align.")
        if upstream_target_state.ndim != 2:
            raise ValueError("target states must have shape [targets, dim].")
        if upstream_target_state.shape[1] != self.dim:
            raise ValueError("target state width does not match module dim.")

        path_features, reachable = self._transport_features(
            student_exercise_mask=student_exercise_mask,
            response_matrix=response_matrix,
            exercise_evidence=exercise_evidence,
            target_state_rows=target_state_rows,
            target_exercise_ids=target_exercise_ids,
        )
        encoded = self.path_encoder(path_features)
        encoded = (
            encoded
            + self.relation_embedding.view(
                1,
                self.NUM_METADATA_RELATIONS,
                1,
                self.dim,
            )
            + self.hop_embedding.view(1, 1, self.hops, self.dim)
        )
        encoded = encoded * reachable.unsqueeze(-1)
        score_input = torch.cat(
            [upstream_target_state, target_requirement],
            dim=-1,
        )
        scores = self.path_score(score_input).reshape(
            -1,
            self.NUM_METADATA_RELATIONS,
            self.hops,
        )
        scores = scores.masked_fill(~reachable, -1.0e4)
        attention = torch.softmax(scores.flatten(1), dim=1).reshape_as(scores)
        any_reachable = reachable.flatten(1).any(dim=1)
        attention = attention * any_reachable.view(-1, 1, 1)
        retrieved = (attention.unsqueeze(-1) * encoded).sum(dim=(1, 2))
        reliability = (
            attention * path_features[..., 4]
        ).sum(dim=(1, 2))
        composed = self.query_composer(
            torch.cat(
                [
                    upstream_target_state,
                    target_requirement,
                    retrieved,
                    upstream_target_state * retrieved,
                ],
                dim=-1,
            )
        )
        return RelationQueryOutput(
            target_student_state=composed,
            reliability=reliability,
            diagnostics={
                "relation_query_reliability": reliability,
                "relation_query_reachable_paths": (
                    reachable.to(dtype=upstream_target_state.dtype).sum(
                        dim=(1, 2)
                    )
                ),
                "relation_query_context_norm": retrieved.norm(dim=-1),
                "relation_query_output_norm": composed.norm(dim=-1),
            },
        )
