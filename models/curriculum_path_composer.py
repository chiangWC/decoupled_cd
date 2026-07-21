from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from data.static_relations import RELATION_SLOTS


@dataclass
class CurriculumPathOutput:
    concept_nodes: torch.Tensor
    exercise_nodes: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class CurriculumPathComposer(nn.Module):
    """Compose typed curriculum paths into item/concept representations."""

    def __init__(
        self,
        *,
        num_exercises: int,
        num_concepts: int,
        num_aux_nodes: int,
        dim: int,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        channels: int = 4,
        hops: int = 4,
    ) -> None:
        super().__init__()
        if channels < 1 or hops < 1:
            raise ValueError("CPC channels and hops must be positive.")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                "CPC edge_index must have shape [2, edges]."
            )
        if (
            edge_type.dim() != 1
            or edge_type.numel() != edge_index.size(1)
        ):
            raise ValueError("CPC edge_type must align with edge_index.")
        if edge_type.numel() == 0:
            raise ValueError("CPC requires at least one graph edge.")
        if (
            int(edge_type.min()) < 0
            or int(edge_type.max()) >= len(RELATION_SLOTS)
        ):
            raise ValueError(
                "CPC edge type is outside the frozen relation schema."
            )

        self.num_exercises = int(num_exercises)
        self.num_concepts = int(num_concepts)
        self.num_aux_nodes = int(num_aux_nodes)
        self.dim = int(dim)
        self.channels = int(channels)
        self.hops = int(hops)
        self.num_nodes = (
            self.num_exercises
            + self.num_concepts
            + self.num_aux_nodes
        )
        if (
            int(edge_index.min()) < 0
            or int(edge_index.max()) >= self.num_nodes
        ):
            raise ValueError("CPC edge index is outside configured nodes.")

        self.register_buffer(
            "edge_index",
            edge_index.to(dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "edge_type",
            edge_type.to(dtype=torch.long),
            persistent=False,
        )
        self.aux_embedding = nn.Embedding(
            max(self.num_aux_nodes, 1),
            dim,
        )
        self.node_type_embedding = nn.Embedding(3, dim)
        self.exercise_stat_projection = nn.Linear(2, dim, bias=False)
        self.relation_logits = nn.Parameter(
            torch.zeros(channels, hops, len(RELATION_SLOTS))
        )
        self.self_transforms = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(dim, dim, bias=False)
                        for _ in range(hops)
                    ]
                )
                for _ in range(channels)
            ]
        )
        self.message_transforms = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(dim, dim, bias=False)
                        for _ in range(hops)
                    ]
                )
                for _ in range(channels)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.LayerNorm(dim) for _ in range(hops)]
                )
                for _ in range(channels)
            ]
        )
        self.channel_logits = nn.Parameter(torch.zeros(channels))
        self.output_projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        nn.init.xavier_uniform_(self.aux_embedding.weight)
        nn.init.xavier_uniform_(self.node_type_embedding.weight)

    def _initial_nodes(
        self,
        *,
        concept_embeddings: torch.Tensor,
        exercise_embeddings: torch.Tensor,
        exercise_evidence: torch.Tensor,
    ) -> torch.Tensor:
        if concept_embeddings.shape != (
            self.num_concepts,
            self.dim,
        ):
            raise ValueError("CPC concept embedding shape mismatch.")
        if exercise_embeddings.shape != (
            self.num_exercises,
            self.dim,
        ):
            raise ValueError("CPC exercise embedding shape mismatch.")
        if (
            exercise_evidence.dim() != 2
            or exercise_evidence.size(0) != self.num_exercises
        ):
            raise ValueError("CPC exercise evidence shape mismatch.")
        if exercise_evidence.size(1) < 4:
            raise ValueError(
                "CPC requires attempts and accuracy evidence."
            )

        dtype = exercise_embeddings.dtype
        attempts = exercise_evidence[:, 0].to(
            dtype=dtype
        ).clamp_min(0.0)
        accuracy = exercise_evidence[:, 3].to(
            dtype=dtype
        ).clamp(0.0, 1.0)
        maximum = attempts.max().clamp_min(1.0)
        confidence = torch.log1p(attempts) / torch.log1p(maximum)
        statistics = torch.stack(
            [accuracy.mul(2.0).sub(1.0), confidence],
            dim=-1,
        )
        exercise_nodes = (
            exercise_embeddings
            + self.exercise_stat_projection(statistics)
            + self.node_type_embedding.weight[0]
        )
        concept_nodes = (
            concept_embeddings + self.node_type_embedding.weight[1]
        )
        if self.num_aux_nodes:
            aux_nodes = (
                self.aux_embedding.weight[: self.num_aux_nodes]
                + self.node_type_embedding.weight[2]
            )
        else:
            aux_nodes = exercise_nodes.new_empty((0, self.dim))
        return torch.cat(
            [exercise_nodes, concept_nodes, aux_nodes],
            dim=0,
        )

    def _aggregate(
        self,
        states: torch.Tensor,
        relation_weights: torch.Tensor,
    ) -> torch.Tensor:
        source, target = self.edge_index
        weights = relation_weights.index_select(0, self.edge_type)
        messages = (
            states.index_select(0, source) * weights.unsqueeze(-1)
        )
        aggregated = states.new_zeros(states.shape)
        aggregated.index_add_(0, target, messages)
        denominator = states.new_zeros((self.num_nodes,))
        denominator.index_add_(0, target, weights)
        return (
            aggregated
            / denominator.clamp_min(1.0e-6).unsqueeze(-1)
        )

    def forward(
        self,
        *,
        concept_embeddings: torch.Tensor,
        exercise_embeddings: torch.Tensor,
        exercise_evidence: torch.Tensor,
    ) -> CurriculumPathOutput:
        initial = self._initial_nodes(
            concept_embeddings=concept_embeddings,
            exercise_embeddings=exercise_embeddings,
            exercise_evidence=exercise_evidence,
        )
        relation_weights = torch.softmax(
            self.relation_logits,
            dim=-1,
        )
        channel_states = []
        for channel in range(self.channels):
            states = initial
            for hop in range(self.hops):
                message = self._aggregate(
                    states,
                    relation_weights[channel, hop],
                )
                states = self.norms[channel][hop](
                    self.self_transforms[channel][hop](states)
                    + self.message_transforms[channel][hop](message)
                )
                states = F.gelu(states)
            channel_states.append(states)
        stacked = torch.stack(channel_states, dim=0)
        channel_weights = torch.softmax(
            self.channel_logits,
            dim=0,
        )
        composed = torch.einsum(
            "c,cnd->nd",
            channel_weights,
            stacked,
        )
        composed = self.output_projection(composed)

        item_end = self.num_exercises
        concept_end = item_end + self.num_concepts
        return CurriculumPathOutput(
            exercise_nodes=composed[:item_end],
            concept_nodes=composed[item_end:concept_end],
            diagnostics={
                "cpc_relation_weights": relation_weights.detach(),
                "cpc_channel_weights": channel_weights.detach(),
                "cpc_output_norm": composed.norm(dim=-1).mean(),
                "cpc_edge_count": composed.new_tensor(
                    float(self.edge_index.size(1))
                ),
            },
        )
