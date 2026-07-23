from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class PathKernelGraph:
    q_transition_t: torch.Tensor
    metadata_transition_t: torch.Tensor
    all_transition_t: torch.Tensor
    num_nodes: int
    num_exercises: int
    source_sha256: str
    edge_sha256: str
    source_variant: str
    mode: str
    audit: dict[str, Any]


@dataclass
class PathKernelOutput:
    target_student_state: torch.Tensor
    reliability: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_node(value: object) -> tuple[str, str]:
    text = str(value)
    if ":" not in text:
        raise ValueError(f"Static relation node lacks a type prefix: {text}")
    kind, identifier = text.split(":", 1)
    if not kind or not identifier:
        raise ValueError(f"Malformed static relation node: {text}")
    return kind, identifier


def _edge_digest(edges: list[tuple[int, int, float, str]]) -> str:
    digest = hashlib.sha256()
    for source, target, weight, channel in sorted(edges):
        digest.update(
            f"{source}:{target}:{weight:.8g}:{channel}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _transition(
    edges: list[tuple[int, int, float]],
    *,
    degree: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    if not edges:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(0, dtype=torch.float32),
            size=(num_nodes, num_nodes),
        ).coalesce()
    source = torch.tensor([edge[0] for edge in edges], dtype=torch.long)
    target = torch.tensor([edge[1] for edge in edges], dtype=torch.long)
    weight = torch.tensor([edge[2] for edge in edges], dtype=torch.float32)
    normalized = weight / degree.index_select(0, source).clamp_min(1.0)
    return torch.sparse_coo_tensor(
        torch.stack([target, source], dim=0),
        normalized,
        size=(num_nodes, num_nodes),
    ).coalesce()


def load_path_kernel_graph(
    path: str | Path,
    *,
    mode: str,
    q_matrix: torch.Tensor,
    exercise_id_map: dict[str, int],
    concept_id_map: dict[str, int],
) -> PathKernelGraph:
    """Bind a provenance-locked relation graph to the current dense IDs."""

    if mode not in {"relational", "q_only"}:
        raise ValueError("Path-kernel mode must be relational or q_only.")
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    raw_metadata = payload.get("metadata_edges")
    if not isinstance(raw_metadata, list):
        raise ValueError("Static relation JSON lacks metadata_edges.")
    num_exercises = len(exercise_id_map)
    num_concepts = len(concept_id_map)
    if tuple(q_matrix.shape) != (num_exercises, num_concepts):
        raise ValueError("Q matrix does not match current dense ID maps.")

    node_specs = {
        _split_node(edge[key])
        for edge in raw_metadata
        for key in ("source", "target")
    }
    aux_nodes = sorted(
        f"{kind}:{identifier}"
        for kind, identifier in node_specs
        if kind not in {"item", "concept"}
    )
    aux_index = {
        node: num_exercises + num_concepts + offset
        for offset, node in enumerate(aux_nodes)
    }

    def bind(value: object) -> int:
        kind, identifier = _split_node(value)
        if kind == "item":
            if identifier not in exercise_id_map:
                raise ValueError(f"Relation graph has unknown item {identifier}.")
            return int(exercise_id_map[identifier])
        if kind == "concept":
            if identifier not in concept_id_map:
                raise ValueError(f"Relation graph has unknown concept {identifier}.")
            return num_exercises + int(concept_id_map[identifier])
        return aux_index[f"{kind}:{identifier}"]

    num_nodes = num_exercises + num_concepts + len(aux_nodes)
    q_edges: list[tuple[int, int, float]] = []
    q_rows, q_columns = torch.nonzero(q_matrix > 0.0, as_tuple=True)
    for item, concept in zip(q_rows.tolist(), q_columns.tolist(), strict=True):
        concept_node = num_exercises + int(concept)
        q_edges.extend(
            [
                (int(item), concept_node, 1.0),
                (concept_node, int(item), 1.0),
            ]
        )

    metadata_edges: list[tuple[int, int, float]] = []
    for edge in raw_metadata:
        source = bind(edge["source"])
        target = bind(edge["target"])
        if mode == "relational":
            metadata_edges.append((source, target, 1.0))
            inverse_weight = 0.5 if bool(edge.get("directed", True)) else 1.0
            metadata_edges.append((target, source, inverse_weight))

    all_edges = [*q_edges, *metadata_edges]
    if not q_edges:
        raise ValueError("Path kernel requires at least one Q edge.")
    degree = torch.zeros(num_nodes, dtype=torch.float32)
    for source, _, weight in all_edges:
        degree[source] += float(weight)
    q_transition_t = _transition(q_edges, degree=degree, num_nodes=num_nodes)
    metadata_transition_t = _transition(
        metadata_edges,
        degree=degree,
        num_nodes=num_nodes,
    )
    all_transition_t = _transition(all_edges, degree=degree, num_nodes=num_nodes)
    digest_edges = [
        (source, target, weight, "q") for source, target, weight in q_edges
    ] + [
        (source, target, weight, "metadata")
        for source, target, weight in metadata_edges
    ]
    return PathKernelGraph(
        q_transition_t=q_transition_t,
        metadata_transition_t=metadata_transition_t,
        all_transition_t=all_transition_t,
        num_nodes=num_nodes,
        num_exercises=num_exercises,
        source_sha256=_sha256_file(source_path),
        edge_sha256=_edge_digest(digest_edges),
        source_variant=str(payload.get("variant", "unknown")),
        mode=mode,
        audit={
            "source_path": str(source_path.resolve()),
            "source_dataset": payload.get("dataset"),
            "source_variant": payload.get("variant"),
            "num_nodes": num_nodes,
            "num_exercises": num_exercises,
            "num_concepts": num_concepts,
            "num_aux_nodes": len(aux_nodes),
            "q_directed_edges": len(q_edges),
            "metadata_edges_available": len(raw_metadata),
            "metadata_directed_edges_used": len(metadata_edges),
            "mode": mode,
        },
    )


class ResponseConditionedPathKernel(nn.Module):
    """Transport train-only responses to a target through curriculum paths."""

    NUM_CHANNELS = 4
    NUM_FEATURES = 6

    def __init__(
        self,
        *,
        dim: int,
        graph: PathKernelGraph,
        hops: int = 4,
        item_ease_shrinkage: float = 20.0,
        aggregation_mode: str = "target_conditioned",
    ) -> None:
        super().__init__()
        if dim <= 0 or hops <= 0 or item_ease_shrinkage <= 0.0:
            raise ValueError("Path-kernel dimensions and shrinkage must be positive.")
        if aggregation_mode not in {"target_conditioned", "student_global"}:
            raise ValueError(
                "Path-kernel aggregation must be target_conditioned or "
                "student_global."
            )
        self.dim = int(dim)
        self.num_nodes = int(graph.num_nodes)
        self.num_exercises = int(graph.num_exercises)
        self.hops = int(hops)
        self.item_ease_shrinkage = float(item_ease_shrinkage)
        self.aggregation_mode = aggregation_mode
        self.source_sha256 = graph.source_sha256
        self.edge_sha256 = graph.edge_sha256
        self.source_variant = graph.source_variant
        self.mode = graph.mode
        self.graph_audit = graph.audit
        self.register_buffer("q_transition_t", graph.q_transition_t, persistent=False)
        self.register_buffer(
            "metadata_transition_t",
            graph.metadata_transition_t,
            persistent=False,
        )
        self.register_buffer(
            "all_transition_t",
            graph.all_transition_t,
            persistent=False,
        )
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.NUM_FEATURES, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
        )
        self.state_composer = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    @staticmethod
    def _propagate(transition_t: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        if transition_t._nnz() == 0:
            return torch.zeros_like(signal)
        return torch.sparse.mm(
            transition_t,
            signal.transpose(0, 1),
        ).transpose(0, 1)

    def _initial_response_signal(
        self,
        *,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        exercise_evidence: torch.Tensor,
    ) -> torch.Tensor:
        if student_exercise_mask.shape != response_matrix.shape:
            raise ValueError("History mask and response matrix must align.")
        if student_exercise_mask.shape[1] != self.num_exercises:
            raise ValueError("History exercise width does not match path graph.")
        if exercise_evidence.shape[0] != self.num_exercises:
            raise ValueError("Exercise evidence does not match path graph.")
        dtype = response_matrix.dtype
        attempts = exercise_evidence[:, 0].to(dtype=dtype)
        correct = exercise_evidence[:, 1].to(dtype=dtype)
        global_rate = (correct.sum() + 1.0) / (attempts.sum() + 2.0)
        ease = (
            correct + self.item_ease_shrinkage * global_rate
        ) / (attempts + self.item_ease_shrinkage)
        mask = student_exercise_mask.to(dtype=dtype)
        residual = (response_matrix - ease.unsqueeze(0)) * mask
        channels = torch.stack(
            [mask, response_matrix * mask, residual, residual.abs()],
            dim=0,
        )
        batch_students = mask.shape[0]
        flattened = channels.reshape(
            self.NUM_CHANNELS * batch_students,
            self.num_exercises,
        )
        padding = flattened.new_zeros(
            flattened.shape[0], self.num_nodes - self.num_exercises
        )
        return torch.cat([flattened, padding], dim=1)

    def _path_features(
        self,
        *,
        student_exercise_mask: torch.Tensor,
        response_matrix: torch.Tensor,
        exercise_evidence: torch.Tensor,
        target_state_rows: torch.Tensor,
        target_exercise_ids: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            initial = self._initial_response_signal(
                student_exercise_mask=student_exercise_mask,
                response_matrix=response_matrix,
                exercise_evidence=exercise_evidence,
            )
            batch_students = student_exercise_mask.shape[0]
            target_rows = target_state_rows.to(dtype=torch.long)
            target_items = target_exercise_ids.to(dtype=torch.long)
            no_metadata = initial
            used_metadata = torch.zeros_like(initial)
            totals = initial.new_zeros(target_rows.numel(), self.NUM_CHANNELS)
            first_hop = initial.new_zeros(target_rows.numel())
            for hop in range(1, self.hops + 1):
                previous_no_metadata = no_metadata
                no_metadata = self._propagate(
                    self.q_transition_t, previous_no_metadata
                )
                used_metadata = self._propagate(
                    self.all_transition_t, used_metadata
                ) + self._propagate(
                    self.metadata_transition_t, previous_no_metadata
                )
                transported_by_node = used_metadata.reshape(
                    self.NUM_CHANNELS,
                    batch_students,
                    self.num_nodes,
                )
                if self.aggregation_mode == "target_conditioned":
                    transported = transported_by_node[
                        :, target_rows, target_items
                    ].transpose(0, 1)
                else:
                    global_transport = transported_by_node[
                        :, :, : self.num_exercises
                    ].sum(dim=-1)
                    transported = global_transport[
                        :, target_rows
                    ].transpose(0, 1)
                mass = transported[:, 0]
                first_reached = (first_hop == 0.0) & (mass > 0.0)
                first_hop[first_reached] = float(hop)
                totals += transported

            mass = totals[:, 0]
            positive = mass > 0.0
            denominator = mass.clamp_min(1.0)
            features = initial.new_zeros(target_rows.numel(), self.NUM_FEATURES)
            features[positive, 0] = totals[positive, 1] / denominator[positive]
            features[positive, 1] = totals[positive, 2] / denominator[positive]
            features[positive, 2] = totals[positive, 3] / denominator[positive]
            features[:, 3] = torch.log1p(mass)
            features[positive, 4] = 1.0 / first_hop[positive]
            features[:, 5] = mass / (mass + 1.0)
            return features

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
    ) -> PathKernelOutput:
        if upstream_target_state.shape != target_requirement.shape:
            raise ValueError("Upstream state and target requirement must align.")
        features = self._path_features(
            student_exercise_mask=student_exercise_mask,
            response_matrix=response_matrix,
            exercise_evidence=exercise_evidence,
            target_state_rows=target_state_rows,
            target_exercise_ids=target_exercise_ids,
        )
        relation_state = self.feature_encoder(features)
        target_student_state = self.state_composer(
            torch.cat(
                [
                    upstream_target_state,
                    target_requirement,
                    relation_state,
                    upstream_target_state * relation_state,
                ],
                dim=-1,
            )
        )
        return PathKernelOutput(
            target_student_state=target_student_state,
            reliability=features[:, 5],
            diagnostics={
                "path_kernel_mass_log": features[:, 3],
                "path_kernel_inverse_first_hop": features[:, 4],
                "path_kernel_reliability": features[:, 5],
                "path_kernel_relation_state_norm": relation_state.norm(dim=-1),
                "path_kernel_output_norm": target_student_state.norm(dim=-1),
            },
        )
