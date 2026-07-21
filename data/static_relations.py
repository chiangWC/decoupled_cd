from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


RELATION_SLOTS = (
    "q_item_to_concept",
    "q_concept_to_item",
    "hierarchy_forward",
    "hierarchy_inverse",
    "similarity",
    "group_item_to_aux",
    "group_aux_to_item",
    "other_forward",
    "other_inverse",
)


@dataclass(frozen=True)
class StaticRelationGraph:
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    num_aux_nodes: int
    source_sha256: str
    edge_sha256: str
    mode: str
    audit: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _edge_digest(edges: list[tuple[int, int, int]]) -> str:
    digest = hashlib.sha256()
    for source, target, relation in sorted(edges):
        digest.update(
            f"{source}:{target}:{relation}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _split_node(value: object) -> tuple[str, str]:
    text = str(value)
    if ":" not in text:
        raise ValueError(
            f"Static relation node lacks a type prefix: {text}"
        )
    prefix, identifier = text.split(":", 1)
    if not prefix or not identifier:
        raise ValueError(f"Malformed static relation node: {text}")
    return prefix, identifier


def _metadata_slot(
    *,
    relation: str,
    source_kind: str,
    target_kind: str,
    directed: bool,
) -> tuple[int, int | None]:
    if not directed:
        if source_kind == "concept" and target_kind == "concept":
            return 4, 4
        return 7, 8
    if "similar" in relation.lower():
        return 4, 4
    if "item" in {source_kind, target_kind} and source_kind != target_kind:
        if source_kind == "item":
            return 5, 6
        return 6, 5
    if source_kind != "item" and target_kind != "item":
        return 2, 3
    return 7, 8


def load_static_relation_graph(
    path: str | Path,
    *,
    mode: str,
    q_matrix_tensor: torch.Tensor,
    exercise_id_map: dict[str, int],
    concept_id_map: dict[str, int],
) -> StaticRelationGraph:
    """Bind a canonical generated relation graph to current dense IDs."""

    if mode not in {"full", "q_only"}:
        raise ValueError(
            "static relation mode must be 'full' or 'q_only'."
        )
    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    raw_edges = payload.get("metadata_edges")
    if not isinstance(raw_edges, list):
        raise ValueError("Static relation JSON lacks metadata_edges.")

    num_exercises = len(exercise_id_map)
    num_concepts = len(concept_id_map)
    if tuple(q_matrix_tensor.shape) != (num_exercises, num_concepts):
        raise ValueError("Q tensor shape does not match current ID mappings.")

    node_specs: set[tuple[str, str]] = set()
    for edge in raw_edges:
        if not isinstance(edge, dict):
            raise ValueError("Every metadata edge must be an object.")
        node_specs.add(_split_node(edge.get("source")))
        node_specs.add(_split_node(edge.get("target")))

    aux_nodes = sorted(
        f"{kind}:{identifier}"
        for kind, identifier in node_specs
        if kind not in {"item", "concept"}
    )
    aux_index = {
        node: num_exercises + num_concepts + offset
        for offset, node in enumerate(aux_nodes)
    }

    def bind_node(value: object) -> tuple[int, str]:
        kind, identifier = _split_node(value)
        if kind == "item":
            if identifier not in exercise_id_map:
                raise ValueError(
                    f"Static relation references unknown item {identifier}."
                )
            return exercise_id_map[identifier], kind
        if kind == "concept":
            if identifier not in concept_id_map:
                raise ValueError(
                    f"Static relation references unknown concept {identifier}."
                )
            return num_exercises + concept_id_map[identifier], kind
        key = f"{kind}:{identifier}"
        return aux_index[key], kind

    edges: set[tuple[int, int, int]] = set()
    q_rows, q_columns = torch.nonzero(
        q_matrix_tensor > 0.0,
        as_tuple=True,
    )
    for item_index, concept_index in zip(
        q_rows.tolist(),
        q_columns.tolist(),
        strict=True,
    ):
        concept_node = num_exercises + concept_index
        edges.add((item_index, concept_node, 0))
        edges.add((concept_node, item_index, 1))

    metadata_edges_used = 0
    if mode == "full":
        for edge in raw_edges:
            source, source_kind = bind_node(edge["source"])
            target, target_kind = bind_node(edge["target"])
            directed = bool(edge.get("directed", True))
            forward, inverse = _metadata_slot(
                relation=str(edge.get("relation", "other")),
                source_kind=source_kind,
                target_kind=target_kind,
                directed=directed,
            )
            edges.add((source, target, forward))
            if inverse is not None:
                edges.add((target, source, inverse))
            metadata_edges_used += 1
    else:
        for edge in raw_edges:
            bind_node(edge["source"])
            bind_node(edge["target"])

    ordered = sorted(edges)
    if not ordered:
        raise ValueError("Static relation graph contains no Q edges.")
    edge_index = torch.tensor(
        [[source, target] for source, target, _ in ordered],
        dtype=torch.long,
    ).transpose(0, 1).contiguous()
    edge_type = torch.tensor(
        [relation for _, _, relation in ordered],
        dtype=torch.long,
    )
    total_nodes = num_exercises + num_concepts + len(aux_nodes)
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= total_nodes:
        raise ValueError("Static relation edge is outside the node range.")

    slot_counts = {
        name: int((edge_type == index).sum().item())
        for index, name in enumerate(RELATION_SLOTS)
    }
    return StaticRelationGraph(
        edge_index=edge_index,
        edge_type=edge_type,
        num_aux_nodes=len(aux_nodes),
        source_sha256=_sha256_file(source_path),
        edge_sha256=_edge_digest(ordered),
        mode=mode,
        audit={
            "source_path": str(source_path.resolve()),
            "source_dataset": payload.get("dataset"),
            "source_variant": payload.get("variant"),
            "num_exercises": num_exercises,
            "num_concepts": num_concepts,
            "num_aux_nodes": len(aux_nodes),
            "num_nodes": total_nodes,
            "q_edges_bidirectional": int(2 * len(q_rows)),
            "metadata_edges_available": len(raw_edges),
            "metadata_edges_used": metadata_edges_used,
            "slot_counts": slot_counts,
        },
    )
