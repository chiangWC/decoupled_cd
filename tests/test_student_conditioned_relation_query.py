from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from data.pipeline import build_history_tensors
from data.static_relations import load_static_relation_graph
from models import TwoStageTKCUKCCDM
from models.student_conditioned_relation_query import (
    StudentConditionedRelationQuery,
)


def _toy_edges(include_metadata: bool) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 1]],
        dtype=torch.long,
    )
    edge_type = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    if include_metadata:
        edge_index = torch.cat(
            [edge_index, torch.tensor([[0], [2]], dtype=torch.long)],
            dim=1,
        )
        edge_type = torch.cat(
            [edge_type, torch.tensor([7], dtype=torch.long)]
        )
    return edge_index, edge_type


def _module(include_metadata: bool) -> StudentConditionedRelationQuery:
    edge_index, edge_type = _toy_edges(include_metadata)
    torch.manual_seed(42)
    return StudentConditionedRelationQuery(
        dim=8,
        num_exercises=3,
        num_concepts=2,
        num_aux_nodes=0,
        edge_index=edge_index,
        edge_type=edge_type,
        hops=4,
    )


def _forward_kwargs() -> dict[str, torch.Tensor]:
    return {
        "upstream_target_state": torch.randn(1, 8),
        "target_requirement": torch.randn(1, 8),
        "student_exercise_mask": torch.tensor([[1.0, 0.0, 0.0]]),
        "response_matrix": torch.tensor([[1.0, 0.0, 0.0]]),
        "exercise_evidence": torch.tensor(
            [
                [10.0, 8.0, 2.0, 0.8, 0.0, 1.0],
                [10.0, 5.0, 5.0, 0.5, 0.0, 1.0],
                [10.0, 4.0, 6.0, 0.4, 0.0, 1.0],
            ]
        ),
        "target_state_rows": torch.tensor([0]),
        "target_exercise_ids": torch.tensor([2]),
    }


def test_full_and_direct_have_identical_trainable_state() -> None:
    full = _module(True)
    direct = _module(False)
    assert full.state_dict().keys() == direct.state_dict().keys()
    for name in full.state_dict():
        assert torch.equal(full.state_dict()[name], direct.state_dict()[name])


def test_relation_graph_changes_output_and_receives_gradient() -> None:
    full = _module(True)
    direct = _module(False)
    kwargs = _forward_kwargs()
    full_output = full(**kwargs)
    direct_output = direct(**kwargs)
    assert full_output.reliability.item() > 0.0
    assert direct_output.reliability.item() == 0.0
    assert not torch.allclose(
        full_output.target_student_state,
        direct_output.target_student_state,
    )
    full_output.target_student_state.sum().backward()
    gradient = full.query_composer[0].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum().item() > 0.0


def test_history_order_is_irrelevant() -> None:
    frame = pd.DataFrame(
        {
            "stu_id": ["s", "s"],
            "exer_id": ["e0", "e1"],
            "label": [1, 0],
            "cpt_seq": ["c0", "c0"],
        }
    )
    mappings = {
        "student_id_map": {"s": 0},
        "exercise_id_map": {"e0": 0, "e1": 1},
        "concept_id_map": {"c0": 0},
    }
    first = build_history_tensors(
        history_interactions=frame,
        **mappings,
    )
    second = build_history_tensors(
        history_interactions=frame.iloc[::-1].reset_index(drop=True),
        **mappings,
    )
    for name in (
        "student_exercise_mask",
        "response_matrix_tensor",
        "student_concept_evidence_tensor",
    ):
        assert torch.equal(first[name], second[name])


def test_loader_keeps_aux_capacity_in_q_only(tmp_path: Path) -> None:
    source = tmp_path / "relations.json"
    source.write_text(
        json.dumps(
            {
                "dataset": "toy",
                "variant": "full",
                "metadata_edges": [
                    {
                        "relation": "group",
                        "source": "item:e0",
                        "target": "group:g0",
                        "directed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    q = torch.tensor([[1.0], [1.0]])
    kwargs = {
        "path": source,
        "q_matrix_tensor": q,
        "exercise_id_map": {"e0": 0, "e1": 1},
        "concept_id_map": {"c0": 0},
    }
    full = load_static_relation_graph(mode="full", **kwargs)
    direct = load_static_relation_graph(mode="q_only", **kwargs)
    assert full.num_aux_nodes == direct.num_aux_nodes == 1
    assert full.audit["metadata_edges_used"] == 1
    assert direct.audit["metadata_edges_used"] == 0
    assert full.edge_type.numel() > direct.edge_type.numel()


def test_base_fingerprint_is_unchanged() -> None:
    model = TwoStageTKCUKCCDM(
        num_students=2,
        num_exercises=3,
        num_concepts=2,
    )
    assert model.architecture_fingerprint == "099906acdba8c3b4"
    assert model.ablation_variant_fingerprint == "099906acdba8c3b4"
