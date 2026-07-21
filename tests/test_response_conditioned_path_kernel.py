from __future__ import annotations

import json

import torch

from models.response_conditioned_path_kernel import (
    ResponseConditionedPathKernel,
    load_path_kernel_graph,
)
from models.two_stage_tkc_ukc import TwoStageTKCUKCCDM


def _graph_file(tmp_path):
    path = tmp_path / "relations.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "synthetic",
                "variant": "full",
                "metadata_edges": [
                    {
                        "relation": "prerequisite",
                        "source": "concept:100",
                        "target": "concept:101",
                        "directed": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _graphs(tmp_path):
    q_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    kwargs = {
        "path": _graph_file(tmp_path),
        "q_matrix": q_matrix,
        "exercise_id_map": {"10": 0, "11": 1},
        "concept_id_map": {"100": 0, "101": 1},
    }
    return (
        load_path_kernel_graph(mode="relational", **kwargs),
        load_path_kernel_graph(mode="q_only", **kwargs),
    )


def test_real_relation_reaches_target_and_direct_does_not(tmp_path) -> None:
    full_graph, direct_graph = _graphs(tmp_path)
    torch.manual_seed(42)
    full = ResponseConditionedPathKernel(dim=8, graph=full_graph, hops=4)
    torch.manual_seed(42)
    direct = ResponseConditionedPathKernel(dim=8, graph=direct_graph, hops=4)
    for (full_name, full_value), (direct_name, direct_value) in zip(
        full.state_dict().items(), direct.state_dict().items(), strict=True
    ):
        assert full_name == direct_name
        assert torch.equal(full_value, direct_value)

    inputs = {
        "upstream_target_state": torch.randn(1, 8),
        "target_requirement": torch.randn(1, 8),
        "student_exercise_mask": torch.tensor([[1.0, 0.0]]),
        "response_matrix": torch.tensor([[1.0, 0.0]]),
        "exercise_evidence": torch.tensor([[10.0, 7.0], [10.0, 5.0]]),
        "target_state_rows": torch.tensor([0]),
        "target_exercise_ids": torch.tensor([1]),
    }
    full_output = full(**inputs)
    direct_output = direct(**inputs)
    assert full_output.reliability.item() > 0.0
    assert direct_output.reliability.item() == 0.0
    assert not torch.allclose(
        full_output.target_student_state,
        direct_output.target_student_state,
    )
    full_output.target_student_state.sum().backward()
    gradients = [
        parameter.grad
        for parameter in full.parameters()
        if parameter.requires_grad
    ]
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)


def test_path_variants_share_architecture_and_initialization(tmp_path) -> None:
    full_graph, direct_graph = _graphs(tmp_path)
    common = {
        "num_students": 2,
        "num_exercises": 2,
        "num_concepts": 2,
        "concept_dim": 8,
        "target_requirement_mode": "factorized_item_control",
    }
    torch.manual_seed(42)
    full = TwoStageTKCUKCCDM(**common, response_path_graph=full_graph)
    torch.manual_seed(42)
    direct = TwoStageTKCUKCCDM(**common, response_path_graph=direct_graph)
    assert full.architecture_fingerprint == direct.architecture_fingerprint
    assert full.initialization_hash() == direct.initialization_hash()
    assert (
        full.active_module_parameter_counts()[
            "response_conditioned_path_kernel"
        ]["active"]
        == direct.active_module_parameter_counts()[
            "response_conditioned_path_kernel"
        ]["active"]
    )


def test_disabled_model_keeps_frozen_fingerprint() -> None:
    model = TwoStageTKCUKCCDM(
        num_students=2,
        num_exercises=3,
        num_concepts=2,
        concept_dim=64,
    )
    assert model.architecture_fingerprint == "099906acdba8c3b4"
