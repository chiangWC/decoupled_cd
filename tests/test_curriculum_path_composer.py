from __future__ import annotations

import hashlib
import json

import torch

from data.static_relations import load_static_relation_graph
from models.curriculum_path_composer import CurriculumPathComposer
from models.two_stage_tkc_ukc import TwoStageTKCUKCCDM


def _write_graph(tmp_path):
    path = tmp_path / "relations.json"
    path.write_text(
        json.dumps({
            "dataset": "toy",
            "variant": "full",
            "metadata_edges": [
                {
                    "relation": "subject_hierarchy",
                    "source": "concept:0",
                    "target": "aux:root",
                    "directed": True,
                },
                {
                    "relation": "subject_hierarchy",
                    "source": "concept:1",
                    "target": "aux:root",
                    "directed": True,
                },
            ],
        }),
        encoding="utf-8",
    )
    return path


def _graphs(tmp_path):
    path = _write_graph(tmp_path)
    q_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    kwargs = {
        "path": path,
        "q_matrix_tensor": q_matrix,
        "exercise_id_map": {"0": 0, "1": 1},
        "concept_id_map": {"0": 0, "1": 1},
    }
    return (
        load_static_relation_graph(mode="full", **kwargs),
        load_static_relation_graph(mode="q_only", **kwargs),
    )


def _parameter_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def test_loader_preserves_capacity_and_removes_metadata(tmp_path):
    full, direct = _graphs(tmp_path)
    assert full.num_aux_nodes == direct.num_aux_nodes == 1
    assert full.audit["metadata_edges_used"] == 2
    assert direct.audit["metadata_edges_used"] == 0
    assert full.edge_index.size(1) > direct.edge_index.size(1)
    assert full.audit["q_edges_bidirectional"] == 4


def test_composer_shape_gradient_and_graph_effect(tmp_path):
    full, direct = _graphs(tmp_path)
    common = {
        "num_exercises": 2,
        "num_concepts": 2,
        "num_aux_nodes": 1,
        "dim": 8,
        "channels": 2,
        "hops": 2,
    }
    torch.manual_seed(42)
    full_model = CurriculumPathComposer(
        edge_index=full.edge_index,
        edge_type=full.edge_type,
        **common,
    )
    torch.manual_seed(42)
    direct_model = CurriculumPathComposer(
        edge_index=direct.edge_index,
        edge_type=direct.edge_type,
        **common,
    )
    assert _parameter_hash(full_model) == _parameter_hash(direct_model)
    concepts = torch.randn(2, 8)
    exercises = torch.randn(2, 8)
    evidence = torch.tensor([
        [10.0, 0.0, 0.0, 0.8],
        [10.0, 0.0, 0.0, 0.2],
    ])
    full_output = full_model(
        concept_embeddings=concepts,
        exercise_embeddings=exercises,
        exercise_evidence=evidence,
    )
    direct_output = direct_model(
        concept_embeddings=concepts,
        exercise_embeddings=exercises,
        exercise_evidence=evidence,
    )
    assert full_output.concept_nodes.shape == (2, 8)
    assert full_output.exercise_nodes.shape == (2, 8)
    assert not torch.allclose(
        full_output.concept_nodes,
        direct_output.concept_nodes,
    )
    full_output.concept_nodes.square().mean().backward()
    assert full_model.relation_logits.grad is not None


def test_two_stage_has_no_raw_semantic_bypass(tmp_path):
    full, _ = _graphs(tmp_path)
    torch.manual_seed(42)
    model = TwoStageTKCUKCCDM(
        num_students=2,
        num_exercises=2,
        num_concepts=2,
        concept_dim=8,
        semantic_node_mode="curriculum_path",
        static_relation_edge_index=full.edge_index,
        static_relation_edge_type=full.edge_type,
        static_relation_num_aux_nodes=full.num_aux_nodes,
        static_relation_variant="full",
        cpc_channels=2,
        cpc_hops=2,
    )
    q = torch.eye(2)
    evidence = torch.tensor([
        [10.0, 0.0, 0.0, 0.8, 0.0],
        [10.0, 0.0, 0.0, 0.2, 0.0],
    ])
    kwargs = {
        "q_matrix": q,
        "concept_graph": torch.eye(2),
        "student_exercise_mask": torch.eye(2),
        "response_matrix": torch.eye(2),
        "student_tkc_mask": torch.eye(2),
        "student_ukc_mask": 1.0 - torch.eye(2),
        "student_concept_evidence": torch.ones(2, 2, 6),
        "exercise_evidence": evidence,
        "target_student_ids": torch.tensor([0, 1]),
        "target_exercise_ids": torch.tensor([0, 1]),
    }
    first = model(**kwargs)
    with torch.no_grad():
        model.curriculum_path_composer.output_projection[0].weight.zero_()
        model.curriculum_path_composer.output_projection[0].bias.zero_()
    second = model(**kwargs)
    assert not torch.allclose(first.probs, second.probs)
    assert first.framework_state.shape == (2, 2, 8)
    assert first.mastery.shape == (2, 2)
