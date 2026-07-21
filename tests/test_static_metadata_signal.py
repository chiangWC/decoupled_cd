from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.audit_static_metadata_signal import (
    RelationEdge,
    aggregate_gate,
    build_graph_kernel,
    relation_features,
    rewire_metadata,
)


def test_degree_preserving_rewire_keeps_directed_degrees() -> None:
    edges = tuple(
        RelationEdge(
            "membership",
            f"item:{item}",
            f"group:{item % 4}",
            True,
        )
        for item in range(20)
    )
    shuffled, audit = rewire_metadata(
        edges,
        dataset="toy",
        replicate=0,
    )
    assert len(shuffled) == len(edges)
    assert audit["degree_sequence_exact"] is True
    assert audit["original_edge_sha256"] != audit["shuffled_edge_sha256"]


def test_relation_feature_requires_metadata_path() -> None:
    q_edges = (
        ("item:0", "concept:0"),
        ("item:1", "concept:1"),
    )
    metadata = (
        RelationEdge("prerequisite", "concept:0", "concept:1", True),
    )
    kernel = build_graph_kernel(q_edges, metadata)
    support = pd.DataFrame(
        {"stu_id": ["0"], "exer_id": ["0"], "label": [1]}
    )
    target = pd.DataFrame({"stu_id": ["0"], "exer_id": ["1"]})
    block = relation_features(
        support=support,
        target=target,
        kernel=kernel,
        item_ease={"0": 0.5, "1": 0.5},
        batch_size=1,
    )
    assert block.shape == (1, 6)
    assert block[0, 0] > 0.0
    assert block[0, 3] > 0.0
    assert block[0, 4] == 1.0 / 3.0


def test_aggregate_gate_does_not_allow_junyi_only() -> None:
    summaries = {
        "assist09": {"deterministic_dataset_pass": False},
        "nips34": {"deterministic_dataset_pass": False},
        "junyi": {
            "deterministic_dataset_pass": True,
            "deltas_full_minus_control": {"target_auc": 0.02},
        },
    }
    result = aggregate_gate(summaries)
    assert result["at_least_two_datasets"] is False
    assert result["assist09_or_nips_required"] is False
    assert result["bootstrap_needed"] is False


def test_aggregate_gate_accepts_two_with_one_large_effect() -> None:
    summaries = {
        "assist09": {
            "deterministic_dataset_pass": True,
            "deltas_full_minus_control": {"target_auc": 0.006},
        },
        "nips34": {"deterministic_dataset_pass": False},
        "junyi": {
            "deterministic_dataset_pass": True,
            "deltas_full_minus_control": {"target_auc": 0.011},
        },
    }
    result = aggregate_gate(summaries)
    assert result["at_least_two_datasets"] is True
    assert result["assist09_or_nips_required"] is True
    assert result["one_target_delta_at_least_0_010"] is True
    assert result["bootstrap_needed"] is True
