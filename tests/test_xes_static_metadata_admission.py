from __future__ import annotations

import pytest

from scripts.audit_xes_static_metadata_admission import materialize_relations


def _questions() -> dict[str, object]:
    return {
        "10": {"kc_routes": ["root----branch----leaf-a"]},
        "11": {"kc_routes": ["root----other----leaf-b"]},
        "12": {"kc_routes": ["root----second----leaf-a"]},
    }


def test_materialize_retains_multiple_proven_paths() -> None:
    result = materialize_relations(
        current_q={0: 0, 1: 1, 2: 0},
        reverse_question_map={0: 10, 1: 11, 2: 12},
        questions=_questions(),
        route_labels={
            "root",
            "branch",
            "other",
            "second",
            "leaf-a",
            "leaf-b",
        },
    )
    assert result["audit"]["mapped_items"] == 3
    assert result["audit"]["mapped_concepts"] == 2
    assert result["audit"]["concepts_with_multiple_parent_paths"] == 1
    assert result["audit"]["incident_current_concepts"] == 2
    targets = {edge["target"] for edge in result["metadata_edges"]}
    assert "concept:0" in targets
    assert "concept:1" in targets


def test_materialize_rejects_concept_leaf_conflict() -> None:
    questions = {
        "10": {"kc_routes": ["root----leaf-a"]},
        "11": {"kc_routes": ["root----leaf-b"]},
    }
    with pytest.raises(RuntimeError, match="maps to"):
        materialize_relations(
            current_q={0: 0, 1: 0},
            reverse_question_map={0: 10, 1: 11},
            questions=questions,
            route_labels={"root", "leaf-a", "leaf-b"},
        )


def test_materialize_rejects_unmapped_route_node() -> None:
    with pytest.raises(RuntimeError, match="absent"):
        materialize_relations(
            current_q={0: 0},
            reverse_question_map={0: 10},
            questions={"10": {"kc_routes": ["root----leaf-a"]}},
            route_labels={"leaf-a"},
        )
