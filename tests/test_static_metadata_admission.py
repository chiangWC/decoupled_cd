from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.audit_static_metadata_admission import (
    TypedGraph,
    add_filtered_group_edges,
    load_q_map,
    parse_concepts,
    protocol_fingerprint_paths,
    read_flat_pairs,
    target_reachability,
)


class StaticMetadataAdmissionTest(unittest.TestCase):
    def test_metadata_edge_is_required_for_reachability(self) -> None:
        graph = TypedGraph.empty()
        graph.add_q({10: {1}, 11: {2}})
        self.assertNotIn(
            2,
            graph.reachable_concepts(
                [10],
                max_hops=4,
                require_metadata=True,
            ),
        )
        graph.add_edge(
            ("item", 10),
            ("metadata:group", "a"),
            metadata=True,
        )
        graph.add_edge(
            ("metadata:group", "a"),
            ("item", 11),
            metadata=True,
        )
        self.assertEqual(
            graph.reachable_concepts(
                [10],
                max_hops=4,
                require_metadata=True,
            )[2],
            3,
        )

    def test_group_filter_excludes_singleton_and_coarse_values(
        self,
    ) -> None:
        graph = TypedGraph.empty()
        report = add_filtered_group_edges(
            graph,
            {
                0: {"singleton", "pair", "coarse"},
                1: {"pair", "coarse"},
                2: {"coarse"},
                3: {"coarse"},
            },
            relation="test",
            num_items=4,
            max_group_fraction=0.5,
        )
        self.assertEqual(report["admitted_values"], 1)
        self.assertEqual(report["incident_items"], 2)

    def test_flat_pair_reader_rejects_odd_token_count(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.txt"
            path.write_text("1 2 3\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_flat_pairs(path)

    def test_target_reachability_uses_validation_only(
        self,
    ) -> None:
        q_map = {10: {1}, 11: {2}, 12: {3}}
        graph = TypedGraph.empty()
        graph.add_q(q_map)
        q_only_graph = TypedGraph.empty()
        q_only_graph.add_q(q_map)
        graph.add_edge(
            ("concept", 1),
            ("concept", 2),
            metadata=True,
        )
        splits = {
            "train": pd.DataFrame(
                {"stu_id": [0], "exer_id": [10]}
            ),
            "valid": pd.DataFrame(
                {"stu_id": [0, 0], "exer_id": [11, 12]}
            ),
            "test": pd.DataFrame(
                {"stu_id": [0], "exer_id": [12]}
            ),
        }
        report = target_reachability(
            graph,
            q_only_graph,
            splits,
            q_map,
            target_scope="exact_zero",
            max_hops=4,
        )
        self.assertEqual(
            report["validation_target_rows"],
            2,
        )
        self.assertEqual(
            report["incremental"]["any_rows"],
            1,
        )
        self.assertEqual(
            report["incremental"]["any_row_fraction"],
            0.5,
        )

    def test_redundant_metadata_path_is_not_incremental(self) -> None:
        q_map = {10: {1}, 11: {1, 2}, 12: {2}}
        graph = TypedGraph.empty()
        graph.add_q(q_map)
        graph.add_edge(
            ("concept", 1),
            ("concept", 2),
            metadata=True,
        )
        q_only_graph = TypedGraph.empty()
        q_only_graph.add_q(q_map)
        splits = {
            "train": pd.DataFrame(
                {"stu_id": [0], "exer_id": [10]}
            ),
            "valid": pd.DataFrame(
                {"stu_id": [0], "exer_id": [12]}
            ),
        }
        report = target_reachability(
            graph,
            q_only_graph,
            splits,
            q_map,
            target_scope="exact_zero",
            max_hops=4,
        )
        self.assertEqual(report["metadata_path"]["any_rows"], 1)
        self.assertEqual(report["q_only"]["any_rows"], 1)
        self.assertEqual(report["incremental"]["any_rows"], 0)

    def test_q_loader_accepts_exploded_multiconcept_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "Q_matrix.csv"
            path.write_text(
                "exer_id,cpt_seq\n0,1\n0,2\n1,\"3,4\"\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_q_map(path),
                {0: {1, 2}, 1: {3, 4}},
            )

    def test_concept_parser_accepts_junyi_code_lists(self) -> None:
        self.assertEqual(parse_concepts([3]), (3,))
        self.assertEqual(parse_concepts([3, 7]), (3, 7))

    def test_protocol_fingerprint_covers_audit_inputs(self) -> None:
        paths = protocol_fingerprint_paths("toy", "/tmp/toy")
        self.assertEqual(
            set(paths),
            {
                "toy_data",
                "toy_train",
                "toy_valid",
                "toy_Q_matrix",
            },
        )


if __name__ == "__main__":
    unittest.main()
