from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from scripts.analyze_graph_family_depletion import (
    assert_aligned,
    interaction,
)
from scripts.run_pyedmine_graph_depletion_gate import RECIPE


def _rows(concept: list[float], random: list[float]) -> pd.DataFrame:
    labels = np.asarray([1, 0, 1, 0])
    concept_array = np.asarray(concept)
    random_array = np.asarray(random)
    concept_loss = -(
        labels * np.log(concept_array)
        + (1 - labels) * np.log(1 - concept_array)
    )
    random_loss = -(
        labels * np.log(random_array)
        + (1 - labels) * np.log(1 - random_array)
    )
    return pd.DataFrame(
        {
            "audit_row_id": [f"r{index}" for index in range(4)],
            "stu_id": [f"s{index}" for index in range(4)],
            "exer_id": [f"e{index}" for index in range(4)],
            "label": labels,
            "concept_prob": concept_array,
            "random_prob": random_array,
            "log_loss_damage": concept_loss - random_loss,
        }
    )


class GraphFamilyDepletionTest(unittest.TestCase):
    def test_graph_interaction_is_direct_paired_difference(self) -> None:
        graph = _rows(
            [0.51, 0.49, 0.51, 0.49], [0.9, 0.1, 0.9, 0.1]
        )
        non_graph = _rows(
            [0.8, 0.2, 0.8, 0.2], [0.9, 0.1, 0.9, 0.1]
        )
        result = interaction(graph, non_graph, replicates=50, seed=2024)
        expected = float(
            graph["log_loss_damage"].mean()
            - non_graph["log_loss_damage"].mean()
        )
        self.assertEqual(result["log_loss_interaction"], expected)
        self.assertIs(result["log_loss_support"], True)

    def test_alignment_rejects_reordered_rows(self) -> None:
        graph = _rows(
            [0.51, 0.49, 0.51, 0.49], [0.9, 0.1, 0.9, 0.1]
        )
        non_graph = graph.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "alignment failed"):
            assert_aligned(graph, non_graph)

    def test_graph_recipes_are_single_frozen_defaults(self) -> None:
        self.assertEqual(RECIPE["RCD"]["batch_size"], 1024)
        self.assertEqual(RECIPE["HyperCD"]["batch_size"], 256)
        self.assertEqual(RECIPE["RCD"]["learning_rate"], 1.0e-4)
        self.assertEqual(RECIPE["HyperCD"]["learning_rate"], 1.0e-4)


if __name__ == "__main__":
    unittest.main()
