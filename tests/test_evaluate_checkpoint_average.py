from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_checkpoint_average import (
    _average_predictions,
    _validate_summary_compatibility,
)


class CheckpointAverageEvaluatorTest(unittest.TestCase):
    def test_probability_average_uses_mean_probability(self) -> None:
        averaged = _average_predictions(
            [
                np.array([0.2, 0.8], dtype=np.float64),
                np.array([0.4, 0.6], dtype=np.float64),
            ],
            "prob",
        )

        np.testing.assert_allclose(averaged, np.array([0.3, 0.7], dtype=np.float64))

    def test_logit_average_uses_mean_log_odds(self) -> None:
        averaged = _average_predictions(
            [
                np.array([0.2, 0.8], dtype=np.float64),
                np.array([0.4, 0.6], dtype=np.float64),
            ],
            "logit",
        )

        expected_logits = np.mean(
            [
                np.log(np.array([0.2, 0.8]) / np.array([0.8, 0.2])),
                np.log(np.array([0.4, 0.6]) / np.array([0.6, 0.4])),
            ],
            axis=0,
        )
        expected = 1.0 / (1.0 + np.exp(-expected_logits))
        np.testing.assert_allclose(averaged, expected)

    def test_requires_at_least_two_summaries(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two"):
            _validate_summary_compatibility(["a.json"], [{"train_interactions": "train.csv"}])

    def test_rejects_mismatched_split_signature(self) -> None:
        first = {
            "train_interactions": "train.csv",
            "valid_interactions": "valid.csv",
            "test_interactions": "test.csv",
            "q_matrix": "q.csv",
            "concept_graph": "graph.csv",
            "graph_mode": "single",
        }
        second = dict(first)
        second["test_interactions"] = "other-test.csv"

        with self.assertRaisesRegex(ValueError, "does not match"):
            _validate_summary_compatibility(["a.json", "b.json"], [first, second])


if __name__ == "__main__":
    unittest.main()
