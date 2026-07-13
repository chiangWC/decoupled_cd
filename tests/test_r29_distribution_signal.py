from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.audit_r29_distribution_signal import (
    prepare_context,
    rff_features,
    rff_parameters,
)


class R29DistributionSignalTest(unittest.TestCase):
    def test_rff_is_fixed_and_has_requested_shape(self) -> None:
        values = np.asarray([[0.2, -1.0], [0.8, 1.0]])
        first = rff_parameters(dim=32, bandwidth=0.5, seed=42)
        second = rff_parameters(dim=32, bandwidth=0.5, seed=42)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertTrue(np.array_equal(first[1], second[1]))
        features = rff_features(values, weights=first[0], phases=first[1])
        self.assertEqual(features.shape, (2, 32))

    def test_pseudo_holdout_is_disjoint_and_deterministic(self) -> None:
        frame = pd.DataFrame(
            {
                "stu_id": [f"s{i // 4}" for i in range(100)],
                "exer_id": [f"e{i % 7}" for i in range(100)],
                "label": [i % 2 for i in range(100)],
                "source_row_id": [f"row-{i}" for i in range(100)],
            }
        )
        first = prepare_context(frame, seed=42, mask_fraction=0.2)
        second = prepare_context(frame, seed=42, mask_fraction=0.2)
        self.assertEqual(first[0]["source_row_id"].tolist(), second[0]["source_row_id"].tolist())
        self.assertEqual(first[1]["source_row_id"].tolist(), second[1]["source_row_id"].tolist())
        self.assertFalse(set(first[0]["source_row_id"]) & set(first[1]["source_row_id"]))


if __name__ == "__main__":
    unittest.main()
