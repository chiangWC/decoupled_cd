from __future__ import annotations

import unittest

from utils.external_benchmark import classify_external_win


class ExternalBenchmarkTest(unittest.TestCase):
    def test_ordinary_and_strict_boundaries(self) -> None:
        self.assertEqual(
            classify_external_win({"S": -0.002, "H": -0.002, "T": 1e-9}),
            (True, False),
        )
        self.assertEqual(
            classify_external_win({"S": 0.0, "H": 0.0, "T": 1e-9}),
            (True, True),
        )

    def test_target_tie_is_always_a_loss(self) -> None:
        self.assertEqual(
            classify_external_win({"S": 0.1, "H": 0.1, "T": 0.0}),
            (False, False),
        )


if __name__ == "__main__":
    unittest.main()
