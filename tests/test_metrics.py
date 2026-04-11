import math
import unittest

from utils.metrics import compute_metrics


class CalibrationMetricsTest(unittest.TestCase):
    def test_compute_metrics_includes_brier_ece_and_bins(self) -> None:
        metrics = compute_metrics(
            labels=[0.0, 0.0, 1.0, 1.0],
            probs=[0.1, 0.4, 0.8, 0.9],
        )

        self.assertAlmostEqual(metrics["brier"], 0.055, places=6)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(0.055), places=6)
        self.assertAlmostEqual(metrics["ece"], 0.2, places=6)
        self.assertEqual(len(metrics["calibration_bins"]), 10)
        self.assertEqual(metrics["calibration_bins"][1]["count"], 1.0)
        self.assertAlmostEqual(metrics["calibration_bins"][1]["accuracy"], 0.0, places=6)
        self.assertAlmostEqual(metrics["calibration_bins"][1]["confidence"], 0.1, places=6)
        self.assertEqual(metrics["calibration_bins"][9]["count"], 1.0)
        self.assertAlmostEqual(metrics["calibration_bins"][9]["accuracy"], 1.0, places=6)
        self.assertAlmostEqual(metrics["calibration_bins"][9]["confidence"], 0.9, places=6)


if __name__ == "__main__":
    unittest.main()
