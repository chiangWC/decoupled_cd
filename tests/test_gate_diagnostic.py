import unittest

import pandas as pd

from scripts.evaluate_gate_diagnostic import (
    aggregate_bin_rows,
    format_coverage_bin,
    parse_bin_edges,
    summary_row,
)


class GateDiagnosticHelpersTest(unittest.TestCase):
    def test_parse_bin_edges_requires_strictly_increasing_edges(self) -> None:
        self.assertEqual(parse_bin_edges("0,0.5,1"), [0.0, 0.5, 1.0])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            parse_bin_edges("0,0.5,0.5,1")

    def test_format_coverage_bin_uses_right_closed_final_bin(self) -> None:
        edges = [0.0, 0.5, 1.0]

        self.assertEqual(format_coverage_bin(0.0, edges), "[0.00,0.50)")
        self.assertEqual(format_coverage_bin(0.5, edges), "[0.50,1.00]")
        self.assertEqual(format_coverage_bin(1.0, edges), "[0.50,1.00]")
        self.assertEqual(format_coverage_bin(1.1, edges), "out_of_range")

    def test_summary_row_uses_none_when_correlation_is_degenerate(self) -> None:
        frame = pd.DataFrame(
            {
                "student_global_coverage": [0.5, 0.5, 0.5],
                "target_coverage": [0.0, 0.5, 1.0],
                "tkc_weight": [0.7, 0.7, 0.7],
                "tkc_contribution_norm": [0.7, 0.7, 0.7],
                "ukc_contribution_norm": [0.3, 0.3, 0.3],
                "effective_tkc_share": [0.7, 0.7, 0.7],
            }
        )

        row = summary_row(frame, dataset_name="toy", model_name="model", tower="single")

        self.assertIsNone(row["corr_student_global_coverage_tkc_weight"])
        self.assertIsNone(row["corr_target_coverage_tkc_weight"])

    def test_aggregate_bin_rows_reports_student_and_target_axes(self) -> None:
        frame = pd.DataFrame(
            {
                "student_global_coverage": [0.1, 0.2, 0.8],
                "target_coverage": [0.0, 0.5, 1.0],
                "coverage_bucket": ["zero", "partial", "full"],
                "tkc_weight": [0.4, 0.6, 0.9],
                "tkc_contribution_norm": [0.4, 0.6, 0.9],
                "ukc_contribution_norm": [0.6, 0.4, 0.1],
                "effective_tkc_share": [0.4, 0.6, 0.9],
            }
        )

        rows = aggregate_bin_rows(
            frame,
            dataset_name="toy",
            model_name="model",
            tower="single",
            student_coverage_bins=[0.0, 0.5, 1.0],
        )

        axes = {row["axis"] for row in rows}
        self.assertEqual(axes, {"student_global_coverage", "target_coverage"})
        student_low = next(row for row in rows if row["axis"] == "student_global_coverage" and row["bin"] == "[0.00,0.50)")
        self.assertEqual(student_low["count"], 2)
        self.assertAlmostEqual(student_low["mean_tkc_weight"], 0.5)
        self.assertAlmostEqual(student_low["mean_effective_tkc_share"], 0.5)


if __name__ == "__main__":
    unittest.main()
