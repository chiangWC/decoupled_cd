from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from scripts import doa_external


class DoaExternalSplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.split_dir = self.root / "splits"
        self.mastery_dir = self.root / "mastery"
        self.split_dir.mkdir()
        self.mastery_dir.mkdir()
        np.save(self.mastery_dir / "mastery.npy", np.zeros((1, 1)))
        (self.mastery_dir / "id_maps.json").write_text(
            json.dumps({"stu_ids": ["student-1"], "cpt_ids": ["concept-1"]}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_main(self, split: str | None) -> Path:
        argv = [
            "doa_external.py",
            "--dataset-name",
            "fixture",
            "--split-dir",
            str(self.split_dir),
            "--mastery-dir",
            str(self.mastery_dir),
            "--model-name",
            "fixture-model",
            "--output-csv",
            str(self.root / "doa.csv"),
        ]
        if split is not None:
            argv.extend(["--split", split])
        frame = pd.DataFrame(
            {
                "stu_id": ["student-1"],
                "cpt_seq": ["concept-1"],
                "label": [1.0],
            }
        )
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(doa_external.pd, "read_csv", return_value=frame) as read_csv,
            mock.patch.object(doa_external.np, "load", return_value=np.zeros((1, 1))),
            mock.patch.object(doa_external, "compute_doa", return_value={"doa": 0.5}),
        ):
            doa_external.main()
        return Path(read_csv.call_args.args[0])

    def test_default_split_reads_test_csv(self) -> None:
        self.assertEqual(self.run_main(None), self.split_dir / "test.csv")

    def test_valid_split_reads_valid_csv(self) -> None:
        self.assertEqual(self.run_main("valid"), self.split_dir / "valid.csv")

    def test_output_binds_split_protocol_and_artifact_hashes(self) -> None:
        self.run_main("valid")

        row = pd.read_csv(self.root / "doa.csv").iloc[0]
        self.assertEqual(row["split"], "valid")
        self.assertEqual(row["doa_seed"], 2024)
        self.assertEqual(row["min_responses"], 1)
        self.assertEqual(row["max_pairs_per_concept"], 100_000)
        self.assertEqual(
            row["mastery_sha256"],
            hashlib.sha256((self.mastery_dir / "mastery.npy").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            row["id_maps_sha256"],
            hashlib.sha256((self.mastery_dir / "id_maps.json").read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
