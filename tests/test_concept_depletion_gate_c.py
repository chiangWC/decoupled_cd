from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
import torch

from data.pool_protocol import sha256_file
from scripts.run_educdm_kancd_concept_depletion_gate import (
    load_official_net_class,
)
from scripts.run_kancd_concept_depletion_gate import (
    FILES,
    verify_protocol_arm,
)
from scripts.summarize_concept_depletion_models import summarize


class ConceptDepletionGateCTest(unittest.TestCase):
    def test_kancd_runner_verifies_frozen_arm_hashes(self) -> None:
        with TemporaryDirectory() as directory:
            protocol = Path(directory)
            arm = protocol / "concept_depleted"
            arm.mkdir()
            for name in FILES:
                pd.DataFrame(
                    {
                        "stu_id": [1],
                        "exer_id": [1],
                        "cpt_seq": [1],
                        "label": [1],
                    }
                ).to_csv(arm / name, index=False)
            hashes = {name: sha256_file(arm / name) for name in FILES}
            manifest = {
                "protocol": "paired_concept_depletion_gate_b",
                "source_test_opened": False,
                "qualified_for_screen": True,
                "generated_hashes": {"concept_depleted": hashes},
            }
            (protocol / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            _, resolved, actual = verify_protocol_arm(
                protocol, "concept_depleted"
            )
            self.assertEqual(resolved, arm)
            self.assertEqual(actual, hashes)

            with (arm / "train.csv").open("a", encoding="utf-8") as handle:
                handle.write("corrupt\n")
            with self.assertRaisesRegex(ValueError, "hashes differ"):
                verify_protocol_arm(protocol, "concept_depleted")

    def test_cross_family_gate_requires_same_three_datasets(self) -> None:
        datasets = ("A", "B", "C", "D", "E")
        support = {
            "ORCDF-NCD": {"A", "B"},
            "SVGCD": {"C", "D"},
            "KaNCD": {"A", "C", "E"},
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for model, supported in support.items():
                path = root / f"{model}.json"
                path.write_text(
                    json.dumps(
                        {
                            "datasets": [
                                {
                                    "dataset": dataset,
                                    "supports_concept_specific_damage": (
                                        dataset in supported
                                    ),
                                    "log_loss_damage_concept_minus_random": (
                                        0.1 if dataset in supported else 0.0
                                    ),
                                }
                                for dataset in datasets
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                paths[model] = path

            payload, table = summarize(paths)

        self.assertEqual(
            payload["common_cross_family_datasets"], ["A", "C"]
        )
        self.assertFalse(payload["admitted"])
        self.assertEqual(int(table["cross_family_support"].sum()), 2)

    def test_cross_family_gate_admits_three_common_datasets(self) -> None:
        datasets = ("A", "B", "C")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for model in ("ORCDF-NCD", "SVGCD", "KaNCD"):
                path = root / f"{model}.json"
                path.write_text(
                    json.dumps(
                        {
                            "datasets": [
                                {
                                    "dataset": dataset,
                                    "supports_concept_specific_damage": True,
                                    "log_loss_damage_concept_minus_random": 0.1,
                                }
                                for dataset in datasets
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                paths[model] = path
            payload, _ = summarize(paths)

        self.assertTrue(payload["admitted"])
        self.assertEqual(payload["common_cross_family_count"], 3)

    def test_official_cross_check_can_reject_primary_admission(self) -> None:
        datasets = ("A", "B", "C")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for model in ("ORCDF-NCD", "SVGCD", "KaNCD"):
                path = root / f"{model}.json"
                path.write_text(
                    json.dumps(
                        {
                            "datasets": [
                                {
                                    "dataset": dataset,
                                    "supports_concept_specific_damage": True,
                                    "log_loss_damage_concept_minus_random": 0.1,
                                }
                                for dataset in datasets
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                paths[model] = path
            official = root / "official.json"
            official.write_text(
                json.dumps(
                    {
                        "datasets": [
                            {
                                "dataset": dataset,
                                "supports_concept_specific_damage": (
                                    dataset == "C"
                                ),
                                "log_loss_damage_concept_minus_random": 0.1,
                            }
                            for dataset in datasets
                        ]
                    }
                ),
                encoding="utf-8",
            )
            payload, table = summarize(paths, official)

        self.assertTrue(payload["admitted"])
        check = payload["official_cross_check"]
        self.assertFalse(check["admitted"])
        self.assertEqual(check["common_cross_family_datasets"], ["C"])
        self.assertEqual(int(table["robust_cross_family_support"].sum()), 1)

    def test_official_educdm_kancd_network_is_callable(self) -> None:
        net_class = load_official_net_class()
        model = net_class(4, 3, 2, "gmf", 5)
        probabilities = model(
            torch.tensor([0, 1]),
            torch.tensor([1, 2]),
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        )

        self.assertEqual(tuple(probabilities.shape), (2,))
        self.assertTrue(bool(torch.isfinite(probabilities).all()))


if __name__ == "__main__":
    unittest.main()
