from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORCDF_ROOT = PROJECT_ROOT / "external" / "ORCDF"
sys.path.insert(0, str(ORCDF_ROOT))

from ORCDF import plugin as orcdf_plugin  # noqa: E402
from ORCDF.model import ORCDFNet  # noqa: E402


def model_kwargs() -> dict[str, object]:
    return {
        "student_n": 3,
        "exer_n": 2,
        "knowledge_n": 2,
        "latent_dim": 2,
        "gcn_layers": 1,
        "keep_prob": 1.0,
        "if_type": "ncd",
        "mode": "all",
        "flip_ratio": 0.0,
        "ssl_temp": 0.5,
        "ssl_weight": 0.0,
        "prednet_len1": 4,
        "prednet_len2": 2,
        "dropout": 0.0,
        "device": "cpu",
    }


def load_main_plugin():
    spec = importlib.util.spec_from_file_location(
        "orcdf_main_plugin_for_test",
        ORCDF_ROOT / "main_plugin.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ORCDFPluginTests(unittest.TestCase):
    def test_zero_weight_has_exact_base_state_and_parameter_values(self) -> None:
        torch.manual_seed(7)
        base = ORCDFNet(**model_kwargs())
        torch.manual_seed(7)
        plugin = orcdf_plugin.DecoupledORCDF(
            **model_kwargs(),
            decouple=False,
            aux_weight=0.0,
        )

        self.assertEqual(base.state_dict().keys(), plugin.state_dict().keys())
        self.assertEqual(
            dict(base.named_parameters()).keys(),
            dict(plugin.named_parameters()).keys(),
        )
        for name, expected in base.state_dict().items():
            self.assertTrue(torch.equal(expected, plugin.state_dict()[name]), name)

    def test_auxiliary_scale_is_a_fixed_buffer_not_a_parameter(self) -> None:
        model = orcdf_plugin.DecoupledORCDF(
            **model_kwargs(),
            aux_weight=0.5,
        )

        parameters = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        self.assertNotIn("aux_scale", parameters)
        self.assertNotIn("aux_scale", buffers)
        self.assertNotIn("mastery_auxiliary.scale", parameters)
        self.assertIn("mastery_auxiliary.scale", buffers)
        self.assertAlmostEqual(
            float(buffers["mastery_auxiliary.scale"]),
            2.126928,
            places=6,
        )

    def test_decouple_gate_joins_optimizer_once_without_changing_base_groups(self) -> None:
        model = orcdf_plugin.DecoupledORCDF(
            **model_kwargs(),
            decouple=True,
            aux_weight=0.5,
        )
        optimizer = torch.optim.Adam(
            [
                {"params": model.extractor.parameters(), "lr": 0.004},
                {"params": model.inter_func.parameters(), "lr": 0.004},
            ]
        )
        base_group_ids = [
            [id(parameter) for parameter in group["params"]]
            for group in optimizer.param_groups
        ]

        orcdf_plugin.add_plugin_parameters_to_optimizer(
            optimizer,
            model,
            lr=0.004,
            weight_decay=0.0,
        )
        orcdf_plugin.add_plugin_parameters_to_optimizer(
            optimizer,
            model,
            lr=0.004,
            weight_decay=0.0,
        )

        self.assertEqual(len(optimizer.param_groups), 3)
        self.assertEqual(
            base_group_ids,
            [[id(p) for p in group["params"]] for group in optimizer.param_groups[:2]],
        )
        gate_occurrences = sum(
            parameter is model.ukc_gate
            for group in optimizer.param_groups
            for parameter in group["params"]
        )
        self.assertEqual(gate_occurrences, 1)

    def test_no_decouple_means_no_plugin_optimizer_group(self) -> None:
        model = orcdf_plugin.DecoupledORCDF(
            **model_kwargs(),
            decouple=False,
            aux_weight=0.5,
        )
        optimizer = torch.optim.Adam(model.extractor.parameters(), lr=0.004)

        orcdf_plugin.add_plugin_parameters_to_optimizer(
            optimizer,
            model,
            lr=0.004,
            weight_decay=0.0,
        )

        self.assertEqual(len(optimizer.param_groups), 1)

    def test_cli_accepts_auxiliary_detach_and_warmup(self) -> None:
        main_plugin = load_main_plugin()
        argv = [
            "main_plugin.py",
            "--plugin-aux-detach-item-difficulty",
            "--plugin-aux-warmup-fraction",
            "0.25",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = main_plugin.parse_all()

        self.assertTrue(args.plugin_aux_detach_item_difficulty)
        self.assertEqual(args.plugin_aux_warmup_fraction, 0.25)


if __name__ == "__main__":
    unittest.main()
