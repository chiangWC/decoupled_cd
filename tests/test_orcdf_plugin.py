from __future__ import annotations

import importlib.util
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from scripts.plugin_campaign import EvaluationArtifactSnapshot


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
    def test_training_initialization_loader_only_allows_exact_plugin_owned_missing_keys(
        self,
    ) -> None:
        main_plugin = load_main_plugin()

        class Objective(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("scale", torch.tensor(2.0))

        class Target(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = torch.nn.Linear(2, 2, bias=False)
                self.mastery_auxiliary = Objective()
                self.ukc_gate = torch.nn.Parameter(torch.tensor(0.5))
                self.flip_calls = 0

            def get_flip_graph(self):
                self.flip_calls += 1

        def checkpoint_bytes(state):
            buffer = io.BytesIO()
            torch.save(state, buffer)
            return buffer.getvalue()

        target = Target()
        expected_backbone = torch.full_like(target.backbone.weight, 3.0)
        main_plugin.load_training_initialization_checkpoint(
            target,
            checkpoint_bytes({"backbone.weight": expected_backbone}),
            "cpu",
        )
        self.assertTrue(torch.equal(target.backbone.weight, expected_backbone))
        self.assertEqual(float(target.mastery_auxiliary.scale), 2.0)
        self.assertEqual(float(target.ukc_gate), 0.5)
        self.assertEqual(target.flip_calls, 1)

        invalid_states = (
            {},
            {
                "backbone.weight": expected_backbone,
                "unexpected": torch.tensor(1.0),
            },
            {"backbone.weight": torch.ones(3, 2)},
            {"backbone.weight": expected_backbone.to(torch.float64)},
        )
        for state in invalid_states:
            with self.subTest(keys=sorted(state)):
                with self.assertRaises(RuntimeError):
                    main_plugin.load_training_initialization_checkpoint(
                        Target(), checkpoint_bytes(state), "cpu"
                    )

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
            "--plugin-mode",
            "train",
            "--plugin-aux-detach-item-difficulty",
            "--plugin-aux-warmup-fraction",
            "0.25",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = main_plugin.parse_all()

        self.assertTrue(args.plugin_aux_detach_item_difficulty)
        self.assertEqual(args.plugin_aux_warmup_fraction, 0.25)
        self.assertEqual(args.plugin_q_matrix_file, Path("Q_matrix.csv"))

    def test_cli_and_configs_bind_baseline_initialization_without_split_checkpoint_in_recipe(
        self,
    ) -> None:
        main_plugin = load_main_plugin()
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = []
            configs = []
            recipes = []
            for split, payload in (("standard", b"standard"), ("holdout", b"holdout")):
                checkpoint = Path(directory) / split / "checkpoint.pth"
                checkpoint.parent.mkdir()
                checkpoint.write_bytes(payload)
                checkpoints.append(checkpoint)
                argv = [
                    "main_plugin.py",
                    "--plugin-mode",
                    "train",
                    "--plugin-init-mode",
                    "baseline-finetune",
                    "--plugin-init-checkpoint",
                    str(checkpoint),
                    "--plugin-lr-multiplier",
                    "0.25",
                ]
                with mock.patch.object(sys, "argv", argv):
                    args = main_plugin.parse_all()
                configs.append(main_plugin.plugin_config(args))
                recipes.append(main_plugin.recipe_config(args))

        self.assertEqual(configs[0]["init_mode"], "baseline-finetune")
        self.assertEqual(configs[0]["base_lr"], recipes[0]["base_lr"])
        self.assertEqual(configs[0]["lr_multiplier"], 0.25)
        self.assertEqual(
            configs[0]["baseline_checkpoint_sha256"],
            hashlib.sha256(b"standard").hexdigest(),
        )
        self.assertNotEqual(
            configs[0]["baseline_checkpoint_sha256"],
            configs[1]["baseline_checkpoint_sha256"],
        )
        self.assertEqual(recipes[0], recipes[1])
        self.assertNotIn("checkpoint", " ".join(recipes[0]))

        missing = [
            "main_plugin.py",
            "--plugin-mode",
            "train",
            "--plugin-init-mode",
            "baseline-finetune",
        ]
        with mock.patch.object(sys, "argv", missing):
            with self.assertRaises(SystemExit):
                main_plugin.parse_all()
        evaluate = [
            "main_plugin.py",
            "--plugin-mode",
            "evaluate",
            "--plugin-checkpoint",
            str(checkpoints[0]),
            "--plugin-eval-split",
            "valid",
            "--plugin-init-mode",
            "baseline-finetune",
            "--plugin-init-checkpoint",
            str(checkpoints[0]),
        ]
        with mock.patch.object(sys, "argv", evaluate):
            with self.assertRaises(SystemExit):
                main_plugin.parse_all()

    def test_baseline_initialization_loads_before_trainer_and_applies_effective_lr(
        self,
    ) -> None:
        main_plugin = load_main_plugin()
        events = []
        snapshot = EvaluationArtifactSnapshot(b"checkpoint", b"{}\n", b"q")
        args = SimpleNamespace(
            plugin_init_mode="baseline-finetune",
            plugin_init_checkpoint=Path("/baseline/checkpoint.pth"),
            plugin_lr_multiplier=0.25,
            plugin_data_protocol="standard",
            plugin_dataset_name="fixture",
            seed=42,
            lr=0.004,
        )
        proc = SimpleNamespace(stu_ids=[], exer_ids=[], cpt_ids=[])

        with (
            mock.patch.object(
                main_plugin,
                "snapshot_evaluation_artifacts",
                side_effect=lambda *_: events.append("snapshot") or snapshot,
            ),
            mock.patch.object(
                main_plugin,
                "validate_training_initialization",
                side_effect=lambda **_: events.append("validate"),
            ),
            mock.patch.object(
                main_plugin,
                "load_training_initialization_checkpoint",
                side_effect=lambda *_: events.append("load"),
            ),
        ):
            trainer_args = main_plugin.prepare_training_initialization(
                args=args,
                model=object(),
                proc=proc,
                device="cpu",
                q_matrix_path=Path("/data/Q_matrix.csv"),
                q_matrix_bytes=b"q",
            )

        events.append("trainer")
        self.assertEqual(events, ["snapshot", "validate", "load", "trainer"])
        self.assertEqual(args.lr, 0.004)
        self.assertEqual(trainer_args.lr, 0.001)

    def test_campaign_protocol_and_recipe_bind_joint_split(self) -> None:
        main_plugin = load_main_plugin()
        self.assertTrue(
            hasattr(main_plugin, "recipe_config"),
            "ORCDF runner must expose recipe_config",
        )
        argv = [
            "main_plugin.py",
            "--plugin-mode",
            "train",
            "--plugin-decouple",
            "--plugin-aux-weight",
            "0.1",
            "--plugin-aux-detach-item-difficulty",
            "--plugin-aux-warmup-fraction",
            "0.25",
            "--plugin-data-protocol",
            "standard",
            "--plugin-dataset-name",
            "assist17",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = main_plugin.parse_all()

        protocol = main_plugin.campaign_protocol(args, b"q-matrix")
        self.assertEqual(protocol["data_protocol"], "standard")
        self.assertEqual(protocol["dataset_name"], "assist17")
        self.assertEqual(
            main_plugin.recipe_config(args),
            {
                "decouple": True,
                "aux_weight": 0.1,
                "aux_detach_item_difficulty": True,
                "aux_warmup_fraction": 0.25,
            },
        )

        args.plugin_dataset_name = None
        with self.assertRaisesRegex(ValueError, "provided together"):
            main_plugin.campaign_protocol(args, b"q-matrix")

    def test_backbone_config_binds_every_orcdf_model_argument_exactly(self) -> None:
        main_plugin = load_main_plugin()
        values = {
            key: value
            for key, value in model_kwargs().items()
            if key not in {"student_n", "exer_n", "knowledge_n", "device"}
        }

        self.assertEqual(main_plugin.backbone_config(SimpleNamespace(**values)), values)

    def test_plugin_q_matrix_path_defaults_under_data_dir_and_must_exist(self) -> None:
        main_plugin = load_main_plugin()
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            args = SimpleNamespace(
                data_dir=str(data_dir),
                plugin_q_matrix_file=Path("Q_matrix.csv"),
            )
            with self.assertRaises(FileNotFoundError):
                main_plugin.resolve_plugin_q_matrix(args)
            q_matrix = data_dir / "Q_matrix.csv"
            q_matrix.write_text("exer_id,cpt_seq\n1,1\n", encoding="utf-8")
            self.assertEqual(main_plugin.resolve_plugin_q_matrix(args), q_matrix)

    def test_cli_requires_explicit_plugin_mode(self) -> None:
        main_plugin = load_main_plugin()

        with mock.patch.object(sys, "argv", ["main_plugin.py"]):
            with self.assertRaises(SystemExit):
                main_plugin.parse_all()

    def test_evaluate_cli_requires_checkpoint_split_ledger_and_selection(self) -> None:
        main_plugin = load_main_plugin()
        incomplete = [
            "main_plugin.py",
            "--plugin-mode",
            "evaluate",
            "--plugin-checkpoint",
            "/tmp/checkpoint.pth",
            "--plugin-eval-split",
            "test",
        ]
        with mock.patch.object(sys, "argv", incomplete):
            with self.assertRaises(SystemExit):
                main_plugin.parse_all()

        complete = incomplete + [
            "--plugin-test-ledger-dir",
            "/tmp/test-ledger",
            "--plugin-selection-json",
            "/tmp/selection.json",
        ]
        with mock.patch.object(sys, "argv", complete):
            args = main_plugin.parse_all()

        self.assertEqual(args.plugin_mode, "evaluate")
        self.assertEqual(args.plugin_eval_split, "test")
        self.assertEqual(args.plugin_selection_json, Path("/tmp/selection.json"))
        self.assertIsNone(args.plugin_frozen_config_id)

    def test_fresh_evaluation_initializes_orcdf_flip_graph(self) -> None:
        main_plugin = load_main_plugin()
        events: list[object] = []

        class RecordingModel:
            def load_state_dict(self, state_dict):
                events.append(("load", state_dict))

            def get_flip_graph(self):
                events.append("flip")

        with mock.patch.object(
            main_plugin.torch,
            "load",
            return_value={"weight": "state"},
        ):
            main_plugin.load_evaluation_checkpoint(
                RecordingModel(),
                b"serialized-checkpoint",
                "cpu",
            )

        self.assertEqual(events, [("load", {"weight": "state"}), "flip"])


if __name__ == "__main__":
    unittest.main()
