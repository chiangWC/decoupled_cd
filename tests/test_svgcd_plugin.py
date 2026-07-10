from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SVGCD_ROOT = PROJECT_ROOT / "external" / "SVGCD"
sys.path.insert(0, str(SVGCD_ROOT))

from SVGCD.model import SVGCDNet  # noqa: E402
from SVGCD import trainer as svgcd_trainer  # noqa: E402


def svgcd_args(**overrides):
    values = {
        "emb_dim": 3,
        "dnn_units": [4, 2],
        "dropout_rate": 0.0,
        "n_gnn_layer": 1,
        "cl_tau": 0.7,
        "cl_weight": 0.5,
        "beta": 0.4,
        "lr": 0.001,
        "weight_decay": 0.0,
        "eps": 1e-8,
        "epochs": 2,
        "log_dir": "/tmp",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def empty_graph(nodes: int) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.zeros((2, 0), dtype=torch.long),
        torch.zeros(0),
        (nodes, nodes),
    ).coalesce()


def load_main_plugin():
    spec = importlib.util.spec_from_file_location(
        "svgcd_main_plugin_for_test",
        SVGCD_ROOT / "main_plugin.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingOptimizer:
    def __init__(self, parameter: torch.nn.Parameter) -> None:
        self.parameter = parameter
        self.gradients: list[float] = []
        self.zero_grad_calls = 0

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1
        self.parameter.grad = None

    def step(self) -> None:
        assert self.parameter.grad is not None
        self.gradients.append(float(self.parameter.grad))


class RecordingScheduler:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class ThreeStageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(1.0))

    def cal_loss_cl(self, **_batch):
        loss = self.value * 1.0
        return loss, {"cl_loss": loss}

    def cal_loss_kl(self, **_batch):
        loss = self.value * 10.0
        return loss, {"kl_loss": loss}

    def cal_loss(self, **_batch):
        loss = self.value * 100.0
        return loss, {"main_loss": loss}


class SVGCDPluginTests(unittest.TestCase):
    def test_three_stage_batch_clears_gradients_and_steps_scheduler_each_time(self) -> None:
        model = ThreeStageModel()
        optimizer = RecordingOptimizer(model.value)
        scheduler = RecordingScheduler()

        main_loss, losses = svgcd_trainer.train_three_stage_batch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            batch={},
        )

        self.assertEqual(optimizer.gradients, [1.0, 10.0, 100.0])
        self.assertEqual(optimizer.zero_grad_calls, 3)
        self.assertEqual(scheduler.steps, 3)
        self.assertEqual(float(main_loss), 100.0)
        self.assertEqual(set(losses), {"cl_loss", "kl_loss", "main_loss"})

    def test_one_cycle_budget_counts_all_three_steps_per_batch(self) -> None:
        model = torch.nn.Linear(1, 1)
        args = svgcd_args(epochs=2)
        trainer = svgcd_trainer.Trainer(
            model,
            loaders=([None] * 4, [], []),
            data_proc=None,
            args=args,
            logger=None,
        )

        self.assertEqual(trainer.scheduler.total_steps, 24)

    def test_zero_weight_has_exact_base_state_and_parameters(self) -> None:
        main_plugin = load_main_plugin()
        graph = empty_graph(5)
        torch.manual_seed(11)
        base = SVGCDNet(3, 2, 2, svgcd_args(), graph, graph, "cpu")
        torch.manual_seed(11)
        plugin = main_plugin.AuxSVGCD(
            3,
            2,
            2,
            svgcd_args(),
            graph,
            graph,
            "cpu",
            aux_weight=0.0,
        )

        self.assertEqual(base.state_dict().keys(), plugin.state_dict().keys())
        self.assertEqual(
            dict(base.named_parameters()).keys(),
            dict(plugin.named_parameters()).keys(),
        )
        for name, expected in base.state_dict().items():
            self.assertTrue(torch.equal(expected, plugin.state_dict()[name]), name)

    def test_auxiliary_scale_is_shared_fixed_buffer(self) -> None:
        main_plugin = load_main_plugin()
        graph = empty_graph(5)
        plugin = main_plugin.AuxSVGCD(
            3,
            2,
            2,
            svgcd_args(),
            graph,
            graph,
            "cpu",
            aux_weight=0.5,
        )

        parameters = dict(plugin.named_parameters())
        buffers = dict(plugin.named_buffers())
        self.assertNotIn("aux_scale", parameters)
        self.assertNotIn("mastery_auxiliary.scale", parameters)
        self.assertIn("mastery_auxiliary.scale", buffers)
        self.assertAlmostEqual(
            float(buffers["mastery_auxiliary.scale"]),
            2.126928,
            places=6,
        )

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
