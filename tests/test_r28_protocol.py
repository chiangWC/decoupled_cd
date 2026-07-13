from __future__ import annotations

from argparse import Namespace
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


class R28ProtocolTests(unittest.TestCase):
    def test_first_screen_builds_exactly_twelve_tasks(self) -> None:
        runner = load_script("run_r28_campaign.py")
        args = Namespace(
            dataset=["moocradar", "xes3g5m"],
            dataset_variant=None,
            state_completer=None,
            completion_objective="none",
            output_root="results/r28/test",
        )
        tasks = runner.build_tasks(args)
        self.assertEqual(len(tasks), 12)
        self.assertEqual(
            {task.state_completer for task in tasks},
            {"relational", "direct_prior", "capacity_mlp"},
        )
        self.assertEqual({task.dataset_variant for task in tasks}, {"standard", "holdout"})

    def test_gpu_selection_respects_running_reservations(self) -> None:
        runner = load_script("run_r28_campaign.py")
        stats = {
            0: {
                "memory_total_mb": 24000,
                "memory_used_mb": 1000,
                "memory_free_mb": 23000,
                "utilization_percent": 10,
            },
            1: {
                "memory_total_mb": 24000,
                "memory_used_mb": 3000,
                "memory_free_mb": 21000,
                "utilization_percent": 40,
            },
        }
        with patch.object(runner, "gpu_stats", return_value=stats):
            selected = runner.choose_gpu(
                candidates=[0, 1],
                expected_peak_mb=9000,
                reservations={0: 15000},
            )
        self.assertEqual(selected, 1)

    def test_difficulty_replacement_builds_clean_twelve_task_screen(self) -> None:
        runner = load_script("run_r28_campaign.py")
        args = Namespace(
            dataset=["moocradar", "xes3g5m"],
            dataset_variant=None,
            state_completer=["difficulty_set", "direct_prior", "difficulty_capacity"],
            completion_objective="none",
            output_root="results/r28/test_difficulty",
        )
        tasks = runner.build_tasks(args)
        self.assertEqual(len(tasks), 12)
        self.assertEqual(
            {task.state_completer for task in tasks},
            {"difficulty_set", "direct_prior", "difficulty_capacity"},
        )

    def test_poe_replacement_builds_clean_twelve_task_screen(self) -> None:
        runner = load_script("run_r28_campaign.py")
        args = Namespace(
            dataset=["moocradar", "xes3g5m"],
            dataset_variant=None,
            state_completer=["poe_ability", "direct_prior", "poe_capacity"],
            completion_objective="none",
            output_root="results/r28/test_poe",
        )
        tasks = runner.build_tasks(args)
        self.assertEqual(len(tasks), 12)

    def test_hierarchical_replacement_builds_clean_twelve_task_screen(self) -> None:
        runner = load_script("run_r28_campaign.py")
        args = Namespace(
            dataset=["moocradar", "xes3g5m"],
            dataset_variant=None,
            state_completer=[
                "hierarchical_bayes",
                "direct_prior",
                "hierarchical_capacity",
            ],
            completion_objective="none",
            output_root="results/r28/test_hierarchical",
        )
        tasks = runner.build_tasks(args)
        self.assertEqual(len(tasks), 12)

    def test_cohort_replacement_builds_clean_twelve_task_screen(self) -> None:
        runner = load_script("run_r28_campaign.py")
        args = Namespace(
            dataset=["moocradar", "xes3g5m"],
            dataset_variant=None,
            state_completer=[
                "cohort_conditioned",
                "direct_prior",
                "cohort_capacity",
            ],
            completion_objective="none",
            output_root="results/r28/test_cohort",
        )
        tasks = runner.build_tasks(args)
        self.assertEqual(len(tasks), 12)

    def test_bipolar_replacement_builds_clean_twelve_task_screen(self) -> None:
        runner = load_script("run_r28_campaign.py")
        args = Namespace(
            dataset=["moocradar", "xes3g5m"],
            dataset_variant=None,
            state_completer=[
                "bipolar_prototype",
                "direct_prior",
                "bipolar_capacity",
            ],
            completion_objective="none",
            output_root="results/r28/test_bipolar",
        )
        tasks = runner.build_tasks(args)
        self.assertEqual(len(tasks), 12)

    def test_standard_and_holdout_resolve_the_same_recipe(self) -> None:
        trainer = load_script("train_r28.py")
        recipes = trainer.load_json(ROOT / "configs/r28_recipes.json")
        base = dict(
            dataset="moocradar",
            epochs=None,
            student_batch_size=None,
            learning_rate=None,
            weight_decay=None,
            concept_dim=None,
            early_stop_patience=None,
            scheduler_patience=None,
            seed=42,
        )
        standard = trainer.resolve_recipe(Namespace(dataset_variant="standard", **base), recipes)
        holdout = trainer.resolve_recipe(Namespace(dataset_variant="holdout", **base), recipes)
        self.assertEqual(standard, holdout)

    def test_factorial_diagnostic_redirects_legacy_test_to_validation(self) -> None:
        text = (ROOT / "scripts/run_r28_factorial_diagnostic.py").read_text()
        self.assertIn('"--test-interactions", str(valid_path)', text)
        self.assertNotIn('"test_rows_read": false', text)
        self.assertIn('"test_rows_read": False', text)


if __name__ == "__main__":
    unittest.main()
