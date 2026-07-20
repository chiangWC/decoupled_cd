from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from data.pool_protocol import canonicalize_interactions, sha256_file
from models.beta_ncd_baseline import (
    AdaptedPosterior,
    BetaNCDConfig,
    BetaNCDBaseline,
    PositiveLinear,
)
from scripts.train_beta_ncd import (
    StudentTask,
    build_q_union,
    evaluate_tasks,
    load_verified_validation_protocol,
    split_optimizer_train,
)


def _tiny_model() -> BetaNCDBaseline:
    torch.manual_seed(42)
    q_matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    model = BetaNCDBaseline(
        config=BetaNCDConfig(num_items=4, num_concepts=3),
        q_matrix=q_matrix,
    )
    model.eval()
    return model


def _task(student: str, *, query_index: int = 0, flipped: bool = False) -> StudentTask:
    support_labels = np.asarray(
        [1.0, 0.0, 1.0] if not flipped else [0.0, 1.0, 0.0],
        dtype=np.float32,
    )
    support = pd.DataFrame(
        {
            "source_row_id": ["s0", "s1", "s2"],
            "split_row_index": [0, 1, 2],
            "stu_id": [student] * 3,
            "exer_id": ["0", "1", "3"],
            "cpt_seq": ["0", "1", "0,1"],
            "label": support_labels.astype(int),
        }
    )
    query = pd.DataFrame(
        {
            "source_row_id": [f"q{query_index}"],
            "split_row_index": [query_index],
            "stu_id": [student],
            "exer_id": ["2"],
            "cpt_seq": ["2"],
            "label": [1],
        }
    )
    return StudentTask(
        student_id=student,
        support_item_ids=np.asarray([0, 1, 3], dtype=np.int64),
        support_labels=support_labels,
        query_item_ids=np.asarray([2], dtype=np.int64),
        query_labels=np.asarray([1.0], dtype=np.float32),
        support_frame=support,
        query_frame=query,
    )


def _canonical_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame, removed = canonicalize_interactions(pd.DataFrame(rows))
    assert removed == 0
    frame.insert(1, "split_row_index", range(len(frame)))
    return frame


class BetaNCDBaselineTest(unittest.TestCase):
    def test_gaussian_kl_is_zero_only_for_equal_distributions(self) -> None:
        zeros = torch.zeros(3)
        self.assertAlmostEqual(
            float(BetaNCDBaseline.gaussian_kl(zeros, zeros, zeros, zeros)),
            0.0,
            places=7,
        )
        shifted = BetaNCDBaseline.gaussian_kl(
            torch.ones(3), zeros, zeros, zeros
        )
        self.assertGreater(float(shifted), 0.0)

    def test_author_variational_initialization_and_learnable_lrs(self) -> None:
        torch.manual_seed(19)
        num_concepts = 2048
        model = BetaNCDBaseline(
            config=BetaNCDConfig(num_items=2, num_concepts=num_concepts),
            q_matrix=torch.ones((2, num_concepts)),
        )
        prior_mean = model.prior_mean.detach()
        prior_log_std = model.prior_log_std.detach()
        # Pinned NormalVariationalNet: randn_like(mean), rand_like(log_std) - 4.
        self.assertLess(abs(float(prior_mean.mean())), 0.08)
        self.assertGreater(float(prior_mean.std()), 0.9)
        self.assertLess(float(prior_mean.std()), 1.1)
        self.assertTrue(torch.all(prior_log_std >= -4.0))
        self.assertTrue(torch.all(prior_log_std < -3.0))
        self.assertGreater(float(prior_log_std.mean()), -3.55)
        self.assertLess(float(prior_log_std.mean()), -3.45)
        torch.testing.assert_close(
            model.inner_lrs.detach(),
            torch.full((3,), 0.1),
        )
        self.assertTrue(model.inner_lrs.requires_grad)

    def test_author_ncd_activation_absolute_weight_and_item_scale(self) -> None:
        model = _tiny_model()
        self.assertIsInstance(model.interaction[1], torch.nn.ReLU)
        self.assertIsInstance(model.interaction[4], torch.nn.ReLU)
        self.assertFalse(
            any(
                isinstance(layer, torch.nn.Sigmoid)
                for layer in model.interaction
            )
        )
        self.assertTrue(hasattr(model, "item_difficulty"))
        self.assertFalse(hasattr(model, "item_discrimination"))
        self.assertGreater(
            float(model.interaction[0].bias.detach().abs().sum()), 0.0
        )

        with torch.no_grad():
            model.knowledge_difficulty.weight.zero_()
            model.item_difficulty.weight.zero_()
        latent = torch.logit(torch.tensor([0.8, 0.2, 0.5]))
        actual = model.ncd_interaction_input(latent, torch.tensor([3]))
        # sigmoid(item_difficulty=0)=0.5, with no x10 multiplier.
        expected = torch.tensor([[0.15, -0.15, 0.0]])
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)

        positive = PositiveLinear(2, 1, bias=False)
        with torch.no_grad():
            positive.weight.copy_(torch.tensor([[-2.0, 3.0]]))
        torch.testing.assert_close(
            positive(torch.ones((1, 2))),
            torch.tensor([[5.0]]),
            atol=0.0,
            rtol=0.0,
        )

    def test_exactly_three_inner_steps_each_use_full_support(self) -> None:
        model = _tiny_model()
        item_ids = torch.tensor([0, 1, 2, 3, 0])
        labels = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
        with patch.object(
            model,
            "predictive_samples",
            wraps=model.predictive_samples,
        ) as predictive:
            model.adapt(
                support_item_ids=item_ids,
                support_labels=labels,
                create_graph=False,
                fixed_noise=True,
            )
        self.assertEqual(predictive.call_count, 3)
        for call in predictive.call_args_list:
            torch.testing.assert_close(call.kwargs["item_ids"], item_ids)

    def test_outer_loss_is_joint_query_log_mean_likelihood(self) -> None:
        model = _tiny_model()
        posterior = AdaptedPosterior(
            mean=torch.zeros(3), log_std=torch.zeros(3)
        )
        samples = torch.tensor(
            [[0.8, 0.25], [0.6, 0.5], [0.4, 0.75], [0.2, 0.1]]
        )
        labels = torch.tensor([1.0, 0.0])
        sample_joint_likelihood = torch.tensor(
            [0.8 * 0.75, 0.6 * 0.5, 0.4 * 0.25, 0.2 * 0.9]
        )
        expected = -torch.log(sample_joint_likelihood.mean()) / 2.0
        with patch.object(
            model,
            "predictive_samples",
            return_value=samples,
        ):
            actual = model.query_meta_loss(
                posterior=posterior,
                query_item_ids=torch.tensor([0, 1]),
                query_labels=labels,
                fixed_noise=True,
            )
        torch.testing.assert_close(actual, expected)

    def test_query_labels_do_not_change_support_adaptation(self) -> None:
        model = _tiny_model()
        support_items = torch.tensor([0, 1, 3])
        support_labels = torch.tensor([1.0, 0.0, 1.0])
        posterior_a = model.adapt(
            support_item_ids=support_items,
            support_labels=support_labels,
            create_graph=True,
            fixed_noise=True,
        )
        posterior_b = model.adapt(
            support_item_ids=support_items,
            support_labels=support_labels,
            create_graph=True,
            fixed_noise=True,
        )
        torch.testing.assert_close(posterior_a.mean, posterior_b.mean)
        torch.testing.assert_close(posterior_a.log_std, posterior_b.log_std)
        loss_positive = model.query_meta_loss(
            posterior=posterior_a,
            query_item_ids=torch.tensor([2, 3]),
            query_labels=torch.tensor([1.0, 1.0]),
            fixed_noise=True,
        )
        loss_negative = model.query_meta_loss(
            posterior=posterior_b,
            query_item_ids=torch.tensor([2, 3]),
            query_labels=torch.tensor([0.0, 0.0]),
            fixed_noise=True,
        )
        self.assertNotAlmostEqual(
            float(loss_positive.detach()),
            float(loss_negative.detach()),
            places=6,
        )

    def test_higher_order_outer_gradient_reaches_shared_parameters(self) -> None:
        model = _tiny_model()
        posterior = model.adapt(
            support_item_ids=torch.tensor([0, 1, 3]),
            support_labels=torch.tensor([1.0, 0.0, 1.0]),
            create_graph=True,
            fixed_noise=True,
        )
        loss = model.query_meta_loss(
            posterior=posterior,
            query_item_ids=torch.tensor([1, 2]),
            query_labels=torch.tensor([0.0, 1.0]),
            fixed_noise=True,
        )
        loss.backward()
        self.assertIsNotNone(model.prior_mean.grad)
        self.assertIsNotNone(model.prior_log_std.grad)
        self.assertIsNotNone(model.inner_lrs.grad)
        self.assertIsNotNone(model.knowledge_difficulty.weight.grad)
        first_layer = model.interaction[0]
        self.assertIsNotNone(first_layer.weight.grad)
        self.assertGreater(
            float(model.knowledge_difficulty.weight.grad.abs().sum()), 0.0
        )

    def test_fixed_noise_makes_support_order_irrelevant(self) -> None:
        model = _tiny_model()
        item_ids = torch.tensor([0, 1, 3])
        labels = torch.tensor([1.0, 0.0, 1.0])
        posterior = model.adapt(
            support_item_ids=item_ids,
            support_labels=labels,
            create_graph=False,
            fixed_noise=True,
        )
        permutation = torch.tensor([2, 0, 1])
        permuted = model.adapt(
            support_item_ids=item_ids[permutation],
            support_labels=labels[permutation],
            create_graph=False,
            fixed_noise=True,
        )
        torch.testing.assert_close(posterior.mean, permuted.mean, atol=1e-7, rtol=1e-6)
        torch.testing.assert_close(
            posterior.log_std, permuted.log_std, atol=1e-7, rtol=1e-6
        )

    def test_heldout_id_and_other_student_do_not_affect_prediction(self) -> None:
        model = _tiny_model()
        task_a = _task("heldout-a", query_index=0)
        renamed = _task("completely-different-id", query_index=0)
        prediction_a, _ = evaluate_tasks(
            model=model, tasks=[task_a], device=torch.device("cpu")
        )
        prediction_renamed, _ = evaluate_tasks(
            model=model, tasks=[renamed], device=torch.device("cpu")
        )
        np.testing.assert_allclose(
            prediction_a["prob"], prediction_renamed["prob"], atol=0.0, rtol=0.0
        )
        self.assertEqual(
            prediction_a.loc[0, "support_sha256"],
            prediction_renamed.loc[0, "support_sha256"],
        )

        distractor = _task("heldout-b", query_index=1, flipped=True)
        prediction_joint, _ = evaluate_tasks(
            model=model,
            tasks=[task_a, distractor],
            device=torch.device("cpu"),
        )
        self.assertEqual(
            float(prediction_a.loc[0, "prob"]),
            float(
                prediction_joint.loc[
                    prediction_joint["stu_id"] == "heldout-a", "prob"
                ].iloc[0]
            ),
        )

    def test_checkpoint_has_no_student_parameters_and_roundtrips(self) -> None:
        model = _tiny_model()
        self.assertFalse(
            any("student" in name.lower() for name in model.state_dict())
        )
        posterior = model.adapt(
            support_item_ids=torch.tensor([0, 1, 3]),
            support_labels=torch.tensor([1.0, 0.0, 1.0]),
            create_graph=False,
            fixed_noise=True,
        )
        expected = model.predict_query(
            posterior=posterior, query_item_ids=torch.tensor([2, 3])
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"model_state_dict": model.state_dict()}, path)
            restored = _tiny_model()
            payload = torch.load(path, map_location="cpu", weights_only=True)
            restored.load_state_dict(payload["model_state_dict"])
            restored_posterior = restored.adapt(
                support_item_ids=torch.tensor([0, 1, 3]),
                support_labels=torch.tensor([1.0, 0.0, 1.0]),
                create_graph=False,
                fixed_noise=True,
            )
            actual = restored.predict_query(
                posterior=restored_posterior,
                query_item_ids=torch.tensor([2, 3]),
            )
        torch.testing.assert_close(expected, actual, atol=0.0, rtol=0.0)

    def test_internal_split_is_deterministic_and_group_atomic(self) -> None:
        frame = _canonical_frame(
            [
                {"stu_id": student, "exer_id": exercise, "cpt_seq": exercise, "label": label}
                for student in ("a", "b")
                for exercise in range(6)
                for label in ((0, 1) if exercise == 2 else (exercise % 2,))
            ]
        )
        support_a, query_a, audit_a = split_optimizer_train(frame)
        support_b, query_b, audit_b = split_optimizer_train(frame.sample(frac=1.0, random_state=9))
        self.assertEqual(
            audit_a["assignment_sha256"], audit_b["assignment_sha256"]
        )
        support_groups = set(
            zip(support_a["stu_id"], support_a["exer_id"], strict=True)
        )
        query_groups = set(
            zip(query_a["stu_id"], query_a["exer_id"], strict=True)
        )
        self.assertFalse(support_groups & query_groups)
        for student in ("a", "b"):
            self.assertIn(student, set(support_a["stu_id"]))
            self.assertIn(student, set(query_a["stu_id"]))

    def test_q_rows_are_unioned_per_exercise(self) -> None:
        train = _canonical_frame(
            [
                {"stu_id": "t", "exer_id": "0", "cpt_seq": "0,1", "label": 1},
                {"stu_id": "t", "exer_id": "1", "cpt_seq": "2", "label": 0},
            ]
        )
        q = pd.DataFrame(
            [
                {"exer_id": "0", "cpt_seq": "0"},
                {"exer_id": "0", "cpt_seq": "1"},
                {"exer_id": "1", "cpt_seq": "2"},
            ]
        )
        q_tensor, item_map, concept_map = build_q_union(
            train=train, q_matrix=q
        )
        row = q_tensor[item_map["0"]]
        self.assertEqual(float(row[concept_map["0"]]), 1.0)
        self.assertEqual(float(row[concept_map["1"]]), 1.0)
        self.assertEqual(int(row.sum()), 2)

    def test_manifest_loader_verifies_test_but_does_not_parse_it(self) -> None:
        with TemporaryDirectory() as directory_name:
            directory = Path(directory_name).resolve()
            frames = {
                "train.csv": _canonical_frame(
                    [
                        {"stu_id": "t0", "exer_id": "0", "cpt_seq": "0", "label": 1},
                        {"stu_id": "t0", "exer_id": "1", "cpt_seq": "1", "label": 0},
                        {"stu_id": "t1", "exer_id": "0", "cpt_seq": "0", "label": 0},
                        {"stu_id": "t1", "exer_id": "2", "cpt_seq": "2", "label": 1},
                    ]
                ),
                "valid_support.csv": _canonical_frame(
                    [
                        {"stu_id": "v0", "exer_id": "0", "cpt_seq": "0", "label": 1},
                        {"stu_id": "v1", "exer_id": "1", "cpt_seq": "1", "label": 0},
                    ]
                ),
                "valid_query.csv": _canonical_frame(
                    [
                        {"stu_id": "v0", "exer_id": "1", "cpt_seq": "1", "label": 0},
                        {"stu_id": "v1", "exer_id": "2", "cpt_seq": "2", "label": 1},
                    ]
                ),
                "test_support.csv": _canonical_frame(
                    [
                        {"stu_id": "z0", "exer_id": "0", "cpt_seq": "0", "label": 1}
                    ]
                ),
                "test_query.csv": _canonical_frame(
                    [
                        {"stu_id": "z0", "exer_id": "2", "cpt_seq": "2", "label": 0}
                    ]
                ),
            }
            for filename, frame in frames.items():
                frame.to_csv(directory / filename, index=False)
            q = pd.DataFrame(
                [
                    {"exer_id": "0", "cpt_seq": "0"},
                    {"exer_id": "1", "cpt_seq": "1"},
                    {"exer_id": "2", "cpt_seq": "2"},
                ]
            )
            q.to_csv(directory / "Q_matrix.csv", index=False)
            files = {}
            for filename in (
                "train.csv",
                "valid_support.csv",
                "valid_query.csv",
                "test_support.csv",
                "test_query.csv",
                "Q_matrix.csv",
            ):
                files[filename] = {
                    "sha256": sha256_file(directory / filename),
                    "rows": len(frames[filename]) if filename in frames else len(q),
                }
            manifest_path = directory / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol": "student_disjoint_support_query",
                        "directory": str(directory),
                        "files": files,
                        "audit": {"seed": 2024},
                    }
                ),
                encoding="utf-8",
            )
            verified = load_verified_validation_protocol(manifest_path)
            self.assertEqual(len(verified.valid_query), 2)
            self.assertFalse(hasattr(verified, "test_support"))
            with (directory / "test_query.csv").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaises(ValueError):
                load_verified_validation_protocol(manifest_path)

    def test_paper_aligned_constants_and_fingerprint_are_fixed(self) -> None:
        config = BetaNCDConfig(num_items=4, num_concepts=3)
        self.assertEqual(config.inner_steps, 3)
        self.assertEqual(config.inner_mc_samples, 4)
        self.assertEqual(config.query_mc_samples, 4)
        self.assertEqual(config.kl_weight, 1e-4)
        self.assertEqual(config.inner_lr, 0.1)
        other_shape = BetaNCDConfig(num_items=99, num_concepts=17)
        self.assertEqual(
            config.architecture_fingerprint(),
            other_shape.architecture_fingerprint(),
        )
        posterior = AdaptedPosterior(
            mean=torch.zeros(3), log_std=torch.zeros(3)
        )
        self.assertEqual(tuple(posterior.mean.shape), (3,))


if __name__ == "__main__":
    unittest.main()
