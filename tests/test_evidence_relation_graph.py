import unittest

import torch

from models.evidence_relation_graph import (
    build_relation_graph,
    node_summary_features,
)


class EvidenceRelationGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = torch.zeros(10, 5, 2)
        self.evidence[..., 0] = 4
        self.evidence[..., 1] = (
            torch.arange(50).reshape(10, 5).remainder(5)
        )
        self.q_matrix = torch.tensor(
            [
                [1, 0, 0, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 1, 1, 0, 1],
                [0, 0, 1, 1, 1],
            ],
            dtype=torch.float32,
        )

    def test_relation_graph_and_epoch_mask(self) -> None:
        left = build_relation_graph(
            self.evidence, epoch=7, training=True
        )
        right = build_relation_graph(
            self.evidence, epoch=7, training=True
        )

        torch.testing.assert_close(
            left.reconstruction_mask, right.reconstruction_mask
        )
        self.assertEqual(int(left.reconstruction_mask.sum()), 10)
        self.assertFalse(
            bool(left.positive_weight[left.reconstruction_mask].any())
        )
        self.assertFalse(
            bool(left.negative_weight[left.reconstruction_mask].any())
        )
        expected = (self.evidence[..., 1] + 1.0) / (
            self.evidence[..., 0] + 2.0
        )
        torch.testing.assert_close(left.target, expected)

    def test_relation_weights_encode_sign_and_capped_reliability(self) -> None:
        evidence = torch.tensor(
            [[[0.0, 0.0], [5.0, 1.0], [20.0, 10.0], [30.0, 20.0]]]
        )

        graph = build_relation_graph(evidence, epoch=None, training=False)

        torch.testing.assert_close(
            graph.positive_weight,
            torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        )
        torch.testing.assert_close(
            graph.negative_weight,
            torch.tensor([[0.0, 0.25, 0.0, 0.0]]),
        )
        self.assertFalse(bool(graph.reconstruction_mask.any()))

    def test_inference_rejects_epoch_to_keep_masking_train_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "epoch must be None"):
            build_relation_graph(self.evidence, epoch=0, training=False)

    def test_node_features_have_fixed_non_id_shapes_and_values(self) -> None:
        students, concepts = node_summary_features(
            self.evidence, self.q_matrix
        )

        self.assertEqual(tuple(students.shape), (10, 3))
        self.assertEqual(tuple(concepts.shape), (5, 4))
        expected_student_accuracy = (
            self.evidence[..., 1].sum(dim=1) + 1.0
        ) / (self.evidence[..., 0].sum(dim=1) + 2.0)
        expected_concept_accuracy = (
            self.evidence[..., 1].sum(dim=0) + 1.0
        ) / (self.evidence[..., 0].sum(dim=0) + 2.0)
        torch.testing.assert_close(students[:, 0], torch.ones(10))
        torch.testing.assert_close(students[:, 1], expected_student_accuracy)
        torch.testing.assert_close(
            students[:, 2],
            torch.log1p(self.evidence[..., 0].sum(dim=1)),
        )
        torch.testing.assert_close(concepts[:, 0], torch.ones(5))
        torch.testing.assert_close(concepts[:, 1], expected_concept_accuracy)
        torch.testing.assert_close(
            concepts[:, 2],
            torch.log1p(self.evidence[..., 0].sum(dim=0)),
        )
        torch.testing.assert_close(
            concepts[:, 3], self.q_matrix.sum(dim=0)
        )

    def test_integer_counts_produce_float_graph_and_summary_outputs(
        self,
    ) -> None:
        evidence = torch.tensor(
            [
                [[0, 0], [2, 1], [4, 3]],
                [[3, 1], [1, 1], [5, 2]],
            ],
            dtype=torch.int64,
        )
        q_matrix = torch.tensor(
            [[1, 0, 1], [0, 1, 1]], dtype=torch.int64
        )

        graph = build_relation_graph(evidence, epoch=None, training=False)
        students, concepts = node_summary_features(evidence, q_matrix)

        self.assertEqual(graph.target.dtype, torch.get_default_dtype())
        self.assertEqual(
            graph.positive_weight.dtype, torch.get_default_dtype()
        )
        self.assertEqual(
            graph.negative_weight.dtype, torch.get_default_dtype()
        )
        self.assertEqual(students.dtype, torch.get_default_dtype())
        self.assertEqual(concepts.dtype, torch.get_default_dtype())
        torch.testing.assert_close(
            graph.target,
            torch.tensor(
                [[0.5, 0.5, 4.0 / 6.0], [2.0 / 5.0, 2.0 / 3.0, 3.0 / 7.0]]
            ),
        )
        torch.testing.assert_close(
            students,
            torch.tensor(
                [
                    [2.0 / 3.0, 5.0 / 8.0, torch.log1p(torch.tensor(6.0))],
                    [1.0, 5.0 / 11.0, torch.log1p(torch.tensor(9.0))],
                ]
            ),
        )
        torch.testing.assert_close(
            concepts,
            torch.tensor(
                [
                    [0.5, 2.0 / 5.0, torch.log1p(torch.tensor(3.0)), 1.0],
                    [1.0, 3.0 / 5.0, torch.log1p(torch.tensor(3.0)), 1.0],
                    [1.0, 6.0 / 11.0, torch.log1p(torch.tensor(9.0)), 2.0],
                ]
            ),
        )

    def test_rejects_nonfinite_attempts(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                evidence = torch.tensor([[[value, 0.0]]])
                with self.assertRaisesRegex(
                    ValueError, "attempts must be finite"
                ):
                    build_relation_graph(evidence, epoch=None, training=False)

    def test_rejects_negative_attempts(self) -> None:
        evidence = torch.tensor([[[-1.0, 0.0]]])
        with self.assertRaisesRegex(ValueError, "attempts must be nonnegative"):
            node_summary_features(evidence, torch.ones(1, 1))

    def test_rejects_nonfinite_correct_counts(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                evidence = torch.tensor([[[1.0, value]]])
                with self.assertRaisesRegex(
                    ValueError, "correct must be finite"
                ):
                    build_relation_graph(evidence, epoch=None, training=False)

    def test_rejects_negative_correct_counts(self) -> None:
        evidence = torch.tensor([[[1.0, -1.0]]])
        with self.assertRaisesRegex(ValueError, "correct must be nonnegative"):
            node_summary_features(evidence, torch.ones(1, 1))

    def test_rejects_correct_counts_above_attempts(self) -> None:
        evidence = torch.tensor([[[1.0, 2.0]]])
        with self.assertRaisesRegex(ValueError, "correct cannot exceed attempts"):
            build_relation_graph(evidence, epoch=None, training=False)

    def test_rejects_nonfinite_or_nonpositive_reliability_cap(self) -> None:
        for value in (
            float("nan"),
            float("inf"),
            float("-inf"),
            0.0,
            -1.0,
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError, "reliability_cap must be finite and positive"
                ):
                    build_relation_graph(
                        self.evidence,
                        epoch=None,
                        training=False,
                        reliability_cap=value,
                    )

    def test_sparse_mask_is_observed_subset_with_floor_and_epoch_variation(
        self,
    ) -> None:
        evidence = torch.zeros(8, 8, 2)
        observed_ids = torch.tensor(
            [0, 3, 7, 12, 18, 25, 31, 40, 48, 55, 63]
        )
        evidence[..., 0].flatten()[observed_ids] = 2
        evidence[..., 1].flatten()[observed_ids] = 1

        first = build_relation_graph(
            evidence, epoch=0, training=True, mask_fraction=0.4
        )
        later = build_relation_graph(
            evidence, epoch=100_000, training=True, mask_fraction=0.4
        )
        observed = evidence[..., 0] > 0

        self.assertTrue(bool((first.reconstruction_mask <= observed).all()))
        self.assertTrue(bool((later.reconstruction_mask <= observed).all()))
        self.assertEqual(int(first.reconstruction_mask.sum()), 4)
        self.assertEqual(int(later.reconstruction_mask.sum()), 4)
        self.assertFalse(
            torch.equal(first.reconstruction_mask, later.reconstruction_mask)
        )


if __name__ == "__main__":
    unittest.main()
