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


if __name__ == "__main__":
    unittest.main()
