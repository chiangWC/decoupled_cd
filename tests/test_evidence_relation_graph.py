import unittest

import torch

from models.evidence_relation_graph import (
    EvidenceRelationGraphCompleter,
    RelationMessageLayer,
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

    def test_completer_supports_noncontiguous_subset_without_id_embedding(
        self,
    ) -> None:
        torch.manual_seed(7)
        model = EvidenceRelationGraphCompleter(4, 3, hidden_dim=5)
        evidence = torch.tensor(
            [
                [[4.0, 4.0], [4.0, 1.0], [2.0, 1.0]],
                [[3.0, 1.0], [2.0, 2.0], [4.0, 3.0]],
                [[1.0, 1.0], [4.0, 0.0], [3.0, 2.0]],
                [[4.0, 2.0], [1.0, 0.0], [2.0, 2.0]],
            ]
        )
        q_matrix = torch.eye(3)

        self.assertFalse(
            any(
                isinstance(module, torch.nn.Embedding)
                for module in model.modules()
            )
        )
        full = model(evidence, q_matrix, epoch=0, training=True)
        subset = model(
            evidence,
            q_matrix,
            student_ids=torch.tensor([3, 1]),
            epoch=0,
            training=True,
        )

        self.assertEqual(tuple(subset.mastery.shape), (2, 3))
        self.assertEqual(tuple(subset.student_state.shape), (2, 5))
        self.assertEqual(tuple(subset.concept_state.shape), (3, 5))
        torch.testing.assert_close(subset.mastery, full.mastery[[3, 1]])
        torch.testing.assert_close(
            subset.reconstruction_mask, full.reconstruction_mask[[3, 1]]
        )
        torch.testing.assert_close(subset.target, full.target[[3, 1]])
        torch.testing.assert_close(
            subset.student_state, full.student_state[[3, 1]]
        )
        torch.testing.assert_close(subset.concept_state, full.concept_state)

        subset.mastery.sum().backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_positive_and_negative_relations_change_predictions(self) -> None:
        torch.manual_seed(11)
        model = EvidenceRelationGraphCompleter(3, 2, hidden_dim=4)
        positive_evidence = torch.tensor(
            [
                [[4.0, 4.0], [4.0, 4.0]],
                [[4.0, 4.0], [4.0, 4.0]],
                [[4.0, 4.0], [4.0, 4.0]],
            ]
        )
        negative_evidence = torch.tensor(
            [
                [[4.0, 0.0], [4.0, 0.0]],
                [[4.0, 0.0], [4.0, 0.0]],
                [[4.0, 0.0], [4.0, 0.0]],
            ]
        )
        q_matrix = torch.eye(2)

        positive = model(
            positive_evidence, q_matrix, None, None, False
        ).mastery
        negative = model(
            negative_evidence, q_matrix, None, None, False
        ).mastery

        self.assertFalse(torch.equal(positive, negative))

    def test_relation_layer_distinguishes_controlled_relation_channels(
        self,
    ) -> None:
        layer = RelationMessageLayer(hidden_dim=2)
        with torch.no_grad():
            layer.positive.weight.copy_(torch.eye(2))
            layer.negative.weight.copy_(
                torch.tensor([[0.0, 1.0], [0.0, 0.0]])
            )
        source = torch.tensor([[1.0, 2.0]])
        present = torch.ones(1, 1)
        absent = torch.zeros(1, 1)

        positive = layer((present, absent), source)
        negative = layer((absent, present), source)

        expected_positive = layer.norm(torch.tensor([[1.0, 2.0]]))
        expected_negative = layer.norm(torch.tensor([[2.0, 0.0]]))
        torch.testing.assert_close(positive, expected_positive)
        torch.testing.assert_close(negative, expected_negative)
        self.assertFalse(torch.equal(positive, negative))

    def test_relation_layer_returns_exact_zero_for_zero_degree_row(
        self,
    ) -> None:
        layer = RelationMessageLayer(hidden_dim=3)
        adjacency = (torch.zeros(2, 4), torch.zeros(2, 4))
        source = torch.randn(4, 3)

        actual = layer(adjacency, source)

        torch.testing.assert_close(actual, torch.zeros(2, 3), rtol=0, atol=0)

    def test_completer_uses_two_synchronous_directed_rounds(self) -> None:
        torch.manual_seed(13)
        model = EvidenceRelationGraphCompleter(3, 2, hidden_dim=4)
        evidence = torch.tensor(
            [
                [[4.0, 4.0], [2.0, 0.0]],
                [[3.0, 1.0], [5.0, 4.0]],
                [[2.0, 2.0], [4.0, 1.0]],
            ]
        )
        q_matrix = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
        inputs: dict[str, list[torch.Tensor]] = {"c2s": [], "s2c": []}
        outputs: dict[str, list[torch.Tensor]] = {"c2s": [], "s2c": []}
        handles = []

        def capture(direction: str):
            def hook(_module, args, output) -> None:
                inputs[direction].append(args[1].detach().clone())
                outputs[direction].append(output.detach().clone())

            return hook

        for layer in model.c2s:
            handles.append(layer.register_forward_hook(capture("c2s")))
        for layer in model.s2c:
            handles.append(layer.register_forward_hook(capture("s2c")))
        try:
            state = model(evidence, q_matrix)
        finally:
            for handle in handles:
                handle.remove()

        student_features, concept_features = node_summary_features(
            evidence, q_matrix
        )
        initial_students = model.student_encoder(student_features)
        initial_concepts = model.concept_encoder(concept_features)
        self.assertEqual(len(model.c2s), 2)
        self.assertEqual(len(model.s2c), 2)
        torch.testing.assert_close(inputs["c2s"][0], initial_concepts)
        torch.testing.assert_close(inputs["s2c"][0], initial_students)
        torch.testing.assert_close(inputs["c2s"][1], outputs["s2c"][0])
        torch.testing.assert_close(inputs["s2c"][1], outputs["c2s"][0])
        torch.testing.assert_close(state.student_state, outputs["c2s"][1])
        torch.testing.assert_close(state.concept_state, outputs["s2c"][1])

    def test_completer_rejects_evidence_shape_mismatching_constructor(
        self,
    ) -> None:
        model = EvidenceRelationGraphCompleter(3, 2, hidden_dim=4)
        q_matrix = torch.eye(2)

        for shape in ((4, 2, 2), (3, 3, 2)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(
                    ValueError, "configured for 3 students and 2 concepts"
                ):
                    model(torch.zeros(shape), q_matrix)


if __name__ == "__main__":
    unittest.main()
