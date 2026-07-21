from __future__ import annotations

import copy
import unittest

import torch

from models.monotone_requirement_surface import (
    SURFACE_MODES,
    HierarchicalLattice1D,
    HierarchicalLattice2D,
    MonotoneRequirementSurface,
    SparseQPositiveLinear,
)


NUM_CONCEPTS = 6


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    readiness = torch.tensor(
        [
            [0.10, 0.80, 0.35, 0.91, 0.27, 0.62],
            [0.72, 0.21, 0.53, 0.44, 0.81, 0.12],
            [0.31, 0.69, 0.47, 0.58, 0.16, 0.88],
            [0.92, 0.24, 0.67, 0.39, 0.55, 0.73],
        ],
        dtype=torch.float32,
    )
    q_mask = torch.tensor(
        [
            [True, True, False, False, False, False],
            [False, True, True, True, False, False],
            [False, False, False, False, False, True],
            [True, True, True, True, False, False],
        ]
    )
    item_offset = torch.tensor([-0.4, 0.2, 0.0, 0.7])
    return readiness, q_mask, item_offset


class MonotoneRequirementSurfaceTest(unittest.TestCase):
    def test_forward_contract_and_frozen_variant_sizes(self) -> None:
        readiness, q_mask, item_offset = _inputs()
        expected = {
            "full": 16,
            "pooled_direct": 16,
            "capacity_additive": 16,
        }
        for mode in SURFACE_MODES:
            with self.subTest(mode=mode):
                model = MonotoneRequirementSurface(
                    num_concepts=NUM_CONCEPTS,
                    mode=mode,
                )
                probabilities = model(readiness, q_mask, item_offset)
                self.assertEqual(probabilities.shape, (readiness.size(0),))
                self.assertTrue(torch.isfinite(probabilities).all())
                self.assertTrue(((probabilities > 0) & (probabilities < 1)).all())
                if mode in expected:
                    self.assertEqual(
                        model.active_variant_parameter_count,
                        expected[mode],
                    )
                else:
                    self.assertGreater(model.active_variant_parameter_count, 16)

    def test_hll_vertices_and_registered_grids_are_monotone(self) -> None:
        one_dimensional = HierarchicalLattice1D(16)
        values_1d = one_dimensional.vertex_values()
        self.assertTrue(torch.all(values_1d[1:] >= values_1d[:-1]))
        self.assertIn("grid", dict(one_dimensional.named_buffers()))

        two_dimensional = HierarchicalLattice2D((4, 4))
        values_2d = two_dimensional.vertex_values()
        self.assertTrue(torch.all(values_2d[1:, :] >= values_2d[:-1, :]))
        self.assertTrue(torch.all(values_2d[:, 1:] >= values_2d[:, :-1]))
        buffers = dict(two_dimensional.named_buffers())
        self.assertIn("grid_x", buffers)
        self.assertIn("grid_y", buffers)

    def test_full_and_capacity_have_identical_initial_predictions(self) -> None:
        readiness, q_mask, item_offset = _inputs()
        multi = q_mask.sum(dim=1) > 1
        full = MonotoneRequirementSurface(
            num_concepts=NUM_CONCEPTS,
            mode="full",
        )
        capacity = MonotoneRequirementSurface(
            num_concepts=NUM_CONCEPTS,
            mode="capacity_additive",
        )
        torch.testing.assert_close(
            full(readiness[multi], q_mask[multi], item_offset[multi]),
            capacity(readiness[multi], q_mask[multi], item_offset[multi]),
            rtol=0.0,
            atol=1e-6,
        )

    def test_q1_path_is_bitwise_common_across_all_modes(self) -> None:
        readiness, q_mask, item_offset = _inputs()
        single = q_mask.sum(dim=1) == 1
        outputs = []
        gradients = []
        for mode in SURFACE_MODES:
            model = MonotoneRequirementSurface(
                num_concepts=NUM_CONCEPTS,
                mode=mode,
            )
            output = model(
                readiness[single],
                q_mask[single],
                item_offset[single],
            )
            output.sum().backward()
            outputs.append(output.detach())
            gradients.append(model.unary_lattice.raw_gates.grad.detach())
        for output in outputs[1:]:
            self.assertTrue(torch.equal(output, outputs[0]))
        for gradient in gradients[1:]:
            self.assertTrue(torch.equal(gradient, gradients[0]))

    def test_masked_values_and_concept_permutation_do_not_change_lattices(self) -> None:
        readiness, q_mask, item_offset = _inputs()
        changed_padding = readiness.clone()
        changed_padding[~q_mask] = torch.linspace(
            -100.0,
            100.0,
            int((~q_mask).sum()),
        )
        permutation = torch.tensor([5, 2, 0, 4, 1, 3])
        for mode in ("full", "pooled_direct", "capacity_additive"):
            with self.subTest(mode=mode):
                model = MonotoneRequirementSurface(
                    num_concepts=NUM_CONCEPTS,
                    mode=mode,
                )
                baseline = model(readiness, q_mask, item_offset)
                self.assertTrue(
                    torch.equal(
                        baseline,
                        model(changed_padding, q_mask, item_offset),
                    )
                )
                permuted = model(
                    readiness[:, permutation],
                    q_mask[:, permutation],
                    item_offset,
                )
                torch.testing.assert_close(baseline, permuted, rtol=0, atol=0)

    def test_every_mode_is_monotone_in_each_active_readiness(self) -> None:
        readiness, q_mask, item_offset = _inputs()
        for mode in SURFACE_MODES:
            model = MonotoneRequirementSurface(
                num_concepts=NUM_CONCEPTS,
                mode=mode,
            )
            baseline = model(readiness, q_mask, item_offset)
            for concept in range(NUM_CONCEPTS):
                active = q_mask[:, concept] & (readiness[:, concept] <= 0.94)
                if not bool(active.any()):
                    continue
                increased = readiness.clone()
                increased[active, concept] += 0.05
                changed = model(increased, q_mask, item_offset)
                self.assertTrue(
                    torch.all(changed[active] > baseline[active]),
                    msg=f"monotonicity failed for {mode}, concept {concept}",
                )

    def test_all_variant_parameters_receive_finite_nonzero_gradients(self) -> None:
        grid = torch.linspace(0.02, 0.98, 16)
        left, right = torch.meshgrid(grid, grid, indexing="ij")
        readiness = torch.stack([left.flatten(), right.flatten()], dim=1)
        q_mask = torch.ones_like(readiness, dtype=torch.bool)
        item_offset = torch.zeros(readiness.size(0))
        for mode in SURFACE_MODES:
            with self.subTest(mode=mode):
                model = MonotoneRequirementSurface(
                    num_concepts=2,
                    mode=mode,
                )
                model(readiness, q_mask, item_offset).sum().backward()
                for parameter in model.variant_parameters():
                    self.assertIsNotNone(parameter.grad)
                    assert parameter.grad is not None
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertEqual(
                        int(torch.count_nonzero(parameter.grad)),
                        parameter.numel(),
                    )
                    self.assertGreater(
                        float(parameter.grad.abs().min()),
                        1e-6,
                        msg=f"effectively frozen initialization for {mode}",
                    )

    def test_sparse_q_positive_linear_matches_dense_value_and_gradients(self) -> None:
        generator = torch.Generator().manual_seed(7)
        sparse = SparseQPositiveLinear(11, 7, seed=143)
        dense = copy.deepcopy(sparse)
        readiness_sparse = torch.rand(
            9,
            11,
            generator=generator,
            requires_grad=True,
        )
        readiness_dense = readiness_sparse.detach().clone().requires_grad_(True)
        q_mask = torch.zeros(9, 11, dtype=torch.bool)
        for row in range(9):
            q_mask[row, torch.randperm(11, generator=generator)[: 1 + row % 4]] = True

        sparse_output = sparse(readiness_sparse, q_mask)
        dense_output = dense.dense_forward(readiness_dense, q_mask)
        torch.testing.assert_close(sparse_output, dense_output, rtol=1e-6, atol=1e-7)

        sparse_output.square().sum().backward()
        dense_output.square().sum().backward()
        torch.testing.assert_close(
            readiness_sparse.grad,
            readiness_dense.grad,
            rtol=1e-5,
            atol=1e-7,
        )
        torch.testing.assert_close(
            sparse.raw_weight.grad,
            dense.raw_weight.grad,
            rtol=1e-5,
            atol=1e-7,
        )
        torch.testing.assert_close(
            sparse.bias.grad,
            dense.bias.grad,
            rtol=1e-5,
            atol=1e-7,
        )

    def test_matches_pinned_ibm_pmlayer_numeric_oracle(self) -> None:
        # Generated before formal training with IBM/pmlayer
        # v1.0.1@89bcceb966b5c2feb506be3da413ed2549a71db1 on CPU.
        one_dimensional = HierarchicalLattice1D(16)
        with torch.no_grad():
            one_dimensional.raw_gates.copy_(torch.linspace(-1.0, 1.0, 16))
        one_inputs = torch.tensor([0.0, 0.125, 0.5, 0.9, 1.0])
        one_expected = torch.tensor(
            [
                0.2689414322376251,
                0.6314241886138916,
                0.9830647706985474,
                0.9999404549598694,
                0.9999926686286926,
            ]
        )
        torch.testing.assert_close(
            one_dimensional(one_inputs),
            one_expected,
            rtol=1e-6,
            atol=2e-7,
        )

        two_dimensional = HierarchicalLattice2D((4, 4))
        with torch.no_grad():
            two_dimensional.raw_gates.copy_(
                torch.linspace(-2.0, 1.0, 16).reshape(4, 4)
            )
        two_inputs = torch.tensor(
            [[0.05, 0.1], [0.2, 0.8], [0.5, 0.5], [0.9, 0.3], [1.0, 1.0]]
        )
        two_expected = torch.tensor(
            [
                0.18983781337738037,
                0.5925215482711792,
                0.7073634266853333,
                0.879988431930542,
                0.9951949715614319,
            ]
        )
        torch.testing.assert_close(
            two_dimensional(two_inputs[:, 0], two_inputs[:, 1]),
            two_expected,
            rtol=1e-6,
            atol=2e-7,
        )

    def test_invalid_empty_q_and_out_of_range_active_readiness_fail(self) -> None:
        model = MonotoneRequirementSurface(
            num_concepts=NUM_CONCEPTS,
            mode="full",
        )
        readiness, q_mask, item_offset = _inputs()
        q_mask[0] = False
        with self.assertRaisesRegex(ValueError, "at least one"):
            model(readiness, q_mask, item_offset)

        _, q_mask, _ = _inputs()
        readiness[0, 0] = 1.1
        with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
            model(readiness, q_mask, item_offset)


if __name__ == "__main__":
    unittest.main()
