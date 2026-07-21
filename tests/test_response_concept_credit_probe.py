from __future__ import annotations

import inspect
import unittest

import torch
import torch.nn.functional as functional

from models.response_concept_credit_probe import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_ROUTING_ITERATIONS,
    DEFAULT_STATE_DIM,
    VARIANTS,
    ResponseConceptCreditProbe,
)


NUM_ITEMS = 12
NUM_CONCEPTS = 5
MAX_Q_CARDINALITY = 3
BATCH_STUDENTS = 3
SUPPORT_ROWS = 4


def _make_model(seed: int = 1234) -> ResponseConceptCreditProbe:
    torch.manual_seed(seed)
    return ResponseConceptCreditProbe(
        num_items=NUM_ITEMS,
        num_concepts=NUM_CONCEPTS,
        max_q_cardinality=MAX_Q_CARDINALITY,
    )


def _inputs() -> dict[str, torch.Tensor]:
    support_q_indices = torch.tensor(
        [
            [
                [1, 2, 0],
                [2, 3, 0],
                [1, 3, 4],
                [0, 0, 0],
            ],
            [
                [1, 0, 0],
                [2, 4, 0],
                [3, 4, 5],
                [1, 5, 0],
            ],
            [
                [2, 3, 0],
                [1, 4, 5],
                [0, 0, 0],
                [0, 0, 0],
            ],
        ],
        dtype=torch.long,
    )
    return {
        "support_item_ids": torch.tensor(
            [
                [1, 2, 3, 0],
                [4, 5, 6, 7],
                [8, 9, 0, 0],
            ],
            dtype=torch.long,
        ),
        "support_q_indices": support_q_indices,
        "support_q_mask": support_q_indices != 0,
        "support_responses": torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ),
        "support_item_ease": torch.tensor(
            [
                [0.72, 0.31, 0.55, 0.0],
                [0.24, 0.66, 0.43, 0.81],
                [0.35, 0.62, 0.0, 0.0],
            ]
        ),
        "support_item_confidence": torch.tensor(
            [
                [0.8, 0.7, 0.9, 0.0],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 0.4, 0.0, 0.0],
            ]
        ),
        "support_group_attempt_confidence": torch.tensor(
            [
                [0.7, 0.6, 0.8, 0.0],
                [0.4, 0.5, 0.6, 0.7],
                [0.8, 0.3, 0.0, 0.0],
            ]
        ),
        "support_mask": torch.tensor(
            [
                [True, True, True, False],
                [True, True, True, True],
                [True, True, False, False],
            ]
        ),
        "query_context_indices": torch.tensor(
            [0, 0, 1, 2, 2],
            dtype=torch.long,
        ),
        "target_q_indices": torch.tensor(
            [
                [1, 0, 0],
                [2, 3, 0],
                [3, 4, 5],
                [4, 0, 0],
                [1, 5, 0],
            ],
            dtype=torch.long,
        ),
        "target_q_mask": torch.tensor(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
                [True, False, False],
                [True, True, False],
            ]
        ),
    }


def _forward(
    model: ResponseConceptCreditProbe,
    variant: str,
    inputs: dict[str, torch.Tensor] | None = None,
):
    return model(variant=variant, **(_inputs() if inputs is None else inputs))


def _clone_inputs(
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in inputs.items()}


class TestResponseConceptCreditProbe(unittest.TestCase):
    def test_frozen_contract_and_output_shapes(self) -> None:
        self.assertEqual(VARIANTS, ("full", "direct", "capacity"))
        self.assertEqual(DEFAULT_EMBEDDING_DIM, 16)
        self.assertEqual(DEFAULT_STATE_DIM, 64)
        self.assertEqual(DEFAULT_HIDDEN_DIM, 64)
        self.assertEqual(DEFAULT_ROUTING_ITERATIONS, 3)

        model = _make_model().eval()
        output = _forward(model, "full")
        self.assertEqual(output.logits.shape, (5,))
        self.assertEqual(output.probs.shape, (5,))
        self.assertEqual(
            output.framework_state.shape,
            (BATCH_STUDENTS, NUM_CONCEPTS, DEFAULT_STATE_DIM),
        )
        self.assertEqual(
            output.mastery.shape,
            (BATCH_STUDENTS, NUM_CONCEPTS),
        )
        self.assertEqual(
            output.routing_responsibility.shape,
            (BATCH_STUDENTS, SUPPORT_ROWS, MAX_Q_CARDINALITY),
        )
        self.assertEqual(
            output.routing_credit.shape,
            output.routing_responsibility.shape,
        )
        self.assertEqual(
            output.routing_entropy.shape,
            (BATCH_STUDENTS, SUPPORT_ROWS),
        )
        self.assertEqual(
            output.routed_mass.shape,
            (BATCH_STUDENTS, SUPPORT_ROWS),
        )
        self.assertEqual(
            output.concept_evidence_mass.shape,
            (BATCH_STUDENTS, NUM_CONCEPTS),
        )
        self.assertEqual(len(output.architecture_fingerprint), 16)
        self.assertNotEqual(
            model.architecture_fingerprint("full"),
            model.architecture_fingerprint("capacity"),
        )
        self.assertFalse(
            any("student" in name and "embedding" in name
                for name, _ in model.named_modules())
        )

    def test_q_mass_mask_and_padding_are_exact(self) -> None:
        inputs = _inputs()
        model = _make_model().eval()
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                output = _forward(model, variant, inputs)
                eligible = (
                    inputs["support_q_mask"]
                    & inputs["support_mask"].unsqueeze(-1)
                )
                self.assertTrue(
                    torch.equal(
                        output.routing_responsibility[~eligible],
                        torch.zeros_like(
                            output.routing_responsibility[~eligible]
                        ),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        output.routing_credit[~eligible],
                        torch.zeros_like(output.routing_credit[~eligible]),
                    )
                )
                responsibility_mass = (
                    output.routing_responsibility.sum(dim=-1)
                )
                expected_responsibility = inputs["support_mask"].to(
                    dtype=responsibility_mass.dtype
                )
                self.assertTrue(
                    torch.allclose(
                        responsibility_mass,
                        expected_responsibility,
                        atol=1e-7,
                        rtol=0.0,
                    )
                )
                expected_credit = (
                    inputs["support_q_mask"].sum(dim=-1)
                    * inputs["support_mask"]
                ).to(dtype=output.routed_mass.dtype)
                self.assertTrue(
                    torch.allclose(
                        output.routed_mass,
                        expected_credit,
                        atol=1e-6,
                        rtol=0.0,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        output.routing_entropy[
                            ~inputs["support_mask"]
                        ],
                        torch.zeros_like(
                            output.routing_entropy[
                                ~inputs["support_mask"]
                            ]
                        ),
                    )
                )

        bad = _clone_inputs(inputs)
        bad["support_q_indices"][0, 3, 0] = 1
        bad["support_q_mask"] = bad["support_q_indices"] != 0
        with self.assertRaisesRegex(ValueError, "Padded support rows"):
            _forward(model, "full", bad)

    def test_zero_initialized_routing_matches_direct_full_copy(self) -> None:
        model = _make_model().eval()
        self.assertTrue(
            torch.equal(
                model.routing_output.weight,
                torch.zeros_like(model.routing_output.weight),
            )
        )
        self.assertTrue(
            torch.equal(
                model.routing_output.bias,
                torch.zeros_like(model.routing_output.bias),
            )
        )
        outputs = {
            variant: _forward(model, variant)
            for variant in VARIANTS
        }
        for variant in ("full", "capacity"):
            self.assertTrue(
                torch.equal(
                    outputs[variant].routing_credit,
                    outputs["direct"].routing_credit,
                )
            )
            self.assertTrue(
                torch.equal(
                    outputs[variant].framework_state,
                    outputs["direct"].framework_state,
                )
            )
            self.assertTrue(
                torch.equal(
                    outputs[variant].logits,
                    outputs["direct"].logits,
                )
            )

    def test_single_concept_support_is_identical_after_router_changes(
        self,
    ) -> None:
        model = _make_model().eval()
        with torch.no_grad():
            model.routing_output.weight.copy_(
                torch.linspace(
                    -0.5,
                    0.5,
                    model.hidden_dim,
                ).unsqueeze(0)
            )
            model.routing_output.bias.fill_(0.7)
        inputs = _inputs()
        singleton = _clone_inputs(inputs)
        first = singleton["support_q_indices"][..., :1].clamp_min(1)
        singleton["support_q_indices"].zero_()
        singleton["support_q_indices"][..., :1] = first
        singleton["support_q_indices"][
            ~singleton["support_mask"]
        ] = 0
        singleton["support_q_mask"] = (
            singleton["support_q_indices"] != 0
        )
        outputs = {
            variant: _forward(model, variant, singleton)
            for variant in VARIANTS
        }
        for variant in ("full", "capacity"):
            self.assertTrue(
                torch.equal(
                    outputs[variant].routing_credit,
                    outputs["direct"].routing_credit,
                )
            )
            self.assertTrue(
                torch.equal(
                    outputs[variant].framework_state,
                    outputs["direct"].framework_state,
                )
            )
            self.assertTrue(
                torch.equal(
                    outputs[variant].probs,
                    outputs["direct"].probs,
                )
            )

    def test_capacity_routing_is_response_invariant_with_zero_jacobian(
        self,
    ) -> None:
        model = _make_model().eval()
        with torch.no_grad():
            model.routing_output.weight.copy_(
                torch.linspace(
                    -0.75,
                    0.75,
                    model.hidden_dim,
                ).unsqueeze(0)
            )
        inputs = _inputs()
        changed = _clone_inputs(inputs)
        changed["support_responses"][inputs["support_mask"]] = (
            1.0 - changed["support_responses"][inputs["support_mask"]]
        )
        original = _forward(model, "capacity", inputs)
        flipped = _forward(model, "capacity", changed)
        self.assertTrue(
            torch.equal(
                original.routing_responsibility,
                flipped.routing_responsibility,
            )
        )
        self.assertTrue(
            torch.equal(original.routing_credit, flipped.routing_credit)
        )
        self.assertTrue(
            torch.equal(original.routing_entropy, flipped.routing_entropy)
        )
        self.assertFalse(
            torch.allclose(
                original.framework_state,
                flipped.framework_state,
                atol=1e-8,
                rtol=0.0,
            )
        )

        differentiable = _clone_inputs(inputs)
        differentiable["support_responses"] = differentiable[
            "support_responses"
        ].requires_grad_(True)
        output = _forward(model, "capacity", differentiable)
        position_weight = torch.tensor(
            [0.0, 1.0, 2.0],
            dtype=output.routing_credit.dtype,
        )
        routed_coordinate = (
            output.routing_credit * position_weight
        ).sum()
        jacobian = torch.autograd.grad(
            routed_coordinate,
            differentiable["support_responses"],
        )[0]
        self.assertTrue(torch.equal(jacobian, torch.zeros_like(jacobian)))

        full_original = _forward(model, "full", inputs)
        full_flipped = _forward(model, "full", changed)
        self.assertFalse(
            torch.allclose(
                full_original.routing_responsibility,
                full_flipped.routing_responsibility,
                atol=1e-8,
                rtol=0.0,
            )
        )

    def test_capacity_focal_route_ignores_other_support_items(self) -> None:
        model = _make_model().eval()
        with torch.no_grad():
            model.routing_output.weight.copy_(
                torch.linspace(
                    -0.75,
                    0.75,
                    model.hidden_dim,
                ).unsqueeze(0)
            )
        inputs = _inputs()
        counterfactual = _clone_inputs(inputs)
        # Keep student zero's focal row zero fixed while changing every static
        # and response channel of its two other valid support rows.
        counterfactual["support_item_ids"][0, 1:3] = torch.tensor([10, 11])
        counterfactual["support_q_indices"][0, 1] = torch.tensor([1, 4, 5])
        counterfactual["support_q_indices"][0, 2] = torch.tensor([2, 4, 0])
        counterfactual["support_q_mask"] = (
            counterfactual["support_q_indices"] != 0
        )
        counterfactual["support_item_ease"][0, 1:3] = torch.tensor(
            [0.95, 0.05]
        )
        counterfactual["support_responses"][0, 1:3] = torch.tensor(
            [1.0, 0.0]
        )
        counterfactual["support_item_confidence"][0, 1:3] = torch.tensor(
            [0.1, 0.2]
        )
        counterfactual[
            "support_group_attempt_confidence"
        ][0, 1:3] = torch.tensor([0.2, 0.1])

        capacity = _forward(model, "capacity", inputs)
        capacity_changed = _forward(model, "capacity", counterfactual)
        self.assertTrue(
            torch.equal(
                capacity.routing_responsibility[0, 0],
                capacity_changed.routing_responsibility[0, 0],
            )
        )
        full = _forward(model, "full", inputs)
        full_changed = _forward(model, "full", counterfactual)
        self.assertFalse(
            torch.allclose(
                full.routing_responsibility[0, 0],
                full_changed.routing_responsibility[0, 0],
                atol=1e-8,
                rtol=0.0,
            )
        )

    def test_nonuniform_routing_is_consumed_by_completion(self) -> None:
        model = _make_model().eval()
        inputs = _inputs()

        def force_position_logits(_module, _args, output):
            positions = torch.arange(
                output.shape[-2],
                dtype=output.dtype,
                device=output.device,
            ).view(1, 1, -1, 1)
            return positions.expand_as(output)

        handle = model.routing_output.register_forward_hook(
            force_position_logits
        )
        try:
            full = _forward(model, "full", inputs)
        finally:
            handle.remove()
        direct = _forward(model, "direct", inputs)
        eligible = (
            inputs["support_q_mask"]
            & inputs["support_mask"].unsqueeze(-1)
        )
        self.assertFalse(
            torch.allclose(
                full.routing_credit[eligible],
                torch.ones_like(full.routing_credit[eligible]),
            )
        )
        self.assertFalse(
            torch.allclose(
                full.framework_state,
                direct.framework_state,
                atol=1e-8,
                rtol=0.0,
            )
        )
        # Concept five is exact-zero for student zero.  Routing must still
        # reach its completed state through the routed observed-state summary.
        self.assertFalse(bool(full.observed_concept_mask[0, 4]))
        self.assertFalse(
            torch.allclose(
                full.framework_state[0, 4],
                direct.framework_state[0, 4],
                atol=1e-8,
                rtol=0.0,
            )
        )

    def test_exact_zero_concepts_receive_completed_student_state(self) -> None:
        model = _make_model().eval()
        inputs = _inputs()
        output = _forward(model, "direct", inputs)
        # Student zero has no support item carrying concept five.
        self.assertFalse(bool(output.observed_concept_mask[0, 4]))
        self.assertEqual(float(output.concept_evidence_mass[0, 4]), 0.0)
        self.assertTrue(torch.isfinite(output.framework_state[0, 4]).all())
        self.assertGreater(
            float(output.framework_state[0, 4].abs().sum()),
            0.0,
        )

        changed = _clone_inputs(inputs)
        changed["support_responses"][0, :3] = (
            1.0 - changed["support_responses"][0, :3]
        )
        changed_output = _forward(model, "direct", changed)
        self.assertFalse(
            torch.allclose(
                output.framework_state[0, 4],
                changed_output.framework_state[0, 4],
                atol=1e-8,
                rtol=0.0,
            )
        )

    def test_diagnosis_has_only_framework_state_as_student_path(self) -> None:
        model = _make_model().eval()
        inputs = _inputs()
        output = _forward(model, "full", inputs)
        diagnosed = model.diagnose(
            framework_state=output.framework_state,
            query_context_indices=inputs["query_context_indices"],
            target_q_indices=inputs["target_q_indices"],
            target_q_mask=inputs["target_q_mask"],
        )
        self.assertTrue(torch.equal(diagnosed, output.logits))
        self.assertEqual(
            tuple(inspect.signature(model.diagnose).parameters),
            (
                "framework_state",
                "query_context_indices",
                "target_q_indices",
                "target_q_mask",
            ),
        )

        perturbed_state = output.framework_state.clone()
        perturbed_state[0, 0] += 5.0
        perturbed = model.diagnose(
            framework_state=perturbed_state,
            query_context_indices=inputs["query_context_indices"],
            target_q_indices=inputs["target_q_indices"],
            target_q_mask=inputs["target_q_mask"],
        )
        self.assertFalse(torch.equal(perturbed, diagnosed))

        def cut_state(_module, _args, output_state):
            return torch.zeros_like(output_state)

        handle = model.completion_network.register_forward_hook(cut_state)
        try:
            first = _forward(model, "full", inputs)
            changed = _clone_inputs(inputs)
            changed["support_responses"][inputs["support_mask"]] = (
                1.0 - changed["support_responses"][inputs["support_mask"]]
            )
            second = _forward(model, "full", changed)
        finally:
            handle.remove()
        self.assertTrue(torch.equal(first.framework_state, second.framework_state))
        self.assertTrue(torch.equal(first.probs, second.probs))

    def test_support_and_q_order_permutations_do_not_change_state(self) -> None:
        model = _make_model().eval()
        with torch.no_grad():
            model.routing_output.weight.copy_(
                torch.linspace(
                    -0.5,
                    0.5,
                    model.hidden_dim,
                ).unsqueeze(0)
            )
        inputs = _inputs()
        original = _forward(model, "full", inputs)

        support_permutation = torch.tensor([2, 0, 3, 1])
        permuted = _clone_inputs(inputs)
        for name in (
            "support_item_ids",
            "support_q_indices",
            "support_q_mask",
            "support_responses",
            "support_item_ease",
            "support_item_confidence",
            "support_group_attempt_confidence",
            "support_mask",
        ):
            permuted[name] = permuted[name][:, support_permutation]
        support_output = _forward(model, "full", permuted)
        self.assertTrue(
            torch.allclose(
                original.framework_state,
                support_output.framework_state,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                original.probs,
                support_output.probs,
                atol=1e-6,
                rtol=1e-6,
            )
        )

        q_permutation = torch.tensor([2, 0, 1])
        permuted_q = _clone_inputs(inputs)
        permuted_q["support_q_indices"] = permuted_q[
            "support_q_indices"
        ][..., q_permutation]
        permuted_q["support_q_mask"] = permuted_q[
            "support_q_mask"
        ][..., q_permutation]
        permuted_q["target_q_indices"] = permuted_q[
            "target_q_indices"
        ][..., q_permutation]
        permuted_q["target_q_mask"] = permuted_q[
            "target_q_mask"
        ][..., q_permutation]
        q_output = _forward(model, "full", permuted_q)
        self.assertTrue(
            torch.allclose(
                original.framework_state,
                q_output.framework_state,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                original.probs,
                q_output.probs,
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_response_bce_has_finite_gradients(self) -> None:
        model = _make_model().train()
        output = _forward(model, "full")
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])
        loss = functional.binary_cross_entropy(output.probs, labels)
        loss.backward()
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(missing, [])
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(
                    torch.isfinite(parameter.grad).all(),
                    msg=name,
                )

    def test_variants_share_one_parameterization(self) -> None:
        model = _make_model()
        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )
        self.assertGreater(parameter_count, 0)
        self.assertNotIn("variant", dict(model.named_parameters()))
        self.assertEqual(
            set(inspect.signature(model.forward).parameters),
            {
                "variant",
                "support_item_ids",
                "support_q_indices",
                "support_q_mask",
                "support_responses",
                "support_item_ease",
                "support_item_confidence",
                "support_group_attempt_confidence",
                "support_mask",
                "query_context_indices",
                "target_q_indices",
                "target_q_mask",
            },
        )


if __name__ == "__main__":
    unittest.main()
