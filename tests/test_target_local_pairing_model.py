from __future__ import annotations

import hashlib
import unittest

import torch

from models.target_local_pairing_probe import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MLP_HIDDEN_DIM,
    DEFAULT_STATE_DIM,
    PLACEMENTS,
    RESPONSE_FEATURE_NAMES,
    SUPPORT_STATISTIC_NAMES,
    SharedItemEncoder,
    StrongOPMSProbe,
    TargetLocalPairingProbe,
)


NUM_ITEMS = 13
NUM_CONCEPTS = 6
ITEM_NUMERIC_DIM = 4
BATCH = 3
SUPPORT = 5


def _make_pair(
    placement: str,
    seed: int = 1234,
) -> TargetLocalPairingProbe:
    torch.manual_seed(seed)
    return TargetLocalPairingProbe(
        num_items=NUM_ITEMS,
        num_concepts=NUM_CONCEPTS,
        item_numeric_dim=ITEM_NUMERIC_DIM,
        placement=placement,
    )


def _make_strong(seed: int = 1234) -> StrongOPMSProbe:
    torch.manual_seed(seed)
    return StrongOPMSProbe(
        num_items=NUM_ITEMS,
        num_concepts=NUM_CONCEPTS,
        item_numeric_dim=ITEM_NUMERIC_DIM,
    )


def _multi_hot(
    leading_shape: tuple[int, ...],
    generator: torch.Generator,
) -> torch.Tensor:
    values = (
        torch.rand(
            *leading_shape,
            NUM_CONCEPTS,
            generator=generator,
        )
        > 0.65
    ).to(dtype=torch.float32)
    values[..., 0] = 1.0
    return values


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(99)
    support_responses = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.37, 0.82],
            [0.0, 0.0, 0.0, 0.0, 0.21],
            [1.0, 0.0, 0.44, 0.73, 0.13],
        ]
    )
    return {
        "support_item_ids": torch.tensor(
            [
                [1, 2, 3, 0, 0],
                [4, 5, 6, 7, 0],
                [8, 9, 0, 0, 0],
            ],
            dtype=torch.long,
        ),
        "support_q_multi_hot": _multi_hot(
            (BATCH, SUPPORT),
            generator,
        ),
        "support_item_numeric": torch.randn(
            BATCH,
            SUPPORT,
            ITEM_NUMERIC_DIM,
            generator=generator,
        ),
        "support_responses": support_responses,
        "support_item_ease": torch.rand(
            BATCH,
            SUPPORT,
            generator=generator,
        ),
        "support_group_attempt_confidence": torch.rand(
            BATCH,
            SUPPORT,
            generator=generator,
        ),
        "support_statistics": torch.randn(
            BATCH,
            len(SUPPORT_STATISTIC_NAMES),
            generator=generator,
        ),
        "support_mask": torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, False],
                [True, True, False, False, False],
            ]
        ),
        "target_item_ids": torch.tensor([1, 4, 8]),
        "target_q_multi_hot": _multi_hot((BATCH,), generator),
        "target_item_numeric": torch.randn(
            BATCH,
            ITEM_NUMERIC_DIM,
            generator=generator,
        ),
        "donor_target_item_ids": torch.tensor([10, 11, 12]),
        "donor_target_q_multi_hot": _multi_hot(
            (BATCH,),
            generator,
        ),
        "donor_target_item_numeric": torch.randn(
            BATCH,
            ITEM_NUMERIC_DIM,
            generator=generator,
        ),
    }


def _pair_forward(
    model: TargetLocalPairingProbe,
    inputs: dict[str, torch.Tensor],
):
    kwargs = dict(inputs)
    if model.placement != "perm_pair":
        kwargs.pop("donor_target_item_ids")
        kwargs.pop("donor_target_q_multi_hot")
        kwargs.pop("donor_target_item_numeric")
    return model(**kwargs)


def _strong_inputs(
    inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in inputs.items()
        if not key.startswith("donor_target_")
    }


def _change_target(
    inputs: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    changed = {
        key: value.clone()
        for key, value in inputs.items()
    }
    id_key = f"{prefix}_item_ids"
    q_key = f"{prefix}_q_multi_hot"
    numeric_key = f"{prefix}_item_numeric"
    changed[id_key] = changed[id_key].roll(1)
    changed[q_key] = changed[q_key].roll(1, dims=-1)
    changed[numeric_key] = changed[numeric_key] + 1.75
    return changed


def _state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


class TestTargetLocalPairingModel(unittest.TestCase):
    def test_frozen_dimensions_and_response_semantics(self) -> None:
        self.assertEqual(DEFAULT_EMBEDDING_DIM, 16)
        self.assertEqual(DEFAULT_STATE_DIM, 32)
        self.assertEqual(DEFAULT_MLP_HIDDEN_DIM, 64)
        self.assertEqual(
            RESPONSE_FEATURE_NAMES,
            (
                "response",
                "response_minus_item_ease",
                "group_attempt_confidence",
            ),
        )
        self.assertEqual(
            SUPPORT_STATISTIC_NAMES,
            (
                "theta_logit",
                "raw_accuracy",
                "log1p_support_rows",
                "log1p_unique_support_items",
                "log1p_correct_rows",
                "log1p_incorrect_rows",
            ),
        )
        model = _make_pair("real_pair")
        self.assertEqual(model.embedding_dim, 16)
        self.assertEqual(model.state_dim, 32)
        self.assertEqual(model.mlp_hidden_dim, 64)
        self.assertEqual(
            model.item_encoder.item_id_embedding.embedding_dim,
            16,
        )
        self.assertEqual(
            model.item_encoder.view_fusion[-1].out_features,
            32,
        )
        self.assertEqual(
            model.interaction_encoder[0].out_features,
            64,
        )

    def test_response_features_are_constructed_inside_model(self) -> None:
        inputs = _inputs()
        output = _pair_forward(_make_pair("real_pair").eval(), inputs)
        expected = torch.stack(
            [
                inputs["support_responses"],
                (
                    inputs["support_responses"]
                    - inputs["support_item_ease"]
                ),
                inputs["support_group_attempt_confidence"],
            ],
            dim=-1,
        )
        self.assertTrue(
            torch.equal(output.response_features, expected)
        )

    def test_one_shared_item_encoder_serves_all_item_roles(self) -> None:
        model = _make_pair("perm_pair").eval()
        shared_encoders = [
            module
            for module in model.modules()
            if isinstance(module, SharedItemEncoder)
        ]
        self.assertEqual(shared_encoders, [model.item_encoder])
        calls: list[tuple[torch.Size, torch.Size]] = []

        def capture_call(_module, args, kwargs, output) -> None:
            del args
            calls.append((kwargs["item_ids"].shape, output.shape))

        handle = model.item_encoder.register_forward_hook(
            capture_call,
            with_kwargs=True,
        )
        try:
            with torch.no_grad():
                model(**_inputs())
        finally:
            handle.remove()
        self.assertEqual(
            calls,
            [
                (
                    torch.Size([BATCH, SUPPORT]),
                    torch.Size([BATCH, SUPPORT, 32]),
                ),
                (torch.Size([BATCH]), torch.Size([BATCH, 32])),
                (torch.Size([BATCH]), torch.Size([BATCH, 32])),
            ],
        )

    def test_opms_pools_raw_item_states_and_empty_pool_is_zero(
        self,
    ) -> None:
        model = _make_pair("real_pair").eval()
        inputs = _inputs()
        no_donor = _strong_inputs(inputs)
        with torch.no_grad():
            support_state = model.item_encoder(
                item_ids=inputs["support_item_ids"],
                q_multi_hot=inputs["support_q_multi_hot"],
                numeric_features=inputs["support_item_numeric"],
            )
            output = model(**no_donor)

        mask = inputs["support_mask"]
        response = inputs["support_responses"]
        expected_correct = []
        expected_incorrect = []
        for row in range(BATCH):
            correct = mask[row] & (response[row] == 1)
            incorrect = mask[row] & (response[row] == 0)
            expected_correct.append(
                support_state[row, correct].mean(dim=0)
                if bool(correct.any())
                else torch.zeros(DEFAULT_STATE_DIM)
            )
            expected_incorrect.append(
                support_state[row, incorrect].mean(dim=0)
                if bool(incorrect.any())
                else torch.zeros(DEFAULT_STATE_DIM)
            )
        self.assertTrue(
            torch.allclose(
                output.correct_pool,
                torch.stack(expected_correct),
            )
        )
        self.assertTrue(
            torch.allclose(
                output.incorrect_pool,
                torch.stack(expected_incorrect),
            )
        )
        self.assertTrue(
            torch.equal(
                output.incorrect_pool[0],
                torch.zeros(DEFAULT_STATE_DIM),
            )
        )
        self.assertTrue(
            torch.equal(
                output.correct_pool[1],
                torch.zeros(DEFAULT_STATE_DIM),
            )
        )

    def test_interaction_encoder_cannot_change_common_opms(self) -> None:
        model = _make_pair("real_pair").eval()
        inputs = _inputs()
        with torch.no_grad():
            before = _pair_forward(model, inputs)
            for parameter in model.interaction_encoder.parameters():
                parameter.add_(5.0)
            after = _pair_forward(model, inputs)
        for field in (
            "opms_state",
            "correct_pool",
            "incorrect_pool",
            "contrast",
            "correct_mass",
            "incorrect_mass",
        ):
            self.assertTrue(
                torch.equal(
                    getattr(before, field),
                    getattr(after, field),
                ),
                msg=field,
            )
        self.assertFalse(
            torch.allclose(before.local_state, after.local_state)
        )

    def test_strong_and_pair_share_identical_raw_opms_path(self) -> None:
        strong = _make_strong(seed=812)
        pair = _make_pair("real_pair", seed=812)
        self.assertEqual(
            _state_hash(strong.item_encoder),
            _state_hash(pair.item_encoder),
        )
        self.assertEqual(
            _state_hash(strong.outcome_state),
            _state_hash(pair.outcome_state),
        )
        inputs = _inputs()
        with torch.no_grad():
            strong_output = strong(**_strong_inputs(inputs))
            pair_output = _pair_forward(pair, inputs)
        for field in (
            "opms_state",
            "target_state",
            "correct_pool",
            "incorrect_pool",
            "contrast",
            "correct_mass",
            "incorrect_mass",
        ):
            self.assertTrue(
                torch.equal(
                    getattr(strong_output, field),
                    getattr(pair_output, field),
                ),
                msg=field,
            )

    def test_strong_opms_keeps_true_target_in_diagnosis(self) -> None:
        model = _make_strong().eval()
        inputs = _strong_inputs(_inputs())
        changed = _change_target(inputs, "target")
        with torch.no_grad():
            first = model(**inputs)
            second = model(**changed)
        self.assertTrue(
            torch.equal(first.opms_state, second.opms_state)
        )
        self.assertFalse(
            torch.allclose(first.target_state, second.target_state)
        )
        self.assertFalse(torch.allclose(first.logits, second.logits))

    def test_placements_have_exact_parameter_and_init_parity(self) -> None:
        models = {
            placement: _make_pair(placement)
            for placement in PLACEMENTS
        }
        signatures = {
            placement: [
                (name, tuple(parameter.shape), parameter.numel())
                for name, parameter in model.named_parameters()
            ]
            for placement, model in models.items()
        }
        reference = signatures[PLACEMENTS[0]]
        self.assertTrue(
            all(value == reference for value in signatures.values())
        )
        self.assertEqual(
            len({_state_hash(model) for model in models.values()}),
            1,
        )

    def test_all_registered_parameters_receive_gradient(self) -> None:
        models: list[torch.nn.Module] = [
            _make_pair(placement)
            for placement in PLACEMENTS
        ]
        models.append(_make_strong())
        for model in models:
            if isinstance(model, TargetLocalPairingProbe):
                output = _pair_forward(model, _inputs())
            else:
                output = model(**_strong_inputs(_inputs()))
            output.logits.sum().backward()
            for name, parameter in model.named_parameters():
                self.assertIsNotNone(
                    parameter.grad,
                    msg=f"{type(model).__name__}:{name}",
                )
                assert parameter.grad is not None
                self.assertTrue(
                    torch.isfinite(parameter.grad).all(),
                    msg=f"{type(model).__name__}:{name}",
                )
                self.assertGreater(
                    float(parameter.grad.abs().sum()),
                    0.0,
                    msg=f"{type(model).__name__}:{name}",
                )

    def test_support_order_and_padding_invariance(self) -> None:
        inputs = _inputs()
        order = torch.tensor([2, 4, 0, 3, 1])
        reordered = {
            key: value.clone()
            for key, value in inputs.items()
        }
        for key in (
            "support_item_ids",
            "support_q_multi_hot",
            "support_item_numeric",
            "support_responses",
            "support_item_ease",
            "support_group_attempt_confidence",
            "support_mask",
        ):
            reordered[key] = inputs[key].index_select(1, order)

        padded = {
            key: value.clone()
            for key, value in inputs.items()
        }
        padded["support_item_ids"] = torch.cat(
            [
                inputs["support_item_ids"],
                torch.zeros(BATCH, 2, dtype=torch.long),
            ],
            dim=1,
        )
        padded["support_q_multi_hot"] = torch.cat(
            [
                inputs["support_q_multi_hot"],
                torch.ones(BATCH, 2, NUM_CONCEPTS),
            ],
            dim=1,
        )
        padded["support_item_numeric"] = torch.cat(
            [
                inputs["support_item_numeric"],
                torch.full(
                    (BATCH, 2, ITEM_NUMERIC_DIM),
                    1.0e6,
                ),
            ],
            dim=1,
        )
        for key, fill in (
            ("support_responses", 0.37),
            ("support_item_ease", 1.0e6),
            ("support_group_attempt_confidence", -1.0e6),
        ):
            padded[key] = torch.cat(
                [
                    inputs[key],
                    torch.full((BATCH, 2), fill),
                ],
                dim=1,
            )
        padded["support_mask"] = torch.cat(
            [
                inputs["support_mask"],
                torch.zeros(BATCH, 2, dtype=torch.bool),
            ],
            dim=1,
        )

        models: list[torch.nn.Module] = [
            _make_pair(placement).eval()
            for placement in PLACEMENTS
        ]
        models.append(_make_strong().eval())
        for model in models:
            with torch.no_grad():
                if isinstance(model, TargetLocalPairingProbe):
                    reference = _pair_forward(model, inputs)
                    order_output = _pair_forward(model, reordered)
                    pad_output = _pair_forward(model, padded)
                else:
                    reference = model(**_strong_inputs(inputs))
                    order_output = model(**_strong_inputs(reordered))
                    pad_output = model(**_strong_inputs(padded))
            for field in (
                "logits",
                "probs",
                "opms_state",
                "target_state",
                "correct_pool",
                "incorrect_pool",
                "contrast",
                "correct_mass",
                "incorrect_mass",
            ):
                expected = getattr(reference, field)
                self.assertTrue(
                    torch.allclose(
                        expected,
                        getattr(order_output, field),
                        atol=1e-6,
                        rtol=1e-6,
                    ),
                    msg=f"{type(model).__name__}:{field}:order",
                )
                self.assertTrue(
                    torch.allclose(
                        expected,
                        getattr(pad_output, field),
                        atol=1e-6,
                        rtol=1e-6,
                    ),
                    msg=f"{type(model).__name__}:{field}:padding",
                )

    def test_real_and_late_target_interventions(self) -> None:
        inputs = _inputs()
        for placement in ("real_pair", "late_fusion"):
            model = _make_pair(placement).eval()
            base = _strong_inputs(inputs)
            changed = _change_target(base, "target")
            captured: list[torch.Tensor] = []

            def capture_pair(_module, _args, output) -> None:
                captured.append(output.detach().clone())

            handle = model.pair_encoder.register_forward_hook(
                capture_pair
            )
            try:
                with torch.no_grad():
                    first = model(**base)
                    second = model(**changed)
            finally:
                handle.remove()
            self.assertTrue(
                torch.equal(
                    first.history_summary,
                    second.history_summary,
                )
            )
            self.assertFalse(torch.allclose(captured[0], captured[1]))
            self.assertFalse(
                torch.allclose(first.local_state, second.local_state)
            )
            self.assertFalse(
                torch.allclose(first.logits, second.logits)
            )
            if placement == "real_pair":
                self.assertEqual(captured[0].ndim, 3)
            else:
                self.assertEqual(captured[0].ndim, 2)

    def test_perm_donor_only_changes_local_branch(self) -> None:
        model = _make_pair("perm_pair").eval()
        inputs = _inputs()
        donor_changed = _change_target(inputs, "donor_target")
        true_changed = _change_target(inputs, "target")
        with torch.no_grad():
            reference = model(**inputs)
            donor_output = model(**donor_changed)
            true_output = model(**true_changed)

        for field in (
            "target_state",
            "opms_state",
            "history_summary",
            "correct_pool",
            "incorrect_pool",
        ):
            self.assertTrue(
                torch.equal(
                    getattr(reference, field),
                    getattr(donor_output, field),
                )
            )
        self.assertFalse(
            torch.allclose(reference.local_state, donor_output.local_state)
        )
        self.assertFalse(
            torch.allclose(reference.logits, donor_output.logits)
        )
        self.assertTrue(
            torch.equal(reference.local_state, true_output.local_state)
        )
        self.assertFalse(
            torch.allclose(reference.target_state, true_output.target_state)
        )
        self.assertFalse(
            torch.allclose(reference.logits, true_output.logits)
        )

    def test_support_statistics_are_consumed_only_by_common_opms(
        self,
    ) -> None:
        model = _make_pair("real_pair").eval()
        inputs = _inputs()
        changed = {
            key: value.clone()
            for key, value in inputs.items()
        }
        changed["support_statistics"] += 2.0
        with torch.no_grad():
            first = _pair_forward(model, inputs)
            second = _pair_forward(model, changed)
        self.assertFalse(
            torch.allclose(first.opms_state, second.opms_state)
        )
        self.assertTrue(
            torch.equal(first.local_state, second.local_state)
        )
        self.assertTrue(
            torch.equal(first.correct_pool, second.correct_pool)
        )

    def test_shared_item_q_mean_and_unk_zero(self) -> None:
        encoder = SharedItemEncoder(
            num_items=NUM_ITEMS,
            num_concepts=NUM_CONCEPTS,
            numeric_dim=ITEM_NUMERIC_DIM,
        ).eval()
        self.assertEqual(encoder.unk_item_id, 0)
        self.assertTrue(
            torch.equal(
                encoder.item_id_embedding.weight[0],
                torch.zeros(DEFAULT_EMBEDDING_DIM),
            )
        )
        q = torch.zeros(2, NUM_CONCEPTS)
        q[0, 1] = 1.0
        q[0, 4] = 1.0
        captured: list[torch.Tensor] = []

        def capture_fusion(_module, args) -> None:
            captured.append(args[0].detach().clone())

        handle = encoder.view_fusion.register_forward_pre_hook(
            capture_fusion
        )
        try:
            with torch.no_grad():
                encoder(
                    item_ids=torch.tensor([0, 0]),
                    q_multi_hot=q,
                    numeric_features=torch.zeros(
                        2,
                        ITEM_NUMERIC_DIM,
                    ),
                )
        finally:
            handle.remove()
        q_view = captured[0][
            :,
            DEFAULT_EMBEDDING_DIM : DEFAULT_EMBEDDING_DIM * 2,
        ]
        expected = (
            encoder.concept_embedding.weight[1]
            + encoder.concept_embedding.weight[4]
        ) / 2.0
        self.assertTrue(torch.allclose(q_view[0], expected))
        self.assertTrue(
            torch.equal(
                q_view[1],
                torch.zeros(DEFAULT_EMBEDDING_DIM),
            )
        )

    def test_invalid_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            _make_pair("unknown")

        inputs = _inputs()
        perm = _make_pair("perm_pair")
        incomplete = dict(inputs)
        incomplete.pop("donor_target_item_numeric")
        with self.assertRaisesRegex(ValueError, "provided together"):
            perm(**incomplete)

        real = _make_pair("real_pair")
        common = _strong_inputs(inputs)
        non_binary = {
            key: value.clone()
            for key, value in common.items()
        }
        non_binary["support_responses"][0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "exactly binary"):
            real(**non_binary)

        bad_statistics = {
            key: value.clone()
            for key, value in common.items()
        }
        bad_statistics["support_statistics"] = torch.randn(
            BATCH,
            len(SUPPORT_STATISTIC_NAMES) - 1,
        )
        with self.assertRaisesRegex(
            ValueError,
            "support_statistics",
        ):
            real(**bad_statistics)

        bad_id = {
            key: value.clone()
            for key, value in common.items()
        }
        bad_id["target_item_ids"][0] = NUM_ITEMS + 1
        with self.assertRaisesRegex(ValueError, "1..num_items"):
            real(**bad_id)


if __name__ == "__main__":
    unittest.main()
