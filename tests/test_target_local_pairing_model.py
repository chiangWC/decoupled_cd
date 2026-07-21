from __future__ import annotations

import hashlib
import unittest

import torch

from models.target_local_pairing_probe import (
    DEFAULT_RESPONSE_FEATURE_NAMES,
    PLACEMENTS,
    SharedItemEncoder,
    TargetLocalPairingProbe,
)


NUM_ITEMS = 13
NUM_CONCEPTS = 6
ITEM_NUMERIC_DIM = 4
OPMS_DIM = 7
RESPONSE_DIM = 3
HIDDEN_DIM = 11


def _make_model(
    placement: str,
    seed: int = 1234,
) -> TargetLocalPairingProbe:
    torch.manual_seed(seed)
    return TargetLocalPairingProbe(
        num_items=NUM_ITEMS,
        num_concepts=NUM_CONCEPTS,
        item_numeric_dim=ITEM_NUMERIC_DIM,
        opms_dim=OPMS_DIM,
        response_feature_dim=RESPONSE_DIM,
        hidden_dim=HIDDEN_DIM,
        placement=placement,
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
    group_accuracy = torch.rand(3, 5, generator=generator)
    residual = (
        torch.rand(3, 5, generator=generator) * 2.0 - 1.0
    )
    confidence = torch.rand(3, 5, generator=generator)
    return {
        "support_item_ids": torch.tensor(
            [
                [1, 2, 3, 0, 0],
                [4, 5, 6, 7, 0],
                [8, 9, 0, 0, 0],
            ],
            dtype=torch.long,
        ),
        "support_q_multi_hot": _multi_hot((3, 5), generator),
        "support_item_numeric": torch.randn(
            3,
            5,
            ITEM_NUMERIC_DIM,
            generator=generator,
        ),
        "support_response_features": torch.stack(
            [group_accuracy, residual, confidence],
            dim=-1,
        ),
        "support_mask": torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, False],
                [True, True, False, False, False],
            ]
        ),
        "target_item_ids": torch.tensor([1, 4, 8]),
        "target_q_multi_hot": _multi_hot((3,), generator),
        "target_item_numeric": torch.randn(
            3,
            ITEM_NUMERIC_DIM,
            generator=generator,
        ),
        "donor_target_item_ids": torch.tensor([10, 11, 12]),
        "donor_target_q_multi_hot": _multi_hot((3,), generator),
        "donor_target_item_numeric": torch.randn(
            3,
            ITEM_NUMERIC_DIM,
            generator=generator,
        ),
        "opms_summary": torch.randn(
            3,
            OPMS_DIM,
            generator=generator,
        ),
    }


def _state_hash(model: TargetLocalPairingProbe) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        contiguous = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _forward(
    model: TargetLocalPairingProbe,
    inputs: dict[str, torch.Tensor],
):
    kwargs = dict(inputs)
    if model.placement != "perm_pair":
        kwargs.pop("donor_target_item_ids")
        kwargs.pop("donor_target_q_multi_hot")
        kwargs.pop("donor_target_item_numeric")
    return model(**kwargs)


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


class TestTargetLocalPairingModel(unittest.TestCase):
    def test_response_features_have_declared_three_channel_semantics(
        self,
    ) -> None:
        self.assertEqual(
            DEFAULT_RESPONSE_FEATURE_NAMES,
            ("group_accuracy", "residual", "confidence"),
        )
        model = _make_model("real_pair")
        self.assertEqual(model.response_feature_dim, 3)

    def test_one_shared_item_encoder_serves_all_item_roles(self) -> None:
        model = _make_model("perm_pair").eval()
        shared_encoders = [
            module
            for module in model.modules()
            if isinstance(module, SharedItemEncoder)
        ]
        self.assertEqual(shared_encoders, [model.item_encoder])
        self.assertFalse(hasattr(model, "target_encoder"))

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
                (torch.Size([3, 5]), torch.Size([3, 5, HIDDEN_DIM])),
                (torch.Size([3]), torch.Size([3, HIDDEN_DIM])),
                (torch.Size([3]), torch.Size([3, HIDDEN_DIM])),
            ],
        )

    def test_q_view_is_the_mean_of_active_concept_embeddings(
        self,
    ) -> None:
        encoder = SharedItemEncoder(
            num_items=NUM_ITEMS,
            num_concepts=NUM_CONCEPTS,
            numeric_dim=ITEM_NUMERIC_DIM,
            hidden_dim=HIDDEN_DIM,
        ).eval()
        item_ids = torch.tensor([0, 0])
        q = torch.zeros(2, NUM_CONCEPTS)
        q[0, 1] = 1.0
        q[0, 4] = 1.0
        numeric = torch.zeros(2, ITEM_NUMERIC_DIM)
        fusion_inputs: list[torch.Tensor] = []

        def capture_fusion_input(_module, args) -> None:
            fusion_inputs.append(args[0].detach().clone())

        handle = encoder.view_fusion.register_forward_pre_hook(
            capture_fusion_input
        )
        try:
            with torch.no_grad():
                encoder(
                    item_ids=item_ids,
                    q_multi_hot=q,
                    numeric_features=numeric,
                )
        finally:
            handle.remove()

        q_view = fusion_inputs[0][
            :,
            HIDDEN_DIM : HIDDEN_DIM * 2,
        ]
        expected = (
            encoder.concept_embedding.weight[1]
            + encoder.concept_embedding.weight[4]
        ) / 2.0
        self.assertTrue(torch.allclose(q_view[0], expected))
        self.assertTrue(torch.equal(q_view[1], torch.zeros(HIDDEN_DIM)))

    def test_unknown_item_zero_is_reserved_and_has_no_id_gradient(
        self,
    ) -> None:
        encoder = SharedItemEncoder(
            num_items=NUM_ITEMS,
            num_concepts=NUM_CONCEPTS,
            numeric_dim=ITEM_NUMERIC_DIM,
            hidden_dim=HIDDEN_DIM,
        )
        self.assertEqual(encoder.unk_item_id, 0)
        self.assertTrue(
            torch.equal(
                encoder.item_id_embedding.weight[0],
                torch.zeros(HIDDEN_DIM),
            )
        )
        item_ids = torch.tensor([0, 1, 0, 2])
        q = torch.zeros(4, NUM_CONCEPTS)
        q[:, 0] = 1.0
        numeric = torch.randn(4, ITEM_NUMERIC_DIM)
        encoder(
            item_ids=item_ids,
            q_multi_hot=q,
            numeric_features=numeric,
        ).sum().backward()
        gradient = encoder.item_id_embedding.weight.grad
        assert gradient is not None
        self.assertTrue(torch.equal(gradient[0], torch.zeros(HIDDEN_DIM)))
        self.assertGreater(float(gradient[1].abs().sum()), 0.0)

    def test_placements_have_identical_parameters_and_initialization(
        self,
    ) -> None:
        models = {
            placement: _make_model(placement)
            for placement in PLACEMENTS
        }
        signatures = {
            placement: [
                (name, tuple(parameter.shape), parameter.numel())
                for name, parameter in model.named_parameters()
            ]
            for placement, model in models.items()
        }
        first_signature = signatures[PLACEMENTS[0]]
        self.assertTrue(
            all(
                signature == first_signature
                for signature in signatures.values()
            )
        )
        counts = {
            placement: sum(
                parameter.numel()
                for parameter in model.parameters()
            )
            for placement, model in models.items()
        }
        self.assertEqual(len(set(counts.values())), 1)
        hashes = {_state_hash(model) for model in models.values()}
        self.assertEqual(len(hashes), 1)

    def test_every_registered_parameter_is_active_in_every_placement(
        self,
    ) -> None:
        active_names: dict[str, set[str]] = {}
        for placement in PLACEMENTS:
            model = _make_model(placement)
            output = _forward(model, _inputs())
            output.logits.sum().backward()

            named_parameters = dict(model.named_parameters())
            active_names[placement] = {
                name
                for name, parameter in named_parameters.items()
                if parameter.grad is not None
            }
            self.assertEqual(
                active_names[placement],
                set(named_parameters),
                msg=f"inactive parameter in {placement}",
            )
            for name, parameter in named_parameters.items():
                assert parameter.grad is not None
                self.assertTrue(
                    torch.isfinite(parameter.grad).all(),
                    msg=f"non-finite gradient for {placement}:{name}",
                )
                self.assertGreater(
                    float(parameter.grad.abs().sum()),
                    0.0,
                    msg=f"zero gradient for {placement}:{name}",
                )
        self.assertTrue(
            all(
                names == active_names[PLACEMENTS[0]]
                for names in active_names.values()
            )
        )

    def test_support_order_and_padding_do_not_change_output(self) -> None:
        inputs = _inputs()
        order = torch.tensor([2, 4, 0, 3, 1])
        padded = {
            key: value.clone()
            for key, value in inputs.items()
        }
        padded["support_item_ids"] = torch.cat(
            [
                inputs["support_item_ids"],
                torch.zeros(3, 3, dtype=torch.long),
            ],
            dim=1,
        )
        padded["support_q_multi_hot"] = torch.cat(
            [
                inputs["support_q_multi_hot"],
                torch.ones(3, 3, NUM_CONCEPTS),
            ],
            dim=1,
        )
        padded["support_item_numeric"] = torch.cat(
            [
                inputs["support_item_numeric"],
                torch.full(
                    (3, 3, ITEM_NUMERIC_DIM),
                    1.0e6,
                ),
            ],
            dim=1,
        )
        padded["support_response_features"] = torch.cat(
            [
                inputs["support_response_features"],
                torch.full((3, 3, RESPONSE_DIM), -1.0e6),
            ],
            dim=1,
        )
        padded["support_mask"] = torch.cat(
            [
                inputs["support_mask"],
                torch.zeros(3, 3, dtype=torch.bool),
            ],
            dim=1,
        )

        reordered = {
            key: value.clone()
            for key, value in inputs.items()
        }
        for key in (
            "support_item_ids",
            "support_q_multi_hot",
            "support_item_numeric",
            "support_response_features",
            "support_mask",
        ):
            reordered[key] = inputs[key].index_select(1, order)

        for placement in PLACEMENTS:
            model = _make_model(placement).eval()
            with torch.no_grad():
                reference = _forward(model, inputs)
                reordered_output = _forward(model, reordered)
                padded_output = _forward(model, padded)
            for field in (
                "logits",
                "probs",
                "history_summary",
                "local_state",
                "target_state",
                "local_target_state",
                "opms_state",
            ):
                expected = getattr(reference, field)
                self.assertTrue(
                    torch.allclose(
                        expected,
                        getattr(reordered_output, field),
                        atol=1e-6,
                        rtol=1e-6,
                    ),
                    msg=f"{placement}:{field} changed after reordering",
                )
                self.assertTrue(
                    torch.allclose(
                        expected,
                        getattr(padded_output, field),
                        atol=1e-6,
                        rtol=1e-6,
                    ),
                    msg=f"{placement}:{field} changed after padding",
                )

    def test_real_pair_target_changes_prepool_pairing(self) -> None:
        model = _make_model("real_pair").eval()
        inputs = _inputs()
        changed = _change_target(inputs, "target")
        for key in (
            "donor_target_item_ids",
            "donor_target_q_multi_hot",
            "donor_target_item_numeric",
        ):
            inputs.pop(key)
            changed.pop(key)

        captured: list[torch.Tensor] = []

        def capture_pair_output(_module, _args, output) -> None:
            captured.append(output.detach().clone())

        handle = model.pair_encoder.register_forward_hook(
            capture_pair_output
        )
        try:
            with torch.no_grad():
                first = model(**inputs)
                second = model(**changed)
        finally:
            handle.remove()

        self.assertEqual(captured[0].ndim, 3)
        self.assertEqual(captured[0].shape[:2], (3, 5))
        self.assertFalse(torch.allclose(captured[0], captured[1]))
        self.assertTrue(
            torch.allclose(first.history_summary, second.history_summary)
        )
        self.assertFalse(
            torch.allclose(first.local_state, second.local_state)
        )
        self.assertFalse(torch.allclose(first.logits, second.logits))

    def test_late_fusion_target_enters_only_after_history_pool(
        self,
    ) -> None:
        model = _make_model("late_fusion").eval()
        inputs = _inputs()
        for key in (
            "donor_target_item_ids",
            "donor_target_q_multi_hot",
            "donor_target_item_numeric",
        ):
            inputs.pop(key)
        changed = _change_target(inputs, "target")
        with torch.no_grad():
            first = model(**inputs)
            second = model(**changed)

        self.assertTrue(
            torch.allclose(first.history_summary, second.history_summary)
        )
        self.assertTrue(
            torch.allclose(first.opms_state, second.opms_state)
        )
        self.assertFalse(
            torch.allclose(first.local_state, second.local_state)
        )
        self.assertFalse(torch.allclose(first.logits, second.logits))

    def test_perm_donor_affects_only_local_branch(self) -> None:
        model = _make_model("perm_pair").eval()
        inputs = _inputs()
        changed_donor = _change_target(inputs, "donor_target")
        changed_true = _change_target(inputs, "target")

        with torch.no_grad():
            reference = model(**inputs)
            donor_output = model(**changed_donor)
            true_output = model(**changed_true)

        self.assertTrue(
            torch.allclose(
                reference.target_state,
                donor_output.target_state,
            )
        )
        self.assertTrue(
            torch.allclose(
                reference.history_summary,
                donor_output.history_summary,
            )
        )
        self.assertTrue(
            torch.allclose(reference.opms_state, donor_output.opms_state)
        )
        self.assertFalse(
            torch.allclose(reference.local_state, donor_output.local_state)
        )
        self.assertFalse(
            torch.allclose(reference.logits, donor_output.logits)
        )

        # With a fixed donor, changing the true target cannot alter the local
        # pairing branch, but it must remain live in common diagnosis.
        self.assertTrue(
            torch.allclose(reference.local_state, true_output.local_state)
        )
        self.assertTrue(
            torch.allclose(
                reference.local_target_state,
                true_output.local_target_state,
            )
        )
        self.assertFalse(
            torch.allclose(reference.target_state, true_output.target_state)
        )
        self.assertFalse(
            torch.allclose(reference.logits, true_output.logits)
        )

    def test_invalid_inputs_raise_clear_errors(self) -> None:
        with self.assertRaises(ValueError):
            _make_model("unknown")

        inputs = _inputs()
        perm = _make_model("perm_pair")
        missing_donor = dict(inputs)
        missing_donor.pop("donor_target_item_numeric")
        with self.assertRaisesRegex(
            ValueError,
            "provided together",
        ):
            perm(**missing_donor)

        real = _make_model("real_pair")
        no_donor = dict(inputs)
        for key in (
            "donor_target_item_ids",
            "donor_target_q_multi_hot",
            "donor_target_item_numeric",
        ):
            no_donor.pop(key)

        wrong_response = dict(no_donor)
        wrong_response["support_response_features"] = torch.randn(
            3,
            5,
            RESPONSE_DIM - 1,
        )
        with self.assertRaisesRegex(
            ValueError,
            "support_response_features",
        ):
            real(**wrong_response)

        wrong_mask = dict(no_donor)
        wrong_mask["support_mask"] = torch.ones(
            3,
            4,
            dtype=torch.bool,
        )
        with self.assertRaisesRegex(ValueError, "support_mask"):
            real(**wrong_mask)

        out_of_range = dict(no_donor)
        out_of_range["target_item_ids"] = torch.tensor(
            [1, 2, NUM_ITEMS + 1]
        )
        with self.assertRaisesRegex(ValueError, "1..num_items"):
            real(**out_of_range)

        float_ids = dict(no_donor)
        float_ids["target_item_ids"] = torch.tensor(
            [1.0, 2.0, 3.0]
        )
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            real(**float_ids)


if __name__ == "__main__":
    unittest.main()
