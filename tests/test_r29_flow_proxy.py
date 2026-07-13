from __future__ import annotations

import unittest

import torch

from scripts.audit_r29_flow_signal import MatchedProxy, state_hash


class R29FlowProxyTest(unittest.TestCase):
    def test_flow_and_capacity_have_identical_parameters_and_initialization(self) -> None:
        torch.manual_seed(42)
        flow = MatchedProxy(flow=True, steps=4)
        torch.manual_seed(42)
        capacity = MatchedProxy(flow=False, steps=4)
        self.assertEqual(state_hash(flow), state_hash(capacity))
        self.assertEqual(
            sum(parameter.numel() for parameter in flow.parameters()),
            sum(parameter.numel() for parameter in capacity.parameters()),
        )

    def test_flow_has_finite_gradients(self) -> None:
        torch.manual_seed(42)
        model = MatchedProxy(flow=True, steps=4)
        features = torch.rand(16, 6)
        logits = model(features)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.rand(16))
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
