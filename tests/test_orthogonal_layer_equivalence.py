import unittest

import torch

from src.legacy.classical_orthogonal_layer import LegacyOrthoLayer
from src.model_definition.classical_orthogonal_layer import OrthoLayer


class OrthoLayerEquivalenceTests(unittest.TestCase):
    def _compare(self, in_features: int, out_features: int, batch: int = 7):
        torch.manual_seed(1729 + in_features * 10 + out_features)
        legacy = LegacyOrthoLayer(in_features, out_features).double()
        optimized = OrthoLayer(in_features, out_features).double()
        optimized.load_state_dict(legacy.state_dict())

        x_legacy = torch.randn(batch, in_features, dtype=torch.float64, requires_grad=True)
        x_optimized = x_legacy.detach().clone().requires_grad_(True)
        weights = torch.randn(batch, out_features, dtype=torch.float64)

        y_legacy = legacy(x_legacy)
        y_optimized = optimized(x_optimized)
        torch.testing.assert_close(y_optimized, y_legacy, rtol=1e-12, atol=1e-12)

        (y_legacy * weights).sum().backward()
        (y_optimized * weights).sum().backward()
        torch.testing.assert_close(x_optimized.grad, x_legacy.grad, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(
            optimized.thetas.grad, legacy.thetas.grad, rtol=1e-11, atol=1e-11
        )
        torch.testing.assert_close(
            optimized.bias.grad, legacy.bias.grad, rtol=1e-12, atol=1e-12
        )

    def test_square(self):
        self._compare(20, 20)

    def test_contracting(self):
        self._compare(21, 20)

    def test_expanding(self):
        self._compare(3, 20)

    def test_small_online_width(self):
        self._compare(6, 5)


if __name__ == "__main__":
    unittest.main()
