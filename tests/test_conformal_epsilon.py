import unittest

import numpy as np

from src.utils.conformal import adaptive_nonconformity, conformal_interval


class ConformalEpsilonTests(unittest.TestCase):
    def test_zero_spread_produces_finite_scores(self):
        truth = np.array([1.0, 2.0])
        predictions = np.array([[1.0, 1.0], [1.0, 1.0]])

        scores = adaptive_nonconformity(truth, predictions, epsilon=0.5)

        np.testing.assert_allclose(scores, [0.0, 2.0])
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_interval_uses_same_stabilized_scale(self):
        predictions = np.array([[1.0, 1.0], [1.0, 1.0]])

        lower, upper = conformal_interval(predictions, q_hat=2.0, epsilon=0.5)

        np.testing.assert_allclose(lower, [0.0, 0.0])
        np.testing.assert_allclose(upper, [2.0, 2.0])

    def test_nonpositive_epsilon_is_rejected(self):
        with self.assertRaises(ValueError):
            adaptive_nonconformity(np.zeros(1), np.zeros((2, 1)), epsilon=0.0)


if __name__ == "__main__":
    unittest.main()
