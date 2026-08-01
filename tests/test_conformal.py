import math
import unittest

import numpy as np

from src.utils.conformal import (
    conformal_metrics,
    finite_sample_quantile,
    grouped_conformal_quantile,
    repeated_trajectory_resplits,
)


class ConformalTests(unittest.TestCase):
    def test_exact_order_statistic(self):
        scores = np.arange(1.0, 11.0)
        # ceil((10 + 1) * .8) = 9
        self.assertEqual(finite_sample_quantile(scores, 0.8), 9.0)

    def test_infinite_small_sample_edge(self):
        self.assertTrue(math.isinf(finite_sample_quantile([1.0, 2.0], 0.9)))

    def test_trajectory_maxima_are_exchangeability_units(self):
        truth = np.array([[0.0, 2.0], [0.0, 4.0], [0.0, 6.0]])
        predictions = np.stack([np.zeros_like(truth), np.zeros_like(truth)])
        q_hat, scores = grouped_conformal_quantile(
            truth, predictions, 0.5, num_units=3, epsilon=1.0
        )
        np.testing.assert_allclose(scores, [2.0, 4.0, 6.0])
        # ceil((3 + 1) * .5) = 2
        self.assertEqual(q_hat, 4.0)

    def test_simultaneous_coverage_is_grouped_by_trajectory(self):
        truth = np.array([[0.0, 0.0], [0.0, 2.0]])
        predictions = np.stack([np.zeros_like(truth), np.zeros_like(truth)])
        metrics = conformal_metrics(
            truth, predictions, q_hat=1.0, num_units=2, epsilon=1.0
        )
        self.assertEqual(metrics["marginal_coverage"], 0.75)
        self.assertEqual(metrics["simultaneous_trajectory_coverage"], 0.5)

    def test_repeated_resplits_keep_whole_trajectories(self):
        calibration_truth = np.arange(12.0).reshape(3, 4)
        test_truth = np.arange(12.0, 24.0).reshape(3, 4)
        truth = np.concatenate((calibration_truth, test_truth))
        predictions = np.stack((truth - 1.0, truth + 1.0))
        summary = repeated_trajectory_resplits(
            calibration_truth,
            predictions[:, :3],
            test_truth,
            predictions[:, 3:],
            0.5,
            trials=5,
            seed=17,
            epsilon=1.0,
        )
        self.assertEqual(summary["trials"], 5)
        self.assertEqual(summary["num_calibration_trajectories"], 3)
        self.assertEqual(summary["num_test_trajectories"], 3)
        self.assertEqual(
            summary["metrics"]["simultaneous_trajectory_coverage"]["mean"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
