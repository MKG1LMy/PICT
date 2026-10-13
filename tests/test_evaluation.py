import unittest

import numpy as np

from tdnirs_posterior.evaluation import (
    evaluate_casewise_posterior,
    evaluate_posterior,
    evaluate_scalar_posterior,
)


class EvaluationTests(unittest.TestCase):
    def test_casewise_nmae_uses_physical_support_width(self) -> None:
        truth = np.zeros((1, 8), dtype=np.float32)
        truth[0, [2, 5]] = [0.001, 1.0]
        draws = np.repeat(truth[:, None, :], 8, axis=1)
        draws[:, :, 2] = 0.002
        mask = np.zeros((1, 8), dtype=bool)
        mask[:, 2] = True
        result = evaluate_casewise_posterior(draws, truth, mask)
        self.assertAlmostEqual(
            float(result["normalized_absolute_error"][0, 2]), 0.001 / 0.049
        )
        self.assertNotAlmostEqual(
            float(result["transformed_normalized_absolute_error"][0, 2]),
            float(result["normalized_absolute_error"][0, 2]),
        )

    def test_joint_thickness_sum_can_be_accurate_when_components_are_not(self) -> None:
        d1_truth = np.asarray([5.0, 6.0])
        d2_truth = np.asarray([5.0, 6.0])
        d1 = np.asarray([[7.0, 7.0, 7.0, 7.0], [8.0, 8.0, 8.0, 8.0]])
        d2 = np.asarray([[3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]])
        report = evaluate_scalar_posterior(
            d1 + d2, d1_truth + d2_truth, (1.0, 25.0)
        )
        self.assertAlmostEqual(report["mae"], 0.0)
        self.assertAlmostEqual(report["normalized_mae"], 0.0)
        self.assertGreater(
            float(np.mean(np.abs(np.median(d1, axis=1) - d1_truth))), 0.0
        )

    def test_reports_joint_draw_marginals_for_active_parameters(self) -> None:
        truth = np.zeros((2, 8), dtype=np.float32)
        truth[:, 2] = [0.01, 0.02]
        truth[:, 5] = [1.0, 1.5]
        draws = np.repeat(truth[:, None, :], 16, axis=1)
        active = np.zeros((2, 8), dtype=bool)
        active[:, [2, 5]] = True
        report = evaluate_posterior(draws, truth, active)
        self.assertEqual(report["cases"], 2)
        self.assertEqual(set(report["parameters"]), {"mua1_mm-1", "musp1_mm-1"})
        self.assertAlmostEqual(report["parameters"]["mua1_mm-1"]["mae"], 0.0)
        self.assertAlmostEqual(report["parameters"]["mua1_mm-1"]["crps"], 0.0)
        self.assertAlmostEqual(
            report["parameters"]["mua1_mm-1"]["normalized_crps"], 0.0
        )
        self.assertAlmostEqual(report["parameters"]["musp1_mm-1"]["coverage_95"], 1.0)
        self.assertEqual(report["draws_per_case"], 16)
        self.assertAlmostEqual(report["aggregate"]["mean_normalized_mae"], 0.0)
        self.assertAlmostEqual(report["aggregate"]["mean_normalized_crps"], 0.0)
        self.assertAlmostEqual(report["aggregate"]["mean_coverage_error_95"], 0.05)


if __name__ == "__main__":
    unittest.main()
