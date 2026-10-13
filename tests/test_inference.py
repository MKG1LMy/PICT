import unittest

import numpy as np
import torch

from tdnirs_posterior.inference import (
    infer_raw_counts,
    sample_native_posterior_batch,
    validate_model_input_grid,
)


class _PosteriorWithOutsideSamples:
    def sample(
        self,
        signal,
        layer_count,
        active_mask,
        known_mask,
        known_values,
        sample_count,
        steps,
    ):
        result = torch.zeros(1, sample_count, 8, device=signal.device)
        split = sample_count // 2
        result[0, :split, 2] = 1.25
        result[0, split:, 2] = torch.linspace(
            -0.4, 0.4, sample_count - split, device=signal.device
        )
        result[0, split:, 5] = torch.linspace(
            0.3, -0.3, sample_count - split, device=signal.device
        )
        return result


class _InferenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.posterior = _PosteriorWithOutsideSamples()

    def encode_view(self, view, sds):
        return torch.zeros(
            view["dtof"].shape[0], 16, device=view["dtof"].device
        )


class _PosteriorWithInvalidSamples:
    def sample(self, signal, layer_count, active_mask, known_mask, known_values, sample_count, steps):
        result = torch.zeros((1, sample_count, 8), device=signal.device)
        result[0, 0, 0] = -3.0
        result[0, 1, 2] = float("nan")
        return result


class _PosteriorWithLowerOpticalExcursion:
    def sample(self, signal, layer_count, active_mask, known_mask, known_values, sample_count, steps):
        result = torch.zeros((1, sample_count, 8), device=signal.device)
        result[0, :, 2] = -2.0
        return result


class InferenceTests(unittest.TestCase):
    def test_fixed_model_time_grid_rejects_mismatched_length_or_bin_width(self) -> None:
        config = {"data": {"time_bins": 2500, "time_bin_ps": 4.0}}
        dtof = np.zeros((4, 2500), dtype=np.float32)
        irf = np.zeros((4, 2500), dtype=np.float32)
        validate_model_input_grid(config, dtof, irf, 4.0)
        with self.assertRaisesRegex(ValueError, "4 ps/bin"):
            validate_model_input_grid(config, dtof, irf, 8.0)
        with self.assertRaisesRegex(ValueError, "exactly 2500 time bins"):
            validate_model_input_grid(config, dtof[:, :2048], irf[:, :2048], 4.0)

    def test_outside_samples_are_preserved_without_top_up_or_clipping(self) -> None:
        length = 128
        bins = np.arange(length, dtype=np.float64)
        irf = 2.0 + 1000.0 * np.exp(-0.5 * ((bins - 20.0) / 2.0) ** 2)
        dtof = np.stack(
            [
                3.0 + 800.0 * np.exp(-0.5 * ((bins - peak) / 5.0) ** 2)
                for peak in (30.0, 34.0, 38.0, 42.0)
            ]
        )
        config = {
            "data": {
                "sds_mm": [10.0, 20.0, 30.0, 40.0],
                "time_bins": length,
                "time_bin_ps": 4.0,
            },
            "simulation": {
                "preprocessing": {
                    "time_alignment_enabled": True,
                    "baseline_window_ps": 20.0,
                    "alignment_anchor_ns": 0.04,
                    "alignment_smooth_bins": 3,
                }
            },
            "inference": {"posterior_samples": 4, "solver_steps": 2},
        }
        samples, metadata = infer_raw_counts(
            _InferenceModel(),
            config,
            dtof,
            irf,
            dt_ps=4.0,
            layer_count=1,
            known_values={},
            sample_count=4,
            seed=7,
        )
        diagnostics = metadata["sampling_diagnostics"]
        self.assertEqual(samples.shape, (4, 8))
        self.assertEqual(diagnostics["generated_samples"], 4)
        self.assertEqual(diagnostics["outside_training_support_samples"], 2)
        self.assertEqual(diagnostics["upper_bound_violations"]["mua1_mm-1"], 2)
        self.assertAlmostEqual(diagnostics["outside_training_support_fraction"], 0.5)
        self.assertTrue(np.all(samples[:2, 2] > 0.05))
        self.assertTrue(
            metadata["preprocessing"]["irf_broadcast_from_single_channel"]
        )

    def test_four_channel_irf_is_not_marked_as_broadcast(self) -> None:
        length = 128
        bins = np.arange(length, dtype=np.float64)
        base_irf = 2.0 + 1000.0 * np.exp(-0.5 * ((bins - 20.0) / 2.0) ** 2)
        irf = np.stack([np.roll(base_irf, index) for index in range(4)])
        dtof = np.stack(
            [
                3.0 + 800.0 * np.exp(-0.5 * ((bins - peak) / 5.0) ** 2)
                for peak in (30.0, 34.0, 38.0, 42.0)
            ]
        )
        config = {
            "data": {
                "sds_mm": [10.0, 20.0, 30.0, 40.0],
                "time_bins": length,
                "time_bin_ps": 4.0,
            },
            "simulation": {
                "preprocessing": {
                    "time_alignment_enabled": False,
                    "baseline_window_ps": 20.0,
                    "alignment_anchor_ns": 0.04,
                    "alignment_smooth_bins": 3,
                }
            },
            "inference": {"posterior_samples": 4, "solver_steps": 2},
        }
        _, metadata = infer_raw_counts(
            _InferenceModel(), config, dtof, irf, 4.0, 1, {}, 4, 9
        )
        self.assertFalse(
            metadata["preprocessing"]["irf_broadcast_from_single_channel"]
        )
        self.assertEqual(
            len(metadata["preprocessing"]["applied_alignment_shift_ps"]), 4
        )

    def test_impossible_and_nonfinite_draws_are_retained_and_flagged(self) -> None:
        model = _InferenceModel()
        model.posterior = _PosteriorWithInvalidSamples()
        active = torch.from_numpy(np.asarray([[True, False, True, False, False, True, False, False]]))
        samples, diagnostics = sample_native_posterior_batch(
            model,
            torch.zeros((1, 4)),
            torch.ones(1, dtype=torch.long),
            active,
            torch.zeros((1, 8), dtype=torch.bool),
            torch.zeros((1, 8)),
            np.zeros((1, 8), dtype=np.float32),
            sample_count=3,
            steps=2,
        )
        self.assertLess(samples[0, 0, 0], 0.0)
        self.assertTrue(np.isnan(samples[0, 1, 2]))
        self.assertEqual(diagnostics[0]["physically_impossible_samples"], 1)
        self.assertEqual(diagnostics[0]["nonfinite_samples"], 1)
        self.assertTrue(diagnostics[0]["numerical_failure"])

    def test_lower_optical_support_excursions_are_reported(self) -> None:
        model = _InferenceModel()
        model.posterior = _PosteriorWithLowerOpticalExcursion()
        active = torch.from_numpy(np.asarray([[False, False, True, False, False, True, False, False]]))
        samples, diagnostics = sample_native_posterior_batch(
            model, torch.zeros((1, 4)), torch.ones(1, dtype=torch.long), active,
            torch.zeros((1, 8), dtype=torch.bool), torch.zeros((1, 8)),
            np.zeros((1, 8), dtype=np.float32), sample_count=3, steps=2,
        )
        self.assertTrue(np.all(samples[0, :, 2] > 0.0))
        self.assertTrue(np.all(samples[0, :, 2] < 0.001))
        self.assertEqual(diagnostics[0]["lower_bound_violations"]["mua1_mm-1"], 3)
        self.assertAlmostEqual(diagnostics[0]["outside_training_support_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
