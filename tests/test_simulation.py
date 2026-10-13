import unittest
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import yaml

from tdnirs_posterior.simulation import (
    generate_channel_irfs,
    generate_true_irf,
    preprocess_observed,
    simulate_measurement,
)
from tdnirs_posterior.simulation.acquisition import _convolve_paired_channels
from tdnirs_posterior.simulation.irf import (
    IRF_CHANNEL_MODES,
    GeneratedIRFSet,
    _weighted_choice,
)


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        project = Path(__file__).resolve().parents[1]
        with (project / "configs" / "smoke.yaml").open("r", encoding="utf-8") as handle:
            cls.config = yaml.safe_load(handle)["simulation"]
        source = sorted((project / "Data").glob("*/mixed-dataset.h5"))[0]
        with h5py.File(source, "r") as h5:
            sds = np.asarray(h5["sds_mm"][0])
            indices = [int(np.flatnonzero(np.isclose(sds, value))[0]) for value in [10, 20, 30, 40]]
            cls.tpsf = np.asarray(h5["tpsf"][0, indices, :], dtype=np.float64)
            cls.dt_ps = float(np.median(np.diff(h5["time_edges_s"][:])) * 1e12)

    def test_irf_is_unit_area_and_delayed(self) -> None:
        generated = generate_true_irf(
            self.dt_ps, self.config["irf"], np.random.default_rng(4)
        )
        self.assertAlmostEqual(float(generated.values.sum()), 1.0, places=10)
        self.assertGreaterEqual(generated.prompt_peak_ps, 1000.0 - self.dt_ps)
        self.assertLessEqual(generated.prompt_peak_ps, 3000.0 + self.dt_ps)

    def test_measurement_is_aligned_and_normalized(self) -> None:
        original = self.tpsf.copy()
        aligned_config = deepcopy(self.config)
        aligned_config["preprocessing"]["time_alignment_enabled"] = True
        result = simulate_measurement(
            self.tpsf,
            self.dt_ps,
            aligned_config,
            np.random.default_rng(123),
        )
        np.testing.assert_array_equal(self.tpsf, original)
        self.assertEqual(result.dtof.shape, (4, 2500))
        self.assertEqual(result.irf.shape, (4, 2500))
        np.testing.assert_allclose(result.dtof.sum(axis=1), 1.0, atol=2e-6)
        np.testing.assert_allclose(result.irf.sum(axis=1), 1.0, atol=2e-6)
        for channel in result.metadata["irf_channels"]:
            self.assertLess(abs(channel["peak_after_preprocessing_ps"] - 500.0), 12.0)
        self.assertTrue(result.metadata["time_alignment_enabled"])
        for channel in result.metadata["irf_channels"]:
            self.assertAlmostEqual(
                channel["applied_alignment_shift_ps"],
                channel["estimated_alignment_shift_ps"],
            )
        self.assertEqual(
            result.metadata["dtof_background_fraction_reference"],
            "full_linear_convolution_window",
        )
        self.assertNotIn("true_irf", result.metadata)

    def test_measurement_can_preserve_system_delay_without_alignment(self) -> None:
        unaligned_config = deepcopy(self.config)
        unaligned_config["preprocessing"]["time_alignment_enabled"] = False
        unaligned = simulate_measurement(
            self.tpsf,
            self.dt_ps,
            unaligned_config,
            np.random.default_rng(123),
        )
        aligned_config = deepcopy(self.config)
        aligned_config["preprocessing"]["time_alignment_enabled"] = True
        aligned = simulate_measurement(
            self.tpsf,
            self.dt_ps,
            aligned_config,
            np.random.default_rng(123),
        )
        np.testing.assert_allclose(unaligned.dtof.sum(axis=1), 1.0, atol=2e-6)
        np.testing.assert_allclose(unaligned.irf.sum(axis=1), 1.0, atol=2e-6)
        self.assertFalse(unaligned.metadata["time_alignment_enabled"])
        for channel in unaligned.metadata["irf_channels"]:
            self.assertEqual(channel["applied_alignment_shift_ps"], 0.0)
            self.assertLess(
                abs(
                    channel["peak_after_preprocessing_ps"]
                    - channel["measured_peak_before_alignment_ps"]
                ),
                2.0 * self.dt_ps,
            )
        self.assertFalse(np.allclose(unaligned.irf, aligned.irf))

    def test_observed_preprocessing_switch_controls_only_time_shift(self) -> None:
        bins = np.arange(128, dtype=np.float64)
        irf = 2.0 + 1000.0 * np.exp(-0.5 * ((bins - 30.0) / 2.0) ** 2)
        dtof = np.stack(
            [
                3.0 + 800.0 * np.exp(-0.5 * ((bins - peak) / 5.0) ** 2)
                for peak in (45.0, 48.0, 51.0, 54.0)
            ]
        )
        preprocessing = {
            "time_alignment_enabled": False,
            "baseline_window_ps": 20.0,
            "alignment_anchor_ns": 0.04,
            "alignment_smooth_bins": 3,
        }
        unaligned = preprocess_observed(dtof, irf, 4.0, preprocessing)
        preprocessing["time_alignment_enabled"] = True
        aligned = preprocess_observed(dtof, irf, 4.0, preprocessing)
        self.assertEqual(unaligned.metadata["applied_alignment_shift_ps"], [0.0] * 4)
        for value in unaligned.metadata["irf_peak_after_preprocessing_ps"]:
            self.assertAlmostEqual(value, 120.0, delta=4.0)
        for value in aligned.metadata["irf_peak_after_preprocessing_ps"]:
            self.assertAlmostEqual(value, 40.0, delta=4.0)
        np.testing.assert_allclose(unaligned.dtof.sum(axis=1), 1.0, atol=2e-6)
        np.testing.assert_allclose(aligned.dtof.sum(axis=1), 1.0, atol=2e-6)

    def test_observed_preprocessing_aligns_each_pair_independently(self) -> None:
        bins = np.arange(128, dtype=np.float64)
        irf_peaks = (20.0, 25.0, 30.0, 35.0)
        irf = np.stack(
            [
                2.0 + 1000.0 * np.exp(-0.5 * ((bins - peak) / 2.0) ** 2)
                for peak in irf_peaks
            ]
        )
        dtof = np.stack(
            [
                3.0 + 800.0 * np.exp(-0.5 * ((bins - peak - 12.0) / 5.0) ** 2)
                for peak in irf_peaks
            ]
        )
        result = preprocess_observed(
            dtof,
            irf,
            4.0,
            {
                "time_alignment_enabled": True,
                "baseline_window_ps": 20.0,
                "alignment_anchor_ns": 0.04,
                "alignment_smooth_bins": 3,
            },
        )
        self.assertEqual(len(set(result.metadata["applied_alignment_shift_ps"])), 4)
        for peak in result.metadata["irf_peak_after_preprocessing_ps"]:
            self.assertAlmostEqual(peak, 40.0, delta=4.0)
        dtof_peaks = [float(np.argmax(channel) * 4.0) for channel in result.dtof]
        for peak in dtof_peaks:
            self.assertAlmostEqual(peak, 88.0, delta=8.0)

    def test_channel_modes_are_reproducible_and_all_shared_is_exact(self) -> None:
        config = deepcopy(self.config["irf"])
        config["channel_model"]["mode_probabilities"] = {
            "all_shared": 1.0,
            "four_similar": 0.0,
            "grouped_2_2": 0.0,
            "grouped_3_1": 0.0,
            "partial_outlier": 0.0,
            "heterogeneous": 0.0,
        }
        first = generate_channel_irfs(self.dt_ps, config, np.random.default_rng(91))
        second = generate_channel_irfs(self.dt_ps, config, np.random.default_rng(91))
        self.assertEqual(first.mode, "all_shared")
        np.testing.assert_array_equal(first.values, second.values)
        for channel in first.values[1:]:
            np.testing.assert_array_equal(channel, first.values[0])

    def test_partial_outlier_has_configured_anomaly_count(self) -> None:
        config = deepcopy(self.config["irf"])
        config["channel_model"]["mode_probabilities"] = {
            "all_shared": 0.0,
            "four_similar": 0.0,
            "grouped_2_2": 0.0,
            "grouped_3_1": 0.0,
            "partial_outlier": 1.0,
            "heterogeneous": 0.0,
        }
        generated = generate_channel_irfs(
            self.dt_ps, config, np.random.default_rng(19)
        )
        self.assertIn(sum(generated.anomaly_mask), (1, 2))
        np.testing.assert_allclose(generated.values.sum(axis=1), 1.0, atol=1e-10)
        self.assertTrue(np.all(np.isfinite(generated.values)))
        self.assertTrue(np.all(generated.values >= 0.0))

    def test_channel_mode_probabilities_match_protocol(self) -> None:
        probabilities = self.config["irf"]["channel_model"]["mode_probabilities"]
        rng = np.random.default_rng(20260913)
        draws = [
            _weighted_choice(rng, probabilities, IRF_CHANNEL_MODES)
            for _ in range(50000)
        ]
        for mode, expected in probabilities.items():
            observed = draws.count(mode) / len(draws)
            self.assertAlmostEqual(observed, expected, delta=0.01)

    def test_paired_convolution_has_no_cross_channel_effect(self) -> None:
        config = deepcopy(self.config["irf"])
        config["channel_model"]["mode_probabilities"] = {
            "all_shared": 0.0,
            "four_similar": 0.0,
            "grouped_2_2": 0.0,
            "grouped_3_1": 0.0,
            "partial_outlier": 0.0,
            "heterogeneous": 1.0,
        }
        generated = generate_channel_irfs(
            self.dt_ps, config, np.random.default_rng(12)
        )
        baseline = _convolve_paired_channels(self.tpsf, generated)
        swapped_channels = list(generated.channels)
        swapped_channels[0], swapped_channels[1] = (
            swapped_channels[1],
            swapped_channels[0],
        )
        swapped = _convolve_paired_channels(
            self.tpsf,
            GeneratedIRFSet(
                channels=tuple(swapped_channels),
                mode=generated.mode,
                group_ids=generated.group_ids,
                anomaly_mask=generated.anomaly_mask,
            ),
        )
        self.assertFalse(np.allclose(baseline[0], swapped[0]))
        self.assertFalse(np.allclose(baseline[1], swapped[1]))
        np.testing.assert_array_equal(baseline[2:], swapped[2:])

    def test_all_shared_true_irfs_have_independent_measured_noise(self) -> None:
        config = deepcopy(self.config)
        config["irf"]["channel_model"]["mode_probabilities"] = {
            "all_shared": 1.0,
            "four_similar": 0.0,
            "grouped_2_2": 0.0,
            "grouped_3_1": 0.0,
            "partial_outlier": 0.0,
            "heterogeneous": 0.0,
        }
        measured = simulate_measurement(
            self.tpsf, self.dt_ps, config, np.random.default_rng(982)
        )
        hashes = {
            channel["true_irf_sha256"]
            for channel in measured.metadata["irf_channels"]
        }
        self.assertEqual(len(hashes), 1)
        self.assertTrue(any(
            not np.array_equal(measured.irf[0], measured.irf[index])
            for index in range(1, 4)
        ))


if __name__ == "__main__":
    unittest.main()
