import unittest
import os
import json
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tdnirs_posterior.parameters import ParameterSchema
from tdnirs_posterior.evaluation_protocol import (
    _oracle_nearest_mode_diagnostics,
    detect_candidate_modes,
    enumerate_prior_scenarios,
    spearman_with_bootstrap,
    thickness_group_labels,
)
from tdnirs_posterior.testing import _group_metrics, _write_mode_svg
from tdnirs_posterior.testing import (
    _deterministic_channel_permutations,
    _deterministic_prior_donors,
    _named_prior_scenarios,
    _prior_intervention_comparison_rows,
    _prior_values_for_mode,
)
from tdnirs_posterior.testing import run_test


class _NativePosterior:
    calls = 0

    def sample(self, signal, layer_count, active_mask, known_mask, known_values, sample_count, steps):
        type(self).calls += 1
        result = torch.rand(
            (signal.shape[0], sample_count, 8),
            dtype=signal.dtype,
            device=signal.device,
        )
        return torch.where(known_mask[:, None, :], known_values[:, None, :], result)


class _TestModel:
    def __init__(self):
        self.posterior = _NativePosterior()

    def eval(self):
        return self

    def encode_view(self, view, sds):
        return torch.zeros(
            (view["dtof"].shape[0], 4), dtype=view["dtof"].dtype
        )


class _ThreeCaseDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        schema = ParameterSchema()
        self.rows = []
        for layer in (1, 2, 3):
            for replicate in range(2):
                active = schema.active_mask(layer)
                truth = np.zeros(8, dtype=np.float32)
                truth[0:2] = 4.0 + replicate
                truth[2:5] = 0.01 + replicate * 0.002
                truth[5:8] = 1.0 + replicate * 0.1
                truth[~active] = 0.0
                irf_metadata = {
                    "irf_channel_mode": "four_similar",
                    "irf_group_ids": [0, 0, 0, 0],
                    "irf_anomaly_mask": [False, False, False, False],
                    "irf_anomaly_count": 0,
                    "irf_delay_range_ps": 30.0 + replicate,
                    "irf_fwhm_cv": 0.05,
                    "irf_pairwise_correlation_mean": 0.98,
                    "irf_pairwise_distance_mean": 0.01 + replicate * 0.001,
                    "irf_channels": [
                        {
                            "family": "single_tail",
                            "actual_fwhm_ps": 100.0 + replicate + channel,
                            "system_delay_ps": 10.0 + channel,
                            "tail_fraction": 0.1,
                            "tail_tau_ps": 300.0,
                            "second_tail_tau_ps": 600.0,
                            "measured_counts": 10000 + replicate,
                            "background_fraction": 0.01,
                            "measured_peak_before_alignment_ps": 20.0,
                            "peak_after_preprocessing_ps": 4.0,
                            "estimated_alignment_shift_ps": -16.0,
                            "applied_alignment_shift_ps": -16.0,
                            "true_irf_sha256": f"hash-{channel}",
                            "artifacts": [],
                            "has_shoulder": False,
                            "has_satellite": False,
                            "has_prepulse": False,
                            "has_echo_train": False,
                        }
                        for channel in range(4)
                    ],
                }
                self.rows.append({
                    "view_a": {
                        "dtof": torch.ones((4, 2, 8)),
                        "irf": torch.ones((4, 2, 8)),
                        "time_ps": torch.arange(8, dtype=torch.float32) * 4.0,
                        "irf_metadata_json": json.dumps(irf_metadata),
                    },
                    "theta": torch.from_numpy(schema.normalize(truth)),
                    "theta_physical": torch.from_numpy(truth),
                    "active_mask": torch.from_numpy(active),
                    "known_mask": torch.zeros(8, dtype=torch.bool),
                    "known_values": torch.zeros(8),
                    "layer_count": torch.tensor(layer),
                    "sample_id": f"case-{layer}-{replicate}",
                })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def set_epoch(self, epoch):
        pass

    def close(self):
        pass


class CompleteTestReportingTests(unittest.TestCase):
    def test_oracle_nearest_mode_is_diagnostic_and_keeps_mode_weight(self) -> None:
        values = np.asarray([[-0.8], [-0.6], [0.55], [0.65], [0.7]])
        labels = np.asarray([0, 0, 1, 1, 1])
        result = _oracle_nearest_mode_diagnostics(
            values,
            labels,
            np.asarray([[-0.7], [0.63]]),
            np.asarray([0.4, 0.6]),
            np.asarray([0.5]),
        )
        self.assertEqual(result["oracle_nearest_truth_component"], 2)
        self.assertAlmostEqual(result["oracle_nearest_truth_component_weight"], 0.6)
        self.assertAlmostEqual(result["oracle_nearest_truth_empirical_fraction"], 0.6)
        self.assertEqual(result["oracle_nearest_truth_mode_draw_count"], 3)
        np.testing.assert_allclose(
            result["truth_rms_distance_normalized"], [1.2, 0.13]
        )

    def test_candidate_mode_svg_shows_joint_and_all_unknown_marginals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mode.svg"
            samples = np.zeros((64, 8), dtype=np.float64)
            samples[:, 0] = np.linspace(-0.8, 0.8, 64)
            samples[:, 2] = np.linspace(-0.4, 0.9, 64)
            samples[:, 5] = np.sin(np.linspace(0.0, 3.0, 64))
            unknown = np.zeros(8, dtype=bool)
            unknown[[0, 2, 5]] = True
            _write_mode_svg(
                path,
                samples,
                unknown,
                {
                    "component_count": 2,
                    "bootstrap_stability": 0.9,
                    "minimum_standardized_separation": 2.5,
                    "weights": [0.4, 0.6],
                    "centers_normalized": [
                        [-0.4, -0.2, 0.1],
                        [0.5, 0.6, 0.8],
                    ],
                },
                "candidate",
            )
            svg = path.read_text(encoding="utf-8")
            self.assertIn('width="1160"', svg)
            self.assertIn("d1_mm", svg)
            self.assertIn("mua1_mm-1", svg)
            self.assertIn("musp1_mm-1", svg)
            self.assertIn("M1: 40.0%", svg)

    def test_complete_protocol_writes_resumable_store_and_all_masks(self) -> None:
        _NativePosterior.calls = 0
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            output = root / "test"
            scientific = {
                "model": {"dummy": True},
                "simulation": {
                    "dummy": True,
                    "irf": {"test_protocol": "four_channel"},
                    "preprocessing": {"time_alignment_enabled": True},
                },
                "data": {
                    "sds_mm": [10.0, 20.0, 30.0, 40.0],
                    "time_bins": 8,
                    "time_bin_ps": 4.0,
                    "time_start_ps": 0.0,
                },
            }
            config = {
                **scientific,
                "seeds": {
                    "data_split": 7,
                    "training": 7,
                    "validation": 7,
                    "evaluation": 7,
                },
                "_project_root": str(root),
                "training": {"num_workers": 0},
                "inference": {"posterior_samples": 2, "solver_steps": 1},
                "evaluation": {
                    "weights": "ema",
                    "device": "cpu",
                    "batch_size": 2,
                    "num_workers": 0,
                    "posterior_samples": 2,
                    "solver_steps": 1,
                    "prior_protocol": "exhaustive_exact",
                    "prior_interventions": {
                        "enabled": True,
                        "scope": "named",
                        "modes": ["shuffled_prior", "mask_only"],
                        "shuffle_seed_offset": 400_000,
                    },
                    "irf_interventions": {
                        "enabled": True,
                        "scope": "no_prior_and_named",
                        "modes": [
                            "shuffled_irf",
                            "fixed_irf",
                            "zero_irf",
                            "channel_permuted_irf",
                        ],
                        "shuffle_seed_offset": 600_000,
                        "fixed_seed_offset": 700_000,
                        "fixed_candidate_count": 2,
                    },
                    "bootstrap_replicates": 2,
                    "multimodality": {"enabled": False},
                },
            }
            payload = {"config": scientific, "dataset_fingerprint": {"x": 1}, "epoch": 1}
            with (
                patch("tdnirs_posterior.testing.load_checkpoint", return_value=(_TestModel(), scientific)),
                patch("tdnirs_posterior.testing.torch.load", return_value=payload),
                patch("tdnirs_posterior.testing._manifest", return_value=object()),
                patch("tdnirs_posterior.testing._dataset_fingerprint", return_value={"x": 1}),
                patch("tdnirs_posterior.testing.H5TPSFDataset", _ThreeCaseDataset),
                patch(
                    "tdnirs_posterior.testing._fixed_irf_medoid",
                    return_value=(
                        np.ones((4, 2, 8), dtype=np.float32),
                        {"sha256": "fixed-test-irf"},
                    ),
                ),
            ):
                report_path = run_test(config, checkpoint, output)
                calls_after_first_run = _NativePosterior.calls
                resumed_path = run_test(config, checkpoint, output)
                config["evaluation"]["batch_size"] = 1
                with self.assertRaisesRegex(ValueError, "incompatible evaluation"):
                    run_test(config, checkpoint, output)
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["scope"]["scenario_count"], 289)
            self.assertEqual(report["scope"]["prior_intervention_scenario_count"], 36)
            self.assertEqual(report["scope"]["irf_intervention_scenario_count"], 84)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(report["channel_subsets"]["subsets"]), 15)
            self.assertTrue((output / "channel_subset_metrics.csv").exists())
            self.assertTrue((output / "channel_subset_effects.csv").exists())
            self.assertIn("channel-subsets", report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {row["thickness_variable"] for row in report["thickness_associations"]},
                {"d1_mm", "d2_mm", "d1_plus_d2_mm"},
            )
            self.assertTrue((output / "posterior.h5").exists())
            self.assertTrue((output / "prior_intervention_effects.csv").exists())
            self.assertTrue((output / "prior_intervention_summary.csv").exists())
            self.assertTrue((output / "prior_intervention_assignments.csv").exists())
            self.assertTrue((output / "irf_intervention_effects.csv").exists())
            self.assertTrue((output / "irf_intervention_summary.csv").exists())
            self.assertTrue((output / "irf_intervention_assignments.csv").exists())
            self.assertTrue((output / "irf_case_metadata.csv").exists())
            self.assertTrue((output / "irf_difficulty_metrics.csv").exists())
            self.assertEqual(len(report["irf_case_metadata"]), 24)
            self.assertEqual(report_path, output / "report.html")
            self.assertEqual(resumed_path, report_path)
            self.assertEqual(_NativePosterior.calls, calls_after_first_run)
            import h5py
            with h5py.File(output / "posterior.h5", "r") as store:
                for name, group in store["channel_subsets"].items():
                    if name == "sds_10_20_30_40":
                        continue
                    for scenario_id, scenario_group in group.items():
                        # The fake encoder intentionally ignores observations:
                        # identical draws prove common random numbers across masks.
                        np.testing.assert_array_equal(scenario_group["samples"][:], store["scenarios"][scenario_id]["samples"][:])
            html_report = report_path.read_text(encoding="utf-8")
            self.assertIn("PICT Model posterior evaluation", html_report)
            self.assertIn("All legal prior masks", html_report)
            self.assertIn("nMAE relative improvement", html_report)
            self.assertIn("Stable candidate multimodal sample distributions", html_report)
            self.assertIn("third-layer starting depth", html_report)
            self.assertIn("oracle nearest mode", html_report)
            self.assertIn("Counterfactual validation of prior values", html_report)
            with (output / "prior_paired_effects.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                paired_rows = list(csv.DictReader(handle))
            with (output / "parameter_metrics.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                parameter_rows = list(csv.DictReader(handle))
            with (output / "case_predictions.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                case_rows = list(csv.DictReader(handle))
            with (output / "prior_intervention_effects.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                intervention_rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["comparison_id"] for row in intervention_rows},
                {
                    "exact_prior_vs_no_prior",
                    "shuffled_prior_vs_no_prior",
                    "mask_only_vs_no_prior",
                    "exact_prior_vs_shuffled_prior",
                    "exact_prior_vs_mask_only",
                },
            )
            self.assertTrue(
                any(row["parameter"] == "__macro__" for row in intervention_rows)
            )
            self.assertIn(
                report["headline"]["prior_value_interventions"]["conclusion"],
                {"consistently_supported", "mixed_evidence", "not_supported"},
            )
            self.assertTrue(
                any(
                    row["parameter"] == "d1_plus_d2_mm"
                    and row["scenario_id"] == "N3_k0_00000000"
                    for row in parameter_rows
                )
            )
            self.assertTrue(
                any(row["parameter"] == "d1_plus_d2_mm" for row in case_rows)
            )
            self.assertIn("surface_depth", report["headline"]["no_prior_by_layer"]["3"])
            parameter_row = next(
                row for row in paired_rows if row["parameter"] != "__joint__"
            )
            self.assertIn("baseline_normalized_mae", parameter_row)
            self.assertIn("current_normalized_mae", parameter_row)
            self.assertIn("relative_improvement_normalized_mae", parameter_row)
            import h5py
            with h5py.File(output / "posterior.h5", "r") as store:
                scenario = store["scenarios"]["N1_k0_00"]
                self.assertIn("lower_bound_violations", scenario)
                self.assertIn("upper_bound_violations", scenario)
                self.assertIn("normalized_exceedance", scenario)
                self.assertIn("sample_status_flags", scenario)
                self.assertIn("bit_definitions", scenario["sample_status_flags"].attrs)
                shuffled = store["prior_interventions"]["shuffled_prior"]["N1_k1_10"]
                mask_only = store["prior_interventions"]["mask_only"]["N1_k1_10"]
                self.assertEqual(shuffled.attrs["status"], "completed")
                self.assertIn("donor_global_case_index", shuffled)
                self.assertIn("known_values_normalized", shuffled)
                np.testing.assert_allclose(mask_only["known_values_normalized"][:], 0.0)
                exact = store["scenarios"]["N1_k1_10"]["samples"][:]
                no_prior = store["scenarios"]["N1_k0_00"]["samples"][:]
                np.testing.assert_allclose(exact[:, :, 5], shuffled["samples"][:, :, 5])
                np.testing.assert_allclose(exact[:, :, 5], mask_only["samples"][:, :, 5])
                np.testing.assert_allclose(exact[:, :, 5], no_prior[:, :, 5])
                self.assertEqual(
                    set(store["irf_interventions"]),
                    {
                        "shuffled_irf",
                        "fixed_irf",
                        "zero_irf",
                        "channel_permuted_irf",
                    },
                )
                for mode in (
                    "shuffled_irf",
                    "fixed_irf",
                    "zero_irf",
                    "channel_permuted_irf",
                ):
                    self.assertIn(
                        "N1_k0_00", store["irf_interventions"][mode]
                    )
                permutation = store["irf_interventions"][
                    "channel_permuted_irf"
                ]["N1_k0_00"]["irf_channel_permutation"][:]
                identity = np.broadcast_to(np.arange(4), permutation.shape)
                self.assertFalse(np.any(permutation == identity))

    def test_channel_permutations_are_reproducible_and_have_no_fixed_points(self) -> None:
        first = _deterministic_channel_permutations(20, 123)
        second = _deterministic_channel_permutations(20, 123)
        np.testing.assert_array_equal(first, second)
        expected = np.broadcast_to(np.arange(4), first.shape)
        self.assertFalse(np.any(first == expected))
        for row in first:
            np.testing.assert_array_equal(np.sort(row), np.arange(4))

    def test_exhaustive_prior_masks_have_expected_counts_and_unknowns(self) -> None:
        self.assertEqual([len(enumerate_prior_scenarios(n)) for n in (1, 2, 3)], [3, 31, 255])
        for layer in (1, 2, 3):
            for scenario in enumerate_prior_scenarios(layer):
                self.assertGreater(len(scenario.unknown_parameters), 0)
                self.assertEqual(len(scenario.known_parameters), scenario.known_count)

    def test_named_prior_interventions_have_expected_count(self) -> None:
        scenarios = [
            scenario
            for layer in (1, 2, 3)
            for scenario in enumerate_prior_scenarios(layer)
        ]
        selected = _named_prior_scenarios(scenarios)
        self.assertEqual(len(selected), 18)
        self.assertTrue(all(scenario.known_count > 0 for scenario in selected))

    def test_prior_donors_are_reproducible_same_layer_derangements(self) -> None:
        layers = np.asarray([1, 1, 1, 2, 2, 3])
        first = _deterministic_prior_donors(layers, 123)
        second = _deterministic_prior_donors(layers, 123)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[:5] != np.arange(5)))
        self.assertTrue(np.all(layers[first[:5]] == layers[:5]))
        self.assertEqual(first[5], -1)

    def test_mask_only_values_are_case_independent_normalized_zero(self) -> None:
        schema = ParameterSchema()
        truth = np.asarray(
            [
                [4.0, 0.0, 0.01, 0.0, 0.0, 1.0, 0.0, 0.0],
                [8.0, 0.0, 0.02, 0.0, 0.0, 2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        known = np.zeros((2, 8), dtype=bool)
        known[:, 2] = True
        normalized, physical = _prior_values_for_mode(
            "mask_only", known, np.arange(2), np.asarray([1, 0]),
            schema.normalize(truth), truth,
            schema.denormalize(np.zeros((1, 8), dtype=np.float32))[0],
        )
        np.testing.assert_allclose(normalized, 0.0)
        self.assertEqual(physical[0, 2], physical[1, 2])
        self.assertGreater(physical[0, 2], 0.0)

    def test_intervention_effect_signs_and_unknown_parameter_scope(self) -> None:
        schema = ParameterSchema()
        scenario = next(
            value
            for value in enumerate_prior_scenarios(1)
            if value.scenario_id == "N1_k1_10"
        )
        truth = np.zeros((2, 8), dtype=np.float32)
        truth[:, 2] = [0.01, 0.02]
        truth[:, 5] = [1.0, 1.5]
        active = np.stack([schema.active_mask(1), schema.active_mask(1)])
        candidate = np.repeat(truth[:, None, :], 16, axis=1)
        reference = candidate.copy()
        reference[:, :, 5] += 0.5
        diagnostics = np.zeros((2, 5), dtype=np.float32)
        diagnostics[:, 0] = 1.0
        rows = _prior_intervention_comparison_rows(
            scenario,
            candidate,
            reference,
            truth,
            active,
            diagnostics,
            diagnostics,
            "exact_prior",
            "mask_only",
            77,
            20,
        )
        parameters = {row["parameter"] for row in rows}
        self.assertNotIn("mua1_mm-1", parameters)
        self.assertIn("musp1_mm-1", parameters)
        self.assertIn("__macro__", parameters)
        macro = next(row for row in rows if row["parameter"] == "__macro__")
        self.assertLess(macro["delta_normalized_mae"], 0.0)
        self.assertGreater(macro["relative_improvement_normalized_mae"], 0.0)

    def test_thickness_bins_are_disjoint_and_include_upper_boundary(self) -> None:
        truth = np.zeros((8, 8), dtype=np.float32)
        layers = np.asarray([2, 2, 2, 2, 3, 3, 3, 3])
        truth[:4, 0] = [0.5, 5.0, 10.0, 20.0]
        truth[4:, 0] = [4.0, 5.0, 10.0, 12.5]
        truth[4:, 1] = [5.0, 5.0, 10.0, 12.5]
        labels = thickness_group_labels(truth, layers)
        self.assertEqual(len(labels), 8)
        self.assertEqual(labels[0], "N2_d1_[0.5,5)")
        self.assertEqual(labels[1], "N2_d1_[5,10)")
        self.assertEqual(labels[7], "N3_d1+d2_[20,25]")

    def test_spearman_does_not_dispatch_to_numpy_corrcoef(self) -> None:
        values = np.asarray([1.0, 3.0, 2.0, 5.0, 4.0])
        with patch("numpy.corrcoef", side_effect=AssertionError("BLAS dispatch")):
            result = spearman_with_bootstrap(values, values, seed=3, replicates=5)
        self.assertAlmostEqual(result["rho"], 1.0)

    @unittest.skipUnless(
        os.environ.get("TDNIRS_TEST_SKLEARN") == "1",
        "Set TDNIRS_TEST_SKLEARN=1 to exercise the platform GMM backend",
    )
    def test_gmm_detects_stable_separated_candidate_modes(self) -> None:
        rng = np.random.default_rng(42)
        samples = np.zeros((400, 8), dtype=np.float64)
        samples[:200, 2] = rng.normal(-0.65, 0.04, 200)
        samples[200:, 2] = rng.normal(0.65, 0.04, 200)
        mask = np.zeros(8, dtype=bool)
        mask[2] = True
        try:
            result = detect_candidate_modes(
                samples,
                mask,
                seed=42,
                normalized_truth=np.zeros(8, dtype=np.float64),
                max_components=3,
                bootstrap_replicates=5,
                minimum_separation=2.0,
            )
        except OSError as error:
            self.skipTest(f"Local BLAS runtime is unavailable: {error}")
        self.assertEqual(result["component_count"], 2)
        self.assertTrue(result["stable_multimodal"])
        self.assertTrue(result["gmm_converged"])
        self.assertEqual(result["bootstrap_requested_replicates"], 5)
        self.assertEqual(result["bootstrap_successful_replicates"], 5)
        self.assertEqual(result["bootstrap_failed_replicates"], 0)
        self.assertEqual(len(result["gmm_fit_diagnostics"]), 3)
        self.assertIn(result["oracle_nearest_truth_component"], (1, 2))
        self.assertGreater(result["oracle_nearest_truth_component_weight"], 0.4)
        self.assertLess(result["oracle_nearest_truth_component_weight"], 0.6)
        self.assertEqual(result["oracle_nearest_truth_mode_draw_count"], 200)

        single = np.zeros((400, 8), dtype=np.float64)
        single[:, 2] = rng.normal(0.0, 0.18, 400)
        single_result = detect_candidate_modes(
            single, mask, seed=43, max_components=3, bootstrap_replicates=5
        )
        self.assertEqual(single_result["component_count"], 1)
        self.assertFalse(single_result["stable_multimodal"])

        outside = samples.copy()
        outside[200:, 2] += 0.9
        outside_result = detect_candidate_modes(
            outside, mask, seed=44, max_components=3, bootstrap_replicates=5
        )
        self.assertTrue(outside_result["stable_multimodal"])
        self.assertGreater(
            max(outside_result["component_outside_training_support_fraction"]), 0.9
        )

        long_tail = np.zeros((600, 8), dtype=np.float64)
        long_tail[:, 2] = rng.normal(0.0, 0.15, 600)
        long_tail[:12, 2] = rng.normal(1.8, 0.05, 12)
        tail_result = detect_candidate_modes(
            long_tail, mask, seed=45, max_components=3, bootstrap_replicates=5
        )
        self.assertFalse(tail_result["stable_multimodal"])

    def test_gmm_retry_is_tracked_without_silently_accepting_first_failure(self) -> None:
        class FakeGaussianMixture:
            def __init__(self, **kwargs):
                self.n_components = kwargs["n_components"]
                self.max_iter = kwargs["max_iter"]

            def fit(self, data):
                self.converged_ = self.max_iter >= 1000
                self.n_iter_ = self.max_iter
                self.weights_ = np.ones(self.n_components) / self.n_components
                self.means_ = np.zeros((self.n_components, data.shape[1]))
                self.covariances_ = np.repeat(
                    np.eye(data.shape[1])[None, :, :], self.n_components, axis=0
                )
                return self

            def bic(self, data):
                return float(self.n_components)

            def predict(self, data):
                return np.zeros(len(data), dtype=np.int64)

        samples = np.zeros((40, 8), dtype=np.float64)
        mask = np.zeros(8, dtype=bool)
        mask[2] = True
        with patch("sklearn.mixture.GaussianMixture", FakeGaussianMixture):
            result = detect_candidate_modes(
                samples,
                mask,
                seed=91,
                max_components=1,
                bootstrap_replicates=2,
                max_iter=3,
                retry_max_iter=1000,
            )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["gmm_retried_candidate_count"], 1)
        self.assertEqual(result["bootstrap_retried_replicates"], 2)
        self.assertEqual(result["bootstrap_failed_replicates"], 0)
        self.assertEqual(len(result["gmm_fit_diagnostics"][0]["attempts"]), 2)

    def test_gmm_final_failure_is_explicit_and_bootstrap_is_marked_skipped(self) -> None:
        class NeverConverges:
            def __init__(self, **kwargs):
                self.n_components = kwargs["n_components"]

            def fit(self, data):
                self.converged_ = False
                self.n_iter_ = 7
                return self

        samples = np.zeros((40, 8), dtype=np.float64)
        mask = np.zeros(8, dtype=bool)
        mask[2] = True
        with patch("sklearn.mixture.GaussianMixture", NeverConverges):
            result = detect_candidate_modes(
                samples,
                mask,
                seed=92,
                max_components=2,
                bootstrap_replicates=3,
            )
        self.assertEqual(result["status"], "gmm_nonconverged")
        self.assertFalse(result["gmm_converged"])
        self.assertEqual(result["gmm_failed_candidate_count"], 2)
        self.assertEqual(result["bootstrap_failed_replicates"], 0)
        self.assertEqual(result["bootstrap_skipped_replicates"], 3)

    def test_groups_unknown_parameters_without_scoring_exact_priors(self) -> None:
        schema = ParameterSchema()
        layers = np.asarray([1, 2, 3], dtype=np.int64)
        active = np.stack([schema.active_mask(int(layer)) for layer in layers])
        truth = np.zeros((3, 8), dtype=np.float32)
        truth[:, 0:2] = 4.0
        truth[:, 2:5] = 0.01
        truth[:, 5:8] = 1.0
        truth[~active] = 0.0
        samples = np.repeat(truth[:, None, :], 8, axis=1)
        known = np.zeros_like(active)
        known[1, 0] = True
        known[2, [0, 2]] = True

        report = _group_metrics(samples, truth, active, known, layers)

        self.assertEqual(set(report["by_layer_count"]), {"1", "2", "3"})
        self.assertEqual(set(report["by_known_count"]), {"0", "1", "2"})
        # d1 is active for N=2 and N=3, but known in both cases, so it must not
        # appear as a perfectly predicted output metric.
        self.assertNotIn(
            "d1_mm", report["all_unknown_parameters"]["parameters"]
        )
        self.assertAlmostEqual(
            report["all_unknown_parameters"]["aggregate"]["mean_normalized_mae"],
            0.0,
        )
        self.assertIn(
            "d1_plus_d2_mm",
            report["all_unknown_parameters"]["parameters"],
        )

        all_n3_thickness_known = known.copy()
        all_n3_thickness_known[2, [0, 1]] = True
        fully_known_report = _group_metrics(
            samples, truth, active, all_n3_thickness_known, layers
        )
        self.assertNotIn(
            "d1_plus_d2_mm",
            fully_known_report["by_layer_count"]["3"]["parameters"],
        )


if __name__ == "__main__":
    unittest.main()
