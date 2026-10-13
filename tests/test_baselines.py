from __future__ import annotations

import inspect
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from tdnirs_posterior.baseline_comparison import compare_deterministic_baselines
from tdnirs_posterior.baseline_evaluation import evaluate_point_predictions
from tdnirs_posterior.baseline_testing import _irf_stratified_rows, _report_html
from tdnirs_posterior.baseline_training import _update_early_stopping
from tdnirs_posterior.cli.training_paths import resolve_training_paths
from tdnirs_posterior.config import load_config
from tdnirs_posterior.model import BaselineConfig, build_baseline, masked_smooth_l1
from tdnirs_posterior.parameters import CANONICAL_PARAMETERS, ParameterSchema


def test_cnn_and_ann_accept_only_dtof_and_layer_count() -> None:
    dtof = torch.rand(3, 4, 2, 32)
    layers = torch.tensor([1, 2, 3])
    cnn = build_baseline(
        BaselineConfig(
            architecture="cnn",
            cnn_channels=(8, 16),
            cnn_pool_bins=4,
            cnn_head_widths=(16,),
        ),
        32,
    )
    ann = build_baseline(BaselineConfig(architecture="ann", ann_widths=(16, 8)), 32)
    assert cnn(dtof, layers).shape == (3, 8)
    assert ann(dtof, layers).shape == (3, 8)
    assert list(inspect.signature(cnn.forward).parameters) == ["dtof", "layer_count", "channel_mask"]
    assert list(inspect.signature(ann.forward).parameters) == ["dtof", "layer_count", "channel_mask"]
    assert not any("irf" in name or "known" in name for name, _ in cnn.named_parameters())
    assert not any("irf" in name or "known" in name for name, _ in ann.named_parameters())


def test_masked_loss_weights_each_case_equally() -> None:
    prediction = torch.ones(2, 8)
    truth = torch.zeros_like(prediction)
    active = torch.zeros_like(prediction, dtype=torch.bool)
    active[0, :2] = True
    active[1, :] = True
    # Every active element has the same error, so differing active dimensions
    # must not change the per-case contribution.
    loss = masked_smooth_l1(prediction, truth, active, beta=0.1)
    expected = torch.nn.functional.smooth_l1_loss(torch.ones(1), torch.zeros(1), beta=0.1)
    assert torch.allclose(loss, expected)


def test_inactive_predictions_do_not_affect_point_metrics() -> None:
    schema = ParameterSchema()
    layers = np.asarray([1, 2, 3])
    active = np.stack([schema.active_mask(int(layer)) for layer in layers])
    truth = np.tile(np.asarray([5.0, 6.0, 0.01, 0.02, 0.03, 1.0, 1.2, 1.4]), (3, 1))
    prediction = truth.copy()
    baseline = evaluate_point_predictions(prediction, truth, active, layers, bootstrap_seed=4, bootstrap_replicates=5)
    prediction[~active] = 1e30
    changed = evaluate_point_predictions(prediction, truth, active, layers, bootstrap_seed=4, bootstrap_replicates=5)
    assert baseline["macro_nmae"] == changed["macro_nmae"]
    parameter_index = {name: index for index, name in enumerate(CANONICAL_PARAMETERS)}
    assert all(
        active[row["case_index"], parameter_index[row["parameter"]]]
        for row in changed["casewise_metrics"]
    )


def test_irf_stratified_baseline_metrics_exclude_inactive_parameters() -> None:
    schema = ParameterSchema()
    layers = np.asarray([1, 1, 2, 2])
    active = np.stack([schema.active_mask(int(layer)) for layer in layers])
    truth = np.zeros((4, 8), dtype=np.float64)
    truth[:, [2, 5]] = [0.01, 1.0]
    truth[2:, [0, 3, 6]] = [5.0, 0.02, 1.2]
    metadata = [
        {
            "irf_channel_mode": "all_shared",
            "irf_delay_range_ps": 0.0,
            "irf_fwhm_cv": 0.0,
            "irf_pairwise_distance_mean": 0.0,
            "irf_anomaly_count": 0,
            "irf_channels": [
                {
                    "family": "single_tail",
                    "has_shoulder": False,
                    "has_satellite": False,
                    "has_prepulse": False,
                    "has_echo_train": False,
                }
                for _ in range(4)
            ],
        }
        for _ in range(4)
    ]
    rows = _irf_stratified_rows(truth, truth, active, layers, metadata)
    assert rows
    active_by_layer = {
        layer: {
            CANONICAL_PARAMETERS[index]
            for index in np.flatnonzero(schema.active_mask(layer))
        }
        for layer in (1, 2, 3)
    }
    assert all(row["parameter"] in active_by_layer[row["layer_count"]] for row in rows)


def test_partial_nonfinite_prediction_invalidates_aggregate_metric() -> None:
    schema = ParameterSchema()
    layers = np.asarray([1, 1])
    active = np.stack([schema.active_mask(1), schema.active_mask(1)])
    truth = np.zeros((2, 8), dtype=np.float64)
    truth[:, [2, 5]] = [0.01, 1.0]
    prediction = truth.copy()
    prediction[0, 2] = np.nan
    result = evaluate_point_predictions(
        prediction, truth, active, layers, bootstrap_seed=8, bootstrap_replicates=5
    )
    mua = next(row for row in result["parameter_metrics"] if row["parameter"] == "mua1_mm-1")
    assert np.isnan(mua["normalized_mae"])
    assert mua["numerical_failure_count"] == 1
    assert result["numerical_failure_cases"] == 1


def test_point_evaluation_includes_surface_depth_without_double_counting() -> None:
    schema = ParameterSchema()
    layers = np.asarray([1, 2, 3, 3])
    active = np.stack([schema.active_mask(int(layer)) for layer in layers])
    truth = np.zeros((4, 8), dtype=np.float32)
    truth[0, [2, 5]] = [0.01, 1.0]
    truth[1, [0, 2, 3, 5, 6]] = [5.0, 0.01, 0.02, 1.0, 1.2]
    truth[2] = [4.0, 5.0, 0.01, 0.02, 0.03, 1.0, 1.2, 1.4]
    truth[3] = [8.0, 7.0, 0.02, 0.03, 0.04, 1.2, 1.4, 1.6]
    prediction = truth.copy()
    prediction[2, 0] += 1.0
    prediction[2, 1] -= 1.0
    result = evaluate_point_predictions(
        prediction, truth, active, layers, bootstrap_seed=7, bootstrap_replicates=5
    )
    total = next(
        row
        for row in result["parameter_metrics"]
        if row["layer_count"] == 3 and row["parameter"] == "d1_plus_d2_mm"
    )
    assert total["mae"] == 0.0
    canonical_n3 = [
        row
        for row in result["parameter_metrics"]
        if row["layer_count"] == 3 and row["parameter"] != "d1_plus_d2_mm"
    ]
    assert np.isclose(result["layer_macro_nmae"]["N3"], np.mean([row["normalized_mae"] for row in canonical_n3]))


def test_cnn_and_ann_use_fixed_defaults():
    cnn = load_config("configs/research_cnn.yaml")
    ann = load_config("configs/research_ann.yaml")
    assert cnn["baseline"]["cnn_channels"] == [32, 64, 128, 256]
    assert cnn["training"]["learning_rate"] == 0.0001
    assert ann["baseline"]["ann_widths"] == [384, 256, 128]
    assert ann["training"]["learning_rate"] == 0.0003
    assert "tuning" not in cnn["baseline"] and "tuning" not in ann["baseline"]


def test_baseline_early_stopping_ignores_warmup_then_starts_fresh() -> None:
    history, best, stale, smoothed, improved = _update_early_stopping(
        0.20,
        [0.10],
        0.10,
        29,
        active=False,
        smoothing_window=5,
        min_delta=0.002,
    )
    assert history == []
    assert np.isinf(best)
    assert stale == 0
    assert smoothed == 0.20
    assert not improved

    history, best, stale, smoothed, improved = _update_early_stopping(
        0.20,
        history,
        best,
        stale,
        active=True,
        smoothing_window=5,
        min_delta=0.002,
    )
    assert history == [0.20]
    assert best == smoothed == 0.20
    assert stale == 0
    assert improved


def test_research_cnn_monitoring_waits_for_warmup() -> None:
    cnn = load_config("configs/research_cnn.yaml")
    early = cnn["training"]["baseline_early_stopping"]
    assert early["start_after_warmup"] is True
    assert early["patience_epochs"] == 50


def test_baseline_launcher_paths_follow_configuration(tmp_path) -> None:
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        "seeds:\n"
        "  data_split: 1\n"
        "  training: 2\n"
        "  validation: 3\n"
        "  evaluation: 4\n"
        "data:\n"
        "  time_bins: 2500\n"
        "  time_bin_ps: 4.0\n"
        "  time_start_ps: 0.0\n"
        "training:\n"
        "  output_dir: custom/run\n",
        encoding="utf-8",
    )
    paths = resolve_training_paths(str(config_path))
    assert set(paths) == {"run_directory"}
    assert paths["run_directory"] == str((tmp_path.parent / "custom/run").resolve())


def test_runtime_baseline_tuning_interface_is_removed() -> None:
    project_metadata = Path("pyproject.toml").read_text(encoding="utf-8")
    training_cli = Path("src/tdnirs_posterior/cli/train_baseline.py").read_text(
        encoding="utf-8"
    )
    assert "tdnirs-tune-baseline" not in project_metadata
    assert "--tuning-result" not in training_cli
    assert not Path("src/tdnirs_posterior/cli/tune_baseline.py").exists()


def test_baseline_association_reports_finite_and_total_case_counts() -> None:
    schema = ParameterSchema()
    layers = np.asarray([2, 2, 2])
    active = np.stack([schema.active_mask(2)] * 3)
    truth = np.zeros((3, 8), dtype=np.float64)
    truth[:, 0] = [2.0, 7.0, 12.0]
    truth[:, [2, 3, 5, 6]] = [0.01, 0.02, 1.0, 1.2]
    prediction = truth.copy()
    prediction[0, 2] = np.nan
    result = evaluate_point_predictions(
        prediction, truth, active, layers, bootstrap_seed=9, bootstrap_replicates=5
    )
    association = next(
        row
        for row in result["thickness_associations"]
        if row["parameter"] == "mua1_mm-1" and row["thickness_variable"] == "d1_mm"
    )
    assert association["count"] == 2
    assert association["total_count"] == 3
    html = _report_html(
        {
            "status": "completed_with_numerical_failures",
            "metrics": {
                "parameter_metrics": [],
                "thickness_metrics": [],
                "thickness_associations": [association],
                "layer_macro_nmae": {},
                "macro_nmae": float("nan"),
            },
            "model": {"architecture": "cnn", "parameter_count": 1},
            "runtime": {"milliseconds_per_case": 1.0},
            "native_output": {"finite_fraction": 2 / 3},
            "limitations": [],
        }
    )
    assert "2/3" in html
    assert "Finite/total cases" in html


def test_baseline_comparison_requires_and_accepts_strict_pairing(tmp_path) -> None:
    schema = ParameterSchema()
    layers = np.asarray([1, 2, 3], dtype=np.int64)
    active = np.stack([schema.active_mask(int(layer)) for layer in layers])
    truth = np.zeros((3, 8), dtype=np.float32)
    truth[0, [2, 5]] = [0.01, 1.0]
    truth[1, [0, 2, 3, 5, 6]] = [5.0, 0.01, 0.02, 1.0, 1.2]
    truth[2] = [4.0, 5.0, 0.01, 0.02, 0.03, 1.0, 1.2, 1.4]
    dtof = np.arange(3 * 4 * 2 * 10, dtype=np.float32).reshape(3, 4, 2, 10)
    sample_ids = np.asarray(["a", "b", "c"], dtype=object)
    fingerprint = {"records_sha256": "same", "source_files_sha256": {}}
    proposed = tmp_path / "proposed"
    proposed.mkdir()
    (proposed / "report.json").write_text(
        json.dumps(
                {
                    "status": "completed",
                    "scope": {"model_variant": "explicit_irf"},
                    "checkpoint": {"weights": "ema"},
                "reproducibility": {
                    "dataset_fingerprint": fingerprint,
                    "observation_seed": 17,
                },
            }
        ),
        encoding="utf-8",
    )
    with h5py.File(proposed / "posterior.h5", "w") as handle:
        cases = handle.create_group("cases")
        cases.create_dataset("truth", data=truth)
        cases.create_dataset("active_mask", data=active)
        cases.create_dataset("layer_count", data=layers)
        cases.create_dataset("dtof_input", data=dtof)
        cases.create_dataset("sample_id", data=sample_ids, dtype=h5py.string_dtype("utf-8"))
        scenarios = handle.create_group("scenarios")
        for index, layer in enumerate((1, 2, 3)):
            from tdnirs_posterior.evaluation_protocol import enumerate_prior_scenarios

            scenario = next(item for item in enumerate_prior_scenarios(layer) if item.known_count == 0)
            group = scenarios.create_group(scenario.scenario_id)
            group.create_dataset("global_case_index", data=np.asarray([index]))
            group.create_dataset("samples", data=np.repeat(truth[index][None, None, :], 4, axis=1))

    baseline_paths = {}
    for architecture in ("cnn", "ann"):
        directory = tmp_path / architecture
        directory.mkdir()
        baseline_paths[architecture] = directory
        (directory / "report.json").write_text(
            json.dumps(
                    {
                        "status": "completed",
                        "model": {"architecture": architecture},
                        "checkpoint": {"weights": "ema"},
                    "reproducibility": {
                        "dataset_fingerprint": fingerprint,
                        "observation_seed": 17,
                    },
                }
            ),
            encoding="utf-8",
        )
        with h5py.File(directory / "predictions.h5", "w") as handle:
            cases = handle.create_group("cases")
            cases.create_dataset("prediction", data=truth)
            cases.create_dataset("truth", data=truth)
            cases.create_dataset("active_mask", data=active)
            cases.create_dataset("layer_count", data=layers)
            cases.create_dataset("dtof_input", data=dtof)
            cases.create_dataset("sample_id", data=sample_ids, dtype=h5py.string_dtype("utf-8"))
    report = compare_deterministic_baselines(
        proposed, baseline_paths["cnn"], baseline_paths["ann"], tmp_path / "comparison"
    )
    assert report.is_file()
    payload = json.loads((report.parent / "comparison.json").read_text(encoding="utf-8"))
    assert set(payload["models"]) == {"Proposed", "CNN", "ANN"}
    assert payload["status"] == "completed"
