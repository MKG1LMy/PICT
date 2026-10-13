import json
from pathlib import Path

import h5py
import numpy as np

from tdnirs_posterior.ablation_comparison import compare_irf_ablation


def _write_test(directory: Path, variant: str, offset: float = 0.0) -> None:
    directory.mkdir()
    report = {
        "status": "completed",
        "scope": {"model_variant": variant},
        "reproducibility": {"dataset_fingerprint": {"sha": "same"}},
        "checkpoint": {
            "path": f"{variant}.pt",
            "sha256": variant,
            "weights": "ema",
        },
    }
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with h5py.File(directory / "posterior.h5", "w") as store:
        store.attrs["evaluation_metadata"] = json.dumps(
            {
                "observation_seed": 1,
                "posterior_seed": 2,
                "posterior_samples": 4,
                "solver_steps": 3,
                "sample_ids": ["a", "b"],
            }
        )
        cases = store.create_group("cases")
        truth = np.zeros((2, 8), dtype=np.float32)
        truth[:, 2] = [0.01, 0.02]
        truth[:, 5] = [1.0, 1.2]
        active = np.zeros((2, 8), dtype=bool)
        active[:, [2, 5]] = True
        cases.create_dataset("truth", data=truth)
        cases.create_dataset("active_mask", data=active)
        cases.create_dataset("sample_id", data=np.asarray([b"a", b"b"]))
        cases.create_dataset(
            "dtof_input", data=np.ones((2, 4, 2, 8), dtype=np.float32)
        )
        scenarios = store.create_group("scenarios")
        scenario = scenarios.create_group("N1_k0_00")
        samples = np.broadcast_to(truth[:, None, :], (2, 4, 8)).copy()
        samples[:, :, 2] += offset
        scenario.create_dataset("samples", data=samples)
        scenario.create_dataset("diagnostics", data=np.zeros((2, 5)))
        scenario.create_dataset("global_case_index", data=np.arange(2))


def test_strict_irf_ablation_comparison_outputs_paired_report(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    dtof = tmp_path / "dtof"
    output = tmp_path / "comparison"
    _write_test(explicit, "explicit_irf", 0.0)
    _write_test(dtof, "dtof_only", 0.001)
    report = compare_irf_ablation(explicit, dtof, output)
    assert report.is_file()
    assert (output / "irf_ablation_metrics.csv").is_file()
    assert (output / "irf_ablation_casewise.csv").is_file()
    payload = json.loads(
        (output / "irf_ablation_comparison.json").read_text(encoding="utf-8")
    )
    assert payload["comparison"] == "explicit_irf_vs_dtof_only"
    assert payload["scenario_count"] == 1
    assert payload["status"] == "completed"


def test_irf_ablation_propagates_numerical_failure_status(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    dtof = tmp_path / "dtof"
    output = tmp_path / "comparison"
    _write_test(explicit, "explicit_irf")
    _write_test(dtof, "dtof_only")
    with h5py.File(dtof / "posterior.h5", "r+") as store:
        store["scenarios"]["N1_k0_00"]["diagnostics"][0, 4] = 0.25
    report = compare_irf_ablation(explicit, dtof, output)
    payload = json.loads(
        (output / "irf_ablation_comparison.json").read_text(encoding="utf-8")
    )
    assert report.is_file()
    assert payload["status"] == "completed_with_numerical_failures"
    assert payload["excluded_numerical_pairs"][0]["sample_ids"] == ["a"]
