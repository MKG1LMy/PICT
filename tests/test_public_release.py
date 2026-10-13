"""Public launcher and advisory deployment contracts; no installation or training."""
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tdnirs_posterior import pipeline

ROOT = Path(__file__).resolve().parents[1]


def deployment():
    spec = importlib.util.spec_from_file_location("public_setup", ROOT / "deploy/setup_environment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("version,expected", [("13.1", False), ("13.0", False), ("12.8", True)])
@pytest.mark.parametrize("label", ["CUDA Version", "CUDA UMD Version"])
def test_driver_support_is_advisory(version, expected, label):
    runner = lambda *a, **kw: SimpleNamespace(returncode=0, stdout=f"{label}: {version}")
    assert bool(deployment().driver_warnings("130", runner)) is expected


def test_missing_driver_does_not_raise():
    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")
    assert deployment().driver_warnings("130", missing)


@pytest.mark.parametrize("install_fails", [False, True])
def test_install_status_is_separate_from_driver_warning(monkeypatch, capsys, install_fails):
    module = deployment()
    monkeypatch.setattr(module, "prepare_temp", lambda: None)
    monkeypatch.setattr(module, "environment_python", lambda *args: Path("target/python.exe"))
    monkeypatch.setattr(module, "find_conda", lambda: "conda")
    monkeypatch.setattr(module, "driver_warnings", lambda cuda: ["Driver update recommended."])
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if "list" in command:
            return SimpleNamespace(stdout=json.dumps({"envs": []}))
        if install_fails:
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(module, "run_command", run)
    assert module.main([]) == int(install_fails)
    output = capsys.readouterr()
    assert "Driver update recommended." in output.out
    if not install_fails:
        assert "Installation completed" in output.out
        assert "--override-channels" in calls[1]
        assert calls[2][:3] == [str(Path("target/python.exe")), "-m", "pip"]
        assert all("run" not in command for command in calls)


def test_pipeline_runs_all_models_and_seeds_with_one_entry(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "run_repetitions", lambda c, m, a: calls.append((c, m, a)))
    pipeline.run_pipeline(ROOT / "configs/pipeline.json", run_epochs=1, no_test=True)
    assert len(calls) == 12
    assert [c[0]["seeds"]["training"][0] for c in calls] == [20736]*4 + [78631]*4 + [26605]*4
    for offset in (0, 4, 8):
        assert [c[1].rsplit(".", 1)[-1] for c in calls[offset:offset+4]] == list(pipeline.MODULES.values())
    assert all("--no-test" in args and "--run-epochs" in args for _, _, args in calls)
    assert len({c["training"]["output_dir"] for c, _, _ in calls}) == 4


@pytest.mark.parametrize("models", [[], ["main", "main"], ["invalid"]])
def test_pipeline_rejects_invalid_model_selection(models):
    with pytest.raises(ValueError, match="unique models"):
        pipeline.run_pipeline(ROOT / "configs/pipeline.json", models=models)


def test_public_text_is_english_and_portable():
    for folder in (ROOT / "src", ROOT / "configs"):
        for path in folder.rglob("*"):
            if path.suffix in (".py", ".yaml", ".html", ".json"):
                text = path.read_text(encoding="utf-8")
                assert not any("\u4e00" <= char <= "\u9fff" for char in text), path
                assert not re.search(r"\b[A-Za-z]:[\\/]", text), path


@pytest.mark.skipif(os.environ.get("TDNIRS_TEST_SKLEARN") != "1", reason="Opt-in native GMM backend")
@pytest.mark.parametrize("second_center,components", [(None, 1), (1.7, 2)])
def test_gmm_retains_single_and_outside_support_modes(second_center, components):
    import numpy as np
    from tdnirs_posterior.evaluation_protocol import detect_candidate_modes
    rng = np.random.default_rng(42)
    samples = np.zeros((400, 8))
    samples[:, 2] = rng.normal(-0.65, 0.04, 400)
    if second_center is not None:
        samples[200:, 2] = rng.normal(second_center, 0.04, 200)
    before = samples.copy()
    unknown = np.zeros(8, dtype=bool)
    unknown[2] = True
    result = detect_candidate_modes(samples, unknown, seed=42, max_components=3,
                                    bootstrap_replicates=5, minimum_separation=2)
    assert result["component_count"] == components
    assert result["stable_multimodal"] == (components == 2)
    np.testing.assert_array_equal(samples, before)
    if second_center is not None:
        assert max(row[0] for row in result["centers_normalized"]) > 1
