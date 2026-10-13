from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tdnirs_posterior.channels import COUNT_WEIGHTS, ChannelValidationLoader, random_mask, subsets
from tdnirs_posterior.config import load_config
from tdnirs_posterior.model import BaselineConfig, build_baseline
from tdnirs_posterior.model.waveform import SignalEncoder, DTOFOnlySignalEncoder
from tdnirs_posterior.evaluation_protocol import thickness_group_labels
from tdnirs_posterior.repeats import run_repetitions
from tdnirs_posterior.training import _run_posterior_validation


def test_mask_distribution_reproducible_and_balanced():
    def generate():
        rng = np.random.default_rng(17)
        return np.stack([random_mask(rng) for _ in range(20000)])
    a, b = generate(), generate()
    assert np.array_equal(a, b)
    assert len(subsets()) == 15
    for k, weight in COUNT_WEIGHTS.items():
        assert abs(np.mean(a.sum(1) == k) - weight) < .015
    for combination in subsets():
        assert np.any(np.all(a == np.isin(np.arange(4), combination), axis=1))


@pytest.mark.parametrize("variant", ["main", "dtof_only", "cnn", "ann"])
def test_missing_slots_cannot_affect_model_or_gradients(variant):
    torch.manual_seed(4)
    x = torch.rand(2, 4, 2, 32, requires_grad=True)
    h = torch.rand_like(x)
    n = torch.tensor([1, 3])
    sds = torch.tensor([10., 20., 30., 40.])
    mask = torch.tensor([[True, False, False, False], [False, True, True, True]])
    if variant in {"main", "dtof_only"}:
        cls = SignalEncoder if variant == "main" else DTOFOnlySignalEncoder
        model = cls(16, 2, 2, 1, 1, 4, 0).eval()
        forward = lambda a, b, m: model(a, b, sds, m) if variant == "main" else model(a, sds, m)
    else:
        model = build_baseline(BaselineConfig(architecture=variant, cnn_channels=(8,), cnn_pool_bins=2,
                               cnn_head_widths=(8,), ann_widths=(8,), dropout=0), 32).eval()
        forward = lambda a, b, m: model(a, n, m)
    y = forward(x, h, mask)
    dirty_x, dirty_h = x.detach().clone(), h.clone()
    dirty_x[~mask], dirty_h[~mask] = float("nan"), float("nan")
    assert torch.allclose(y, forward(dirty_x, dirty_h, mask), atol=1e-6)
    y.square().sum().backward()
    assert torch.all(x.grad[~mask] == 0)
    assert torch.isfinite(y).all()
    with pytest.raises(ValueError):
        forward(x, h, torch.zeros_like(mask))


def test_fixed_validation_subsets_share_views_and_cover_combinations():
    raw = {"layer_count": torch.tensor([1]*12+[2]*12+[3]*12),
           "view_a": {"dtof": torch.rand(36,4,2,8)}, "view_b": {"dtof": torch.rand(36,4,2,8)}}
    for k in range(1,5):
        a = next(iter(ChannelValidationLoader([raw], k)))
        b = next(iter(ChannelValidationLoader([raw], k)))
        assert torch.equal(a["view_a"]["channel_mask"], b["view_a"]["channel_mask"])
        assert torch.equal(a["view_a"]["channel_mask"], a["view_b"]["channel_mask"])
        assert a["view_a"]["dtof"] is raw["view_a"]["dtof"]
        assert len(torch.unique(a["view_a"]["channel_mask"][:12], dim=0)) == len(subsets(k))


def test_compact_inference_preserves_sds_identity_and_pair_order():
    from tdnirs_posterior.model import ICAPModel, ModelConfig
    from tdnirs_posterior.inference import infer_raw_counts
    cfg = load_config("configs/smoke.yaml")
    cfg["inference"]["solver_steps"] = 1
    model = ICAPModel(ModelConfig(d_model=16,n_heads=2,waveform_latents=2,waveform_layers=1,
                       signal_layers=1,posterior_layers=1,frequency_bins=4,dropout=0)).eval()
    t = np.arange(2500)
    h = np.stack([np.exp(-((t-500)/25)**2), np.exp(-((t-600)/30)**2)])*10000
    y = np.stack([np.exp(-((t-700)/70)**2), np.exp(-((t-850)/90)**2)])*10000
    a, meta = infer_raw_counts(model,cfg,y,h,4.,1,{},sample_count=2,seed=31,sds_mm=[20,40])
    b, _ = infer_raw_counts(model,cfg,y[::-1].copy(),h[::-1].copy(),4.,1,{},sample_count=2,seed=31,sds_mm=[40,20])
    np.testing.assert_array_equal(a,b)
    assert meta["channel_mask"] == [False,True,False,True]
    with pytest.raises(ValueError, match="unique subset"):
        infer_raw_counts(model,cfg,y,h,4.,1,{},sample_count=2,sds_mm=[20,20])


def test_extended_thickness_bins_include_boundaries_once():
    truth = np.zeros((8,8))
    truth[:,0] = [1,10,15,20,25,25.001,30,35]
    labels = thickness_group_labels(truth, np.full(8,3))
    assert "[20,25]" in labels[4]
    assert "(25,30]" in labels[5] and "(25,30]" in labels[6]
    assert "(30,35]" in labels[7]


def test_posterior_weighting_and_protected_channel_counts():
    def result(k):
        return {"score":float(k), "eligible":k>=3, "finite_fraction":1., "mean_coverage_error_90":.01,
            "prior_guardrail":{"passed":k>=3,"relative_changes_from_no_prior":{},"maximum_allowed_relative":.05},
            "group_scores":{"N1":float(k)},
            "score_components":{"no_prior_macro_normalized_crps":float(k), "prior_macro_normalized_crps":float(k), "weights":{"no_prior":.7,"prior":.3}}}
    # Preserve the real callable signature used by the wrapper's binding.
    from tdnirs_posterior.training import _run_posterior_validation_single
    import inspect
    def fake(*args, **kwargs):
        return result(kwargs["loader"].count)
    fake.__signature__ = inspect.signature(_run_posterior_validation_single)
    with patch("tdnirs_posterior.training._run_posterior_validation_single", fake):
        report = _run_posterior_validation(None, [], torch.device("cpu"), torch.tensor([10.,20.,30.,40.]),
                  False, 7, 2, 1, .15)
    assert report["eligible"]  # C1/C2's eligibility alone is diagnostic.
    assert report["score"] == pytest.approx(.15+2*.15+3*.3+4*.4)
    assert set(report["group_scores"]) == {"C3/N1","C4/N1"}


def test_repeat_queue_is_serial_resumable_and_keeps_fixed_seeds(tmp_path):
    cfg = load_config("configs/research.yaml")
    cfg["seeds"]["training"] = [1601, 915]
    cfg["training"]["output_dir"] = str(tmp_path / "runs")
    fixed = {k:v for k,v in cfg["seeds"].items() if k != "training"}
    calls = []
    def fake_process(command, check):
        calls.append(command)
        target = Path(command[command.index("--output-dir")+1])
        snapshot = json.loads((target/"resolved_config.json").read_text())
        assert {k:v for k,v in snapshot["seeds"].items() if k != "training"} == fixed
        (target/"training_summary.json").write_text(json.dumps({"status":"completed","best_checkpoint":str(target/"best.pt")}))
        (target/"test").mkdir()
        (target/"test/report.json").write_text(json.dumps({"status":"completed","metrics":{"macro_nmae":.1}}))
        (target/"test/report.html").write_text("<html>test</html>")
    with patch("tdnirs_posterior.repeats.subprocess.run", fake_process):
        run_repetitions(cfg, "tdnirs_posterior.cli.train", ["--config",cfg["_config_path"]])
        run_repetitions(cfg, "tdnirs_posterior.cli.train", ["--config",cfg["_config_path"]])
    assert len(calls) == 2
    assert [int(c[c.index("--seed")+1]) for c in calls] == [1601,915]
    summary = json.loads((tmp_path/"runs/repeat_summary.json").read_text())
    assert len(summary["runs"]) == 2
    cfg["simulation"]["irf"]["fwhm_ps"] = [50,1400]
    with pytest.raises(ValueError, match="different configuration"):
        run_repetitions(cfg, "tdnirs_posterior.cli.train", [])
