import copy
import itertools
import math
import json
from unittest.mock import patch

import pytest
import torch

from tdnirs_posterior.model import ICAPModel, ModelConfig
from tdnirs_posterior.model.stable_attention import StableCrossAttention
from tdnirs_posterior.precision import checked_optimizer_step, TrainingNumerics


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("scale", [1., 1e4, 1e8])
def test_bounded_attention_matches_double_reference_at_large_input_scales(scale):
    torch.manual_seed(18)
    model = StableCrossAttention(32, 4, dropout=0.)
    model.diagnostics_enabled = True
    reference = copy.deepcopy(model).double()
    x = [torch.randn(2, 7, 32) * scale for _ in range(3)]
    a = [v.clone().requires_grad_() for v in x]
    b = [v.double().requires_grad_() for v in x]
    y, _ = model(*a)
    z, _ = reference(*b)
    y.square().mean().backward()
    z.square().mean().backward()
    torch.testing.assert_close(y.double(), z, atol=2e-6, rtol=2e-5)
    for p, q in zip(a, b):
        assert torch.isfinite(p.grad).all()
        torch.testing.assert_close(p.grad.double() * scale, q.grad * scale, atol=1e-7, rtol=2e-3)
    assert model.diagnostics["logit_abs_max"] <= math.sqrt(8) + 1e-6


def test_cross_attention_gradcheck_and_zero_projection():
    model = StableCrossAttention(8, 2, dropout=0.).double()
    inputs = tuple(torch.randn(1, 2, 8, dtype=torch.float64, requires_grad=True) for _ in range(3))
    assert torch.autograd.gradcheck(lambda *x: model(*x)[0], inputs)
    with torch.no_grad():
        model.in_proj_weight.zero_()
        model.in_proj_bias.zero_()
    model(*inputs)[0].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("variant", ["explicit_irf", "dtof_only"])
def test_all_15_channel_subsets_have_finite_gradients_and_ignore_invalid_slots(variant):
    torch.manual_seed(2)
    model = ICAPModel(ModelConfig(variant=variant, d_model=16, n_heads=4, waveform_latents=4,
                                 waveform_layers=1, signal_layers=1, posterior_layers=1,
                                 frequency_bins=4, dropout=0.))
    masks = []
    for count in range(1, 5):
        for subset in itertools.combinations(range(4), count):
            masks.append([i in subset for i in range(4)])
    mask = torch.tensor(masks)
    dtof, irf = torch.rand(15, 4, 2, 32), torch.rand(15, 4, 2, 32)
    sds = torch.tensor([10., 20., 30., 40.])
    def encode(d, h):
        if variant == "explicit_irf": return model.signal_encoder(d, h, sds, mask)
        return model.signal_encoder(d, sds, mask)
    out = encode(dtof, irf)
    dirty_d, dirty_h = dtof.clone(), irf.clone()
    dirty_d[~mask], dirty_h[~mask] = float("nan"), float("nan")
    torch.testing.assert_close(out, encode(dirty_d, dirty_h))
    (out * torch.linspace(-1, 1, out.shape[-1])).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    state = model.state_dict()
    # Strict loading requires the normalized-output protocol marker.
    old = {k: v for k, v in state.items() if not k.endswith("normalized_output_protocol")}
    with pytest.raises(RuntimeError, match="normalized_output_protocol"):
        model.load_state_dict(old)


def test_large_finite_gradient_is_measured_without_overflow_but_not_applied():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model.weight.grad = torch.full_like(model.weight, 1e25)
    before = model.weight.detach().clone()
    details = {}
    with pytest.raises(FloatingPointError, match="Explosive gradient norm"):
        checked_optimizer_step(model, optimizer, torch.amp.GradScaler("cpu", enabled=False), 1.,
                               epoch=5, batch_index=7, diagnostics=details)
    assert math.isfinite(details["pre_clip_norm"])
    assert details["nonfinite_parameters"] == []
    assert details["top_parameters"][0]["name"] == "weight"
    torch.testing.assert_close(before, model.weight)
    assert not optimizer.state


def test_numerical_monitor_excludes_validation_and_has_failure_context():
    model = StableCrossAttention(8, 2)
    monitor = TrainingNumerics(model)
    x = torch.randn(2, 3, 8)
    model(x, x, x)
    monitor.step_details.update(pre_clip_norm=3., top_parameters=[])
    monitor.observe(3., 2)
    record = monitor.finish()
    assert record["gradient_peak"]["batch"] == 2
    assert record["attention"][""]["logit_abs_max"] <= 2.000001
    assert not model.diagnostics_enabled
    failure = monitor.failure({"sample_id": ["a"], "view_a": {"channel_mask": torch.ones(1, 4, dtype=torch.bool)}})
    assert failure["sample_ids"] == ["a"]
    assert failure["view_a_channel_mask"] == [[True] * 4]
    monitor.begin_batch({"sample_id": ["first"]}, 0, 2)
    monitor.begin_batch({"sample_id": ["second"]}, 1, 2)
    assert [v["sample_ids"] for v in monitor.failure(None)["accumulation_batches"]] == [["first"], ["second"]]
    monitor.begin_batch({"sample_id": ["third"]}, 2, 2)
    assert len(monitor.failure(None)["accumulation_batches"]) == 1


def test_finite_gradient_clipping_matches_reference():
    torch.manual_seed(4)
    model = torch.nn.Linear(5, 3)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
    for p, q in zip(model.parameters(), reference.parameters()):
        p.grad = torch.randn_like(p) * 100
        q.grad = p.grad.clone()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
    reference_optimizer.step()
    checked_optimizer_step(model, optimizer, torch.amp.GradScaler("cpu", enabled=False),
                           1., epoch=1, batch_index=1)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cross_attention_stays_fp32_inside_bf16_autocast():
    module = StableCrossAttention(32, 4).cuda()
    module.diagnostics_enabled = True
    x = (torch.randn(2, 8, 32, device="cuda") * 1e5).requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output, _ = module(x, x, x)
        output.square().mean().backward()
    assert output.dtype == torch.float32
    assert module.diagnostics["logit_abs_max"].dtype == torch.float32
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("baseline", [False, True])
def test_training_failure_file_preserves_accumulation_context(tmp_path, baseline):
    from tdnirs_posterior.config import load_config
    from tdnirs_posterior.training import train
    from tdnirs_posterior.baseline_training import train_baseline
    cfg = load_config("configs/smoke.yaml")
    cfg["model"] = dict(d_model=8, n_heads=2, waveform_latents=2, waveform_layers=1,
                        signal_layers=1, posterior_layers=1, frequency_bins=4, dropout=0.)
    cfg["baseline"] = dict(architecture="cnn", cnn_channels=[4], cnn_pool_bins=2,
                           cnn_head_widths=[4], dropout=0., loss_beta=.1)
    cfg["training"].update(epochs=1, max_train_samples=2, max_val_samples=2,
                           micro_batch_size=1, effective_batch_size=2, num_workers=0,
                           amp=False, output_dir=str(tmp_path))
    cfg["training"]["posterior_validation"]["enabled"] = False
    cfg["evaluation"]["auto_run"] = False
    module = "tdnirs_posterior.baseline_training" if baseline else "tdnirs_posterior.training"
    def inject(model, *args, **kwargs):
        parameter = next(p for p in model.parameters() if p.grad is not None)
        parameter.grad.fill_(float("inf"))
        return checked_optimizer_step(model, *args, **kwargs)
    with patch("torch.cuda.is_available", return_value=False), patch(module + ".checked_optimizer_step", side_effect=inject), patch(module + "._dataset_fingerprint", return_value={"test": True}):
        with pytest.raises(FloatingPointError, match="Non-finite gradient norm"):
            if baseline: train_baseline(cfg, auto_test=False)
            else: train(cfg)
    report = json.loads((tmp_path / "training_failure.json").read_text())
    assert report["epoch"] == 1 and report["batch"] == 2
    assert report["training_seed"] == cfg["seeds"]["training"]
    assert len(report["accumulation_batches"]) == 2
    assert all(len(x["sample_ids"]) == 1 for x in report["accumulation_batches"])
    assert report["last_gradient_check"]["nonfinite_parameters"]
    assert not (tmp_path / "last.pt").exists()
