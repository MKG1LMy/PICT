from copy import deepcopy

import pytest
import torch

from tdnirs_posterior.config import load_config
from tdnirs_posterior.lr_schedule import WarmupPlateau
from tdnirs_posterior.training import _epoch_progress_line, _resume_signature
from tdnirs_posterior.baseline_training import _signature


def setup_schedule(epochs=10000, warmup=2, **overrides):
    cfg = dict(epochs=epochs, warmup_epochs=warmup, lr_schedule=dict(
        type="plateau", factor=.5, patience_epochs=1, min_delta_relative=.002,
        cooldown_epochs=1, min_lr=.00005, early_stop_grace_epochs=2))
    cfg["lr_schedule"].update(overrides)
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=.0002)
    return opt, WarmupPlateau(opt, cfg, updates_per_epoch=2)


def test_warmup_and_plateau_independent_of_maximum_epochs():
    a, sa = setup_schedule(epochs=4000)
    b, sb = setup_schedule(epochs=10000)
    rates = []
    for step in range(4):
        assert a.param_groups[0]["lr"] == b.param_groups[0]["lr"]
        rates.append(a.param_groups[0]["lr"])
        assert not sa.can_stop
        # Changing validation scores during warmup cannot trigger reductions.
        sa.step_metric(10 + step)
        sb.step_metric(10 + step)
        sa.step()
        sb.step()
    assert rates == pytest.approx([.00005, .0001, .00015, .0002])
    assert sa.can_stop and sb.can_stop
    for _ in range(20):
        assert sa.step_metric(1) == sb.step_metric(1)
        sa.step()
        sb.step()
        assert a.param_groups[0]["lr"] == b.param_groups[0]["lr"]
    assert a.param_groups[0]["lr"] == .00005


def test_plateau_cooldown_grace_floor_and_improvement():
    opt, schedule = setup_schedule(warmup=0)
    assert not schedule.step_metric(1)
    assert not schedule.step_metric(.9)  # genuine improvement resets plateau
    assert not schedule.step_metric(.9)
    assert schedule.step_metric(.9)  # patience=1: second bad epoch
    assert opt.param_groups[0]["lr"] == .0001
    assert not schedule.can_stop
    assert not schedule.step_metric(.9)  # cooldown
    assert not schedule.can_stop
    assert not schedule.step_metric(.9)
    assert schedule.can_stop
    assert schedule.step_metric(.9)
    assert opt.param_groups[0]["lr"] == .00005
    for _ in range(15):
        assert not schedule.step_metric(.9)  # no false reductions at floor
    assert schedule.can_stop
    assert schedule.reductions == 2


@pytest.mark.parametrize("split", [1, 4, 9])
def test_scheduler_resume_exact_in_warmup_cooldown_and_plateau(split):
    opt, schedule = setup_schedule()
    for step in range(split):
        schedule.step()
        if step % 2 == 1:
            schedule.step_metric(1)
    opt_state, state = deepcopy(opt.state_dict()), deepcopy(schedule.state_dict())
    restored_opt, restored = setup_schedule()
    restored_opt.load_state_dict(opt_state)
    restored.load_state_dict(state)
    for step in range(split, 40):
        assert opt.param_groups[0]["lr"] == restored_opt.param_groups[0]["lr"]
        schedule.step()
        restored.step()
        if step % 2 == 1:
            assert schedule.step_metric(1) == restored.step_metric(1)
        assert schedule.state_dict() == restored.state_dict()
        assert schedule.diagnostics() == restored.diagnostics()


def test_nonfinite_metric_and_invalid_config_rejected():
    _, schedule = setup_schedule()
    with pytest.raises(ValueError, match="finite"):
        schedule.step_metric(float("nan"))
    for kwargs in (dict(factor=0), dict(min_lr=-1), dict(patience_epochs=-1)):
        with pytest.raises(ValueError):
            setup_schedule(**kwargs)


def test_all_resume_signatures_protect_scheduler_config():
    cfg = load_config("configs/research_cnn.yaml")
    cfg["seeds"]["training"] = 20734
    for signature in (_resume_signature, _signature):
        original = deepcopy(signature(cfg))
        for key, value in (("warmup_epochs", 200), ("lr_schedule", {"type": "different"})):
            modified = deepcopy(cfg)
            modified["training"][key] = value
            assert signature(modified) != original


def test_console_explains_warmup_and_lr_observation_even_when_stale_full():
    record = dict(epoch=1, train_loss=1., validation_loss=1., validation_loss_smoothed=1.,
                  learning_rate=.0001, epoch_seconds=1., stale_epochs=50,
                  posterior_validation_enabled=False)
    record["lr_schedule"] = {"phase": "warmup"}
    assert "early_stop=warmup" in _epoch_progress_line(record, 10000, 50)
    record["lr_schedule"] = {"phase": "plateau", "early_stop_grace_remaining": 20}
    assert "early_stop=lr_observe:20" in _epoch_progress_line(record, 10000, 50)


@pytest.mark.parametrize("architecture", ["explicit_irf", "dtof_only", "cnn", "ann"])
def test_real_updates_resume_after_lr_reduction(tmp_path, architecture):
    from unittest.mock import patch
    from tdnirs_posterior.baseline_training import train_baseline
    from tdnirs_posterior.training import train

    cfg = load_config("configs/smoke.yaml")
    is_baseline = architecture in {"cnn", "ann"}
    cfg["model"] = dict(variant=architecture if not is_baseline else "explicit_irf",
                        d_model=8, n_heads=2, waveform_latents=2, waveform_layers=1,
                        signal_layers=1, posterior_layers=1, frequency_bins=4, dropout=.1)
    cfg["baseline"] = dict(architecture=architecture, cnn_channels=[4], cnn_pool_bins=2,
                           cnn_head_widths=[4], ann_widths=[4], dropout=.1, loss_beta=.1)
    cfg["training"].update(epochs=8, micro_batch_size=2, effective_batch_size=2,
                           num_workers=0, amp=False, max_train_samples=3, max_val_samples=3,
                           baseline_early_stopping={"patience_epochs":1})
    cfg["training"]["lr_schedule"].update(patience_epochs=0, cooldown_epochs=0,
                                          min_lr=.0001, early_stop_grace_epochs=2)
    cfg["training"]["early_stopping"]["patience_epochs"] = 1
    cfg["evaluation"]["auto_run"] = False
    single = dict(macro_nmae=.2, layer_nmae={"N1":.1,"N2":.2,"N3":.3},
                  core_layer_nmae={"N1":.1,"N2":.2,"N3":.3},
                  finite_fraction=1., numerical_failure_cases=0)
    validation = {**single, "by_channel_count": {str(k):single for k in range(1,5)}}
    module = "tdnirs_posterior.baseline_training" if is_baseline else "tdnirs_posterior.training"

    def run(resume=None, run_epochs=None):
        if is_baseline:
            return train_baseline(cfg, resume=resume, run_epochs=run_epochs, auto_test=False)
        cfg["training"].pop("run_epochs", None)
        if run_epochs is not None:
            cfg["training"]["run_epochs"] = run_epochs
        return train(cfg, resume=resume)

    outputs = {}
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with patch("torch.cuda.is_available", return_value=False), patch(
            module + ("._validate" if is_baseline else "._run_validation"),
            return_value=validation if is_baseline else {"flow":1.}
        ), patch(module + "._dataset_fingerprint", return_value={"test":True}):
            for label in ("continuous", "resumed"):
                cfg["training"]["output_dir"] = str(tmp_path / label)
                if label == "continuous":
                    run()
                else:
                    run(run_epochs=2)
                    path = tmp_path / label / "last.pt"
                    partial = torch.load(path, weights_only=False)
                    assert partial["scheduler"]["reductions"] == 1
                    assert partial["stale_epochs"] >= 1  # resume must respect remaining grace
                    run(resume=path)
                outputs[label] = torch.load(tmp_path / label / "last.pt", weights_only=False)
        for key in ("model", "ema"):
            for name, value in outputs["continuous"][key].items():
                torch.testing.assert_close(value, outputs["resumed"][key][name], rtol=0, atol=0)
        assert outputs["continuous"]["scheduler"] == outputs["resumed"]["scheduler"]
        assert outputs["resumed"]["scheduler"]["reductions"] == 1
        assert outputs["resumed"]["next_epoch"] == 4  # not stopped immediately at epoch 2
        assert outputs["continuous"]["metrics_record"]["learning_rate"] == outputs["resumed"]["metrics_record"]["learning_rate"]
    finally:
        torch.set_num_threads(old_threads)
