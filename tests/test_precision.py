from __future__ import annotations

from unittest import mock

import pytest
import torch

from tdnirs_posterior.precision import (
    PRECISION_PROTOCOL,
    checked_optimizer_step,
    use_bf16_amp,
)


def test_bf16_policy_never_silently_falls_back_to_fp16() -> None:
    with mock.patch("torch.cuda.is_bf16_supported", return_value=True):
        assert use_bf16_amp({"amp": True}, torch.device("cuda"))
    with mock.patch("torch.cuda.is_bf16_supported", return_value=False):
        with pytest.raises(RuntimeError, match="Set amp: false"):
            use_bf16_amp({"amp": True}, torch.device("cuda"))
    assert not use_bf16_amp({"amp": False}, torch.device("cuda"))
    assert not use_bf16_amp({"amp": True}, torch.device("cpu"))
    assert "bfloat16" in PRECISION_PROTOCOL


def test_checked_step_advances_optimizer_only_with_finite_gradients() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    loss = model(torch.ones(1, 2)).square().sum()
    loss.backward()
    norm = checked_optimizer_step(
        model, optimizer, scaler, 1.0, epoch=1, batch_index=1
    )
    assert norm >= 0.0
    assert {int(state["step"]) for state in optimizer.state.values()} == {1}

    optimizer.zero_grad(set_to_none=True)
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    with pytest.raises(FloatingPointError, match="epoch 2, batch 3"):
        checked_optimizer_step(model, optimizer, scaler, 1.0, epoch=2, batch_index=3)
    assert {int(state["step"]) for state in optimizer.state.values()} == {1}


def test_checked_step_rejects_enabled_scaler_before_any_update() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = mock.Mock()
    scaler.is_enabled.return_value = True
    with pytest.raises(RuntimeError, match="must not use a gradient scaler"):
        checked_optimizer_step(model, optimizer, scaler, 1.0, epoch=1, batch_index=1)
    scaler.step.assert_not_called()
