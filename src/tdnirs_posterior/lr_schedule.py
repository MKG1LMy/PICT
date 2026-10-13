"""Update-based warmup and epoch-based plateau control, independent of run length."""
from __future__ import annotations

import math
from typing import Any

import torch


class WarmupPlateau:
    def __init__(self, optimizer: torch.optim.Optimizer, training: dict[str, Any], updates_per_epoch: int):
        if "warmup_fraction" in training:
            raise ValueError("Use training.warmup_epochs instead of warmup_fraction")
        cfg = training["lr_schedule"]
        if cfg["type"] != "plateau":
            raise ValueError("training.lr_schedule.type must be plateau")
        warmup_epochs = training["warmup_epochs"]
        if isinstance(warmup_epochs, bool) or int(warmup_epochs) != warmup_epochs or warmup_epochs < 0:
            raise ValueError("warmup_epochs must be a non-negative integer")
        for key in ("patience_epochs", "cooldown_epochs", "early_stop_grace_epochs"):
            if isinstance(cfg[key], bool) or int(cfg[key]) != cfg[key] or cfg[key] < 0:
                raise ValueError(f"lr_schedule.{key} must be a non-negative integer")
        minimum = float(cfg["min_lr"])
        threshold = float(cfg["min_delta_relative"])
        factor = float(cfg["factor"])
        if not math.isfinite(factor) or not 0 < factor < 1:
            raise ValueError("lr_schedule.factor must be in (0,1)")
        if not math.isfinite(minimum) or minimum <= 0 or any(minimum > group["lr"] for group in optimizer.param_groups):
            raise ValueError("min_lr must be positive and no greater than the peak learning rate")
        if not math.isfinite(threshold) or not 0 <= threshold < 1:
            raise ValueError("min_delta_relative must be in [0,1)")
        self.optimizer = optimizer
        self.peak_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.warmup_steps = int(warmup_epochs) * max(1, updates_per_epoch)
        self.completed_updates = 0
        self.grace_epochs = int(cfg["early_stop_grace_epochs"])
        self.epochs_since_reduction = self.grace_epochs
        self.reductions = 0
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=float(cfg["factor"]),
            patience=int(cfg["patience_epochs"]), threshold=threshold,
            threshold_mode="rel", cooldown=int(cfg["cooldown_epochs"]),
            min_lr=minimum,
        )
        self._set_warmup_lr()

    @property
    def warmup_complete(self) -> bool:
        return self.completed_updates >= self.warmup_steps

    @property
    def can_stop(self) -> bool:
        return self.warmup_complete and self.epochs_since_reduction >= self.grace_epochs

    def _set_warmup_lr(self) -> None:
        scale = min(1.0, (self.completed_updates + 1) / max(self.warmup_steps, 1))
        for group, peak in zip(self.optimizer.param_groups, self.peak_lrs, strict=True):
            group["lr"] = peak * scale

    def step(self) -> None:
        """Call only after a successful optimizer update."""
        self.completed_updates += 1
        if self.completed_updates <= self.warmup_steps:
            self._set_warmup_lr()

    def step_metric(self, metric: float) -> bool:
        if not math.isfinite(metric):
            raise ValueError("Learning-rate validation metric must be finite")
        if not self.warmup_complete:
            return False
        before = [float(group["lr"]) for group in self.optimizer.param_groups]
        self.plateau.step(metric)
        reduced = any(group["lr"] < old for group, old in zip(self.optimizer.param_groups, before, strict=True))
        if reduced:
            self.reductions += 1
            self.epochs_since_reduction = 0
        else:
            self.epochs_since_reduction += 1
        return reduced

    def diagnostics(self) -> dict[str, Any]:
        return {
            "phase": "plateau" if self.warmup_complete else "warmup",
            "completed_updates": self.completed_updates,
            "warmup_steps": self.warmup_steps,
            "reductions": self.reductions,
            "bad_epochs": self.plateau.num_bad_epochs,
            "cooldown_remaining": self.plateau.cooldown_counter,
            "early_stop_grace_remaining": max(0, self.grace_epochs - self.epochs_since_reduction),
            "early_stop_allowed": self.can_stop,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "completed_updates": self.completed_updates,
            "epochs_since_reduction": self.epochs_since_reduction,
            "reductions": self.reductions,
            "plateau": self.plateau.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        # The optimizer restores the actual learning rate; configuration is checked
        # by the caller's resume signature before restoring either object.
        self.completed_updates = int(state["completed_updates"])
        self.epochs_since_reduction = int(state["epochs_since_reduction"])
        self.reductions = int(state["reductions"])
        self.plateau.load_state_dict(state["plateau"])
