"""Numerical precision policy shared by training and evaluation.

FP16's narrow exponent range caused repeated gradient overflows in the
four-channel model.  BF16 has the FP32 exponent range and needs no dynamic
gradient scaler; on GPUs without BF16 support, disable AMP explicitly.
"""

from __future__ import annotations

import torch
import math


PRECISION_PROTOCOL = "cuda-bfloat16-bounded-cross-attention-stable-gradient-v2"
AUTOCAST_DTYPE = torch.bfloat16


def use_bf16_amp(settings: dict, device: torch.device) -> bool:
    if not bool(settings.get("amp", True)) or device.type != "cuda":
        return False
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "CUDA AMP now requires BF16 support. Set amp: false to use FP32 "
            "on this GPU; FP16 is disabled because it can overflow during training."
        )
    return True


def checked_optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    max_grad_norm: float,
    *,
    epoch: int,
    batch_index: int,
    diagnostics: dict | None = None,
) -> float:
    """Step exactly once, or fail before corrupting weights/EMA/scheduler state."""
    if scaler.is_enabled():
        raise RuntimeError("BF16/FP32 training must not use a gradient scaler")
    scaler.unscale_(optimizer)
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    parameters = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
    if not parameters:
        raise FloatingPointError("No gradients: optimizer step was not applied")
    # FP64 reductions distinguish genuinely nonfinite elements from FP32
    # sum-of-squares overflow. Do not mutate gradients before this check.
    norms = torch.stack([torch.linalg.vector_norm(p.grad, dtype=torch.float64) for _, p in parameters])
    norm_value = float(torch.linalg.vector_norm(norms).detach().cpu())
    values = norms.detach().cpu().tolist()
    ranked = sorted(zip((n for n, _ in parameters), values), key=lambda x: x[1] if math.isfinite(x[1]) else math.inf, reverse=True)
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({"pre_clip_norm": norm_value if math.isfinite(norm_value) else str(norm_value),
                            "clip_factor": min(1., max_grad_norm / (norm_value + 1e-6)) if math.isfinite(norm_value) else None,
                            "top_parameters": [{"name": n, "norm": v if math.isfinite(v) else str(v)} for n, v in ranked[:5]],
                            "nonfinite_parameters": [n for n, v in ranked if not math.isfinite(v)]})
    if not math.isfinite(norm_value):
        raise FloatingPointError(
            f"Non-finite gradient norm at epoch {epoch}, batch {batch_index}; "
            f"optimizer step was not applied (norm={norm_value}, parameters={[n for n, v in ranked if not math.isfinite(v)][:5]})."
        )
    # Emergency numerical tripwire, not a scientific-quality threshold. A
    # finite but catastrophic gradient must not be silently clipped/accepted.
    if norm_value > max_grad_norm * 1e6:
        raise FloatingPointError(
            f"Explosive gradient norm at epoch {epoch}, batch {batch_index}; "
            f"optimizer step was not applied (norm={norm_value:.6g}, largest={ranked[0][0]})."
        )
    with torch.no_grad():
        factor = min(1.0, max_grad_norm / (norm_value + 1e-6))
        torch._foreach_mul_([p.grad for _, p in parameters], factor)
    scaler.step(optimizer)
    scaler.update()
    return norm_value


class TrainingNumerics:
    """Small epoch summaries; attention tensors are detached and never serialized."""
    def __init__(self, model):
        from .model.stable_attention import StableCrossAttention
        self.attention = {n: m for n, m in model.named_modules() if isinstance(m, StableCrossAttention)}
        self.start_epoch()

    def start_epoch(self):
        self.step_details = {}
        self.norms = []
        self.peak = {}
        self.accumulation_batches = []
        for module in self.attention.values():
            module.diagnostics.clear()
            module.diagnostics_enabled = True

    def begin_batch(self, raw, batch_index, accumulation):
        if batch_index % accumulation == 0:
            self.accumulation_batches = []
            self.step_details.clear()
        context = {"batch": batch_index + 1, "sample_ids": list(raw.get("sample_id", []))}
        for view in ("view_a", "view_b"):
            mask = raw.get(view, {}).get("channel_mask")
            if mask is not None:
                context[view + "_channel_mask"] = mask.cpu().tolist()
        self.accumulation_batches.append(context)

    def observe(self, norm, batch_index):
        self.norms.append(norm)
        if not self.peak or norm > self.peak["pre_clip_norm"]:
            self.peak = {**self.step_details, "batch": batch_index}

    def finish(self):
        attention = {}
        for name, module in self.attention.items():
            module.diagnostics_enabled = False
            attention[name] = {k: float(v.cpu()) for k, v in module.diagnostics.items()}
        return {"optimizer_updates": len(self.norms),
                "gradient_norm_mean": sum(self.norms) / len(self.norms) if self.norms else None,
                "gradient_peak": self.peak, "attention": attention}

    def failure(self, raw):
        result = {"last_gradient_check": self.step_details, "epoch_so_far": self.finish(),
                  "accumulation_batches": self.accumulation_batches}
        if raw is not None:
            result["sample_ids"] = list(raw.get("sample_id", []))
            for view in ("view_a", "view_b"):
                mask = raw.get(view, {}).get("channel_mask")
                if mask is not None:
                    result[view + "_channel_mask"] = mask.cpu().tolist()
        return result
