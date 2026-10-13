from __future__ import annotations

import argparse
import gc
import json
import time

import torch

from ..config import load_config
from ..model import ICAPModel, ModelConfig


def _profile(config: dict, length: int, batch_size: int, device: torch.device) -> dict:
    model = ICAPModel(ModelConfig.from_mapping(config["model"])).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    dtof_a = torch.rand(batch_size, 4, 2, length, device=device)
    irf_a = torch.rand(batch_size, 4, 2, length, device=device)
    dtof_b = torch.rand(batch_size, 4, 2, length, device=device)
    irf_b = torch.rand(batch_size, 4, 2, length, device=device)
    theta = torch.zeros(batch_size, 8, device=device)
    active = torch.ones(batch_size, 8, dtype=torch.bool, device=device)
    known = torch.zeros_like(active)
    known_values = torch.zeros_like(theta)
    layers = torch.full((batch_size,), 3, dtype=torch.long, device=device)
    sds = torch.tensor([10.0, 20.0, 30.0, 40.0], device=device)
    batch = {
        "view_a": {"dtof": dtof_a, "irf": irf_a},
        "view_b": {"dtof": dtof_b, "irf": irf_b},
        "theta": theta,
        "active_mask": active,
        "known_mask": known,
        "known_values": known_values,
        "layer_count": layers,
    }
    from tdnirs_posterior.precision import AUTOCAST_DTYPE, use_bf16_amp
    amp = use_bf16_amp(config["training"], device)

    def training_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=AUTOCAST_DTYPE, enabled=amp):
            step_loss, _ = model.training_loss(
                batch, sds, float(config["training"]["consistency_weight"])
            )
        step_loss.backward()
        optimizer.step()
        return step_loss

    # Warm up CUDA kernels before timing and reset the optimizer state so the
    # measured step includes the memory cost that persists during training.
    training_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    loss = training_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    else:
        peak = float("nan")
        reserved = float("nan")
    elapsed = time.perf_counter() - start
    result = {
        "time_bins": length,
        "batch_size": batch_size,
        "device": str(device),
        "seconds": elapsed,
        "peak_allocated_gib": peak,
        "peak_reserved_gib": reserved,
        "loss": float(loss.detach().cpu()),
    }
    del model, optimizer, batch, dtof_a, dtof_b, irf_a, irf_b, theta, loss
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile PICT Model input resolutions")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[1024, 2048, 2500])
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = int(args.batch_size or config["training"]["micro_batch_size"])
    for resolution in args.resolutions:
        print(json.dumps(_profile(config, resolution, batch, device)))


if __name__ == "__main__":
    main()
