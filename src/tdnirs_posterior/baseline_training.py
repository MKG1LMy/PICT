from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .baseline_evaluation import PHYSICAL_WIDTHS
from .config import config_seed, project_path
from .channels import COUNT_WEIGHTS, PROTOCOL as CHANNEL_PROTOCOL, ChannelValidationLoader
from .data.dataset import H5TPSFDataset
from .model import BaselineConfig, build_baseline, masked_smooth_l1
from .lr_schedule import WarmupPlateau
from .parameters import ParameterSchema
from .precision import AUTOCAST_DTYPE, PRECISION_PROTOCOL, checked_optimizer_step, use_bf16_amp, TrainingNumerics
from .thickness_focus import focus_policy, strata, configure_training_focus
from .log_time import timestamp
from .training import (
    ExponentialMovingAverage,
    _atomic_torch_save,
    _capture_rng_state,
    _dataset_fingerprint,
    _manifest,
    _posterior_validation_indices,
    _reconcile_metrics,
    _restore_rng_state,
    _seed_everything,
)


BASELINE_PROTOCOL = "dtof_only_two_measurement_deterministic_regression"


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _signature(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    return {
        "protocol": BASELINE_PROTOCOL,
        "precision_protocol": PRECISION_PROTOCOL,
        "channel_protocol": CHANNEL_PROTOCOL,
        "seeds": {
            name: config_seed(config, name)
            for name in ("data_split", "training", "validation")
        },
        "data": {
            key: config["data"][key]
            for key in (
                "sds_mm",
                "time_bins",
                "time_bin_ps",
                "time_start_ps",
                "split",
            )
        },
        "simulation": config["simulation"],
        "baseline": config["baseline"],
        "training": {
            key: training.get(key)
            for key in (
                "epochs",
                "micro_batch_size",
                "effective_batch_size",
                "learning_rate",
                "weight_decay",
                "warmup_epochs",
                "lr_schedule",
                "grad_clip",
                "ema_decay",
                "thickness_focus",
                "baseline_early_stopping",
                "amp",
                "max_train_samples",
                "max_val_samples",
                "output_dir",
            )
        },
    }


def _update_early_stopping(
    score: float,
    history: list[float],
    best_validation: float,
    stale: int,
    *,
    active: bool,
    smoothing_window: int,
    min_delta: float,
) -> tuple[list[float], float, int, float, bool]:
    """Update baseline selection state without letting warm-up consume patience."""
    if not active:
        return [], float("inf"), 0, score, False
    history = (history + [score])[-smoothing_window:]
    smoothed = float(np.mean(history))
    improved = (
        not math.isfinite(best_validation)
        or smoothed < best_validation * (1.0 - min_delta)
    )
    if improved:
        return history, smoothed, 0, smoothed, True
    return history, best_validation, stale + 1, smoothed, False


def _to_device(raw: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "loss_weight": raw.get("loss_weight", torch.ones(len(raw["theta"]))).to(device, non_blocking=True),
        "theta": raw["theta"].to(device, non_blocking=True),
        "theta_physical": raw["theta_physical"].to(device, non_blocking=True),
        "active_mask": raw["active_mask"].to(device, non_blocking=True),
        "layer_count": raw["layer_count"].to(device, non_blocking=True),
        "view_a": {key: value.to(device, non_blocking=True) for key, value in raw["view_a"].items() if key in {"dtof", "channel_mask"}},
        "view_b": {key: value.to(device, non_blocking=True) for key, value in raw["view_b"].items() if key in {"dtof", "channel_mask"}},
    }


@torch.no_grad()
def _validate_single(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    focus=None,
) -> dict[str, Any]:
    model.eval()
    focus = focus_policy(focus)
    schema = ParameterSchema()
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    layers: list[np.ndarray] = []
    for raw in loader:
        batch = _to_device(raw, device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=AUTOCAST_DTYPE):
            prediction = model(batch["view_a"]["dtof"], batch["layer_count"], batch["view_a"].get("channel_mask"))
        predictions.append(prediction.float().cpu().numpy())
        truths.append(batch["theta_physical"].float().cpu().numpy())
        masks.append(batch["active_mask"].bool().cpu().numpy())
        layers.append(batch["layer_count"].cpu().numpy())
    normalized_prediction = np.concatenate(predictions)
    physical_prediction = schema.denormalize(normalized_prediction)
    truth = np.concatenate(truths)
    active = np.concatenate(masks)
    layer_count = np.concatenate(layers)
    active_finite = (~active) | np.isfinite(physical_prediction)
    failed_cases = ~np.all(active_finite, axis=1)
    if np.any(failed_cases):
        return {
            "macro_nmae": float("inf"),
            "layer_nmae": {f"N{layer}": float("inf") for layer in (1, 2, 3)},
            "core_layer_nmae": {f"N{layer}": float("inf") for layer in (1, 2, 3)},
            "finite_fraction": float(np.mean(~failed_cases)),
            "numerical_failure_cases": int(np.sum(failed_cases)),
        }
    scores: dict[str, float] = {}
    core_scores, by_thickness = {}, {}
    for layer in (1, 2, 3):
        selected = layer_count == layer
        parameter_scores = []
        for parameter in np.flatnonzero(np.any(active[selected], axis=0)):
            parameter_scores.append(
                float(
                    np.mean(
                        np.abs(
                            physical_prediction[selected, parameter]
                            - truth[selected, parameter]
                        )
                    )
                    / PHYSICAL_WIDTHS[parameter]
                )
            )
        scores[f"N{layer}"] = float(np.mean(parameter_scores)) if parameter_scores else float("inf")
        core_scores[f"N{layer}"] = scores[f"N{layer}"]
        by_thickness[f"N{layer}"] = {}
        if focus["enabled"] and np.any(selected):
            weighted = 0.0
            core_scores[f"N{layer}"] = float("inf")
            for name, (mask, weight) in strata(truth[selected], layer, focus).items():
                indices = np.flatnonzero(selected)[mask]
                parameters = np.flatnonzero(np.any(active[indices], axis=0))
                error = np.abs(physical_prediction[indices][:, parameters] - truth[indices][:, parameters]) / PHYSICAL_WIDTHS[parameters]
                value = float(error.mean())
                by_thickness[f"N{layer}"][name] = {"cases": len(indices), "nmae": value, "weight": weight}
                weighted += weight * value
                if name in {"core", "all"}:
                    core_scores[f"N{layer}"] = value
            scores[f"N{layer}"] = weighted
    return {
        "macro_nmae": float(np.mean(list(scores.values()))),
        "layer_nmae": scores,
        "core_layer_nmae": core_scores,
        "by_thickness": by_thickness,
        "finite_fraction": 1.0,
        "numerical_failure_cases": 0,
    }


def _validate(model, loader, device, amp_enabled, focus=None):
    reports = {str(k): _validate_single(model, ChannelValidationLoader(loader, k), device, amp_enabled, focus)
               for k in COUNT_WEIGHTS}
    return {**reports["4"], "by_channel_count": reports,
            "macro_nmae": sum(COUNT_WEIGHTS[k] * reports[str(k)]["macro_nmae"] for k in COUNT_WEIGHTS),
            "finite_fraction": min(r["finite_fraction"] for r in reports.values()),
            "numerical_failure_cases": sum(r["numerical_failure_cases"] for r in reports.values())}


def train_baseline(
    config: dict[str, Any],
    *,
    resume: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
    init_weights: str = "raw",
    run_epochs: int | None = None,
    auto_test: bool = True,
) -> Path:
    session_started_at = timestamp()
    if resume is not None and init_checkpoint is not None:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
    training_seed = config_seed(config, "training")
    validation_seed = config_seed(config, "validation")
    _seed_everything(training_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = config["training"]
    focus = focus_policy(training.get("thickness_focus"))
    amp_enabled = use_bf16_amp(training, device)
    index = _manifest(config)
    fingerprint = _dataset_fingerprint(index)
    output = project_path(config, training["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "metrics.jsonl"
    best_path = output / "best.pt"
    last_path = output / "last.pt"
    summary_path = output / "training_summary.json"
    if resume is None and any(path.exists() for path in (log_path, best_path, last_path)):
        raise FileExistsError(f"Baseline artifacts already exist in {output}; use --resume or a new output_dir")

    train_data = H5TPSFDataset(index, "train", config["simulation"], config["data"]["sds_mm"], training_seed, training.get("max_train_samples"), paired_views=True)
    focus_counts = configure_training_focus(train_data, focus)
    print(json.dumps({"event": "training_start", "timestamp": session_started_at,
                      "seed": training_seed, "output": str(output), "thickness_focus": focus,
                      "training_strata": focus_counts}), flush=True)
    validation_data = H5TPSFDataset(index, "validation", config["simulation"], config["data"]["sds_mm"], validation_seed + 1, None, paired_views=False)
    batch_size = int(training["micro_batch_size"])
    effective = int(training["effective_batch_size"])
    if effective < batch_size or effective % batch_size:
        raise ValueError("effective_batch_size must be an integer multiple of micro_batch_size")
    accumulation = effective // batch_size
    loader_args = {"batch_size": batch_size, "num_workers": int(training.get("num_workers", 0)), "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    validation_source: H5TPSFDataset | Subset = validation_data
    validation_limit = training.get("max_val_samples")
    if validation_limit is not None:
        per_layer = max(1, math.ceil(int(validation_limit) / 3))
        indices = _posterior_validation_indices(
            validation_data, per_layer, validation_seed + 910000, focus=focus
        )
        validation_source = Subset(validation_data, indices)
    validation_loader = DataLoader(validation_source, shuffle=False, **loader_args)
    baseline_config = BaselineConfig.from_mapping(config["baseline"])
    model = build_baseline(baseline_config, int(config["data"]["time_bins"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    epochs = int(training["epochs"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    scheduler = WarmupPlateau(optimizer, training, updates_per_epoch)
    warmup_steps = scheduler.warmup_steps
    scaler = torch.amp.GradScaler(device.type, enabled=False)
    ema = ExponentialMovingAverage(model, float(training.get("ema_decay", 0.9985)))
    early = training.get("baseline_early_stopping", {})
    patience = int(early.get("patience_epochs", 50))
    smoothing_window = int(early.get("smoothing_window", 5))
    min_delta = float(early.get("min_delta_relative", 0.002))
    start_after_warmup = bool(early.get("start_after_warmup", True))
    beta = float(config["baseline"].get("loss_beta", 0.1))
    start_epoch = 0
    global_step = 0
    stale = 0
    best_validation = float("inf")
    best_channel_scores = {}
    history: list[float] = []
    skipped_steps = 0
    signature = _signature(config)
    initialization: dict[str, Any] | None = None

    if init_checkpoint is not None:
        if init_weights not in {"raw", "ema"}:
            raise ValueError("init_weights must be 'raw' or 'ema'")
        source_path = project_path(config, init_checkpoint).resolve()
        source = torch.load(source_path, map_location=device, weights_only=False)
        source_baseline = source.get("config", {}).get("baseline", {})
        if source_baseline.get("architecture") != baseline_config.architecture:
            raise ValueError("Initialization checkpoint baseline architecture is incompatible")
        source_state = source["ema"] if init_weights == "ema" else source["model"]
        model.load_state_dict(source_state)
        ema = ExponentialMovingAverage(model, float(training.get("ema_decay", 0.9985)))
        initialization = {
            "source_checkpoint": str(source_path),
            "source_weights": init_weights,
            "source_epoch": int(source.get("epoch", -1)) + 1,
            "new_experiment": True,
        }

    if resume is not None:
        checkpoint_path = project_path(config, resume)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("resume_signature") != signature:
            raise ValueError("Baseline checkpoint configuration is incompatible")
        if checkpoint.get("dataset_fingerprint") != fingerprint:
            raise ValueError("Baseline checkpoint dataset differs from the current manifest")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"], float(checkpoint["ema_decay"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["next_epoch"])
        global_step = int(checkpoint["global_step"])
        stale = int(checkpoint["stale_epochs"])
        best_validation = float(checkpoint["best_validation"])
        best_channel_scores = checkpoint.get("best_channel_scores", {})
        history = list(checkpoint["monitor_history"])
        skipped_steps = int(checkpoint.get("skipped_optimizer_steps", 0))
        initialization = checkpoint.get("initialization")
        _restore_rng_state(checkpoint["rng_state"])
        _reconcile_metrics(log_path, checkpoint)

    requested_epochs = int(run_epochs) if run_epochs is not None else epochs - start_epoch
    if requested_epochs <= 0 or start_epoch >= epochs:
        raise ValueError("No baseline training epochs remain")
    end_epoch = min(epochs, start_epoch + requested_epochs)
    completed_epoch = start_epoch - 1
    stop_reason = "run_epoch_limit"
    numerics = TrainingNumerics(model)
    raw = None
    batch_index = -1
    try:
        for epoch in range(start_epoch, end_epoch):
            numerics.start_epoch()
            raw, batch_index = None, -1
            epoch_started_at = timestamp()
            started = time.perf_counter()
            train_data.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            sample_count = 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for batch_index, raw in enumerate(train_loader):
                numerics.begin_batch(raw, batch_index, accumulation)
                batch = _to_device(raw, device)
                count = int(batch["theta"].shape[0])
                group_start = (batch_index // accumulation) * accumulation * batch_size
                group_count = min(effective, len(train_data) - group_start)
                with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=AUTOCAST_DTYPE):
                    prediction_a = model(batch["view_a"]["dtof"], batch["layer_count"], batch["view_a"].get("channel_mask"))
                    prediction_b = model(batch["view_b"]["dtof"], batch["layer_count"], batch["view_b"].get("channel_mask"))
                    loss = 0.5 * (
                        masked_smooth_l1(prediction_a, batch["theta"], batch["active_mask"], beta, batch["loss_weight"])
                        + masked_smooth_l1(prediction_b, batch["theta"], batch["active_mask"], beta, batch["loss_weight"])
                    )
                    scaled = loss * (count / group_count)
                if not torch.isfinite(scaled).item():
                    raise FloatingPointError(
                        f"Non-finite baseline training loss at epoch {epoch + 1}, "
                        f"batch {batch_index + 1}; optimizer step was not applied."
                    )
                scaler.scale(scaled).backward()
                update = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
                if update:
                    grad_norm = checked_optimizer_step(
                        model, optimizer, scaler, float(training["grad_clip"]),
                        epoch=epoch + 1, batch_index=batch_index + 1,
                        diagnostics=numerics.step_details,
                    )
                    numerics.observe(grad_norm, batch_index + 1)
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    ema.update(model)
                    global_step += 1
                loss_sum += float(loss.detach().cpu()) * count
                sample_count += count
            numerical_record = numerics.finish()
            with ema.apply_to(model):
                validation = _validate(model, validation_loader, device, amp_enabled, focus)
            score = float(validation["macro_nmae"])
            if not math.isfinite(score):
                raise RuntimeError("EMA validation macro nMAE is not finite")
            early_stopping_active = not start_after_warmup or global_step >= warmup_steps
            previous_best, previous_stale = best_validation, stale
            history, best_validation, stale, smoothed, improved = _update_early_stopping(
                score,
                history,
                best_validation,
                stale,
                active=early_stopping_active,
                smoothing_window=smoothing_window,
                min_delta=min_delta,
            )
            channel_scores = {f"C{k}/{n}": value for k in (3, 4)
                              for n, value in validation["by_channel_count"][str(k)]["core_layer_nmae"].items()}
            tolerance = float(training.get("posterior_validation", {}).get("max_group_regression_relative", 0.05))
            channel_guard_passed = validation["finite_fraction"] == 1.0 and all(
                value <= best_channel_scores[key] * (1 + tolerance) + 1e-6
                for key, value in channel_scores.items() if key in best_channel_scores)
            if improved and not channel_guard_passed:
                best_validation, stale, improved = previous_best, previous_stale + 1, False
            if improved:
                best_channel_scores = channel_scores
            lr_reduced = scheduler.step_metric(smoothed)
            record = {
                "epoch_started_at": epoch_started_at,
                "epoch_finished_at": timestamp(),
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": loss_sum / max(sample_count, 1),
                "numerics": numerical_record,
                "ema_macro_nmae": score,
                "ema_layer_nmae": validation["layer_nmae"],
                "ema_by_channel_count": validation["by_channel_count"],
                "channel_guard_passed": channel_guard_passed,
                "ema_finite_fraction": validation["finite_fraction"],
                "ema_numerical_failure_cases": validation["numerical_failure_cases"],
                "smoothed_macro_nmae": smoothed,
                "best_validation": best_validation,
                "stale_epochs": stale,
                "patience_epochs": patience,
                "early_stopping_active": early_stopping_active,
                "early_stopping_starts_after_warmup": start_after_warmup,
                "warmup_steps": warmup_steps,
                "warmup_steps_completed": min(global_step, warmup_steps),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "lr_reduced": lr_reduced,
                "lr_schedule": scheduler.diagnostics(),
                "epoch_seconds": time.perf_counter() - started,
                "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
            }
            payload = {
                "model": model.state_dict(),
                "ema": ema.shadow,
                "ema_decay": ema.decay,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "config": config,
                "resume_signature": signature,
                "dataset_fingerprint": fingerprint,
                "epoch": epoch,
                "next_epoch": epoch + 1,
                "global_step": global_step,
                "best_validation": best_validation,
                "stale_epochs": stale,
                "monitor_history": history,
                "best_channel_scores": best_channel_scores,
                "early_stopping_active": early_stopping_active,
                "warmup_steps": warmup_steps,
                "skipped_optimizer_steps": skipped_steps,
                "metrics_record": record,
                "initialization": initialization,
                "rng_state": _capture_rng_state(),
            }
            if improved:
                _atomic_torch_save(payload, best_path)
            _atomic_torch_save(payload, last_path)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            monitor_text = (
                f"stale={stale}/{patience}"
                if early_stopping_active
                else f"warmup={min(global_step, warmup_steps)}/{warmup_steps} | early_stop=inactive"
            )
            if scheduler.warmup_complete and not scheduler.can_stop:
                monitor_text += f" | early_stop=lr_observe:{scheduler.diagnostics()['early_stop_grace_remaining']}"
            print(
                f"[{record['epoch_finished_at']}] "
                f"Epoch {epoch + 1:04d}/{epochs:04d} | train={record['train_loss']:.5f} | "
                f"ema_nMAE={score:.5f} | {monitor_text} | "
                f"C4={validation['by_channel_count']['4']['macro_nmae']:.5f} | channel_guard={'ok' if channel_guard_passed else 'regressed'} | "
                f"lr={record['learning_rate']:.3e} | {record['epoch_seconds']:.1f}s",
                flush=True,
            )
            completed_epoch = epoch
            if scheduler.can_stop and early_stopping_active and stale >= patience:
                stop_reason = "early_stopping"
                break
    except BaseException as error:
        failure = {"timestamp": timestamp(), "epoch": epoch + 1, "batch": batch_index + 1,
                   "training_seed": training_seed, "error": str(error), **numerics.failure(raw)}
        with (output / "training_failure.json").open("w", encoding="utf-8") as handle:
            json.dump(failure, handle, ensure_ascii=False, indent=2)
        print(json.dumps({"event": "training_failed", "timestamp": timestamp(),
                          "session_started_at": session_started_at, "error": str(error)}), flush=True)
        raise
    finally:
        train_data.close()
        validation_data.close()
    if completed_epoch + 1 >= epochs:
        stop_reason = "max_epochs"
    epoch_records = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                epoch_records.append(json.loads(line))
    total_training_seconds = float(sum(float(row.get("epoch_seconds", 0.0)) for row in epoch_records))
    summary = {
        "session_started_at": session_started_at,
        "session_finished_at": timestamp(),
        "thickness_focus": focus,
        "training_strata": focus_counts,
        "status": "paused" if stop_reason == "run_epoch_limit" else "completed",
        "stop_reason": stop_reason,
        "architecture": baseline_config.architecture,
        "completed_epoch": completed_epoch,
        "configured_epochs": epochs,
        "best_validation": best_validation,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_elapsed_seconds": total_training_seconds,
        "completed_epochs": len(epoch_records),
        "early_stopping_starts_after_warmup": start_after_warmup,
        "warmup_steps": warmup_steps,
        "patience_epochs": patience,
        "initialization": initialization,
    }
    _atomic_json_write(summary_path, summary)
    print(json.dumps({"event": "training_complete", **summary}), flush=True)
    if summary["status"] == "completed" and auto_test and bool(config.get("evaluation", {}).get("auto_run", True)):
        from .baseline_testing import run_baseline_test

        run_baseline_test(config, best_path)
    return best_path
