from __future__ import annotations

import hashlib
import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .config import config_seed, project_path
from .channels import COUNT_WEIGHTS, PROTOCOL as CHANNEL_PROTOCOL, ChannelValidationLoader
from .data import DatasetIndex, dataset_policy, prepare_dataset_index
from .data.dataset import H5TPSFDataset
from .evaluation import evaluate_posterior
from .inference import sample_native_posterior_batch
from .model import ICAPModel, ModelConfig
from .lr_schedule import WarmupPlateau
from .precision import AUTOCAST_DTYPE, PRECISION_PROTOCOL, checked_optimizer_step, use_bf16_amp, TrainingNumerics
from .thickness_focus import (focus_policy, core_mask, strata, dataset_truth,
                              configure_training_focus, FocusValidationLoader)
from .log_time import timestamp


TRAINING_CONDITIONING_PROTOCOL = "two_measurements_x_no_prior_and_nonempty_prior"


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _materialize_fixed_dataset(
    dataset: H5TPSFDataset,
) -> tuple[tuple[dict[str, Any], ...], int]:
    """Generate deterministic validation observations once and close HDF5 handles."""
    samples: list[dict[str, Any]] = []
    try:
        for index in range(len(dataset)):
            samples.append(dataset[index])
    finally:
        dataset.close()
    cached = tuple(samples)
    return cached, sum(_tensor_bytes(sample) for sample in cached)


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, value in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)

    def load_state_dict(self, state: dict[str, torch.Tensor], decay: float) -> None:
        if set(state) != set(self.shadow):
            missing = sorted(set(self.shadow) - set(state))
            unexpected = sorted(set(state) - set(self.shadow))
            raise ValueError(
                "EMA state does not match the model: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for name, value in state.items():
            if value.shape != self.shadow[name].shape:
                raise ValueError(f"EMA tensor shape mismatch for {name}")
            self.shadow[name].copy_(value.to(self.shadow[name].device))
        self.decay = float(decay)

    @contextmanager
    def apply_to(self, model: torch.nn.Module):
        """Temporarily swap EMA weights into ``model`` without a full-model copy."""

        def swap() -> None:
            with torch.no_grad():
                for name, value in model.state_dict().items():
                    if name not in self.shadow:
                        continue
                    temporary = value.detach().clone()
                    value.copy_(self.shadow[name])
                    self.shadow[name].copy_(temporary)

        swap()
        try:
            yield model
        finally:
            swap()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda", [])
    if torch.cuda.is_available() and cuda_states:
        # Training uses the current CUDA device only. Restoring its saved state
        # keeps resume portable when the destination workstation exposes a
        # different number of otherwise unused GPUs.
        current_device = torch.cuda.current_device()
        source_device = min(current_device, len(cuda_states) - 1)
        torch.cuda.set_rng_state(cuda_states[source_device].cpu(), current_device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_fingerprint(index: DatasetIndex) -> dict[str, Any]:
    records = hashlib.sha256()
    for record in index.records:
        records.update(
            json.dumps(
                {
                    "file": record.file,
                    "source_index": record.source_index,
                    "sample_id": record.sample_id,
                    "layer_count": record.layer_count,
                    "status": record.status,
                    "category": record.category,
                    "split": record.split,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        records.update(b"\n")
    source_files = {
        name: _sha256(index.root / name)
        for name in sorted({record.file for record in index.records})
    }
    return {"records_sha256": records.hexdigest(), "source_files_sha256": source_files}


def _resume_signature(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    data_signature = {
        "sds_mm": config["data"]["sds_mm"],
        "time_bins": config["data"]["time_bins"],
        "time_bin_ps": config["data"]["time_bin_ps"],
        "time_start_ps": config["data"]["time_start_ps"],
        "split": config["data"]["split"],
    }
    return {
        "conditioning_protocol": TRAINING_CONDITIONING_PROTOCOL,
        "channel_protocol": CHANNEL_PROTOCOL,
        "precision_protocol": PRECISION_PROTOCOL,
        "seeds": {
            name: config_seed(config, name)
            for name in ("data_split", "training", "validation")
        },
        "data": data_signature,
        "simulation": config["simulation"],
        "model": config["model"],
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
                "consistency_weight",
                "ema_decay",
                "thickness_focus",
                "early_stopping",
                "posterior_validation",
                "amp",
                "max_train_samples",
                "max_val_samples",
                "output_dir",
            )
        },
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _reconcile_metrics(log_path: Path, checkpoint: dict[str, Any]) -> None:
    """Make the JSONL log agree with the last atomically committed epoch."""
    committed_epoch = int(checkpoint["epoch"])
    saved_record = checkpoint.get("metrics_record")
    if not isinstance(saved_record, dict):
        raise ValueError("Resume checkpoint is missing its metrics record")
    records_by_epoch: dict[int, dict[str, Any]] = {}
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if line_number == len(lines):
                    continue  # Interrupted append at the end of the file.
                raise ValueError(f"Invalid metrics JSON at line {line_number}")
            record_epoch = int(record["epoch"])
            if record_epoch < committed_epoch:
                records_by_epoch[record_epoch] = record
    records_by_epoch[committed_epoch] = saved_record
    temporary = log_path.with_name(f"{log_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for epoch in sorted(records_by_epoch):
            handle.write(json.dumps(records_by_epoch[epoch]) + "\n")
    temporary.replace(log_path)


def _checkpoint_payload(
    *,
    config: dict[str, Any],
    dataset_fingerprint: dict[str, Any],
    model: ICAPModel,
    ema: ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupPlateau,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    epoch: int,
    global_step: int,
    best_validation: float,
    validation_loss: float,
    stale_epochs: int,
    monitor_history: list[float],
    best_posterior_score: float,
    best_posterior_group_scores: dict[str, float],
    posterior_stale_evaluations: int,
    posterior_evaluations: int,
    skipped_optimizer_steps: int,
    metrics_record: dict[str, Any],
    initialization: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        # EMA is the primary evaluation/inference weight set.
        "ema": ema.shadow,
        "ema_decay": ema.decay,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "config": config,
        "resume_signature": _resume_signature(config),
        "dataset_fingerprint": dataset_fingerprint,
        "epoch": epoch,
        "next_epoch": epoch + 1,
        "global_step": global_step,
        "best_validation": best_validation,
        "validation_loss": validation_loss,
        "stale_epochs": stale_epochs,
        "monitor_history": list(monitor_history),
        "best_posterior_score": best_posterior_score,
        "best_posterior_group_scores": dict(best_posterior_group_scores),
        "posterior_stale_evaluations": posterior_stale_evaluations,
        "posterior_evaluations": posterior_evaluations,
        "skipped_optimizer_steps": skipped_optimizer_steps,
        "metrics_record": metrics_record,
        "initialization": initialization,
        "runtime": {
            "torch_version": torch.__version__,
            "device_type": next(model.parameters()).device.type,
            "amp_enabled": amp_enabled,
            "amp_dtype": "bfloat16" if amp_enabled else "float32",
            "gradient_scaler_enabled": scaler.is_enabled(),
            "cuda_device_count": torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0,
        },
        "rng_state": _capture_rng_state(),
    }


def _manifest(config: dict[str, Any]) -> DatasetIndex:
    root = project_path(config, config["data"]["root"])
    directory = project_path(config, config["data"]["manifest_dir"])
    manifest = directory / "manifest.jsonl"
    summary = directory / "summary.json"
    split_seed = config_seed(config, "data_split")
    expected_policy = dataset_policy(config["data"], split_seed)
    rebuild = not manifest.exists() or not summary.exists()
    if not rebuild:
        try:
            saved_summary = json.loads(summary.read_text(encoding="utf-8"))
            rebuild = saved_summary.get("dataset_policy") != expected_policy
            if not rebuild:
                indexed = DatasetIndex.read(manifest, root)
                discovered = {p.relative_to(root).as_posix() for p in root.glob("*/mixed-dataset.h5")}
                rebuild = {r.file for r in indexed.records} != discovered
        except (json.JSONDecodeError, OSError):
            rebuild = True
    if rebuild:
        index = prepare_dataset_index(root, config["data"], split_seed)
        index.write(directory)
    return DatasetIndex.read(manifest, root)


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "loss_weight" in batch:
        result["loss_weight"] = batch["loss_weight"].to(device, non_blocking=True)
    for key in (
        "theta",
        "theta_physical",
        "active_mask",
        "known_mask",
        "known_values",
        "layer_count",
    ):
        result[key] = batch[key].to(device, non_blocking=True)
    for view_name in ("view_a", "view_b"):
        result[view_name] = {
            "dtof": batch[view_name]["dtof"].to(device, non_blocking=True),
            "irf": batch[view_name]["irf"].to(device, non_blocking=True),
        }
        if "channel_mask" in batch[view_name]:
            result[view_name]["channel_mask"] = batch[view_name]["channel_mask"].to(device, non_blocking=True)
    return result


FLOW_VALIDATION_PRIOR_INDICES: dict[int, dict[str, tuple[int, ...]]] = {
    1: {"no_prior": (), "known_mua1": (2,), "known_musp1": (5,)},
    2: {
        "no_prior": (),
        "known_d1": (0,),
        "known_layer1_optical": (2, 5),
    },
    3: {
        "no_prior": (),
        "known_all_thickness": (0, 1),
        "known_layer1_optical": (2, 5),
        "known_thickness_layer1_optical": (0, 1, 2, 5),
    },
}


POSTERIOR_VALIDATION_PRIOR_INDICES: dict[int, dict[str, tuple[int, ...]]] = {
    1: {"no_prior": (), "known_mua1": (2,), "known_musp1": (5,)},
    2: {
        "no_prior": (),
        "k1_known_d1": (0,),
        "k1_known_mua2": (3,),
        "k2_known_layer1_optical": (2, 5),
        "k2_known_layer2_optical": (3, 6),
        "k3_known_d1_layer1_optical": (0, 2, 5),
        "k3_known_cross_layer_optical": (2, 3, 6),
        "k4_known_all_optical": (2, 3, 5, 6),
        "k4_known_d1_cross_layer_optical": (0, 2, 3, 6),
    },
    3: {
        "no_prior": (),
        "k1_known_d1": (0,),
        "k1_known_mua3": (4,),
        "k2_known_all_thickness": (0, 1),
        "k2_known_layer3_optical": (4, 7),
        "k3_known_layer1_structure": (0, 2, 5),
        "k3_known_d2_layer3_optical": (1, 4, 7),
        "k4_known_thickness_layer1_optical": (0, 1, 2, 5),
        "k4_known_deep_two_layers_optical": (3, 4, 6, 7),
        "k5_known_thickness_all_absorption": (0, 1, 2, 3, 4),
        "k5_known_d2_all_scattering_mua3": (1, 4, 5, 6, 7),
        "k6_known_all_optical": (2, 3, 4, 5, 6, 7),
        "k6_known_surface_two_layers": (0, 1, 2, 3, 5, 6),
        "k7_unknown_mua3": (0, 1, 2, 3, 5, 6, 7),
        "k7_unknown_d2": (0, 2, 3, 4, 5, 6, 7),
    },
}


def _fixed_known_mask(
    active: torch.Tensor,
    layer: int,
    scenario: str,
    panel: dict[int, dict[str, tuple[int, ...]]],
) -> torch.Tensor:
    result = torch.zeros_like(active, dtype=torch.bool)
    indices = panel[layer][scenario]
    if indices:
        result[:, list(indices)] = True
    return result


def _run_validation_single(
    model: ICAPModel,
    loader: DataLoader,
    device: torch.device,
    sds: torch.Tensor,
    consistency_weight: float,
    amp_enabled: bool,
    seed: int,
) -> dict[str, Any]:
    """Evaluate fixed observations with explicit scenario/layer macro averaging."""

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.eval()
    accumulators: dict[tuple[str, int], dict[str, float]] = {
        (scenario, layer): {
            "flow_numerator": 0.0,
            "unknown_terms": 0.0,
            "consistency_sum": 0.0,
            "samples": 0.0,
        }
        for layer, scenarios in FLOW_VALIDATION_PRIOR_INDICES.items()
        for scenario in scenarios
    }
    try:
        with torch.no_grad():
            for raw in loader:
                batch = _device_batch(raw, device)
                with torch.autocast(
                    device_type=device.type, enabled=amp_enabled, dtype=AUTOCAST_DTYPE
                ):
                    signal_a = model.encode_view(batch["view_a"], sds)
                    signal_b = model.encode_view(batch["view_b"], sds)
                    consistency_per_sample = 1.0 - F.cosine_similarity(
                        signal_a, signal_b, dim=-1
                    )
                    layers = batch["layer_count"]
                    for layer in (1, 2, 3):
                        selected = layers == layer
                        sample_count = int(selected.sum().item())
                        if not sample_count:
                            continue
                        consistency_sum = float(
                            consistency_per_sample[selected].sum().detach().cpu()
                        )
                        active_mask = batch["active_mask"][selected]
                        for scenario in FLOW_VALIDATION_PRIOR_INDICES[layer]:
                            known_mask = _fixed_known_mask(
                                active_mask,
                                layer,
                                scenario,
                                FLOW_VALIDATION_PRIOR_INDICES,
                            )
                            known_values = torch.where(
                                known_mask,
                                batch["theta"][selected],
                                torch.zeros_like(batch["theta"][selected]),
                            )
                            flow = model.posterior.flow_matching_loss(
                                batch["theta"][selected],
                                signal_a[selected],
                                layers[selected],
                                active_mask,
                                known_mask,
                                known_values,
                            )
                            unknown_terms = int(
                                (
                                    active_mask.bool() & ~known_mask.bool()
                                ).sum().item()
                            )
                            accumulator = accumulators[(scenario, layer)]
                            accumulator["flow_numerator"] += (
                                float(flow.detach().cpu()) * unknown_terms
                            )
                            accumulator["unknown_terms"] += unknown_terms
                            accumulator["consistency_sum"] += consistency_sum
                            accumulator["samples"] += sample_count
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    by_scenario: dict[str, Any] = {}
    all_groups: list[dict[str, float]] = []
    no_prior_groups: list[dict[str, float]] = []
    for layer, scenarios in FLOW_VALIDATION_PRIOR_INDICES.items():
        for scenario in scenarios:
            by_layer: dict[str, dict[str, float]] = {}
            accumulator = accumulators[(scenario, layer)]
            if not accumulator["samples"] or not accumulator["unknown_terms"]:
                continue
            flow = accumulator["flow_numerator"] / accumulator["unknown_terms"]
            consistency = accumulator["consistency_sum"] / accumulator["samples"]
            metrics = {
                "flow": float(flow),
                "consistency": float(consistency),
                "total": float(flow + consistency_weight * consistency),
                "samples": int(accumulator["samples"]),
                "unknown_terms": int(accumulator["unknown_terms"]),
            }
            by_layer[str(layer)] = metrics
            all_groups.append(metrics)
            if scenario == "no_prior":
                no_prior_groups.append(metrics)
            key = f"N{layer}/{scenario}"
            by_scenario[key] = {
                "flow": metrics["flow"],
                "consistency": metrics["consistency"],
                "total": metrics["total"],
                "by_layer_count": by_layer,
            }
    if not all_groups:
        return {
            "flow": float("nan"),
            "consistency": float("nan"),
            "total": float("nan"),
            "by_scenario": by_scenario,
        }
    return {
        "flow": float(np.mean([value["flow"] for value in no_prior_groups])),
        "consistency": float(
            np.mean([value["consistency"] for value in no_prior_groups])
        ),
        "total": float(np.mean([value["total"] for value in no_prior_groups])),
        "by_scenario": by_scenario,
    }


def _focused_flow_validation(model, loader, device, sds, consistency_weight, amp_enabled, seed, focus):
    if not focus["enabled"]:
        return _run_validation_single(model, loader, device, sds, consistency_weight, amp_enabled, seed)
    groups = {name: _run_validation_single(model, FocusValidationLoader(loader, focus, core),
              device, sds, consistency_weight, amp_enabled, seed)
              for name, core in (("core", True), ("other", False))}
    scenarios = {}
    keys = set().union(*(r["by_scenario"] for r in groups.values()))
    for key in sorted(keys):
        present = {name: r["by_scenario"][key] for name, r in groups.items() if key in r["by_scenario"]}
        weights = {name: focus[name + "_weight"] for name in present}
        total = sum(weights.values())
        scenarios[key] = {metric: sum(weights[n] * v[metric] for n, v in present.items()) / total
                          for metric in ("flow", "consistency", "total")}
        scenarios[key]["by_thickness"] = present
    no_prior = [v for k, v in scenarios.items() if k.endswith("/no_prior")]
    return {**{metric: float(np.mean([v[metric] for v in no_prior])) for metric in ("flow", "consistency", "total")},
            "by_scenario": scenarios, "by_thickness": groups, "thickness_focus": focus}


def _run_validation(model, loader, device, sds, consistency_weight, amp_enabled, seed, focus=None):
    focus = focus_policy(focus)
    reports = {str(k): _focused_flow_validation(model, ChannelValidationLoader(loader, k),
               device, sds, consistency_weight, amp_enabled, seed, focus) for k in COUNT_WEIGHTS}
    result = dict(reports["4"])
    for key in ("flow", "consistency", "total"):
        result[key] = sum(COUNT_WEIGHTS[k] * reports[str(k)][key] for k in COUNT_WEIGHTS)
    result["by_channel_count"] = reports
    return result


def _run_posterior_validation(*args, **kwargs):
    import inspect
    bound = inspect.signature(_run_posterior_validation_single).bind(*args, **kwargs)
    bound.apply_defaults()
    options = dict(bound.arguments)
    loader = options.pop("loader")
    reports = {str(k): _run_posterior_validation_single(
        loader=ChannelValidationLoader(loader, k), **options) for k in COUNT_WEIGHTS}
    result = dict(reports["4"])
    result["by_channel_count"] = reports
    result["channel_weights"] = COUNT_WEIGHTS
    result["score"] = sum(COUNT_WEIGHTS[k] * reports[str(k)]["score"] for k in COUNT_WEIGHTS)
    result["score_components"] = {
        "channel_weights": COUNT_WEIGHTS,
        "by_channel_count": {k: r["score_components"] for k, r in reports.items()},
        "no_prior_macro_normalized_crps": sum(COUNT_WEIGHTS[k] * reports[str(k)]["score_components"]["no_prior_macro_normalized_crps"] for k in COUNT_WEIGHTS),
        "prior_macro_normalized_crps": sum(COUNT_WEIGHTS[k] * reports[str(k)]["score_components"]["prior_macro_normalized_crps"] for k in COUNT_WEIGHTS),
        "weights": reports["4"]["score_components"]["weights"],
    }
    result["eligible"] = all(reports[str(k)]["eligible"] for k in (3, 4)) and all(
        reports[str(k)]["finite_fraction"] == 1.0 and math.isfinite(reports[str(k)]["score"])
        for k in COUNT_WEIGHTS)
    result["finite_fraction"] = min(r["finite_fraction"] for r in reports.values())
    result["mean_coverage_error_90"] = max(reports[str(k)]["mean_coverage_error_90"] for k in (3, 4))
    result["prior_guardrail"] = {**result["prior_guardrail"],
        "passed": all(reports[str(k)]["prior_guardrail"]["passed"] for k in (3, 4)),
        "relative_changes_from_no_prior": {f"C{k}/{key}": v for k in (3, 4)
            for key, v in reports[str(k)]["prior_guardrail"]["relative_changes_from_no_prior"].items()}}
    result["group_scores"] = {f"C{k}/{key}": v for k in (3, 4)
                              for key, v in reports[str(k)]["group_scores"].items()}
    result["all_channel_group_scores"] = {f"C{k}/{key}": v for k in COUNT_WEIGHTS
                              for key, v in reports[str(k)]["group_scores"].items()}
    return result


def _posterior_validation_indices(
    dataset: H5TPSFDataset, cases_per_layer: int, seed: int, focus=None
) -> list[int]:
    rng = np.random.default_rng(seed)
    focus = focus_policy(focus)
    truth, layers = dataset_truth(dataset) if focus["enabled"] else (None, None)
    result: list[int] = []
    for layer in (1, 2, 3):
        candidates = np.asarray(
            [
                index
                for index, record in enumerate(dataset.records)
                if record.layer_count == layer
            ],
            dtype=np.int64,
        )
        if not candidates.size:
            continue
        if focus["enabled"] and layer > 1:
            core = core_mask(truth[candidates], layers[candidates], focus)
            pools = [candidates[core], candidates[~core]]
            for pool in pools:
                rng.shuffle(pool)
            count = min(cases_per_layer, len(candidates))
            ncore = int(round(count * focus["core_weight"]))
            if count >= 2 and all(len(p) for p in pools):
                ncore = min(max(ncore, 1), count - 1)
            ncore = min(ncore, len(pools[0]))
            nother = min(count - ncore, len(pools[1]))
            ncore = min(count - nother, len(pools[0]))
            result.extend(pools[0][:ncore].tolist() + pools[1][:nother].tolist())
            continue
        rng.shuffle(candidates)
        result.extend(candidates[: min(cases_per_layer, candidates.size)].tolist())
    return sorted(result)


def _posterior_group_regression_check(
    candidate: dict[str, float],
    reference: dict[str, float],
    maximum_relative: float,
) -> dict[str, Any]:
    relative_changes: dict[str, float] = {}
    passed = not reference or set(candidate) == set(reference)
    for key, reference_value in reference.items():
        candidate_value = candidate.get(key, float("inf"))
        relative_changes[key] = float(
            (candidate_value - reference_value) / max(abs(reference_value), 1e-8)
        )
        if candidate_value > (
            reference_value * (1.0 + maximum_relative) + 1e-6
        ):
            passed = False
    return {
        "passed": passed,
        "maximum_allowed_relative": maximum_relative,
        "relative_changes_from_selected_best": relative_changes,
    }


def _run_posterior_validation_single(
    model: ICAPModel,
    loader: DataLoader,
    device: torch.device,
    sds: torch.Tensor,
    amp_enabled: bool,
    seed: int,
    sample_count: int,
    solver_steps: int,
    max_coverage_error_90: float,
    max_prior_regression_relative: float = 0.05,
    no_prior_score_weight: float = 0.7,
    prior_score_weight: float = 0.3,
    focus: dict | None = None,
) -> dict[str, Any]:
    """Run a deterministic, lightweight posterior check on a stratified subset."""
    focus = focus_policy(focus)

    if (
        not math.isfinite(no_prior_score_weight)
        or not math.isfinite(prior_score_weight)
        or no_prior_score_weight < 0
        or prior_score_weight < 0
        or no_prior_score_weight + prior_score_weight <= 0
    ):
        raise ValueError("Posterior score weights must be finite and nonnegative")
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    scenario_keys = [
        f"N{layer}/{scenario}"
        for layer, scenarios in POSTERIOR_VALIDATION_PRIOR_INDICES.items()
        for scenario in scenarios
    ]
    payloads: dict[str, dict[str, list[Any]]] = {
        key: {"samples": [], "known": [], "truth": [], "active": [], "diagnostics": []}
        for key in scenario_keys
    }
    case_count = 0
    model.eval()
    try:
        with torch.no_grad():
            for batch_index, raw in enumerate(loader):
                dtof = raw["view_a"]["dtof"].to(device, non_blocking=True)
                irf = raw["view_a"]["irf"].to(device, non_blocking=True)
                layers = raw["layer_count"].to(device, non_blocking=True)
                active = raw["active_mask"].to(device, non_blocking=True).bool()
                truth = raw["theta_physical"].cpu().numpy().astype(np.float32)
                normalized_truth = raw["theta"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type, enabled=amp_enabled, dtype=AUTOCAST_DTYPE
                ):
                    signal = model.encode_view(
                        {"dtof": dtof, "irf": irf, "channel_mask": raw["view_a"]["channel_mask"].to(device)}, sds
                    )
                case_count += len(truth)
                for scenario_index, (layer, scenarios) in enumerate(
                    POSTERIOR_VALIDATION_PRIOR_INDICES.items()
                ):
                    selected = layers == layer
                    if not torch.any(selected):
                        continue
                    for local_index, scenario in enumerate(scenarios):
                        key = f"N{layer}/{scenario}"
                        selected_active = active[selected]
                        known = _fixed_known_mask(
                            selected_active,
                            layer,
                            scenario,
                            POSTERIOR_VALIDATION_PRIOR_INDICES,
                        )
                        selected_truth = truth[selected.cpu().numpy()]
                        selected_normalized_truth = normalized_truth[selected]
                        known_normalized = torch.where(
                            known, selected_normalized_truth, torch.zeros_like(selected_normalized_truth)
                        )
                        known_physical = np.where(
                            known.cpu().numpy(), selected_truth, 0.0
                        ).astype(np.float32)
                        posterior_seed = seed + scenario_index * 1_000_003 + local_index * 100_003 + batch_index
                        torch.manual_seed(posterior_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(posterior_seed)
                        samples, diagnostics = sample_native_posterior_batch(
                            model=model,
                            signal=signal[selected],
                            layer_count=layers[selected],
                            active_mask=selected_active,
                            known_mask=known,
                            known_normalized=known_normalized,
                            known_physical=known_physical,
                            sample_count=sample_count,
                            steps=solver_steps,
                        )
                        payloads[key]["samples"].append(samples)
                        payloads[key]["known"].append(known.cpu().numpy())
                        payloads[key]["truth"].append(selected_truth)
                        payloads[key]["active"].append(selected_active.cpu().numpy())
                        payloads[key]["diagnostics"].extend(diagnostics)
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    reports: dict[str, Any] = {}
    no_prior_scores: dict[int, float] = {}
    prior_scores_by_count: dict[int, dict[int, list[float]]] = {}
    coverage_errors: dict[str, float] = {}
    group_scores: dict[str, float] = {}
    finite_fractions: list[float] = []
    for scenario, payload in payloads.items():
        samples = np.concatenate(payload["samples"], axis=0)
        known = np.concatenate(payload["known"], axis=0).astype(bool)
        scenario_truth = np.concatenate(payload["truth"], axis=0)
        scenario_active = np.concatenate(payload["active"], axis=0).astype(bool)
        diagnostics = payload["diagnostics"]
        finite_cases = np.asarray([not value["numerical_failure"] for value in diagnostics])
        finite_fraction = float(np.mean([value["finite_fraction"] for value in diagnostics]))
        finite_fractions.append(finite_fraction)
        layer = int(scenario.split("/", 1)[0][1:])
        if np.any(finite_cases):
            metrics = evaluate_posterior(
                samples[finite_cases], scenario_truth[finite_cases],
                (scenario_active & ~known)[finite_cases], joint_seed=seed + layer,
            )
            aggregate = metrics["aggregate"]
            group_score = float(aggregate["mean_normalized_crps"])
            coverage_error = float(aggregate["mean_coverage_error_90"])
        else:
            metrics = {"aggregate": {}}
            group_score = float("inf")
            coverage_error = float("inf")
        thickness_metrics = {}
        guard_score = group_score
        if focus["enabled"] and layer > 1:
            weighted_score = 0.0
            guard_score = float("inf")
            for name, (mask, weight) in strata(scenario_truth, layer, focus).items():
                valid = mask & finite_cases
                detail = evaluate_posterior(samples[valid], scenario_truth[valid],
                    (scenario_active & ~known)[valid], joint_seed=seed + layer) if np.any(valid) else {"aggregate": {}}
                value = float(detail["aggregate"].get("mean_normalized_crps", float("inf")))
                thickness_metrics[name] = {"cases": int(mask.sum()), "weight": weight, "metrics": detail}
                weighted_score += weight * value
                if name == "core":
                    guard_score = value
                    coverage_error = float(detail["aggregate"].get("mean_coverage_error_90", float("inf")))
            group_score = weighted_score
        group_scores[scenario] = guard_score
        coverage_errors[scenario] = coverage_error
        if scenario.endswith("/no_prior"):
            no_prior_scores[layer] = group_score
        else:
            scenario_name = scenario.split("/", 1)[1]
            known_count = len(
                POSTERIOR_VALIDATION_PRIOR_INDICES[layer][scenario_name]
            )
            prior_scores_by_count.setdefault(layer, {}).setdefault(
                known_count, []
            ).append(group_score)
        reports[scenario] = {
            "layer_count": layer,
            "known_count": int(known.sum(axis=1)[0]),
            "known_indices": list(
                POSTERIOR_VALIDATION_PRIOR_INDICES[layer][
                    scenario.split("/", 1)[1]
                ]
            ),
            "metrics": metrics,
            "weighted_normalized_crps": group_score,
            "guard_normalized_crps": guard_score,
            "by_thickness": thickness_metrics,
            "finite_fraction": finite_fraction,
            "outside_training_support_fraction": float(np.mean([value["outside_training_support_fraction"] for value in diagnostics])),
            "physically_impossible_fraction": float(np.mean([value["physically_impossible_fraction"] for value in diagnostics])),
        }
    no_prior_score = float(
        np.mean([no_prior_scores.get(layer, float("inf")) for layer in (1, 2, 3)])
    )
    prior_scores_by_layer = {
        layer: float(
            np.mean(
                [
                    np.mean(scores)
                    for _, scores in sorted(prior_scores_by_count.get(layer, {}).items())
                ]
            )
        )
        if prior_scores_by_count.get(layer)
        else float("inf")
        for layer in (1, 2, 3)
    }
    prior_score = float(np.mean(list(prior_scores_by_layer.values())))
    score_weight_total = no_prior_score_weight + prior_score_weight
    score = float(
        (
            no_prior_score_weight * no_prior_score
            + prior_score_weight * prior_score
        )
        / score_weight_total
    )
    coverage_error = float(
        np.mean(
            [
                coverage_errors.get(f"N{layer}/no_prior", float("inf"))
                for layer in (1, 2, 3)
            ]
        )
    )
    finite_fraction = float(np.mean(finite_fractions))
    prior_relative_changes: dict[str, float] = {}
    prior_guardrail_passed = True
    for key, value in group_scores.items():
        if key.endswith("/no_prior"):
            continue
        layer_key = key.split("/", 1)[0]
        baseline_payload = payloads[f"{layer_key}/no_prior"]
        prior_payload = payloads[key]
        baseline_samples = np.concatenate(baseline_payload["samples"], axis=0)
        prior_known = np.concatenate(prior_payload["known"], axis=0).astype(bool)
        prior_truth = np.concatenate(prior_payload["truth"], axis=0)
        prior_active = np.concatenate(prior_payload["active"], axis=0).astype(bool)
        baseline_diagnostics = baseline_payload["diagnostics"]
        prior_diagnostics = prior_payload["diagnostics"]
        paired_finite = np.asarray([
            not baseline_item["numerical_failure"] and not prior_item["numerical_failure"]
            for baseline_item, prior_item in zip(baseline_diagnostics, prior_diagnostics)
        ])
        if focus["enabled"]:
            layer = int(layer_key[1:])
            paired_finite &= core_mask(prior_truth, np.full(len(prior_truth), layer), focus)
        if np.any(paired_finite):
            baseline = float(
                evaluate_posterior(
                    baseline_samples[paired_finite],
                    prior_truth[paired_finite],
                    (prior_active & ~prior_known)[paired_finite],
                    joint_seed=seed,
                )["aggregate"]["mean_normalized_crps"]
            )
        else:
            baseline = float("inf")
            prior_guardrail_passed = False
        relative = (
            float((value - baseline) / max(abs(baseline), 1e-8))
            if math.isfinite(value) and math.isfinite(baseline)
            else float("inf")
        )
        prior_relative_changes[key] = relative
        reports[key]["matched_no_prior_normalized_crps"] = baseline
        if value > baseline * (1.0 + max_prior_regression_relative) + 1e-6:
            prior_guardrail_passed = False
    eligible = bool(
        math.isfinite(score)
        and coverage_error <= max_coverage_error_90
        and finite_fraction == 1.0
        and prior_guardrail_passed
    )
    return {
        "status": "completed",
        "thickness_focus": focus,
        "guard_scope": "core_thickness_and_N1" if focus["enabled"] else "all_thickness",
        "score": score,
        "score_components": {
            "no_prior_macro_normalized_crps": no_prior_score,
            "no_prior_by_layer": no_prior_scores,
            "prior_macro_normalized_crps": prior_score,
            "prior_macro_by_layer": prior_scores_by_layer,
            "prior_scores_by_layer_and_known_count": prior_scores_by_count,
            "weights": {
                "no_prior": no_prior_score_weight / score_weight_total,
                "prior": prior_score_weight / score_weight_total,
            },
        },
        "mean_coverage_error_90": coverage_error,
        "maximum_group_coverage_error_90": float(
            max(coverage_errors.values(), default=float("inf"))
        ),
        "coverage_errors_by_scenario": coverage_errors,
        "finite_fraction": finite_fraction,
        "eligible": eligible,
        "prior_guardrail": {
            "passed": prior_guardrail_passed,
            "maximum_allowed_relative": max_prior_regression_relative,
            "relative_changes_from_no_prior": prior_relative_changes,
        },
        "cases": int(case_count),
        "posterior_samples": sample_count,
        "solver_steps": solver_steps,
        "group_scores": group_scores,
        "scenarios": reports,
    }


def _largest_relative_violation(
    changes: dict[str, Any], maximum_allowed: float
) -> tuple[str, float, int] | None:
    violations: list[tuple[str, float]] = []
    for key, raw_value in changes.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > maximum_allowed:
            violations.append((str(key), value))
    if not violations:
        return None
    violations.sort(key=lambda item: item[1], reverse=True)
    key, value = violations[0]
    return key, value, len(violations)


def _posterior_progress_status(posterior: dict[str, Any]) -> str:
    regression = posterior.get("group_regression_check", {})
    if not regression.get("passed", True):
        violation = _largest_relative_violation(
            regression.get("relative_changes_from_selected_best", {}),
            float(regression.get("maximum_allowed_relative", 0.0)),
        )
        if violation is not None:
            key, relative, count = violation
            suffix = f";+{count - 1}" if count > 1 else ""
            return f"regressed:{key},{relative:+.1%}{suffix}"
        return "regressed"
    if posterior.get("eligible"):
        return "ok"
    finite_fraction = float(posterior.get("finite_fraction", 1.0))
    if finite_fraction < 1.0:
        return f"guarded:finite={finite_fraction:.1%}"
    prior_guardrail = posterior.get("prior_guardrail", {})
    if not prior_guardrail.get("passed", True):
        violation = _largest_relative_violation(
            prior_guardrail.get("relative_changes_from_no_prior", {}),
            float(prior_guardrail.get("maximum_allowed_relative", 0.0)),
        )
        if violation is not None:
            key, relative, count = violation
            suffix = f";+{count - 1}" if count > 1 else ""
            return f"guarded:prior:{key},{relative:+.1%}{suffix}"
        return "guarded:prior"
    coverage_error = posterior.get("mean_coverage_error_90")
    if coverage_error is not None:
        return f"guarded:coverage90={float(coverage_error):.3f}"
    return "guarded"


def _epoch_progress_line(
    record: dict[str, Any], total_epochs: int, patience: int
) -> str:
    posterior = record.get("posterior_validation")
    if posterior is not None and posterior.get("status") == "failed":
        posterior_text = "failed"
    elif posterior is not None and posterior.get("score") is not None:
        status = _posterior_progress_status(posterior)
        posterior_text = f"{float(posterior['score']):.5f}({status})"
        channels = posterior.get("by_channel_count", {})
        if "4" in channels and "3" in channels:
            posterior_text += f"[C4={channels['4']['score']:.5f},C3={channels['3']['score']:.5f}]"
    elif record.get("best_posterior_score") is not None:
        posterior_text = f"best:{float(record['best_posterior_score']):.5f}"
    else:
        posterior_text = "-"
    posterior_enabled = bool(record.get("posterior_validation_enabled", True))
    posterior_patience = int(record.get("posterior_patience_evaluations", 0))
    posterior_stale = int(record.get("posterior_stale_evaluations", 0))
    posterior_evaluations = int(record.get("posterior_evaluations", 0))
    ema_stale = int(record["stale_epochs"])
    lr_state = record.get("lr_schedule", {})
    if lr_state.get("phase") == "warmup":
        early_stop_text = "warmup"
    elif lr_state.get("early_stop_grace_remaining", 0) > 0:
        early_stop_text = f"lr_observe:{lr_state['early_stop_grace_remaining']}"
    elif ema_stale < patience:
        early_stop_text = f"ema_wait:{patience - ema_stale}"
    elif posterior_enabled and posterior_stale < posterior_patience:
        early_stop_text = f"posterior_wait:{posterior_patience - posterior_stale}"
    else:
        early_stop_text = "trigger"
    posterior_counter = (
        f" | post_stale={posterior_stale}/{posterior_patience}(eval={posterior_evaluations})"
        if posterior_enabled
        else " | post_stale=disabled"
    )
    return (
        (f"[{record['epoch_finished_at']}] " if "epoch_finished_at" in record else "")
        +
        f"Epoch {int(record['epoch']) + 1:03d}/{int(total_epochs):03d}"
        f" | train={float(record['train_loss']):.5f}"
        f" | ema_flow={float(record['validation_loss']):.5f}"
        f" | smooth={float(record['validation_loss_smoothed']):.5f}"
        f" | ncrps={posterior_text}"
        f" | ema_stale={ema_stale}/{int(patience)}"
        f"{posterior_counter}"
        f" | early_stop={early_stop_text}"
        f" | lr={float(record['learning_rate']):.3e}"
        f" | {float(record['epoch_seconds']):.1f}s"
    )


def train(
    config: dict[str, Any],
    resume: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
    init_weights: str = "raw",
) -> Path:
    session_started_at = timestamp()
    if resume is not None and init_checkpoint is not None:
        raise ValueError("resume and init_checkpoint are mutually exclusive")
    training_seed = config_seed(config, "training")
    validation_seed = config_seed(config, "validation")
    _seed_everything(training_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = use_bf16_amp(config["training"], device)
    index = _manifest(config)
    train_cfg = config["training"]
    focus = focus_policy(train_cfg.get("thickness_focus"))
    output = project_path(config, train_cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "metrics.jsonl"
    best_loss_path = output / "best_loss.pt"
    best_posterior_path = output / "best_posterior.pt"
    last_path = output / "last.pt"
    artifacts = (
        log_path,
        best_loss_path,
        best_posterior_path,
        last_path,
    )
    if resume is None and any(path.exists() for path in artifacts):
        raise FileExistsError(
            f"Training artifacts already exist in {output}. Use --resume to continue "
            "or choose a new --output-dir for a fresh experiment."
        )
    dataset_fingerprint = _dataset_fingerprint(index)
    train_data = H5TPSFDataset(
        index,
        "train",
        config["simulation"],
        config["data"]["sds_mm"],
        training_seed,
        train_cfg.get("max_train_samples"),
        paired_views=True,
    )
    focus_counts = configure_training_focus(train_data, focus)
    print(json.dumps({"event": "training_start", "timestamp": session_started_at,
                      "seed": training_seed, "output": str(output), "thickness_focus": focus,
                      "training_strata": focus_counts}), flush=True)
    val_data = H5TPSFDataset(
        index,
        "validation",
        config["simulation"],
        config["data"]["sds_mm"],
        validation_seed + 1,
        train_cfg.get("max_val_samples"),
        paired_views=True,
    )
    posterior_cfg = train_cfg.get("posterior_validation", {})
    posterior_enabled = bool(posterior_cfg.get("enabled", False))
    posterior_indices: list[int] = []
    if posterior_enabled:
        posterior_indices = _posterior_validation_indices(
            val_data,
            int(posterior_cfg.get("cases_per_layer", 64)),
            validation_seed
            + int(posterior_cfg.get("subset_seed_offset", 400_000)),
            focus=focus,
        )
        if not posterior_indices:
            train_data.close()
            val_data.close()
            raise ValueError("Posterior validation subset is empty")
    cache_started = time.perf_counter()
    try:
        validation_cache, validation_cache_bytes = _materialize_fixed_dataset(
            val_data
        )
    except Exception:
        train_data.close()
        raise
    validation_cache_seconds = time.perf_counter() - cache_started
    print(
        json.dumps(
            {
                "event": "validation_cache_ready",
                "cases": len(validation_cache),
                "tensor_mib": validation_cache_bytes / 1024**2,
                "elapsed_seconds": validation_cache_seconds,
            }
        ),
        flush=True,
    )
    batch_size = int(train_cfg["micro_batch_size"])
    effective_batch_size = int(train_cfg["effective_batch_size"])
    if batch_size <= 0 or effective_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if effective_batch_size < batch_size or effective_batch_size % batch_size != 0:
        raise ValueError(
            "effective_batch_size must be an integer multiple of micro_batch_size"
        )
    train_loader_args = {
        "batch_size": batch_size,
        "num_workers": int(train_cfg.get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    cached_loader_args = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, **train_loader_args)
    val_loader = DataLoader(validation_cache, shuffle=False, **cached_loader_args)
    posterior_loader: DataLoader | None = None
    if posterior_enabled:
        posterior_loader = DataLoader(
            Subset(validation_cache, posterior_indices),
            shuffle=False,
            **{
                **cached_loader_args,
                "batch_size": int(posterior_cfg.get("batch_size", 3)),
            },
        )
    model = ICAPModel(ModelConfig.from_mapping(config["model"])).to(device)
    initialization: dict[str, Any] | None = None
    if init_checkpoint is not None:
        if init_weights not in {"raw", "ema"}:
            raise ValueError("init_weights must be 'raw' or 'ema'")
        init_path = project_path(config, init_checkpoint).resolve()
        if not init_path.exists():
            raise FileNotFoundError(f"Initialization checkpoint does not exist: {init_path}")
        initial = torch.load(init_path, map_location="cpu", weights_only=False)
        initial_config = initial.get("config", {})
        if initial_config.get("model") != config["model"]:
            raise ValueError(
                "Initialization checkpoint model configuration is incompatible"
            )
        if initial_config.get("simulation") != config["simulation"]:
            raise ValueError(
                "Initialization checkpoint simulation configuration is incompatible"
            )
        for key in ("sds_mm", "time_bins", "time_bin_ps", "time_start_ps"):
            if initial_config.get("data", {}).get(key) != config["data"][key]:
                raise ValueError(
                    f"Initialization checkpoint data.{key} is incompatible"
                )
        state = initial["model"]
        if init_weights == "ema":
            if "ema" not in initial:
                raise ValueError("Initialization checkpoint does not contain EMA weights")
            state = {**state, **initial["ema"]}
        model.load_state_dict(state)
        initialization = {
            "checkpoint": str(init_path),
            "checkpoint_sha256": _sha256(init_path),
            "weights": init_weights,
            "source_epoch": initial.get("epoch"),
            "source_best_validation": initial.get("best_validation"),
        }
        print(
            json.dumps(
                {
                    "event": "warm_start",
                    **initialization,
                }
            ),
            flush=True,
        )
        del initial
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    accumulation = effective_batch_size // batch_size
    scheduler = WarmupPlateau(optimizer, train_cfg, math.ceil(len(train_loader) / accumulation))
    # BF16 has FP32's exponent range and must not use a dynamic loss scaler.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ema = ExponentialMovingAverage(
        model, decay=float(train_cfg.get("ema_decay", 0.9985))
    )
    sds = torch.tensor(config["data"]["sds_mm"], device=device, dtype=torch.float32)
    early_cfg = train_cfg["early_stopping"]
    patience = int(early_cfg["patience_epochs"])
    smoothing_window = int(early_cfg["smoothing_window"])
    min_delta_relative = float(early_cfg["min_delta_relative"])
    raw_validation_interval = int(early_cfg["raw_validation_interval_epochs"])
    if patience <= 0 or smoothing_window <= 0 or min_delta_relative < 0:
        raise ValueError("Invalid early-stopping configuration")
    if raw_validation_interval < 0:
        raise ValueError("raw_validation_interval_epochs cannot be negative")
    posterior_interval = int(posterior_cfg.get("interval_epochs", 10))
    posterior_start = int(posterior_cfg.get("start_epoch", posterior_interval))
    posterior_patience = int(posterior_cfg.get("patience_evaluations", 7))
    posterior_min_delta = float(
        posterior_cfg.get("min_delta_relative", min_delta_relative)
    )
    max_group_regression = float(
        posterior_cfg.get("max_group_regression_relative", 0.05)
    )
    no_prior_score_weight = float(
        posterior_cfg.get("no_prior_score_weight", 0.7)
    )
    prior_score_weight = float(posterior_cfg.get("prior_score_weight", 0.3))
    if posterior_enabled and (
        posterior_interval <= 0
        or posterior_start <= 0
        or posterior_patience <= 0
        or max_group_regression < 0
        or not math.isfinite(no_prior_score_weight)
        or not math.isfinite(prior_score_weight)
        or no_prior_score_weight < 0
        or prior_score_weight < 0
        or no_prior_score_weight + prior_score_weight <= 0
    ):
        raise ValueError("Invalid posterior-validation schedule")
    best_validation = float("inf")
    stale = 0
    monitor_history: list[float] = []
    best_posterior_score = float("inf")
    best_posterior_group_scores: dict[str, float] = {}
    posterior_stale = 0
    posterior_evaluations = 0
    global_step = 0
    skipped_optimizer_steps = 0
    start_epoch = 0
    if resume is not None:
        resume_path = last_path if str(resume) == "auto" else project_path(config, resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("resume_signature") != _resume_signature(config):
            raise ValueError(
                "The current configuration is incompatible with the resume checkpoint. "
                "Only num_workers and run_epochs may change when resuming."
            )
        if checkpoint.get("dataset_fingerprint") != dataset_fingerprint:
            raise ValueError(
                "The manifest or source HDF5 contents differ from the resume checkpoint"
            )
        required = {
            "model",
            "ema",
            "optimizer",
            "scheduler",
            "scaler",
            "rng_state",
            "next_epoch",
            "global_step",
            "best_validation",
            "stale_epochs",
            "monitor_history",
            "best_posterior_score",
            "best_posterior_group_scores",
            "posterior_stale_evaluations",
            "posterior_evaluations",
            "skipped_optimizer_steps",
            "metrics_record",
            "initialization",
        }
        missing = sorted(required - set(checkpoint))
        if missing:
            raise ValueError(f"Resume checkpoint is missing state: {missing}")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"], checkpoint["ema_decay"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["next_epoch"])
        global_step = int(checkpoint["global_step"])
        best_validation = float(checkpoint["best_validation"])
        stale = int(checkpoint["stale_epochs"])
        monitor_history = [float(value) for value in checkpoint["monitor_history"]]
        if not monitor_history:
            raise ValueError("Resume checkpoint has an empty monitor history")
        best_posterior_score = float(checkpoint["best_posterior_score"])
        best_posterior_group_scores = {
            str(key): float(value)
            for key, value in checkpoint["best_posterior_group_scores"].items()
        }
        posterior_stale = int(checkpoint["posterior_stale_evaluations"])
        posterior_evaluations = int(checkpoint["posterior_evaluations"])
        skipped_optimizer_steps = int(checkpoint["skipped_optimizer_steps"])
        initialization = checkpoint["initialization"]
        _restore_rng_state(checkpoint["rng_state"])
        _reconcile_metrics(log_path, checkpoint)
        print(
            json.dumps(
                {
                    "event": "resume",
                    "checkpoint": str(resume_path),
                    "next_epoch": start_epoch,
                    "global_step": global_step,
                    "best_validation": best_validation,
                    "stale_epochs": stale,
                    "best_posterior_score": best_posterior_score,
                    "posterior_stale_evaluations": posterior_stale,
                }
            )
        )
        del checkpoint
    optimizer.zero_grad(set_to_none=True)
    total_epochs = int(train_cfg["epochs"])
    run_epochs = int(train_cfg.get("run_epochs", total_epochs - start_epoch))
    if run_epochs <= 0:
        raise ValueError("run_epochs must be positive")
    end_epoch = min(total_epochs, start_epoch + run_epochs)
    if start_epoch >= total_epochs:
        raise ValueError(
            f"Checkpoint has already reached training.epochs={total_epochs}; "
            "increase epochs only by starting a new experiment"
        )
    posterior_plateaued = (
        posterior_evaluations >= posterior_patience
        and posterior_stale >= posterior_patience
    )
    if scheduler.can_stop and stale >= patience and (not posterior_enabled or posterior_plateaued):
        raise ValueError(
            f"Checkpoint already reached early stopping patience={patience}; "
            "there is no unfinished training state to resume"
        )
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
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            epoch_started = time.perf_counter()
            train_started = epoch_started
            train_data.set_epoch(epoch)
            model.train()
            epoch_loss_sum = 0.0
            epoch_sample_count = 0
            epoch_component_sums: dict[str, torch.Tensor] = {}
            for batch_index, raw in enumerate(train_loader):
                numerics.begin_batch(raw, batch_index, accumulation)
                batch = _device_batch(raw, device)
                batch_samples = int(batch["theta"].shape[0])
                group_start_batch = (batch_index // accumulation) * accumulation
                group_start_sample = group_start_batch * batch_size
                group_samples = min(
                    effective_batch_size, len(train_data) - group_start_sample
                )
                with torch.autocast(
                    device_type=device.type, enabled=amp_enabled, dtype=AUTOCAST_DTYPE
                ):
                    loss, loss_components = model.training_loss(
                        batch, sds, float(train_cfg["consistency_weight"])
                    )
                    # Weight every sample equally, including a partial final
                    # accumulation group in a filtered dataset.
                    scaled_loss = loss * (batch_samples / group_samples)
                if not torch.isfinite(scaled_loss).item():
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch {epoch + 1}, "
                        f"batch {batch_index + 1}; optimizer step was not applied."
                    )
                scaler.scale(scaled_loss).backward()
                update = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
                if update:
                    grad_norm = checked_optimizer_step(
                        model, optimizer, scaler, float(train_cfg["grad_clip"]),
                        epoch=epoch + 1, batch_index=batch_index + 1,
                        diagnostics=numerics.step_details,
                    )
                    numerics.observe(grad_norm, batch_index + 1)
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    ema.update(model)
                    global_step += 1
                epoch_loss_sum += float(loss.detach().cpu()) * batch_samples
                epoch_sample_count += batch_samples
                for name, value in loss_components.items():
                    contribution = value.detach().float() * batch_samples
                    if name in epoch_component_sums:
                        epoch_component_sums[name] += contribution
                    else:
                        epoch_component_sums[name] = contribution

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            numerical_record = numerics.finish()
            train_seconds = time.perf_counter() - train_started
            validation_started = time.perf_counter()
            raw_validation: dict[str, Any] | None = None
            if raw_validation_interval and (
                epoch == 0 or (epoch + 1) % raw_validation_interval == 0
            ):
                raw_validation = _run_validation(
                    model,
                    val_loader,
                    device,
                    sds,
                    float(train_cfg["consistency_weight"]),
                    amp_enabled,
                    validation_seed + 100_000,
                    focus=focus,
                )
            posterior_result: dict[str, Any] | None = None
            posterior_seconds = 0.0
            posterior_due = bool(
                posterior_enabled
                and posterior_loader is not None
                and epoch + 1 >= posterior_start
                and (epoch + 1 - posterior_start) % posterior_interval == 0
            )
            with ema.apply_to(model):
                ema_validation = _run_validation(
                    model,
                    val_loader,
                    device,
                    sds,
                    float(train_cfg["consistency_weight"]),
                    amp_enabled,
                    validation_seed + 100_000,
                    focus=focus,
                )
                if posterior_due:
                    posterior_started = time.perf_counter()
                    posterior_evaluations += 1
                    posterior_result = _run_posterior_validation(
                            model=model,
                            loader=posterior_loader,
                            device=device,
                            sds=sds,
                            amp_enabled=amp_enabled,
                            seed=validation_seed
                            + int(
                                posterior_cfg.get(
                                    "posterior_seed_offset", 500_000
                                )
                            ),
                            sample_count=int(
                                posterior_cfg.get("posterior_samples", 64)
                            ),
                            solver_steps=int(
                                posterior_cfg.get("solver_steps", 20)
                            ),
                            max_coverage_error_90=float(
                                posterior_cfg.get(
                                    "max_coverage_error_90", 0.15
                                )
                            ),
                            max_prior_regression_relative=max_group_regression,
                            no_prior_score_weight=no_prior_score_weight,
                            prior_score_weight=prior_score_weight,
                            focus=focus,
                        )
                    posterior_seconds = time.perf_counter() - posterior_started
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            validation_seconds = time.perf_counter() - validation_started
            validation = float(ema_validation["flow"])
            if not math.isfinite(validation):
                raise RuntimeError("EMA validation flow loss is not finite")
            monitor_history.append(validation)
            monitor_history = monitor_history[-smoothing_window:]
            smoothed_validation = float(np.mean(monitor_history))
            improved = bool(
                not math.isfinite(best_validation)
                or smoothed_validation
                < best_validation * (1.0 - min_delta_relative)
            )
            if improved:
                best_validation = smoothed_validation
                stale = 0
            else:
                stale += 1

            if not scheduler.warmup_complete:
                stale = 0
            lr_reduced = scheduler.step_metric(smoothed_validation)

            posterior_improved = False
            if posterior_result is not None:
                score_value = posterior_result.get("score")
                eligible = bool(posterior_result.get("eligible", False))
                candidate_group_scores = {
                    str(key): float(value)
                    for key, value in posterior_result.get(
                        "group_scores", {}
                    ).items()
                }
                regression_check = _posterior_group_regression_check(
                    candidate_group_scores,
                    best_posterior_group_scores,
                    max_group_regression,
                )
                posterior_result["group_regression_check"] = regression_check
                groups_preserved = bool(regression_check["passed"])
                if score_value is not None:
                    score_value = float(score_value)
                posterior_improved = bool(
                    eligible
                    and groups_preserved
                    and score_value is not None
                    and (
                        not math.isfinite(best_posterior_score)
                        or score_value
                        < best_posterior_score
                        * (1.0 - posterior_min_delta)
                    )
                )
                if posterior_improved:
                    best_posterior_score = float(score_value)
                    best_posterior_group_scores = candidate_group_scores
                    posterior_stale = 0
                else:
                    posterior_stale += 1

            epoch_seconds = time.perf_counter() - epoch_started
            record = {
                "epoch_started_at": epoch_started_at,
                "epoch_finished_at": timestamp(),
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": epoch_loss_sum / max(epoch_sample_count, 1),
                "numerics": numerical_record,
                "train_loss_components": {
                    name: float(value.cpu()) / max(epoch_sample_count, 1)
                    for name, value in epoch_component_sums.items()
                },
                "validation_loss": validation,
                "validation_loss_smoothed": smoothed_validation,
                "ema_validation": ema_validation,
                "raw_validation": raw_validation,
                "posterior_validation": posterior_result,
                "stale_epochs": stale,
                "posterior_stale_evaluations": posterior_stale,
                "posterior_evaluations": posterior_evaluations,
                "posterior_patience_evaluations": posterior_patience,
                "posterior_validation_enabled": posterior_enabled,
                "best_validation": best_validation,
                "best_posterior_score": (
                    best_posterior_score
                    if math.isfinite(best_posterior_score)
                    else None
                ),
                "best_posterior_group_scores": best_posterior_group_scores,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "lr_reduced": lr_reduced,
                "lr_schedule": scheduler.diagnostics(),
                "train_seconds": train_seconds,
                "validation_seconds": validation_seconds,
                "posterior_validation_seconds": posterior_seconds,
                "epoch_seconds": epoch_seconds,
                "train_samples_per_second": len(train_data) / max(train_seconds, 1e-9),
                "micro_batch_size": batch_size,
                "effective_batch_size": effective_batch_size,
                "gradient_accumulation_steps": accumulation,
                "skipped_optimizer_steps": skipped_optimizer_steps,
                "peak_cuda_allocated_gib": (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else None
                ),
                "peak_cuda_reserved_gib": (
                    torch.cuda.max_memory_reserved(device) / 1024**3
                    if device.type == "cuda"
                    else None
                ),
            }
            payload = _checkpoint_payload(
                config=config,
                dataset_fingerprint=dataset_fingerprint,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                amp_enabled=amp_enabled,
                epoch=epoch,
                global_step=global_step,
                best_validation=best_validation,
                validation_loss=validation,
                stale_epochs=stale,
                monitor_history=monitor_history,
                best_posterior_score=best_posterior_score,
                best_posterior_group_scores=best_posterior_group_scores,
                posterior_stale_evaluations=posterior_stale,
                posterior_evaluations=posterior_evaluations,
                skipped_optimizer_steps=skipped_optimizer_steps,
                metrics_record=record,
                initialization=initialization,
            )
            if improved:
                _atomic_torch_save(payload, best_loss_path)
            if posterior_improved:
                _atomic_torch_save(payload, best_posterior_path)
            _atomic_torch_save(payload, last_path)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(_epoch_progress_line(record, total_epochs, patience), flush=True)
            completed_epoch = epoch
            posterior_plateaued = (
                posterior_evaluations >= posterior_patience
                and posterior_stale >= posterior_patience
            )
            if scheduler.can_stop and stale >= patience and (
                not posterior_enabled or posterior_plateaued
            ):
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
    if completed_epoch + 1 >= total_epochs:
        stop_reason = "max_epochs"
    selected_best = (
        best_posterior_path
        if best_posterior_path.exists()
        else best_loss_path
        if best_loss_path.exists()
        else last_path
    )
    summary = {
        "session_started_at": session_started_at,
        "session_finished_at": timestamp(),
        "thickness_focus": focus,
        "training_strata": focus_counts,
        "status": "completed" if stop_reason != "run_epoch_limit" else "paused",
        "stop_reason": stop_reason,
        "completed_epoch": completed_epoch,
        "next_epoch": completed_epoch + 1,
        "configured_epochs": total_epochs,
        "best_checkpoint": str(selected_best),
        "best_loss_checkpoint": str(best_loss_path)
        if best_loss_path.exists()
        else None,
        "best_posterior_checkpoint": str(best_posterior_path)
        if best_posterior_path.exists()
        else None,
        "last_checkpoint": str(last_path),
        "best_validation": best_validation,
        "stale_epochs": stale,
        "early_stopping_patience_epochs": patience,
        "best_posterior_score": best_posterior_score
        if math.isfinite(best_posterior_score)
        else None,
        "best_posterior_group_scores": best_posterior_group_scores,
        "posterior_stale_evaluations": posterior_stale,
        "posterior_evaluations": posterior_evaluations,
        "validation_cache": {
            "cases": len(validation_cache),
            "tensor_bytes": validation_cache_bytes,
            "elapsed_seconds": validation_cache_seconds,
            "shared_by_ema_and_posterior_validation": posterior_enabled,
        },
        "initialization": initialization,
    }
    summary_path = output / "training_summary.json"
    temporary_summary = summary_path.with_name(f"{summary_path.name}.tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    temporary_summary.replace(summary_path)
    print(json.dumps({"event": "training_complete", **summary}), flush=True)
    return selected_best
