from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .data.dataset import waveform_features
from .model import ICAPModel, ModelConfig
from .parameters import CANONICAL_PARAMETERS, ParameterSchema
from .simulation import preprocess_observed


def validate_model_input_grid(
    config: dict[str, Any],
    dtof_counts: np.ndarray,
    irf_counts: np.ndarray,
    dt_ps: float,
) -> None:
    """Enforce the fixed temporal grid on which this checkpoint was trained."""
    expected_bins = int(config["data"]["time_bins"])
    expected_dt_ps = float(config["data"]["time_bin_ps"])
    if not np.isfinite(dt_ps) or not np.isclose(
        float(dt_ps), expected_dt_ps, rtol=1e-6, atol=1e-6
    ):
        raise ValueError(
            f"This checkpoint requires {expected_dt_ps:g} ps/bin; resample the "
            "DTOF and IRF before inference"
        )
    dtof = np.asarray(dtof_counts)
    irf = np.asarray(irf_counts)
    dtof_bins = int(dtof.shape[-1]) if dtof.ndim else 0
    irf_bins = int(irf.shape[-1]) if irf.ndim else 0
    if dtof_bins != expected_bins or irf_bins != expected_bins:
        raise ValueError(
            f"This checkpoint requires exactly {expected_bins} time bins; crop or "
            "resample the DTOF and IRF before inference"
        )


def load_checkpoint(
    path: str | Path,
    device: torch.device,
    weights: str = "ema",
) -> tuple[ICAPModel, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = payload["config"]
    if "channel_model" not in config.get("simulation", {}).get("irf", {}):
        raise ValueError(
            "Checkpoint predates the four-channel paired-IRF scientific protocol"
        )
    model = ICAPModel(ModelConfig.from_mapping(config["model"]))
    if weights not in {"raw", "ema"}:
        raise ValueError("weights must be 'raw' or 'ema'")
    state = payload["model"]
    if weights == "ema":
        if "ema" not in payload:
            raise ValueError("Checkpoint does not contain EMA weights")
        state = {**state, **payload["ema"]}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, config


@torch.no_grad()
def sample_native_posterior_batch(
    model: ICAPModel,
    signal: torch.Tensor,
    layer_count: torch.Tensor,
    active_mask: torch.Tensor,
    known_mask: torch.Tensor,
    known_normalized: torch.Tensor,
    known_physical: np.ndarray,
    sample_count: int,
    steps: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Draw the model's native posterior once, without rejection or clipping."""
    requested = int(sample_count)
    if requested <= 0:
        raise ValueError("sample_count must be positive")
    batch = int(signal.shape[0])
    if layer_count.shape != (batch,):
        raise ValueError("layer_count must have shape [batch]")
    for name, value in (
        ("active_mask", active_mask),
        ("known_mask", known_mask),
        ("known_normalized", known_normalized),
    ):
        if value.shape != (batch, len(CANONICAL_PARAMETERS)):
            raise ValueError(f"{name} must have shape [batch,8]")
    known_physical = np.asarray(known_physical, dtype=np.float32)
    if known_physical.shape != (batch, len(CANONICAL_PARAMETERS)):
        raise ValueError("known_physical must have shape [batch,8]")

    schema = ParameterSchema()
    normalized_batch = model.posterior.sample(
        signal,
        layer_count,
        active_mask,
        known_mask,
        known_normalized,
        sample_count=requested,
        steps=int(steps),
    ).float().cpu().numpy()
    samples = schema.denormalize(normalized_batch)
    active_np = active_mask.bool().cpu().numpy()
    known_np = known_mask.bool().cpu().numpy()
    for case_index in range(batch):
        samples[case_index][:, ~active_np[case_index]] = 0.0
        samples[case_index][:, known_np[case_index]] = known_physical[
            case_index, known_np[case_index]
        ]
    # Keep the decoder's native coordinates for support diagnostics.  A
    # physical round trip can underflow very small log-space optical values and
    # would otherwise hide genuine lower-support excursions.
    final_normalized_batch = normalized_batch.copy()
    for case_index in range(batch):
        final_normalized_batch[case_index][:, ~active_np[case_index]] = 0.0
        known_values = known_normalized[case_index].float().cpu().numpy()
        final_normalized_batch[case_index][:, known_np[case_index]] = known_values[
            known_np[case_index]
        ]

    diagnostics: list[dict[str, Any]] = []
    for case_index in range(batch):
        active = active_np[case_index]
        normalized = final_normalized_batch[case_index]
        physical = samples[case_index]
        finite_dimensions = np.isfinite(normalized) & np.isfinite(physical)
        finite_rows = np.all((~active) | finite_dimensions, axis=-1)
        outside_dimensions = active & finite_dimensions & (
            (normalized < -1.0) | (normalized > 1.0)
        )
        outside_rows = finite_rows & np.any(outside_dimensions, axis=-1)
        layer = int(layer_count[case_index].item())
        study_rows = (
            finite_rows & (physical[:, 0] + physical[:, 1] > 35.0)
            if layer == 3
            else np.zeros(requested, dtype=bool)
        )
        impossible_dimensions = active & finite_dimensions & (physical <= 0.0)
        impossible_rows = finite_rows & np.any(impossible_dimensions, axis=-1)
        lower_violations = np.sum(
            active & finite_dimensions & (normalized < -1.0), axis=0
        )
        upper_violations = np.sum(
            active & finite_dimensions & (normalized > 1.0), axis=0
        )
        exceedance = np.maximum(np.abs(normalized) - 1.0, 0.0)
        active_exceedance = exceedance[:, active]
        positive_exceedance = active_exceedance[active_exceedance > 0.0]
        diagnostics.append(
            {
                "requested_samples": requested,
                "generated_samples": requested,
                "finite_samples": int(finite_rows.sum()),
                "finite_fraction": float(finite_rows.mean()),
                "outside_training_support_samples": int(outside_rows.sum()),
                "outside_training_support_fraction": float(outside_rows.mean()),
                "outside_study_constraint_samples": int(study_rows.sum()),
                "outside_study_constraint_fraction": float(study_rows.mean()),
                "physically_impossible_samples": int(impossible_rows.sum()),
                "physically_impossible_fraction": float(impossible_rows.mean()),
                "nonfinite_samples": int((~finite_rows).sum()),
                "numerical_failure": bool(np.any(~finite_rows)),
                "lower_bound_violations": {
                    CANONICAL_PARAMETERS[index]: int(lower_violations[index])
                    for index in np.flatnonzero(active_np[case_index])
                },
                "upper_bound_violations": {
                    CANONICAL_PARAMETERS[index]: int(upper_violations[index])
                    for index in np.flatnonzero(active_np[case_index])
                },
                "normalized_exceedance": {
                    "median": float(np.median(positive_exceedance))
                    if positive_exceedance.size
                    else 0.0,
                    "q90": float(np.quantile(positive_exceedance, 0.9))
                    if positive_exceedance.size
                    else 0.0,
                    "maximum": float(np.max(positive_exceedance))
                    if positive_exceedance.size
                    else 0.0,
                },
            }
        )
    return samples.astype(np.float32), diagnostics


def _posterior_summary(samples: np.ndarray, active: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index in np.flatnonzero(active):
        values = samples[:, index]
        result[CANONICAL_PARAMETERS[index]] = {
            "median": float(np.median(values)),
            "q05": float(np.quantile(values, 0.05)),
            "q10": float(np.quantile(values, 0.10)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "q90": float(np.quantile(values, 0.90)),
            "q95": float(np.quantile(values, 0.95)),
        }
    return result


def _correlation_matrix(values: np.ndarray) -> list[list[float]]:
    """Compute a small correlation matrix without invoking a BLAS backend."""
    values = np.asarray(values, dtype=np.float64)
    width = values.shape[1]
    if width == 1:
        return [[1.0]]
    centered = values - np.mean(values, axis=0, keepdims=True)
    square_sums = np.sum(centered * centered, axis=0)
    result = np.zeros((width, width), dtype=np.float64)
    for row in range(width):
        for column in range(width):
            denominator = float(np.sqrt(square_sums[row] * square_sums[column]))
            if denominator > 0.0:
                result[row, column] = float(
                    np.sum(centered[:, row] * centered[:, column]) / denominator
                )
            elif row == column:
                result[row, column] = 1.0
    return result.tolist()


@torch.no_grad()
def infer_raw_counts(
    model: ICAPModel,
    config: dict[str, Any],
    dtof_counts: np.ndarray,
    irf_counts: np.ndarray,
    dt_ps: float,
    layer_count: int,
    known_values: Mapping[str, float],
    sample_count: int | None = None,
    seed: int | None = None,
    sds_mm: list[float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if seed is not None:
        torch.manual_seed(int(seed))
    device = next(model.parameters()).device
    from .channels import SDS
    supplied = list(SDS) if sds_mm is None else list(map(float, sds_mm))
    if not supplied or len(supplied) != len(set(supplied)) or any(s not in SDS for s in supplied):
        raise ValueError("sds_mm must be a nonempty unique subset of [10,20,30,40]")
    if np.asarray(dtof_counts).ndim != 2 or len(dtof_counts) != len(supplied):
        raise ValueError("Each DTOF row must match one supplied SDS")
    validate_model_input_grid(config, dtof_counts, irf_counts, dt_ps)
    prepared = preprocess_observed(
        dtof_counts,
        irf_counts,
        dt_ps,
        config["simulation"]["preprocessing"],
    )
    slots = [SDS.index(s) for s in supplied]
    dtof_array = np.zeros((4, prepared.dtof.shape[-1]), dtype=np.float32)
    irf_array = np.zeros_like(dtof_array)
    dtof_array[slots], irf_array[slots] = prepared.dtof, prepared.irf
    channel_mask = torch.zeros((1, 4), dtype=torch.bool, device=device)
    channel_mask[:, slots] = True
    dtof = torch.from_numpy(waveform_features(dtof_array))[None].to(device)
    irf = torch.from_numpy(waveform_features(irf_array))[None].to(device)
    schema = ParameterSchema()
    active_np = schema.active_mask(layer_count)
    known_mask_np, known_physical = schema.known_arrays(layer_count, known_values)
    known_normalized = np.where(
        known_mask_np, schema.normalize(known_physical), 0.0
    ).astype(np.float32)
    active = torch.from_numpy(active_np)[None].to(device)
    known_mask = torch.from_numpy(known_mask_np)[None].to(device)
    known = torch.from_numpy(known_normalized)[None].to(device)
    layers = torch.tensor([layer_count], device=device, dtype=torch.long)
    sds = torch.tensor(config["data"]["sds_mm"], device=device, dtype=torch.float32)
    signal = model.encode_view({"dtof": dtof, "irf": irf, "channel_mask": channel_mask}, sds)
    requested = int(sample_count or config["inference"]["posterior_samples"])
    if requested <= 0:
        raise ValueError("sample_count must be positive")
    samples_batch, batch_diagnostics = sample_native_posterior_batch(
        model=model,
        signal=signal,
        layer_count=layers,
        active_mask=active,
        known_mask=known_mask,
        known_normalized=known,
        known_physical=known_physical[None, :],
        sample_count=requested,
        steps=int(config["inference"]["solver_steps"]),
    )
    samples = samples_batch[0]
    diagnostics = batch_diagnostics[0]
    finite_rows = np.all((~active_np) | np.isfinite(samples), axis=1)
    finite_samples = samples[finite_rows]
    active_values = finite_samples[:, active_np]
    correlation = (
        _correlation_matrix(active_values)
        if finite_samples.shape[0] >= 2
        else []
    )
    metadata = {
        "status": "numerical_failure"
        if diagnostics["numerical_failure"]
        else "completed",
        "layer_count": layer_count,
        "sds_mm": supplied,
        "channel_mask": channel_mask[0].cpu().tolist(),
        "active_parameters": [
            CANONICAL_PARAMETERS[index] for index in np.flatnonzero(active_np)
        ],
        "known_parameters": sorted(known_values),
        "summary": _posterior_summary(finite_samples, active_np)
        if finite_samples.size
        else {},
        "correlation": correlation,
        "preprocessing": prepared.metadata,
        "requested_samples": requested,
        "sampling_diagnostics": diagnostics,
    }
    return samples, metadata


def save_inference(
    output: str | Path, samples: np.ndarray, metadata: dict[str, Any]
) -> tuple[Path, Path]:
    base = Path(output)
    base.parent.mkdir(parents=True, exist_ok=True)
    npz_path = base.with_suffix(".npz")
    json_path = base.with_suffix(".json")
    np.savez_compressed(npz_path, posterior_samples=samples)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return npz_path, json_path
