from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import platform
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import config_seed, project_path
from .data.dataset import H5TPSFDataset, waveform_features
from .evaluation import (
    evaluate_casewise_posterior,
    evaluate_casewise_scalar_posterior,
    evaluate_posterior,
    evaluate_scalar_posterior,
)
from .evaluation_protocol import (
    PriorScenario,
    detect_candidate_modes,
    enumerate_prior_scenarios,
    is_deep_multimodality_scenario,
    scenario_mask,
    spearman_with_bootstrap,
    thickness_group_labels,
)
from .inference import load_checkpoint, sample_native_posterior_batch
from .parameters import CANONICAL_PARAMETERS, ParameterSchema
from .precision import AUTOCAST_DTYPE, PRECISION_PROTOCOL, use_bf16_amp
from .reporting import render_test_report
from .simulation import simulate_measured_irf
from .training import _dataset_fingerprint, _manifest


DIAGNOSTIC_COLUMNS = (
    "finite_fraction",
    "outside_training_support_fraction",
    "outside_study_constraint_fraction",
    "physically_impossible_fraction",
    "nonfinite_fraction",
)
EXCEEDANCE_COLUMNS = ("median", "q90", "maximum")
SAMPLE_STATUS_BITS = {
    "nonfinite": 1,
    "outside_training_support": 2,
    "outside_study_constraint": 4,
    "physically_impossible": 8,
}
N3_SURFACE_DEPTH_NAME = "d1_plus_d2_mm"
N3_SURFACE_DEPTH_BOUNDS_MM = (1.0, 35.0)
PRIOR_INTERVENTION_MODES = ("shuffled_prior", "mask_only")
IRF_INTERVENTION_MODES = (
    "shuffled_irf",
    "fixed_irf",
    "zero_irf",
    "channel_permuted_irf",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_scientific_config(
    config: dict[str, Any], checkpoint_config: dict[str, Any]
) -> None:
    for key in ("model", "simulation"):
        if config[key] != checkpoint_config[key]:
            raise ValueError(f"Current {key} configuration differs from the checkpoint")
    for key in ("sds_mm", "time_bins", "time_bin_ps", "time_start_ps"):
        if config["data"][key] != checkpoint_config["data"][key]:
            raise ValueError(f"Current data.{key} differs from the checkpoint configuration")


def _group_metrics(
    samples: np.ndarray,
    truth: np.ndarray,
    active: np.ndarray,
    known: np.ndarray,
    layers: np.ndarray,
    thickness_groups: np.ndarray | None = None,
    *,
    joint_seed: int = 0,
) -> dict[str, Any]:
    unknown = active & ~known
    selected_all = np.ones(len(samples), dtype=bool)

    def evaluate_group(selected: np.ndarray, seed: int) -> dict[str, Any]:
        result = evaluate_posterior(
            samples[selected], truth[selected], unknown[selected], joint_seed=seed
        )
        surface_depth_unknown = (
            (layers[selected] == 3)
            & ~(known[selected, 0] & known[selected, 1])
        )
        if np.any(surface_depth_unknown):
            selected_samples = samples[selected][surface_depth_unknown]
            selected_truth = truth[selected][surface_depth_unknown]
            result["parameters"][N3_SURFACE_DEPTH_NAME] = evaluate_scalar_posterior(
                selected_samples[:, :, 0] + selected_samples[:, :, 1],
                selected_truth[:, 0] + selected_truth[:, 1],
                N3_SURFACE_DEPTH_BOUNDS_MM,
            )
        return result

    groups: dict[str, Any] = {
        "all_unknown_parameters": evaluate_group(selected_all, joint_seed),
        "by_layer_count": {},
        "by_known_count": {},
        "by_thickness_group": {},
    }
    for layer in (1, 2, 3):
        selected = layers == layer
        if np.any(selected):
            groups["by_layer_count"][str(layer)] = evaluate_group(
                selected, joint_seed + layer
            )
    counts = known.sum(axis=1)
    for count in sorted(np.unique(counts).tolist()):
        selected = counts == count
        groups["by_known_count"][str(int(count))] = evaluate_group(
            selected, joint_seed + int(count)
        )
    if thickness_groups is not None:
        for label in sorted(set(thickness_groups) - {"not_applicable"}):
            selected = thickness_groups == label
            if np.any(selected):
                groups["by_thickness_group"][str(label)] = evaluate_group(
                    selected, joint_seed
                )
    return groups


def _diagnostic_row(item: dict[str, Any]) -> list[float]:
    return [
        float(item["finite_fraction"]),
        float(item["outside_training_support_fraction"]),
        float(item["outside_study_constraint_fraction"]),
        float(item["physically_impossible_fraction"]),
        float(item["nonfinite_samples"]) / max(int(item["generated_samples"]), 1),
    ]


def _diagnostic_parameter_rows(
    item: dict[str, Any], sample_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray(
        [item["lower_bound_violations"].get(name, 0) for name in CANONICAL_PARAMETERS],
        dtype=np.int32,
    )
    upper = np.asarray(
        [item["upper_bound_violations"].get(name, 0) for name in CANONICAL_PARAMETERS],
        dtype=np.int32,
    )
    exceedance = np.asarray(
        [item["normalized_exceedance"][name] for name in EXCEEDANCE_COLUMNS],
        dtype=np.float32,
    )
    if np.any(lower < 0) or np.any(upper < 0) or np.any(lower > sample_count) or np.any(upper > sample_count):
        raise ValueError("Invalid per-parameter support violation counts")
    return lower, upper, exceedance


def _sample_status_flags(
    samples: np.ndarray, active: np.ndarray, layers: np.ndarray
) -> np.ndarray:
    schema = ParameterSchema()
    normalized = schema.normalize(samples)
    active_3d = np.broadcast_to(active[:, None, :], samples.shape)
    finite_dimensions = np.isfinite(samples) & np.isfinite(normalized)
    finite_rows = np.all((~active_3d) | finite_dimensions, axis=2)
    flags = np.zeros(samples.shape[:2], dtype=np.uint8)
    flags[~finite_rows] |= SAMPLE_STATUS_BITS["nonfinite"]
    outside = finite_rows & np.any(
        active_3d & finite_dimensions & ((normalized < -1.0) | (normalized > 1.0)),
        axis=2,
    )
    flags[outside] |= SAMPLE_STATUS_BITS["outside_training_support"]
    impossible = finite_rows & np.any(
        active_3d & finite_dimensions & (samples <= 0.0), axis=2
    )
    flags[impossible] |= SAMPLE_STATUS_BITS["physically_impossible"]
    study = finite_rows & (layers[:, None] == 3) & (
        samples[:, :, 0] + samples[:, :, 1] > 35.0
    )
    flags[study] |= SAMPLE_STATUS_BITS["outside_study_constraint"]
    return flags


def _diagnostics_from_matrix(matrix: np.ndarray) -> dict[str, Any]:
    return {
        "case_count": int(matrix.shape[0]),
        **{
            name: float(np.mean(matrix[:, index]))
            for index, name in enumerate(DIAGNOSTIC_COLUMNS)
        },
        "numerical_failure_case_indices": np.flatnonzero(matrix[:, 4] > 0.0).tolist(),
    }


def _bootstrap_mean(values: np.ndarray, seed: int, replicates: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(values[rng.integers(0, len(values), size=len(values))]))
        for _ in range(int(replicates))
    ]
    return np.quantile(means, [0.025, 0.975]).tolist()


def _named_prior_scenarios(
    scenarios: list[PriorScenario],
) -> list[PriorScenario]:
    """Return the non-empty paper scenarios used for prior interventions."""
    return [
        scenario
        for scenario in scenarios
        if scenario.known_count > 0 and bool(scenario.aliases)
    ]


def _deterministic_prior_donors(
    layers: np.ndarray, seed: int
) -> np.ndarray:
    """Build one reproducible, within-layer derangement for all prior masks."""
    layers = np.asarray(layers, dtype=np.int64)
    donors = np.full(len(layers), -1, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    for layer in sorted(set(map(int, layers))):
        indices = np.flatnonzero(layers == layer)
        if len(indices) < 2:
            continue
        shuffled = rng.permutation(indices)
        donors[shuffled] = np.roll(shuffled, -1)
    return donors


def _deterministic_channel_permutations(case_count: int, seed: int) -> np.ndarray:
    """Return reproducible four-channel cyclic permutations without fixed points."""
    rng = np.random.default_rng(int(seed))
    shifts = rng.integers(1, 4, size=int(case_count))
    channels = np.arange(4, dtype=np.int64)
    return np.stack([np.roll(channels, int(shift)) for shift in shifts])


def _fixed_irf_medoid(
    config: dict[str, Any],
    dt_ps: float,
    output_bins: int,
    seed: int,
    candidate_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select a deterministic, physically sampled IRF closest to median shape."""
    if candidate_count < 2:
        raise ValueError("Fixed-IRF medoid requires at least two candidates")
    rng = np.random.default_rng(int(seed))
    fixed_config = deepcopy(config)
    fixed_config["simulation"]["irf"]["channel_model"]["mode_probabilities"] = {
        "all_shared": 0.0,
        "four_similar": 1.0,
        "grouped_2_2": 0.0,
        "grouped_3_1": 0.0,
        "partial_outlier": 0.0,
        "heterogeneous": 0.0,
    }
    curves: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    grid = np.linspace(0, output_bins - 1, min(128, output_bins))
    source_grid = np.arange(output_bins)
    descriptors: list[np.ndarray] = []
    for _ in range(int(candidate_count)):
        curve, item = simulate_measured_irf(
            dt_ps,
            output_bins,
            fixed_config["simulation"]["irf"],
            fixed_config["simulation"]["preprocessing"],
            rng,
        )
        curves.append(curve)
        metadata.append(item)
        sampled = np.stack(
            [np.interp(grid, source_grid, channel) for channel in curve]
        )
        channels = item["irf_channels"]
        descriptors.append(
            np.concatenate(
                (
                    (
                        sampled
                        / np.maximum(sampled.max(axis=1, keepdims=True), 1e-12)
                    ).reshape(-1),
                    np.asarray(
                        [
                            np.mean(
                                [np.log1p(float(value["actual_fwhm_ps"])) for value in channels]
                            ),
                            np.mean([float(value["tail_fraction"]) for value in channels]),
                            np.mean(
                                [
                                    sum(
                                        float(artifact["fraction"])
                                        for artifact in value["artifacts"]
                                    )
                                    for value in channels
                                ]
                            ),
                            np.mean(
                                [np.log1p(float(value["measured_counts"])) for value in channels]
                            ),
                            np.mean(
                                [np.log1p(float(value["background_fraction"])) for value in channels]
                            ),
                            float(item["irf_delay_range_ps"]),
                            float(item["irf_fwhm_cv"]),
                            float(item["irf_pairwise_distance_mean"]),
                        ]
                    ),
                )
            )
        )
    matrix = np.stack(descriptors)
    center = np.median(matrix, axis=0)
    scale = np.median(np.abs(matrix - center), axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    distance = np.mean(((matrix - center) / scale) ** 2, axis=1)
    selected = int(np.argmin(distance))
    curve = curves[selected]
    details = {
        **metadata[selected],
        "candidate_count": int(candidate_count),
        "selected_candidate_index": selected,
        "selection": "closest standardized descriptor to componentwise median",
        "required_channel_mode": "four_similar",
        "seed": int(seed),
        "sha256": hashlib.sha256(curve.tobytes()).hexdigest(),
    }
    return waveform_features(curve), details


def _quartile_labels(values: np.ndarray, prefix: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    edges = np.quantile(values, [0.25, 0.5, 0.75])
    bins = np.searchsorted(edges, values, side="right")
    return np.asarray([f"{prefix}_Q{value + 1}" for value in bins], dtype=object)


def _irf_strata(metadata: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not metadata:
        return {}

    def any_artifact(item: dict[str, Any], key: str) -> str:
        return "present" if any(bool(channel[key]) for channel in item["irf_channels"]) else "absent"

    delay = np.asarray([float(item["irf_delay_range_ps"]) for item in metadata])
    fwhm_cv = np.asarray([float(item["irf_fwhm_cv"]) for item in metadata])
    distance = np.asarray(
        [float(item["irf_pairwise_distance_mean"]) for item in metadata]
    )
    return {
        "channel_mode": np.asarray(
            [item["irf_channel_mode"] for item in metadata], dtype=object
        ),
        "channel_delay_range": np.asarray(
            [
                "[0,100) ps" if value < 100.0 else
                "[100,300) ps" if value < 300.0 else
                "[300,600) ps" if value < 600.0 else
                "[600,+inf) ps"
                for value in delay
            ],
            dtype=object,
        ),
        "channel_fwhm_cv": np.asarray(
            [
                "[0,0.10)" if value < 0.10 else
                "[0.10,0.25)" if value < 0.25 else
                "[0.25,0.50)" if value < 0.50 else
                "[0.50,+inf)"
                for value in fwhm_cv
            ],
            dtype=object,
        ),
        "waveform_heterogeneity": _quartile_labels(distance, "heterogeneity"),
        "anomaly_channel_count": np.asarray(
            [str(int(item["irf_anomaly_count"])) for item in metadata], dtype=object
        ),
        "shoulder": np.asarray(
            [any_artifact(item, "has_shoulder") for item in metadata], dtype=object
        ),
        "delayed_secondary_peak": np.asarray(
            [any_artifact(item, "has_satellite") for item in metadata], dtype=object
        ),
        "prepulse": np.asarray(
            [any_artifact(item, "has_prepulse") for item in metadata], dtype=object
        ),
        "echo_train": np.asarray(
            [any_artifact(item, "has_echo_train") for item in metadata], dtype=object
        ),
        "complex_tail": np.asarray(
            [
                "present"
                if any(
                    channel["family"] in {"double_tail", "stretched_tail"}
                    for channel in item["irf_channels"]
                )
                else "absent"
                for item in metadata
            ],
            dtype=object,
        ),
    }


def _flatten_irf_metadata(
    metadata: list[dict[str, Any]],
    sample_ids: list[str],
    layers: np.ndarray,
    sds_mm: list[float],
) -> list[dict[str, Any]]:
    """Create an auditable one-row-per-case-and-channel metadata table."""
    rows: list[dict[str, Any]] = []
    for case_index, (item, sample_id) in enumerate(zip(metadata, sample_ids)):
        group_ids = item["irf_group_ids"]
        anomaly_mask = item["irf_anomaly_mask"]
        for channel_index, channel in enumerate(item["irf_channels"]):
            artifacts = channel.get("artifacts", [])
            rows.append(
                {
                    "global_case_index": case_index,
                    "sample_id": sample_id,
                    "layer_count": int(layers[case_index]),
                    "channel_index": channel_index,
                    "sds_mm": float(sds_mm[channel_index]),
                    "channel_mode": item["irf_channel_mode"],
                    "group_id": int(group_ids[channel_index]),
                    "is_anomaly": bool(anomaly_mask[channel_index]),
                    "family": channel["family"],
                    "actual_fwhm_ps": channel["actual_fwhm_ps"],
                    "system_delay_ps": channel["system_delay_ps"],
                    "tail_fraction": channel["tail_fraction"],
                    "tail_tau_ps": channel["tail_tau_ps"],
                    "second_tail_tau_ps": channel["second_tail_tau_ps"],
                    "artifact_types": ";".join(
                        str(artifact["kind"]) for artifact in artifacts
                    ),
                    "artifact_metadata_json": json.dumps(
                        artifacts, sort_keys=True, separators=(",", ":")
                    ),
                    "measured_counts": channel["measured_counts"],
                    "background_fraction": channel["background_fraction"],
                    "measured_peak_before_alignment_ps": channel[
                        "measured_peak_before_alignment_ps"
                    ],
                    "peak_after_preprocessing_ps": channel[
                        "peak_after_preprocessing_ps"
                    ],
                    "estimated_alignment_shift_ps": channel[
                        "estimated_alignment_shift_ps"
                    ],
                    "applied_alignment_shift_ps": channel[
                        "applied_alignment_shift_ps"
                    ],
                    "true_irf_sha256": channel["true_irf_sha256"],
                    "bundle_delay_range_ps": item["irf_delay_range_ps"],
                    "bundle_fwhm_cv": item["irf_fwhm_cv"],
                    "bundle_pairwise_correlation_mean": item[
                        "irf_pairwise_correlation_mean"
                    ],
                    "bundle_pairwise_distance_mean": item[
                        "irf_pairwise_distance_mean"
                    ],
                }
            )
    return rows


def _prior_values_for_mode(
    mode: str,
    known: np.ndarray,
    target_indices: np.ndarray,
    donor_indices: np.ndarray,
    normalized_truth: np.ndarray,
    truth: np.ndarray,
    neutral_physical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "shuffled_prior":
        if np.any(donor_indices < 0):
            raise ValueError("Shuffled-prior intervention requires a donor for every case")
        source_normalized = normalized_truth[donor_indices]
        source_physical = truth[donor_indices]
    elif mode == "mask_only":
        source_normalized = np.zeros((len(target_indices), truth.shape[1]), dtype=np.float32)
        source_physical = np.broadcast_to(
            np.asarray(neutral_physical, dtype=np.float32), source_normalized.shape
        )
    else:
        raise ValueError(f"Unsupported prior intervention mode: {mode}")
    return (
        np.where(known, source_normalized, 0.0).astype(np.float32),
        np.where(known, source_physical, 0.0).astype(np.float32),
    )


def _representative_case_indices(
    samples: np.ndarray,
    truth: np.ndarray,
    unknown: np.ndarray,
    outside_fraction: np.ndarray,
    thickness_groups: np.ndarray,
    limit: int,
) -> np.ndarray:
    """Select deterministic high-error, wide, domain-outside, and bin examples."""
    if limit <= 0 or not len(samples):
        return np.asarray([], dtype=np.int64)
    schema = ParameterSchema()
    normalized = schema.normalize(samples)
    normalized_truth = schema.normalize(truth)
    median = np.median(normalized, axis=1)
    q05 = np.quantile(normalized, 0.05, axis=1)
    q95 = np.quantile(normalized, 0.95, axis=1)
    errors = np.sum(np.abs(median - normalized_truth) * unknown, axis=1) / np.maximum(unknown.sum(axis=1), 1)
    widths = np.sum((q95 - q05) * unknown, axis=1) / np.maximum(unknown.sum(axis=1), 1)
    selected: list[int] = []

    def add(index: int) -> None:
        if index not in selected and len(selected) < limit:
            selected.append(index)

    quota = max(1, limit // 4)
    for score in (errors, widths, outside_fraction):
        for index in np.argsort(-score, kind="stable")[:quota]:
            add(int(index))
    for label in sorted(set(map(str, thickness_groups)) - {"not_applicable"}):
        candidates = np.flatnonzero(thickness_groups == label)
        if len(candidates):
            add(int(candidates[np.argmax(errors[candidates])]))
    for index in np.argsort(-errors, kind="stable"):
        add(int(index))
    return np.asarray(selected, dtype=np.int64)


def _energy_rows(
    samples: np.ndarray, truth: np.ndarray, unknown: np.ndarray
) -> np.ndarray:
    schema = ParameterSchema()
    normalized = schema.normalize(samples) / 2.0
    target = schema.normalize(truth) / 2.0
    result = np.full(len(samples), np.nan)
    for case in range(len(samples)):
        mask = unknown[case]
        dimension = int(mask.sum())
        if not dimension:
            continue
        draws = normalized[case][:, mask]
        scale = np.sqrt(dimension)
        first = np.linalg.norm(draws - target[case, mask], axis=1).mean() / scale
        second = np.linalg.norm(draws - np.roll(draws, 1, axis=0), axis=1).mean() / scale
        result[case] = first - 0.5 * second
    return result


def _irf_difficulty_rows(
    samples: np.ndarray,
    truth: np.ndarray,
    active: np.ndarray,
    layers: np.ndarray,
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize paired-input posterior accuracy by four-channel IRF difficulty."""
    casewise = evaluate_casewise_posterior(samples, truth, active)
    energy = _energy_rows(samples, truth, active)
    rows: list[dict[str, Any]] = []
    for stratum_type, labels in _irf_strata(metadata).items():
        for stratum_value in sorted(set(map(str, labels))):
            in_stratum = labels == stratum_value
            for layer in (1, 2, 3):
                selected_cases = in_stratum & (layers == layer)
                if int(selected_cases.sum()) < 4:
                    continue
                layer_parameters = np.flatnonzero(
                    np.any(active[selected_cases], axis=0)
                )
                for parameter_index in layer_parameters:
                    selected = selected_cases & active[:, parameter_index]
                    rows.append(
                        {
                            "stratum_type": stratum_type,
                            "stratum_value": stratum_value,
                            "layer_count": layer,
                            "parameter": CANONICAL_PARAMETERS[parameter_index],
                            "case_count": int(selected.sum()),
                            "mean_normalized_mae": float(
                                np.mean(
                                    casewise["normalized_absolute_error"][
                                        selected, parameter_index
                                    ]
                                )
                            ),
                            "mean_normalized_crps": float(
                                np.mean(
                                    casewise["normalized_crps"][
                                        selected, parameter_index
                                    ]
                                )
                            ),
                            "mean_normalized_width_90": float(
                                np.mean(
                                    casewise["normalized_width_90"][
                                        selected, parameter_index
                                    ]
                                )
                            ),
                        }
                    )
                selected_active = active[selected_cases]
                denominator = np.maximum(selected_active.sum(axis=1), 1)
                macro_nmae = np.sum(
                    np.where(
                        selected_active,
                        casewise["normalized_absolute_error"][selected_cases],
                        0.0,
                    ),
                    axis=1,
                ) / denominator
                macro_ncrps = np.sum(
                    np.where(
                        selected_active,
                        casewise["normalized_crps"][selected_cases],
                        0.0,
                    ),
                    axis=1,
                ) / denominator
                macro_width = np.sum(
                    np.where(
                        selected_active,
                        casewise["normalized_width_90"][selected_cases],
                        0.0,
                    ),
                    axis=1,
                ) / denominator
                rows.append(
                    {
                        "stratum_type": stratum_type,
                        "stratum_value": stratum_value,
                        "layer_count": layer,
                        "parameter": "__macro__",
                        "case_count": int(selected_cases.sum()),
                        "mean_normalized_mae": float(np.mean(macro_nmae)),
                        "mean_normalized_crps": float(np.mean(macro_ncrps)),
                        "mean_normalized_width_90": float(np.mean(macro_width)),
                        "mean_energy_score": float(np.mean(energy[selected_cases])),
                    }
                )
    return rows


def _paired_prior_rows(
    scenario: PriorScenario,
    samples: np.ndarray,
    baseline: np.ndarray,
    truth: np.ndarray,
    active: np.ndarray,
    seed: int,
    bootstrap_replicates: int,
) -> list[dict[str, Any]]:
    known = scenario_mask(scenario, len(truth))
    unknown = active & ~known
    current = evaluate_casewise_posterior(samples, truth, unknown)
    reference = evaluate_casewise_posterior(baseline, truth, unknown)
    current_energy = _energy_rows(samples, truth, unknown)
    reference_energy = _energy_rows(baseline, truth, unknown)
    rows = []
    for index in np.flatnonzero(np.any(unknown, axis=0)):
        selected = unknown[:, index]
        current_values = {
            "normalized_mae": current["normalized_absolute_error"][selected, index],
            "normalized_crps": current["normalized_crps"][selected, index],
            "normalized_width_90": current["normalized_width_90"][selected, index],
        }
        reference_values = {
            "normalized_mae": reference["normalized_absolute_error"][selected, index],
            "normalized_crps": reference["normalized_crps"][selected, index],
            "normalized_width_90": reference["normalized_width_90"][selected, index],
        }
        deltas = {
            f"delta_{name}": current_values[name] - reference_values[name]
            for name in current_values
        }
        row: dict[str, Any] = {
            "scenario_id": scenario.scenario_id,
            "aliases": ";".join(scenario.aliases),
            "layer_count": scenario.layer_count,
            "known_count": scenario.known_count,
            "known_parameters": scenario.known_parameters,
            "unknown_parameters": scenario.unknown_parameters,
            "prior_type": scenario.prior_type,
            "optical_type": scenario.optical_type,
            "parameter": CANONICAL_PARAMETERS[index],
            "cases": int(selected.sum()),
        }
        for offset, (name, values) in enumerate(deltas.items()):
            row[name] = float(np.nanmean(values))
            row[f"{name}_ci95"] = _bootstrap_mean(
                values, seed + index * 100 + offset, bootstrap_replicates
            )
        for name in current_values:
            current_mean = float(np.nanmean(current_values[name]))
            reference_mean = float(np.nanmean(reference_values[name]))
            row[f"current_{name}"] = current_mean
            row[f"baseline_{name}"] = reference_mean
            row[f"relative_improvement_{name}"] = (
                (reference_mean - current_mean) / reference_mean
                if reference_mean > 0.0
                else float("nan")
            )
        row["improved_case_fraction"] = float(
            np.mean(deltas["delta_normalized_crps"] < 0.0)
        )
        current_miss = 1.0 - current["covered_90"][selected, index]
        reference_miss = 1.0 - reference["covered_90"][selected, index]
        row["overconfident_case_fraction"] = float(np.nanmean(current_miss))
        row["baseline_overconfident_case_fraction"] = float(np.nanmean(reference_miss))
        row["delta_overconfident_case_fraction"] = float(
            np.nanmean(current_miss - reference_miss)
        )
        rows.append(row)
    if scenario.layer_count == 3 and not np.all(known[:, 0] & known[:, 1]):
        current_surface = evaluate_casewise_scalar_posterior(
            samples[:, :, 0] + samples[:, :, 1],
            truth[:, 0] + truth[:, 1],
            N3_SURFACE_DEPTH_BOUNDS_MM,
        )
        reference_surface = evaluate_casewise_scalar_posterior(
            baseline[:, :, 0] + baseline[:, :, 1],
            truth[:, 0] + truth[:, 1],
            N3_SURFACE_DEPTH_BOUNDS_MM,
        )
        current_values = {
            "normalized_mae": current_surface["normalized_absolute_error"],
            "normalized_crps": current_surface["normalized_crps"],
            "normalized_width_90": current_surface["normalized_width_90"],
        }
        reference_values = {
            "normalized_mae": reference_surface["normalized_absolute_error"],
            "normalized_crps": reference_surface["normalized_crps"],
            "normalized_width_90": reference_surface["normalized_width_90"],
        }
        deltas = {
            f"delta_{name}": current_values[name] - reference_values[name]
            for name in current_values
        }
        surface_row: dict[str, Any] = {
            "scenario_id": scenario.scenario_id,
            "aliases": ";".join(scenario.aliases),
            "layer_count": scenario.layer_count,
            "known_count": scenario.known_count,
            "known_parameters": scenario.known_parameters,
            "unknown_parameters": scenario.unknown_parameters,
            "prior_type": scenario.prior_type,
            "optical_type": scenario.optical_type,
            "parameter": N3_SURFACE_DEPTH_NAME,
            "cases": len(truth),
        }
        for offset, (name, values) in enumerate(deltas.items()):
            surface_row[name] = float(np.mean(values))
            surface_row[f"{name}_ci95"] = _bootstrap_mean(
                values, seed + 8_000 + offset, bootstrap_replicates
            )
        for name in current_values:
            current_mean = float(np.mean(current_values[name]))
            reference_mean = float(np.mean(reference_values[name]))
            surface_row[f"current_{name}"] = current_mean
            surface_row[f"baseline_{name}"] = reference_mean
            surface_row[f"relative_improvement_{name}"] = (
                (reference_mean - current_mean) / reference_mean
                if reference_mean > 0.0
                else float("nan")
            )
        surface_row["improved_case_fraction"] = float(
            np.mean(deltas["delta_normalized_crps"] < 0.0)
        )
        current_miss = 1.0 - current_surface["covered_90"]
        reference_miss = 1.0 - reference_surface["covered_90"]
        surface_row["overconfident_case_fraction"] = float(
            np.mean(current_miss)
        )
        surface_row["baseline_overconfident_case_fraction"] = float(
            np.mean(reference_miss)
        )
        surface_row["delta_overconfident_case_fraction"] = float(
            np.mean(current_miss - reference_miss)
        )
        rows.append(surface_row)
    joint_delta = current_energy - reference_energy
    current_energy_mean = float(np.nanmean(current_energy))
    reference_energy_mean = float(np.nanmean(reference_energy))
    rows.append(
        {
            "scenario_id": scenario.scenario_id,
            "aliases": ";".join(scenario.aliases),
            "layer_count": scenario.layer_count,
            "known_count": scenario.known_count,
            "known_parameters": scenario.known_parameters,
            "unknown_parameters": scenario.unknown_parameters,
            "prior_type": scenario.prior_type,
            "optical_type": scenario.optical_type,
            "parameter": "__joint__",
            "cases": int(np.isfinite(joint_delta).sum()),
            "delta_energy_score": float(np.nanmean(joint_delta)),
            "current_energy_score": current_energy_mean,
            "baseline_energy_score": reference_energy_mean,
            "relative_improvement_energy_score": (
                (reference_energy_mean - current_energy_mean) / reference_energy_mean
                if reference_energy_mean > 0.0
                else float("nan")
            ),
            "delta_energy_score_ci95": _bootstrap_mean(
                joint_delta, seed + 9_999, bootstrap_replicates
            ),
            "improved_case_fraction": float(np.nanmean(joint_delta < 0.0)),
        }
    )
    return rows


def _prior_intervention_comparison_rows(
    scenario: PriorScenario,
    candidate_samples: np.ndarray,
    reference_samples: np.ndarray,
    truth: np.ndarray,
    active: np.ndarray,
    candidate_diagnostics: np.ndarray,
    reference_diagnostics: np.ndarray,
    candidate_condition: str,
    reference_condition: str,
    seed: int,
    bootstrap_replicates: int,
) -> list[dict[str, Any]]:
    """Compare two paired conditions on the same cases and unknown dimensions."""
    rows = _paired_prior_rows(
        scenario,
        candidate_samples,
        reference_samples,
        truth,
        active,
        seed,
        bootstrap_replicates,
    )
    known = scenario_mask(scenario, len(truth))
    unknown = active & ~known
    candidate = evaluate_casewise_posterior(candidate_samples, truth, unknown)
    reference = evaluate_casewise_posterior(reference_samples, truth, unknown)
    counts = np.maximum(unknown.sum(axis=1), 1)

    def case_mean(values: np.ndarray) -> np.ndarray:
        return np.sum(np.where(unknown, values, 0.0), axis=1) / counts

    candidate_values = {
        "normalized_mae": case_mean(candidate["normalized_absolute_error"]),
        "normalized_crps": case_mean(candidate["normalized_crps"]),
        "normalized_width_90": case_mean(candidate["normalized_width_90"]),
    }
    reference_values = {
        "normalized_mae": case_mean(reference["normalized_absolute_error"]),
        "normalized_crps": case_mean(reference["normalized_crps"]),
        "normalized_width_90": case_mean(reference["normalized_width_90"]),
    }
    macro: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "aliases": ";".join(scenario.aliases),
        "layer_count": scenario.layer_count,
        "known_count": scenario.known_count,
        "known_parameters": scenario.known_parameters,
        "unknown_parameters": scenario.unknown_parameters,
        "prior_type": scenario.prior_type,
        "optical_type": scenario.optical_type,
        "parameter": "__macro__",
        "cases": len(truth),
    }
    for offset, name in enumerate(candidate_values):
        delta = candidate_values[name] - reference_values[name]
        candidate_mean = float(np.nanmean(candidate_values[name]))
        reference_mean = float(np.nanmean(reference_values[name]))
        macro[f"delta_{name}"] = float(np.nanmean(delta))
        macro[f"delta_{name}_ci95"] = _bootstrap_mean(
            delta, seed + 20_000 + offset, bootstrap_replicates
        )
        macro[f"current_{name}"] = candidate_mean
        macro[f"baseline_{name}"] = reference_mean
        macro[f"relative_improvement_{name}"] = (
            (reference_mean - candidate_mean) / reference_mean
            if reference_mean > 0.0
            else float("nan")
        )
    macro["improved_case_fraction"] = float(
        np.mean(
            candidate_values["normalized_crps"]
            < reference_values["normalized_crps"]
        )
    )
    candidate_miss = 1.0 - case_mean(candidate["covered_90"])
    reference_miss = 1.0 - case_mean(reference["covered_90"])
    macro["overconfident_case_fraction"] = float(np.nanmean(candidate_miss))
    macro["baseline_overconfident_case_fraction"] = float(
        np.nanmean(reference_miss)
    )
    macro["delta_overconfident_case_fraction"] = float(
        np.nanmean(candidate_miss - reference_miss)
    )
    candidate_energy = _energy_rows(candidate_samples, truth, unknown)
    reference_energy = _energy_rows(reference_samples, truth, unknown)
    energy_delta = candidate_energy - reference_energy
    candidate_energy_mean = float(np.nanmean(candidate_energy))
    reference_energy_mean = float(np.nanmean(reference_energy))
    macro["delta_energy_score"] = float(np.nanmean(energy_delta))
    macro["delta_energy_score_ci95"] = _bootstrap_mean(
        energy_delta, seed + 29_999, bootstrap_replicates
    )
    macro["current_energy_score"] = candidate_energy_mean
    macro["baseline_energy_score"] = reference_energy_mean
    macro["relative_improvement_energy_score"] = (
        (reference_energy_mean - candidate_energy_mean) / reference_energy_mean
        if reference_energy_mean > 0.0
        else float("nan")
    )
    rows.append(macro)

    comparison_id = f"{candidate_condition}_vs_{reference_condition}"
    for row in rows:
        row["comparison_id"] = comparison_id
        row["candidate_condition"] = candidate_condition
        row["reference_condition"] = reference_condition
        for column, name in enumerate(DIAGNOSTIC_COLUMNS):
            candidate_mean = float(np.mean(candidate_diagnostics[:, column]))
            reference_mean = float(np.mean(reference_diagnostics[:, column]))
            row[f"candidate_{name}"] = candidate_mean
            row[f"reference_{name}"] = reference_mean
            row[f"delta_{name}"] = candidate_mean - reference_mean
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not columns:
            return
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict, tuple)) else value for key, value in row.items()})


def _write_mode_svg(
    path: Path,
    normalized_samples: np.ndarray,
    unknown_mask: np.ndarray,
    mode: dict[str, Any],
    title: str,
) -> None:
    """Write a dependency-free joint scatter plus every unknown marginal."""
    indices = np.flatnonzero(unknown_mask)
    values = np.asarray(normalized_samples, dtype=np.float64)[:, indices]
    values = values[np.all(np.isfinite(values), axis=1)]
    if not len(values) or not len(indices):
        return
    centers_array = np.asarray(mode.get("centers_normalized", []), dtype=np.float64)
    if centers_array.ndim == 1 and centers_array.size:
        centers_array = centers_array[None, :]
    weights = np.asarray(mode.get("weights", []), dtype=np.float64)
    palette = ("#b24c63", "#c89020", "#527a68", "#765aa6")
    width = 1160
    marginal_rows = math.ceil(len(indices) / 2)
    height = max(470, 115 + marginal_rows * 118)
    left, top, plot_w, plot_h = 68, 76, 450, 310
    first = values[:, 0]
    second = values[:, 1] if values.shape[1] > 1 else np.zeros(len(values))
    x_min, x_max = np.quantile(first, [0.005, 0.995])
    y_min, y_max = np.quantile(second, [0.005, 0.995]) if values.shape[1] > 1 else (0.0, 1.0)
    if len(centers_array):
        x_min = min(float(x_min), float(np.min(centers_array[:, 0])))
        x_max = max(float(x_max), float(np.max(centers_array[:, 0])))
        if values.shape[1] > 1:
            y_min = min(float(y_min), float(np.min(centers_array[:, 1])))
            y_max = max(float(y_max), float(np.max(centers_array[:, 1])))
    if x_max <= x_min:
        x_min, x_max = x_min - 0.5, x_max + 0.5
    if y_max <= y_min:
        y_min, y_max = y_min - 0.5, y_max + 0.5
    marks: list[str] = []
    if values.shape[1] > 1:
        for row in values[:: max(1, len(values) // 900)]:
            x = left + plot_w * np.clip((row[0] - x_min) / (x_max - x_min), 0, 1)
            y = top + plot_h * (1 - np.clip((row[1] - y_min) / (y_max - y_min), 0, 1))
            marks.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.4" fill="#2774ae" opacity="0.35"/>')
    else:
        counts, _ = np.histogram(first, bins=36, range=(x_min, x_max))
        maximum = max(int(counts.max()), 1)
        for index, count in enumerate(counts):
            x = left + plot_w * index / len(counts)
            bar_w = plot_w / len(counts)
            bar_h = plot_h * count / maximum
            marks.append(f'<rect x="{x:.2f}" y="{top + plot_h - bar_h:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#2774ae" opacity="0.65"/>')
    center_marks: list[str] = []
    for component, (center, weight) in enumerate(zip(centers_array, weights)):
        color = palette[component % len(palette)]
        x = left + plot_w * np.clip((center[0] - x_min) / (x_max - x_min), 0, 1)
        if values.shape[1] > 1:
            y = top + plot_h * (1 - np.clip((center[1] - y_min) / (y_max - y_min), 0, 1))
            center_marks.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{color}" stroke="white" stroke-width="2"><title>M{component + 1}, weight={weight:.3f}</title></circle>')
        else:
            center_marks.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top + plot_h}" stroke="{color}" stroke-width="3"><title>M{component + 1}, weight={weight:.3f}</title></line>')

    marginal_marks: list[str] = []
    panel_w, panel_h = 255, 72
    for offset, parameter_index in enumerate(indices):
        panel_x = 585 + (offset % 2) * 280
        panel_y = 82 + (offset // 2) * 118
        series = values[:, offset]
        low, high = np.quantile(series, [0.005, 0.995])
        if len(centers_array):
            low = min(float(low), float(np.min(centers_array[:, offset])))
            high = max(float(high), float(np.max(centers_array[:, offset])))
        if high <= low:
            low, high = low - 0.5, high + 0.5
        counts, _ = np.histogram(series, bins=32, range=(low, high))
        maximum = max(int(counts.max()), 1)
        bars = []
        for bin_index, count in enumerate(counts):
            bar_x = panel_x + panel_w * bin_index / len(counts)
            bar_width = panel_w / len(counts) + 0.2
            bar_height = panel_h * count / maximum
            bars.append(
                f'<rect x="{bar_x:.2f}" y="{panel_y + panel_h - bar_height:.2f}" '
                f'width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#80acd7" opacity="0.78"/>'
            )
        center_lines = []
        for component, center in enumerate(centers_array):
            center_x = panel_x + panel_w * np.clip(
                (center[offset] - low) / (high - low), 0, 1
            )
            color = palette[component % len(palette)]
            center_lines.append(
                f'<line x1="{center_x:.2f}" x2="{center_x:.2f}" '
                f'y1="{panel_y}" y2="{panel_y + panel_h}" stroke="{color}" stroke-width="2.5"/>'
            )
        marginal_marks.append(
            f'<text x="{panel_x}" y="{panel_y - 9}" font-family="sans-serif" '
            f'font-size="12" font-weight="600">{html.escape(CANONICAL_PARAMETERS[parameter_index])}</text>'
            f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" '
            f'fill="#f7f9fc" stroke="#cbd5e1"/>{"".join(bars)}{"".join(center_lines)}'
            f'<text x="{panel_x}" y="{panel_y + panel_h + 15}" font-family="sans-serif" font-size="10">{low:.2f}</text>'
            f'<text x="{panel_x + panel_w}" y="{panel_y + panel_h + 15}" text-anchor="end" font-family="sans-serif" font-size="10">{high:.2f}</text>'
        )
    y_label = CANONICAL_PARAMETERS[indices[1]] if len(indices) > 1 else "frequency"
    legend = "".join(
        f'<circle cx="{585 + component * 112}" cy="43" r="5" fill="{palette[component % len(palette)]}"/>'
        f'<text x="{595 + component * 112}" y="47" font-family="sans-serif" font-size="11">M{component + 1}: {weight:.1%}</text>'
        for component, weight in enumerate(weights)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="30" y="30" font-family="sans-serif" font-size="18" font-weight="600">{html.escape(title)}</text>
{legend}<text x="585" y="65" font-family="sans-serif" font-size="11">marginal posterior; colored lines are GMM centers</text>
<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#f7f9fc" stroke="#9aa7b4"/>{''.join(marks)}{''.join(center_marks)}
<text x="{left + plot_w / 2}" y="{top + plot_h + 32}" text-anchor="middle" font-family="sans-serif">{html.escape(CANONICAL_PARAMETERS[indices[0]])} (normalized)</text>
<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" text-anchor="middle" font-family="sans-serif">{html.escape(y_label)} (normalized)</text>
<text x="{left}" y="{top + plot_h + 58}" font-family="sans-serif" font-size="11">components={mode.get('component_count')} · stability={mode.get('bootstrap_stability', 0):.1%} · min separation={mode.get('minimum_standardized_separation', 0):.2f}</text>
{''.join(marginal_marks)}</svg>'''
    path.write_text(svg, encoding="utf-8")


def _aggregate_prior_effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("layer_count"), row.get("known_count"), row.get("prior_type"),
            row.get("optical_type"), row.get("parameter"),
        )
        grouped[key].append(row)
    result = []
    metrics = (
        "delta_normalized_mae", "delta_normalized_crps",
        "delta_normalized_width_90", "delta_energy_score",
        "relative_improvement_normalized_mae",
        "relative_improvement_normalized_crps",
        "relative_improvement_normalized_width_90",
        "relative_improvement_energy_score",
        "improved_case_fraction",
    )
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        row = {
            "layer_count": key[0], "known_count": key[1], "prior_type": key[2],
            "optical_type": key[3], "parameter": key[4], "mask_count": len(values),
        }
        for metric in metrics:
            observed = [float(value[metric]) for value in values if value.get(metric) is not None and np.isfinite(value[metric])]
            if observed:
                row[f"macro_mean_{metric}"] = float(np.mean(observed))
        result.append(row)
    return result


def _aggregate_intervention_effects(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("comparison_id"), row.get("candidate_condition"),
            row.get("reference_condition"), row.get("layer_count"),
            row.get("known_count"), row.get("prior_type"),
            row.get("optical_type"), row.get("parameter"),
            row.get("stratum_type", "all"), row.get("stratum_value", "all"),
        )
        grouped[key].append(row)
    metrics = (
        "delta_normalized_mae", "delta_normalized_crps",
        "delta_normalized_width_90", "delta_energy_score",
        "relative_improvement_normalized_mae",
        "relative_improvement_normalized_crps",
        "relative_improvement_normalized_width_90",
        "relative_improvement_energy_score", "improved_case_fraction",
        "delta_overconfident_case_fraction", "candidate_finite_fraction",
        "candidate_outside_training_support_fraction",
        "candidate_outside_study_constraint_fraction",
        "candidate_physically_impossible_fraction",
    )
    result: list[dict[str, Any]] = []
    for key, values in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        row = {
            "comparison_id": key[0], "candidate_condition": key[1],
            "reference_condition": key[2], "layer_count": key[3],
            "known_count": key[4], "prior_type": key[5],
            "optical_type": key[6], "parameter": key[7],
            "stratum_type": key[8], "stratum_value": key[9],
            "scenario_count": len(values),
        }
        for metric in metrics:
            observed = [
                float(value[metric]) for value in values
                if value.get(metric) is not None and np.isfinite(value[metric])
            ]
            if observed:
                row[f"macro_mean_{metric}"] = float(np.mean(observed))
        result.append(row)
    return result


def _flatten_metric_rows(
    scenario: PriorScenario, metrics: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parameter_rows: list[dict[str, Any]] = []
    thickness_rows: list[dict[str, Any]] = []
    groups = [("all", metrics["all_unknown_parameters"])]
    groups.extend((f"layer_N={key}", value) for key, value in metrics["by_layer_count"].items())
    groups.extend((f"thickness={key}", value) for key, value in metrics["by_thickness_group"].items())
    for group_name, report in groups:
        for parameter, values in report["parameters"].items():
            row = {
                "scenario_id": scenario.scenario_id,
                "aliases": ";".join(scenario.aliases),
                "layer_count": scenario.layer_count,
                "known_count": scenario.known_count,
                "prior_type": scenario.prior_type,
                "group": group_name,
                "parameter": parameter,
                **values,
            }
            parameter_rows.append(row)
            if group_name.startswith("thickness="):
                thickness_rows.append(row)
    return parameter_rows, thickness_rows


def _prepare_store(
    path: Path,
    signature: str,
    truth: np.ndarray,
    active: np.ndarray,
    layers: np.ndarray,
    sample_ids: list[str],
    thickness_groups: np.ndarray,
) -> h5py.File:
    store = h5py.File(path, "a")
    existing = store.attrs.get("evaluation_signature")
    if existing is not None and str(existing) != signature:
        store.close()
        raise ValueError("Existing posterior.h5 belongs to an incompatible evaluation")
    store.attrs["evaluation_signature"] = signature
    if "cases" not in store:
        cases = store.create_group("cases")
        cases.create_dataset("truth", data=truth)
        cases.create_dataset("active_mask", data=active)
        cases.create_dataset("layer_count", data=layers)
        string = h5py.string_dtype("utf-8")
        cases.create_dataset("sample_id", data=np.asarray(sample_ids, dtype=object), dtype=string)
        cases.create_dataset("thickness_group", data=np.asarray(thickness_groups, dtype=object), dtype=string)
    store.require_group("scenarios")
    store.require_group("prior_interventions")
    store.require_group("irf_interventions")
    return store


@torch.no_grad()
def run_test(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    weights: str | None = None,
    max_test_samples: int | None = None,
) -> Path:
    """Run the deterministic exhaustive native-posterior test protocol."""
    started = time.perf_counter()
    checkpoint = project_path(config, checkpoint_path).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    cfg = config["evaluation"]
    time_alignment_enabled = bool(
        config["simulation"]["preprocessing"]["time_alignment_enabled"]
    )
    irf_intervention_stage = (
        "after independent channel-paired IRF time alignment"
        if time_alignment_enabled
        else "after paired preprocessing without time alignment"
    )
    if cfg.get("prior_protocol", "exhaustive_exact") != "exhaustive_exact":
        raise ValueError("evaluation.prior_protocol must be 'exhaustive_exact'")
    selected_weights = str(weights or cfg.get("weights", "ema"))
    requested_device = str(cfg.get("device", "auto"))
    device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available() else
        "cpu" if requested_device == "auto" else requested_device
    )
    model, checkpoint_config = load_checkpoint(checkpoint, device, selected_weights)
    _validate_scientific_config(config, checkpoint_config)
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    index = _manifest(config)
    fingerprint = _dataset_fingerprint(index)
    expected = checkpoint_payload.get("dataset_fingerprint")
    if expected is not None and fingerprint != expected:
        raise ValueError("The current manifest or HDF5 data differs from the checkpoint")

    seed = config_seed(config, "evaluation")
    observation_seed = seed + int(cfg.get("observation_seed_offset", 200_000))
    posterior_seed = seed + int(cfg.get("posterior_seed_offset", 300_000))
    requested_max = max_test_samples if max_test_samples is not None else cfg.get("max_test_samples")
    dataset = H5TPSFDataset(
        index, "test", config["simulation"], config["data"]["sds_mm"],
        observation_seed, requested_max, paired_views=False,
    )
    dataset.set_epoch(0)
    batch_size = int(cfg.get("batch_size", 3))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    sds = torch.tensor(config["data"]["sds_mm"], device=device, dtype=torch.float32)
    amp_enabled = use_bf16_amp(cfg, device)
    signals: list[np.ndarray] = []
    dtofs: list[np.ndarray] = []
    irfs: list[np.ndarray] = []
    irf_metadata: list[dict[str, Any]] = []
    time_axis_reference: np.ndarray | None = None
    truths: list[np.ndarray] = []
    active_masks: list[np.ndarray] = []
    layer_counts: list[np.ndarray] = []
    sample_ids: list[str] = []
    model.eval()
    try:
        for raw in loader:
            dtof_cpu = raw["view_a"]["dtof"]
            irf_cpu = raw["view_a"]["irf"]
            if time_axis_reference is None:
                time_axis_reference = (
                    raw["view_a"]["time_ps"][0].numpy().astype(np.float64)
                )
            with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=AUTOCAST_DTYPE):
                encoded = model.encode_view(
                    {"dtof": dtof_cpu.to(device), "irf": irf_cpu.to(device)},
                    sds,
                )
            signals.append(encoded.float().cpu().numpy())
            dtofs.append(dtof_cpu.numpy().astype(np.float32))
            irfs.append(irf_cpu.numpy().astype(np.float32))
            batch_metadata = raw["view_a"]["irf_metadata_json"]
            irf_metadata.extend(json.loads(str(value)) for value in batch_metadata)
            truths.append(raw["theta_physical"].numpy().astype(np.float32))
            active_masks.append(raw["active_mask"].numpy().astype(bool))
            layer_counts.append(raw["layer_count"].numpy().astype(np.int64))
            sample_ids.extend(map(str, raw["sample_id"]))
    finally:
        dataset.close()
    signal_all = np.concatenate(signals)
    dtof_all = np.concatenate(dtofs)
    irf_all = np.concatenate(irfs)
    truth_all = np.concatenate(truths)
    active_all = np.concatenate(active_masks)
    layers_all = np.concatenate(layer_counts)
    thickness_groups = thickness_group_labels(truth_all, layers_all)
    present_layers = set(int(value) for value in np.unique(layers_all))
    scenarios = [
        scenario
        for layer in (1, 2, 3)
        if layer in present_layers
        for scenario in enumerate_prior_scenarios(layer)
    ]
    intervention_cfg = cfg.get("prior_interventions", {})
    interventions_enabled = bool(intervention_cfg.get("enabled", True))
    intervention_scope = str(intervention_cfg.get("scope", "named"))
    if intervention_scope != "named":
        raise ValueError("evaluation.prior_interventions.scope must be 'named'")
    configured_intervention_modes = tuple(
        str(value)
        for value in intervention_cfg.get("modes", PRIOR_INTERVENTION_MODES)
    )
    if len(set(configured_intervention_modes)) != len(configured_intervention_modes) or any(
        value not in PRIOR_INTERVENTION_MODES
        for value in configured_intervention_modes
    ):
        raise ValueError(
            "evaluation.prior_interventions.modes must contain unique values from "
            f"{PRIOR_INTERVENTION_MODES}"
        )
    if interventions_enabled and set(configured_intervention_modes) != set(
        PRIOR_INTERVENTION_MODES
    ):
        raise ValueError(
            "Enabled prior interventions require both shuffled_prior and mask_only"
        )
    intervention_modes = (
        configured_intervention_modes if interventions_enabled else ()
    )
    intervention_scenarios = (
        _named_prior_scenarios(scenarios) if interventions_enabled else []
    )
    irf_cfg = cfg.get("irf_interventions", {})
    model_variant = str(checkpoint_config.get("model", {}).get("variant", "explicit_irf"))
    irf_interventions_enabled = bool(irf_cfg.get("enabled", False))
    if model_variant != "explicit_irf":
        irf_interventions_enabled = False
    irf_scope = str(irf_cfg.get("scope", "no_prior_and_named"))
    if irf_scope != "no_prior_and_named":
        raise ValueError(
            "evaluation.irf_interventions.scope must be 'no_prior_and_named'"
        )
    configured_irf_modes = tuple(
        str(value)
        for value in irf_cfg.get("modes", IRF_INTERVENTION_MODES)
    )
    if len(set(configured_irf_modes)) != len(configured_irf_modes) or any(
        value not in IRF_INTERVENTION_MODES for value in configured_irf_modes
    ):
        raise ValueError(
            "evaluation.irf_interventions.modes must contain unique values from "
            f"{IRF_INTERVENTION_MODES}"
        )
    if irf_interventions_enabled and set(configured_irf_modes) != set(
        IRF_INTERVENTION_MODES
    ):
        raise ValueError(
            "Enabled IRF interventions require shuffled_irf, fixed_irf, zero_irf, "
            "and channel_permuted_irf"
        )
    irf_modes = configured_irf_modes if irf_interventions_enabled else ()
    irf_scenarios = (
        [scenario for scenario in scenarios if scenario.known_count == 0]
        + _named_prior_scenarios(scenarios)
        if irf_interventions_enabled
        else []
    )
    irf_seed = seed + int(irf_cfg.get("shuffle_seed_offset", 600_000))
    fixed_irf_seed = seed + int(irf_cfg.get("fixed_seed_offset", 700_000))
    irf_donor_global = _deterministic_prior_donors(layers_all, irf_seed)
    irf_channel_permutations = _deterministic_channel_permutations(
        len(layers_all), irf_seed + 1
    )
    fixed_irf_features: np.ndarray | None = None
    fixed_irf_metadata: dict[str, Any] = {}
    if irf_interventions_enabled:
        if time_axis_reference is None:
            raise ValueError("Cannot construct fixed IRF for an empty test set")
        dt_ps = float(np.median(np.diff(time_axis_reference)))
        fixed_irf_features, fixed_irf_metadata = _fixed_irf_medoid(
            config,
            dt_ps,
            int(irf_all.shape[-1]),
            fixed_irf_seed,
            int(irf_cfg.get("fixed_candidate_count", 4096)),
        )
        fixed_irf_metadata["dt_ps"] = dt_ps
    irf_signal_by_mode: dict[str, np.ndarray] = {}
    for mode in irf_modes:
        if mode == "shuffled_irf":
            if np.any(irf_donor_global < 0):
                raise ValueError(
                    "Shuffled-IRF intervention requires at least two cases per layer"
                )
            intervention_irf = irf_all[irf_donor_global]
        elif mode == "fixed_irf":
            if fixed_irf_features is None:
                raise RuntimeError("Fixed IRF was not constructed")
            intervention_irf = np.broadcast_to(
                fixed_irf_features[None, ...], irf_all.shape
            )
        elif mode == "zero_irf":
            intervention_irf = np.zeros_like(irf_all)
        elif mode == "channel_permuted_irf":
            intervention_irf = np.stack(
                [
                    irf_all[index, irf_channel_permutations[index]]
                    for index in range(len(irf_all))
                ]
            )
        else:  # pragma: no cover - configuration validation owns this branch.
            raise ValueError(f"Unsupported IRF intervention mode: {mode}")
        encoded_batches: list[np.ndarray] = []
        for start in range(0, len(dtof_all), batch_size):
            stop = min(start + batch_size, len(dtof_all))
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
                dtype=AUTOCAST_DTYPE,
            ):
                encoded = model.encode_view(
                    {
                        "dtof": torch.from_numpy(dtof_all[start:stop]).to(device),
                        "irf": torch.from_numpy(
                            np.asarray(intervention_irf[start:stop]).copy()
                        ).to(device),
                    },
                    sds,
                )
            encoded_batches.append(encoded.float().cpu().numpy())
        irf_signal_by_mode[mode] = np.concatenate(encoded_batches)
    intervention_seed = seed + int(
        intervention_cfg.get("shuffle_seed_offset", 400_000)
    )
    donor_global = _deterministic_prior_donors(layers_all, intervention_seed)
    schema = ParameterSchema()
    normalized_truth_all = schema.normalize(truth_all)
    neutral_physical = schema.denormalize(
        np.zeros((1, len(CANONICAL_PARAMETERS)), dtype=np.float32)
    )[0]
    sample_count = int(cfg.get("posterior_samples", config["inference"]["posterior_samples"]))
    solver_steps = int(cfg.get("solver_steps", config["inference"]["solver_steps"]))
    destination = project_path(config, output_dir) if output_dir else checkpoint.parent / "test"
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = _sha256(checkpoint)
    from .evaluation_protocol import GMM_INITIALIZATION
    signature_payload = {
        "gmm_initialization": GMM_INITIALIZATION,
        "precision_protocol": PRECISION_PROTOCOL,
        "channel_subset_protocol": "all15_named_exact_40_30_15_15",
        "protocol": "four-channel-paired-irf-native-posterior-v4",
        "checkpoint": checkpoint_hash,
        "weights": selected_weights,
        "dataset": fingerprint,
        "observation_seed": observation_seed,
        "posterior_seed": posterior_seed,
        "posterior_samples": sample_count,
        "solver_steps": solver_steps,
        "batch_size": batch_size,
        "device": str(device),
        "amp_enabled": amp_enabled,
        "preprocessing": config["simulation"]["preprocessing"],
        "irf_generation": config["simulation"]["irf"],
        "irf_channel_pairing_rule": "dtof_channel_c_uses_only_irf_channel_c",
        "sample_ids": sample_ids,
        "prior_interventions": {
            "enabled": interventions_enabled,
            "scope": intervention_scope,
            "modes": intervention_modes,
            "shuffle_seed": intervention_seed,
            "scenario_ids": [
                scenario.scenario_id for scenario in intervention_scenarios
            ],
            "donor_global_case_index": donor_global.tolist(),
            "mask_only_normalized_value": 0.0,
        },
        "irf_interventions": {
            "enabled": irf_interventions_enabled,
            "scope": irf_scope,
            "modes": irf_modes,
            "shuffle_seed": irf_seed,
            "fixed_seed": fixed_irf_seed,
            "scenario_ids": [scenario.scenario_id for scenario in irf_scenarios],
            "donor_global_case_index": irf_donor_global.tolist(),
            "channel_permutations": irf_channel_permutations.tolist(),
            "fixed_irf": fixed_irf_metadata,
            "intervention_stage": irf_intervention_stage,
        },
        "model_variant": model_variant,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    store = _prepare_store(
        destination / "posterior.h5", signature, truth_all, active_all,
        layers_all, sample_ids, thickness_groups,
    )
    cases_group = store["cases"]
    if "dtof_input" not in cases_group:
        cases_group.create_dataset(
            "dtof_input", data=dtof_all, compression="gzip", compression_opts=1
        )
        cases_group.create_dataset(
            "paired_irf_input", data=irf_all, compression="gzip", compression_opts=1
        )
        string = h5py.string_dtype("utf-8")
        cases_group.create_dataset(
            "irf_metadata_json",
            data=np.asarray(
                [json.dumps(item, sort_keys=True) for item in irf_metadata],
                dtype=object,
            ),
            dtype=string,
        )
    if fixed_irf_features is not None:
        irf_root = store["irf_interventions"]
        if "fixed_irf_input" not in cases_group:
            cases_group.create_dataset(
                "fixed_irf_input", data=fixed_irf_features
            )
        irf_root.attrs["fixed_irf_metadata"] = json.dumps(
            fixed_irf_metadata, sort_keys=True
        )
    store.attrs["evaluation_metadata"] = json.dumps(
        signature_payload, sort_keys=True, ensure_ascii=False
    )
    try:
        for ordinal, scenario in enumerate(scenarios):
            selected_global = np.flatnonzero(layers_all == scenario.layer_count)
            group = store["scenarios"].require_group(scenario.scenario_id)
            group.attrs["metadata"] = json.dumps(scenario.to_dict())
            case_count = len(selected_global)
            if "samples" not in group:
                group.create_dataset(
                    "samples", shape=(case_count, sample_count, 8), dtype="f4",
                    chunks=(1, min(sample_count, 128), 8), compression="gzip", compression_opts=1,
                )
                group.create_dataset("diagnostics", shape=(case_count, len(DIAGNOSTIC_COLUMNS)), dtype="f4")
                group.create_dataset("lower_bound_violations", shape=(case_count, 8), dtype="i4")
                group.create_dataset("upper_bound_violations", shape=(case_count, 8), dtype="i4")
                group.create_dataset("normalized_exceedance", shape=(case_count, len(EXCEEDANCE_COLUMNS)), dtype="f4")
                status = group.create_dataset(
                    "sample_status_flags", shape=(case_count, sample_count), dtype="u1",
                    chunks=(1, min(sample_count, 256)), compression="gzip", compression_opts=1,
                )
                status.attrs["bit_definitions"] = json.dumps(SAMPLE_STATUS_BITS)
                group.create_dataset("global_case_index", data=selected_global)
                group.attrs["completed_cases"] = 0
                group.attrs["complete"] = False
            completed = int(group.attrs.get("completed_cases", 0))
            for start in range(completed, case_count, batch_size):
                stop = min(start + batch_size, case_count)
                global_indices = selected_global[start:stop]
                active_np = active_all[global_indices]
                known_np = scenario_mask(scenario, len(global_indices))
                known_normalized_np = np.where(
                    known_np, normalized_truth_all[global_indices], 0.0
                ).astype(np.float32)
                known_physical_np = np.where(
                    known_np, truth_all[global_indices], 0.0
                ).astype(np.float32)
                # Common random numbers across no-prior, exact-prior, and
                # counterfactual conditions isolate the effect of conditioning.
                local_seed = (
                    posterior_seed + scenario.layer_count * 1_000_003 + start
                )
                torch.manual_seed(local_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(local_seed)
                samples, diagnostics = sample_native_posterior_batch(
                    model,
                    torch.from_numpy(signal_all[global_indices]).to(device),
                    torch.from_numpy(layers_all[global_indices]).to(device),
                    torch.from_numpy(active_np).to(device),
                    torch.from_numpy(known_np).to(device),
                    torch.from_numpy(known_normalized_np).to(device),
                    known_physical_np,
                    sample_count,
                    solver_steps,
                )
                group["samples"][start:stop] = samples
                group["diagnostics"][start:stop] = np.asarray(
                    [_diagnostic_row(item) for item in diagnostics], dtype=np.float32
                )
                parameter_diagnostics = [
                    _diagnostic_parameter_rows(item, sample_count) for item in diagnostics
                ]
                group["lower_bound_violations"][start:stop] = np.stack(
                    [item[0] for item in parameter_diagnostics]
                )
                group["upper_bound_violations"][start:stop] = np.stack(
                    [item[1] for item in parameter_diagnostics]
                )
                group["normalized_exceedance"][start:stop] = np.stack(
                    [item[2] for item in parameter_diagnostics]
                )
                group["sample_status_flags"][start:stop] = _sample_status_flags(
                    samples, active_np, layers_all[global_indices]
                )
                group.attrs["completed_cases"] = stop
                store.flush()
            group.attrs["complete"] = True
            store.flush()
            print(json.dumps({"event": "test_progress", "completed_scenarios": ordinal + 1, "total_scenarios": len(scenarios), "scenario": scenario.scenario_id}), flush=True)

        total_irf_scenarios = len(irf_modes) * len(irf_scenarios)
        completed_irf_scenarios = 0
        for mode in irf_modes:
            mode_root = store["irf_interventions"].require_group(mode)
            mode_root.attrs["definition"] = {
                "shuffled_irf": "same-layer deterministic donor IRF after preprocessing",
                "fixed_irf": "one deterministic four_similar measured-IRF bundle medoid",
                "zero_irf": "all-zero IRF model input; destructive stress test",
                "channel_permuted_irf": (
                    "case-matched four-channel IRF bundle with a deterministic "
                    "fixed-point-free SDS-to-IRF channel permutation"
                ),
            }[mode]
            for scenario in irf_scenarios:
                selected_global = np.flatnonzero(
                    layers_all == scenario.layer_count
                )
                group = mode_root.require_group(scenario.scenario_id)
                group.attrs["metadata"] = json.dumps(scenario.to_dict())
                group.attrs["intervention_mode"] = mode
                case_count = len(selected_global)
                if "samples" not in group:
                    group.create_dataset(
                        "samples",
                        shape=(case_count, sample_count, 8),
                        dtype="f4",
                        chunks=(1, min(sample_count, 128), 8),
                        compression="gzip",
                        compression_opts=1,
                    )
                    group.create_dataset(
                        "diagnostics",
                        shape=(case_count, len(DIAGNOSTIC_COLUMNS)),
                        dtype="f4",
                    )
                    status = group.create_dataset(
                        "sample_status_flags",
                        shape=(case_count, sample_count),
                        dtype="u1",
                        chunks=(1, min(sample_count, 256)),
                        compression="gzip",
                        compression_opts=1,
                    )
                    status.attrs["bit_definitions"] = json.dumps(
                        SAMPLE_STATUS_BITS
                    )
                    group.create_dataset(
                        "global_case_index", data=selected_global
                    )
                    group.create_dataset(
                        "irf_source_global_case_index",
                        data=(
                            irf_donor_global[selected_global]
                            if mode == "shuffled_irf"
                            else np.full(case_count, -1, dtype=np.int64)
                        ),
                    )
                    group.create_dataset(
                        "irf_channel_permutation",
                        data=(
                            irf_channel_permutations[selected_global]
                            if mode == "channel_permuted_irf"
                            else np.broadcast_to(
                                np.arange(4, dtype=np.int64), (case_count, 4)
                            )
                        ),
                    )
                    group.attrs["completed_cases"] = 0
                    group.attrs["complete"] = False
                completed = int(group.attrs.get("completed_cases", 0))
                for start in range(completed, case_count, batch_size):
                    stop = min(start + batch_size, case_count)
                    global_indices = selected_global[start:stop]
                    active_np = active_all[global_indices]
                    known_np = scenario_mask(scenario, len(global_indices))
                    known_normalized_np = np.where(
                        known_np,
                        normalized_truth_all[global_indices],
                        0.0,
                    ).astype(np.float32)
                    known_physical_np = np.where(
                        known_np, truth_all[global_indices], 0.0
                    ).astype(np.float32)
                    local_seed = (
                        posterior_seed
                        + scenario.layer_count * 1_000_003
                        + start
                    )
                    torch.manual_seed(local_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(local_seed)
                    samples, diagnostics = sample_native_posterior_batch(
                        model,
                        torch.from_numpy(
                            irf_signal_by_mode[mode][global_indices]
                        ).to(device),
                        torch.from_numpy(layers_all[global_indices]).to(device),
                        torch.from_numpy(active_np).to(device),
                        torch.from_numpy(known_np).to(device),
                        torch.from_numpy(known_normalized_np).to(device),
                        known_physical_np,
                        sample_count,
                        solver_steps,
                    )
                    group["samples"][start:stop] = samples
                    group["diagnostics"][start:stop] = np.asarray(
                        [_diagnostic_row(item) for item in diagnostics],
                        dtype=np.float32,
                    )
                    group["sample_status_flags"][start:stop] = (
                        _sample_status_flags(
                            samples, active_np, layers_all[global_indices]
                        )
                    )
                    group.attrs["completed_cases"] = stop
                    store.flush()
                group.attrs["complete"] = True
                store.flush()
                completed_irf_scenarios += 1
                print(
                    json.dumps(
                        {
                            "event": "irf_intervention_progress",
                            "completed_scenarios": completed_irf_scenarios,
                            "total_scenarios": total_irf_scenarios,
                            "mode": mode,
                            "scenario": scenario.scenario_id,
                        }
                    ),
                    flush=True,
                )

        total_intervention_scenarios = len(intervention_modes) * len(
            intervention_scenarios
        )
        completed_intervention_scenarios = 0
        for mode in intervention_modes:
            mode_root = store["prior_interventions"].require_group(mode)
            mode_root.attrs["definition"] = (
                "same-layer deterministic whole-case donor values"
                if mode == "shuffled_prior"
                else "known mask with normalized values fixed to zero"
            )
            for scenario in intervention_scenarios:
                selected_global = np.flatnonzero(
                    layers_all == scenario.layer_count
                )
                selected_donors = donor_global[selected_global]
                group = mode_root.require_group(scenario.scenario_id)
                group.attrs["metadata"] = json.dumps(scenario.to_dict())
                group.attrs["intervention_mode"] = mode
                group.attrs["shuffle_seed"] = intervention_seed
                case_count = len(selected_global)
                available = mode != "shuffled_prior" or bool(
                    case_count and np.all(selected_donors >= 0)
                )
                if not available:
                    group.attrs["status"] = "unavailable_insufficient_same_layer_cases"
                    group.attrs["completed_cases"] = 0
                    group.attrs["complete"] = True
                    completed_intervention_scenarios += 1
                    print(
                        json.dumps(
                            {
                                "event": "test_prior_intervention_progress",
                                "mode": mode,
                                "completed_scenarios": completed_intervention_scenarios,
                                "total_scenarios": total_intervention_scenarios,
                                "scenario": scenario.scenario_id,
                                "status": group.attrs["status"],
                            }
                        ),
                        flush=True,
                    )
                    continue
                known_all = scenario_mask(scenario, case_count)
                known_normalized_all, known_physical_all = _prior_values_for_mode(
                    mode,
                    known_all,
                    selected_global,
                    selected_donors,
                    normalized_truth_all,
                    truth_all,
                    neutral_physical,
                )
                if "samples" not in group:
                    group.create_dataset(
                        "samples", shape=(case_count, sample_count, 8), dtype="f4",
                        chunks=(1, min(sample_count, 128), 8), compression="gzip",
                        compression_opts=1,
                    )
                    group.create_dataset(
                        "diagnostics",
                        shape=(case_count, len(DIAGNOSTIC_COLUMNS)), dtype="f4",
                    )
                    group.create_dataset(
                        "lower_bound_violations", shape=(case_count, 8), dtype="i4"
                    )
                    group.create_dataset(
                        "upper_bound_violations", shape=(case_count, 8), dtype="i4"
                    )
                    group.create_dataset(
                        "normalized_exceedance",
                        shape=(case_count, len(EXCEEDANCE_COLUMNS)), dtype="f4",
                    )
                    status = group.create_dataset(
                        "sample_status_flags", shape=(case_count, sample_count),
                        dtype="u1", chunks=(1, min(sample_count, 256)),
                        compression="gzip", compression_opts=1,
                    )
                    status.attrs["bit_definitions"] = json.dumps(SAMPLE_STATUS_BITS)
                    group.create_dataset("global_case_index", data=selected_global)
                    group.create_dataset("donor_global_case_index", data=selected_donors)
                    group.create_dataset(
                        "known_values_normalized", data=known_normalized_all
                    )
                    group.create_dataset("known_values_physical", data=known_physical_all)
                    group.attrs["completed_cases"] = 0
                    group.attrs["complete"] = False
                    group.attrs["status"] = "in_progress"
                completed = int(group.attrs.get("completed_cases", 0))
                for start in range(completed, case_count, batch_size):
                    stop = min(start + batch_size, case_count)
                    global_indices = selected_global[start:stop]
                    active_np = active_all[global_indices]
                    known_np = known_all[start:stop]
                    local_seed = (
                        posterior_seed
                        + scenario.layer_count * 1_000_003
                        + start
                    )
                    torch.manual_seed(local_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(local_seed)
                    samples, diagnostics = sample_native_posterior_batch(
                        model,
                        torch.from_numpy(signal_all[global_indices]).to(device),
                        torch.from_numpy(layers_all[global_indices]).to(device),
                        torch.from_numpy(active_np).to(device),
                        torch.from_numpy(known_np).to(device),
                        torch.from_numpy(known_normalized_all[start:stop]).to(device),
                        known_physical_all[start:stop],
                        sample_count,
                        solver_steps,
                    )
                    group["samples"][start:stop] = samples
                    group["diagnostics"][start:stop] = np.asarray(
                        [_diagnostic_row(item) for item in diagnostics], dtype=np.float32
                    )
                    parameter_diagnostics = [
                        _diagnostic_parameter_rows(item, sample_count)
                        for item in diagnostics
                    ]
                    group["lower_bound_violations"][start:stop] = np.stack(
                        [item[0] for item in parameter_diagnostics]
                    )
                    group["upper_bound_violations"][start:stop] = np.stack(
                        [item[1] for item in parameter_diagnostics]
                    )
                    group["normalized_exceedance"][start:stop] = np.stack(
                        [item[2] for item in parameter_diagnostics]
                    )
                    group["sample_status_flags"][start:stop] = _sample_status_flags(
                        samples, active_np, layers_all[global_indices]
                    )
                    group.attrs["completed_cases"] = stop
                    store.flush()
                group.attrs["status"] = "completed"
                group.attrs["complete"] = True
                store.flush()
                completed_intervention_scenarios += 1
                print(
                    json.dumps(
                        {
                            "event": "test_prior_intervention_progress",
                            "mode": mode,
                            "completed_scenarios": completed_intervention_scenarios,
                            "total_scenarios": total_intervention_scenarios,
                            "scenario": scenario.scenario_id,
                            "status": "completed",
                        }
                    ),
                    flush=True,
                )

        scenario_reports: dict[str, Any] = {}
        parameter_rows: list[dict[str, Any]] = []
        thickness_rows: list[dict[str, Any]] = []
        case_prediction_rows: list[dict[str, Any]] = []
        prior_rows: list[dict[str, Any]] = []
        intervention_effect_rows: list[dict[str, Any]] = []
        intervention_assignment_rows: list[dict[str, Any]] = []
        irf_effect_rows: list[dict[str, Any]] = []
        irf_assignment_rows: list[dict[str, Any]] = []
        irf_reports: dict[str, dict[str, Any]] = {
            mode: {} for mode in irf_modes
        }
        intervention_reports: dict[str, dict[str, Any]] = {
            mode: {} for mode in intervention_modes
        }
        multimodality_rows: list[dict[str, Any]] = []
        representative_mode_plots: list[dict[str, str]] = []
        domain_diagnostic_rows: list[dict[str, Any]] = []
        for scenario in scenarios:
            group = store["scenarios"][scenario.scenario_id]
            samples = group["samples"][:]
            diagnostics_matrix = group["diagnostics"][:]
            global_indices = group["global_case_index"][:]
            known = scenario_mask(scenario, len(global_indices))
            finite_cases = diagnostics_matrix[:, 4] == 0.0
            if np.any(finite_cases):
                metrics = _group_metrics(
                    samples[finite_cases], truth_all[global_indices][finite_cases],
                    active_all[global_indices][finite_cases], known[finite_cases],
                    layers_all[global_indices][finite_cases], thickness_groups[global_indices][finite_cases],
                    joint_seed=posterior_seed + scenario.layer_count,
                )
            else:
                metrics = {"all_unknown_parameters": {"aggregate": {}, "parameters": {}, "joint": {}}, "by_layer_count": {}, "by_known_count": {}, "by_thickness_group": {}}
            native = _diagnostics_from_matrix(diagnostics_matrix)
            native["numerical_failure_sample_ids"] = [
                sample_ids[global_indices[local]]
                for local in native["numerical_failure_case_indices"]
            ]
            lower_counts = group["lower_bound_violations"][:]
            upper_counts = group["upper_bound_violations"][:]
            exceedance = group["normalized_exceedance"][:]
            native["per_parameter_lower_violation_probability"] = {
                name: float(np.sum(lower_counts[:, index]) / (len(global_indices) * sample_count))
                for index, name in enumerate(CANONICAL_PARAMETERS)
                if np.any(active_all[global_indices, index])
            }
            native["per_parameter_upper_violation_probability"] = {
                name: float(np.sum(upper_counts[:, index]) / (len(global_indices) * sample_count))
                for index, name in enumerate(CANONICAL_PARAMETERS)
                if np.any(active_all[global_indices, index])
            }
            normalized_samples = schema.normalize(samples)
            active_values = np.broadcast_to(
                active_all[global_indices, None, :], normalized_samples.shape
            )
            exceedance_values = np.maximum(np.abs(normalized_samples) - 1.0, 0.0)
            positive_exceedance = exceedance_values[
                active_values & np.isfinite(exceedance_values) & (exceedance_values > 0.0)
            ]
            native["normalized_exceedance"] = {
                "median": float(np.median(positive_exceedance)) if positive_exceedance.size else 0.0,
                "q90": float(np.quantile(positive_exceedance, 0.9)) if positive_exceedance.size else 0.0,
                "maximum": float(np.max(positive_exceedance)) if positive_exceedance.size else 0.0,
            }
            scenario_reports[scenario.scenario_id] = {
                "metadata": scenario.to_dict(),
                "cases": len(global_indices),
                "evaluated_cases": int(finite_cases.sum()),
                "native_diagnostics": native,
                "metrics": metrics,
            }
            flat, thick = _flatten_metric_rows(scenario, metrics)
            parameter_rows.extend(flat)
            thickness_rows.extend(thick)
            unknown = active_all[global_indices] & ~known
            finite_samples = np.where(np.isfinite(samples), samples, np.nan)
            medians = np.nanmedian(finite_samples, axis=1)
            q05 = np.nanquantile(finite_samples, 0.05, axis=1)
            q95 = np.nanquantile(finite_samples, 0.95, axis=1)
            # A case containing any nonfinite active posterior draw is a
            # numerical failure.  Do not make its point summary look valid by
            # silently reducing over only the remaining finite draws.
            medians[~finite_cases] = np.nan
            q05[~finite_cases] = np.nan
            q95[~finite_cases] = np.nan
            for local, global_index in enumerate(global_indices):
                diagnostic_row = {
                    "scenario_id": scenario.scenario_id,
                    "sample_id": sample_ids[global_index],
                    "layer_count": scenario.layer_count,
                    "finite_fraction": float(diagnostics_matrix[local, 0]),
                    "outside_training_support_fraction": float(diagnostics_matrix[local, 1]),
                    "outside_study_constraint_fraction": float(diagnostics_matrix[local, 2]),
                    "physically_impossible_fraction": float(diagnostics_matrix[local, 3]),
                    "nonfinite_fraction": float(diagnostics_matrix[local, 4]),
                    "normalized_exceedance_median": float(exceedance[local, 0]),
                    "normalized_exceedance_q90": float(exceedance[local, 1]),
                    "normalized_exceedance_maximum": float(exceedance[local, 2]),
                }
                for parameter_index, parameter_name in enumerate(CANONICAL_PARAMETERS):
                    if active_all[global_index, parameter_index]:
                        diagnostic_row[f"{parameter_name}_lower_probability"] = float(
                            lower_counts[local, parameter_index] / sample_count
                        )
                        diagnostic_row[f"{parameter_name}_upper_probability"] = float(
                            upper_counts[local, parameter_index] / sample_count
                        )
                domain_diagnostic_rows.append(diagnostic_row)
                for parameter_index in np.flatnonzero(unknown[local]):
                    case_prediction_rows.append({
                        "scenario_id": scenario.scenario_id,
                        "sample_id": sample_ids[global_index],
                        "layer_count": scenario.layer_count,
                        "known_count": scenario.known_count,
                        "prior_type": scenario.prior_type,
                        "thickness_group": thickness_groups[global_index],
                        "parameter": CANONICAL_PARAMETERS[parameter_index],
                        "truth": float(truth_all[global_index, parameter_index]),
                        "posterior_median": float(medians[local, parameter_index]),
                        "absolute_error": float(abs(medians[local, parameter_index] - truth_all[global_index, parameter_index])),
                        "width_90": float(q95[local, parameter_index] - q05[local, parameter_index]),
                        "finite_fraction": float(diagnostics_matrix[local, 0]),
                        "outside_training_support_fraction": float(diagnostics_matrix[local, 1]),
                        "outside_study_constraint_fraction": float(diagnostics_matrix[local, 2]),
                        "physically_impossible_fraction": float(diagnostics_matrix[local, 3]),
                        "numerical_failure": bool(not finite_cases[local]),
                    })
                if scenario.layer_count == 3 and not (
                    known[local, 0] and known[local, 1]
                ):
                    surface_draws = (
                        finite_samples[local, :, 0] + finite_samples[local, :, 1]
                    )
                    surface_truth = float(
                        truth_all[global_index, 0] + truth_all[global_index, 1]
                    )
                    surface_median = float(np.nanmedian(surface_draws))
                    case_prediction_rows.append({
                        "scenario_id": scenario.scenario_id,
                        "sample_id": sample_ids[global_index],
                        "layer_count": scenario.layer_count,
                        "known_count": scenario.known_count,
                        "prior_type": scenario.prior_type,
                        "thickness_group": thickness_groups[global_index],
                        "parameter": N3_SURFACE_DEPTH_NAME,
                        "truth": surface_truth,
                        "posterior_median": surface_median,
                        "absolute_error": abs(surface_median - surface_truth),
                        "width_90": float(
                            np.nanquantile(surface_draws, 0.95)
                            - np.nanquantile(surface_draws, 0.05)
                        ),
                        "finite_fraction": float(diagnostics_matrix[local, 0]),
                        "outside_training_support_fraction": float(
                            diagnostics_matrix[local, 1]
                        ),
                        "outside_study_constraint_fraction": float(
                            diagnostics_matrix[local, 2]
                        ),
                        "physically_impossible_fraction": float(
                            diagnostics_matrix[local, 3]
                        ),
                        "numerical_failure": bool(not finite_cases[local]),
                    })

        bootstrap_replicates = int(cfg.get("bootstrap_replicates", 1000))
        for ordinal, scenario in enumerate(scenarios):
            if scenario.known_count == 0:
                continue
            group = store["scenarios"][scenario.scenario_id]
            indices = group["global_case_index"][:]
            diagnostics_matrix = group["diagnostics"][:]
            baseline_group = next(
                store["scenarios"][candidate.scenario_id]
                for candidate in scenarios
                if candidate.layer_count == scenario.layer_count and candidate.known_count == 0
            )
            valid = (diagnostics_matrix[:, 4] == 0.0) & (baseline_group["diagnostics"][:, 4] == 0.0)
            if np.any(valid):
                prior_rows.extend(
                    _paired_prior_rows(
                        scenario, group["samples"][:][valid], baseline_group["samples"][:][valid],
                        truth_all[indices][valid], active_all[indices][valid],
                        posterior_seed + ordinal, bootstrap_replicates,
                    )
                )

        for mode in intervention_modes:
            for global_index, sample_id in enumerate(sample_ids):
                donor_index = int(donor_global[global_index])
                available = mode != "shuffled_prior" or donor_index >= 0
                if mode == "shuffled_prior" and available:
                    source_physical = truth_all[donor_index].tolist()
                    source_normalized = normalized_truth_all[donor_index].tolist()
                    donor_sample_id = sample_ids[donor_index]
                elif mode == "mask_only":
                    source_physical = neutral_physical.tolist()
                    source_normalized = [0.0] * len(CANONICAL_PARAMETERS)
                    donor_sample_id = ""
                else:
                    source_physical = []
                    source_normalized = []
                    donor_sample_id = ""
                intervention_assignment_rows.append(
                    {
                        "mode": mode,
                        "target_global_case_index": global_index,
                        "target_sample_id": sample_id,
                        "layer_count": int(layers_all[global_index]),
                        "donor_global_case_index": donor_index if available and mode == "shuffled_prior" else "",
                        "donor_sample_id": donor_sample_id,
                        "available": available,
                        "source_values_physical": source_physical,
                        "source_values_normalized": source_normalized,
                    }
                )

        comparison_specs = (
            ("exact_prior", "no_prior"),
            ("shuffled_prior", "no_prior"),
            ("mask_only", "no_prior"),
            ("exact_prior", "shuffled_prior"),
            ("exact_prior", "mask_only"),
        )
        for scenario_offset, scenario in enumerate(intervention_scenarios):
            exact_group = store["scenarios"][scenario.scenario_id]
            indices = exact_group["global_case_index"][:]
            baseline_group = next(
                store["scenarios"][candidate.scenario_id]
                for candidate in scenarios
                if candidate.layer_count == scenario.layer_count
                and candidate.known_count == 0
            )
            condition_groups: dict[str, Any] = {
                "no_prior": baseline_group,
                "exact_prior": exact_group,
            }
            for mode in intervention_modes:
                group = store["prior_interventions"][mode][scenario.scenario_id]
                status = str(group.attrs.get("status", "unknown"))
                if status != "completed":
                    intervention_reports[mode][scenario.scenario_id] = {
                        "metadata": scenario.to_dict(),
                        "status": status,
                        "cases": len(indices),
                        "evaluated_cases": 0,
                    }
                    continue
                condition_groups[mode] = group
                diagnostics_matrix = group["diagnostics"][:]
                finite = diagnostics_matrix[:, 4] == 0.0
                known = scenario_mask(scenario, len(indices))
                metrics = (
                    _group_metrics(
                        group["samples"][:][finite], truth_all[indices][finite],
                        active_all[indices][finite], known[finite],
                        layers_all[indices][finite],
                        thickness_groups[indices][finite],
                        joint_seed=posterior_seed + scenario.layer_count,
                    )
                    if np.any(finite)
                    else {
                        "all_unknown_parameters": {
                            "aggregate": {}, "parameters": {}, "joint": {}
                        },
                        "by_layer_count": {}, "by_known_count": {},
                        "by_thickness_group": {},
                    }
                )
                native = _diagnostics_from_matrix(diagnostics_matrix)
                native["numerical_failure_sample_ids"] = [
                    sample_ids[indices[local]]
                    for local in native["numerical_failure_case_indices"]
                ]
                intervention_reports[mode][scenario.scenario_id] = {
                    "metadata": scenario.to_dict(),
                    "status": status,
                    "cases": len(indices),
                    "evaluated_cases": int(finite.sum()),
                    "native_diagnostics": native,
                    "metrics": metrics,
                }
            for comparison_offset, (candidate_name, reference_name) in enumerate(
                comparison_specs
            ):
                if candidate_name not in condition_groups or reference_name not in condition_groups:
                    continue
                candidate_group = condition_groups[candidate_name]
                reference_group = condition_groups[reference_name]
                candidate_diagnostics = candidate_group["diagnostics"][:]
                reference_diagnostics = reference_group["diagnostics"][:]
                valid = (
                    (candidate_diagnostics[:, 4] == 0.0)
                    & (reference_diagnostics[:, 4] == 0.0)
                )
                if not np.any(valid):
                    continue
                intervention_effect_rows.extend(
                    _prior_intervention_comparison_rows(
                        scenario,
                        candidate_group["samples"][:][valid],
                        reference_group["samples"][:][valid],
                        truth_all[indices][valid],
                        active_all[indices][valid],
                        candidate_diagnostics[valid],
                        reference_diagnostics[valid],
                        candidate_name,
                        reference_name,
                        posterior_seed + scenario_offset * 100 + comparison_offset,
                        bootstrap_replicates,
                    )
                )

        strata_all = _irf_strata(irf_metadata) if irf_modes else {}
        for mode_offset, mode in enumerate(irf_modes):
            for scenario_offset, scenario in enumerate(irf_scenarios):
                candidate_group = store["irf_interventions"][mode][
                    scenario.scenario_id
                ]
                reference_group = store["scenarios"][scenario.scenario_id]
                indices = candidate_group["global_case_index"][:]
                candidate_diagnostics = candidate_group["diagnostics"][:]
                reference_diagnostics = reference_group["diagnostics"][:]
                valid = (
                    (candidate_diagnostics[:, 4] == 0.0)
                    & (reference_diagnostics[:, 4] == 0.0)
                )
                known = scenario_mask(scenario, len(indices))
                metrics = (
                    _group_metrics(
                        candidate_group["samples"][:][valid],
                        truth_all[indices][valid],
                        active_all[indices][valid],
                        known[valid],
                        layers_all[indices][valid],
                        thickness_groups[indices][valid],
                        joint_seed=posterior_seed + scenario.layer_count,
                    )
                    if np.any(valid)
                    else {
                        "all_unknown_parameters": {
                            "aggregate": {}, "parameters": {}, "joint": {}
                        },
                        "by_layer_count": {},
                        "by_known_count": {},
                        "by_thickness_group": {},
                    }
                )
                native = _diagnostics_from_matrix(candidate_diagnostics)
                native["numerical_failure_sample_ids"] = [
                    sample_ids[indices[local]]
                    for local in native["numerical_failure_case_indices"]
                ]
                irf_reports[mode][scenario.scenario_id] = {
                    "metadata": scenario.to_dict(),
                    "status": "completed",
                    "cases": len(indices),
                    "evaluated_cases": int(valid.sum()),
                    "native_diagnostics": native,
                    "metrics": metrics,
                }
                if not np.any(valid):
                    continue
                seed_base = (
                    posterior_seed
                    + 3_000_000
                    + mode_offset * 100_000
                    + scenario_offset * 100
                )
                overall = _prior_intervention_comparison_rows(
                    scenario,
                    candidate_group["samples"][:][valid],
                    reference_group["samples"][:][valid],
                    truth_all[indices][valid],
                    active_all[indices][valid],
                    candidate_diagnostics[valid],
                    reference_diagnostics[valid],
                    mode,
                    "paired_irf",
                    seed_base,
                    bootstrap_replicates,
                )
                for row in overall:
                    row["stratum_type"] = "all"
                    row["stratum_value"] = "all"
                irf_effect_rows.extend(overall)
                for stratum_offset, (stratum_name, labels) in enumerate(
                    strata_all.items()
                ):
                    local_labels = labels[indices]
                    for label_offset, label in enumerate(
                        sorted(set(map(str, local_labels)))
                    ):
                        selected = valid & (local_labels == label)
                        if int(selected.sum()) < 4:
                            continue
                        rows = _prior_intervention_comparison_rows(
                            scenario,
                            candidate_group["samples"][:][selected],
                            reference_group["samples"][:][selected],
                            truth_all[indices][selected],
                            active_all[indices][selected],
                            candidate_diagnostics[selected],
                            reference_diagnostics[selected],
                            mode,
                            "paired_irf",
                            seed_base
                            + (stratum_offset + 1) * 10_000
                            + label_offset * 100,
                            bootstrap_replicates,
                        )
                        for row in rows:
                            row["stratum_type"] = stratum_name
                            row["stratum_value"] = label
                        irf_effect_rows.extend(rows)

        for mode in irf_modes:
            for target_index, sample_id in enumerate(sample_ids):
                donor_index = (
                    int(irf_donor_global[target_index])
                    if mode == "shuffled_irf"
                    else -1
                )
                irf_assignment_rows.append(
                    {
                        "mode": mode,
                        "target_global_case_index": target_index,
                        "target_sample_id": sample_id,
                        "layer_count": int(layers_all[target_index]),
                        "donor_global_case_index": (
                            donor_index if donor_index >= 0 else ""
                        ),
                        "donor_sample_id": (
                            sample_ids[donor_index] if donor_index >= 0 else ""
                        ),
                        "fixed_irf_sha256": (
                            fixed_irf_metadata.get("sha256", "")
                            if mode == "fixed_irf"
                            else ""
                        ),
                        "channel_permutation": (
                            ";".join(
                                map(str, irf_channel_permutations[target_index])
                            )
                            if mode == "channel_permuted_irf"
                            else ""
                        ),
                        "intervention_stage": irf_intervention_stage,
                    }
                )

        mode_cfg = cfg.get("multimodality", {})
        mode_case_limit = int(mode_cfg.get("cases_per_scenario", 12))
        baseline_mode_indices: dict[int, np.ndarray] = {}
        baseline_mode_results: dict[tuple[int, int], dict[str, Any]] = {}
        if bool(mode_cfg.get("enabled", True)):
            for ordinal, scenario in enumerate(scenarios):
                if not is_deep_multimodality_scenario(scenario):
                    continue
                group = store["scenarios"][scenario.scenario_id]
                indices = group["global_case_index"][:]
                matrix = group["diagnostics"][:]
                valid_local = np.flatnonzero(matrix[:, 4] == 0.0)
                all_valid_local = valid_local.copy()
                if len(valid_local) > mode_case_limit:
                    known_for_selection = scenario_mask(scenario, len(indices))
                    chosen = _representative_case_indices(
                        group["samples"][:][valid_local],
                        truth_all[indices][valid_local],
                        (active_all[indices] & ~known_for_selection)[valid_local],
                        matrix[valid_local, 1],
                        thickness_groups[indices][valid_local],
                        mode_case_limit,
                    )
                    valid_local = valid_local[chosen]
                if scenario.known_count == 0:
                    baseline_mode_indices[scenario.layer_count] = valid_local.copy()
                elif scenario.layer_count in baseline_mode_indices and mode_case_limit > 1:
                    baseline = baseline_mode_indices[scenario.layer_count]
                    shared = baseline[np.isin(baseline, all_valid_local)]
                    target_shared = min(len(shared), max(1, mode_case_limit // 2))
                    combined = list(map(int, shared[:target_shared]))
                    combined.extend(int(value) for value in valid_local if int(value) not in combined)
                    valid_local = np.asarray(combined[:mode_case_limit], dtype=np.int64)
                known = scenario_mask(scenario, len(indices))
                samples = group["samples"]
                for local in valid_local:
                    normalized = schema.normalize(samples[local])
                    mode = detect_candidate_modes(
                        normalized, active_all[indices[local]] & ~known[local],
                        seed=posterior_seed + ordinal * 1000 + int(local),
                        normalized_truth=schema.normalize(truth_all[indices[local]]),
                        max_components=int(mode_cfg.get("max_components", 4)),
                        bootstrap_replicates=int(mode_cfg.get("bootstrap_replicates", 20)),
                        minimum_weight=float(mode_cfg.get("minimum_weight", 0.05)),
                        minimum_separation=float(mode_cfg.get("minimum_separation", 2.0)),
                        max_iter=int(mode_cfg.get("max_iter", 300)),
                        retry_max_iter=int(mode_cfg.get("retry_max_iter", 1000)),
                    )
                    if mode.get("status") == "completed":
                        unknown_indices = np.flatnonzero(
                            active_all[indices[local]] & ~known[local]
                        )
                        physical_centers = []
                        base = schema.normalize(truth_all[indices[local]])
                        for center in mode["centers_normalized"]:
                            vector = base.copy()
                            vector[unknown_indices] = np.asarray(center)
                            physical = schema.denormalize(vector)
                            physical_centers.append(
                                {
                                    CANONICAL_PARAMETERS[index]: float(physical[index])
                                    for index in unknown_indices
                                }
                            )
                            if scenario.layer_count == 3 and not (
                                known[local, 0] and known[local, 1]
                            ):
                                physical_centers[-1][N3_SURFACE_DEPTH_NAME] = float(
                                    physical[0] + physical[1]
                                )
                        mode["centers_physical"] = physical_centers
                        if mode.get("stable_multimodal"):
                            plot_directory = destination / "multimodality_plots"
                            plot_directory.mkdir(exist_ok=True)
                            digest = hashlib.sha256(
                                f"{scenario.scenario_id}:{sample_ids[indices[local]]}".encode("utf-8")
                            ).hexdigest()[:12]
                            relative_plot = Path("multimodality_plots") / f"candidate-{digest}.svg"
                            _write_mode_svg(
                                destination / relative_plot,
                                normalized,
                                active_all[indices[local]] & ~known[local],
                                mode,
                                f"{sample_ids[indices[local]]} / {scenario.scenario_id}",
                            )
                            mode["plot_file"] = relative_plot.as_posix()
                            representative_mode_plots.append({
                                "scenario_id": scenario.scenario_id,
                                "sample_id": sample_ids[indices[local]],
                                "plot_file": relative_plot.as_posix(),
                            })
                    mode_row = {
                        "scenario_id": scenario.scenario_id,
                        "aliases": ";".join(scenario.aliases),
                        "sample_id": sample_ids[indices[local]],
                        "layer_count": scenario.layer_count,
                        "thickness_group": thickness_groups[indices[local]],
                        "outside_training_support_fraction": float(matrix[local, 1]),
                        **mode,
                    }
                    baseline_mode = baseline_mode_results.get((scenario.layer_count, int(local)))
                    if baseline_mode is not None:
                        mode_row["baseline_component_count"] = baseline_mode.get("component_count")
                        mode_row["delta_component_count"] = (
                            int(mode.get("component_count", 0)) - int(baseline_mode.get("component_count", 0))
                            if mode.get("status") == "completed" and baseline_mode.get("status") == "completed"
                            else None
                        )
                        mode_row["baseline_largest_mode_weight"] = max(baseline_mode.get("weights", [float("nan")]))
                        mode_row["delta_largest_mode_weight"] = (
                            max(mode.get("weights", [float("nan")]))
                            - max(baseline_mode.get("weights", [float("nan")]))
                        )
                    multimodality_rows.append(mode_row)
                    if scenario.known_count == 0:
                        baseline_mode_results[(scenario.layer_count, int(local))] = mode

        thickness_associations: list[dict[str, Any]] = []
        for layer in (2, 3):
            if layer not in present_layers:
                continue
            scenario = next(s for s in scenarios if s.layer_count == layer and s.known_count == 0)
            group = store["scenarios"][scenario.scenario_id]
            indices = group["global_case_index"][:]
            valid = group["diagnostics"][:, 4] == 0.0
            samples = group["samples"][:][valid]
            truth = truth_all[indices][valid]
            active = active_all[indices][valid]
            casewise = evaluate_casewise_posterior(samples, truth, active)
            thickness_variables = (
                [("d1_mm", truth[:, 0])]
                if layer == 2
                else [
                    ("d1_plus_d2_mm", truth[:, 0] + truth[:, 1]),
                    ("d1_mm", truth[:, 0]),
                    ("d2_mm", truth[:, 1]),
                ]
            )
            for thickness_offset, (thickness_name, thickness) in enumerate(thickness_variables):
                for parameter_index in np.flatnonzero(np.any(active, axis=0)):
                    for metric_offset, metric_name in enumerate(("normalized_absolute_error", "normalized_crps", "normalized_width_90")):
                        result = spearman_with_bootstrap(
                            thickness, casewise[metric_name][:, parameter_index],
                            posterior_seed + layer * 1000 + thickness_offset * 100 + parameter_index * 3 + metric_offset,
                            bootstrap_replicates,
                        )
                        thickness_associations.append({
                            "layer_count": layer,
                            "thickness_variable": thickness_name,
                            "parameter": CANONICAL_PARAMETERS[parameter_index],
                            "metric": metric_name,
                            **result,
                        })
            if layer == 3:
                surface_casewise = evaluate_casewise_scalar_posterior(
                    samples[:, :, 0] + samples[:, :, 1],
                    truth[:, 0] + truth[:, 1],
                    N3_SURFACE_DEPTH_BOUNDS_MM,
                )
                for thickness_offset, (thickness_name, thickness) in enumerate(
                    thickness_variables
                ):
                    for metric_offset, metric_name in enumerate(
                        (
                            "normalized_absolute_error",
                            "normalized_crps",
                            "normalized_width_90",
                        )
                    ):
                        result = spearman_with_bootstrap(
                            thickness,
                            surface_casewise[metric_name],
                            posterior_seed
                            + layer * 1000
                            + thickness_offset * 100
                            + 50
                            + metric_offset,
                            bootstrap_replicates,
                        )
                        thickness_associations.append(
                            {
                                "layer_count": layer,
                                "thickness_variable": thickness_name,
                                "parameter": N3_SURFACE_DEPTH_NAME,
                                "metric": metric_name,
                                **result,
                            }
                        )

        irf_difficulty_rows: list[dict[str, Any]] = []
        for layer in (1, 2, 3):
            no_prior = next(
                (
                    scenario
                    for scenario in scenarios
                    if scenario.layer_count == layer and scenario.known_count == 0
                ),
                None,
            )
            if no_prior is None:
                continue
            no_prior_group = store["scenarios"][no_prior.scenario_id]
            indices = no_prior_group["global_case_index"][:]
            valid = no_prior_group["diagnostics"][:, 4] == 0.0
            if not np.any(valid):
                continue
            irf_difficulty_rows.extend(
                _irf_difficulty_rows(
                    no_prior_group["samples"][:][valid],
                    truth_all[indices][valid],
                    active_all[indices][valid],
                    layers_all[indices][valid],
                    [irf_metadata[index] for index in indices[valid]],
                )
            )

        mask_rows = [scenario.to_dict() for scenario in scenarios]
        irf_case_metadata_rows = _flatten_irf_metadata(
            irf_metadata, sample_ids, layers_all, config["data"]["sds_mm"]
        )
        _write_csv(destination / "prior_masks.csv", mask_rows)
        _write_csv(destination / "parameter_metrics.csv", parameter_rows)
        _write_csv(destination / "case_predictions.csv", case_prediction_rows)
        _write_csv(destination / "thickness_metrics.csv", thickness_rows)
        _write_csv(destination / "prior_paired_effects.csv", prior_rows)
        prior_summary_rows = _aggregate_prior_effects(prior_rows)
        _write_csv(destination / "prior_summary.csv", prior_summary_rows)
        _write_csv(
            destination / "prior_intervention_effects.csv",
            intervention_effect_rows,
        )
        intervention_summary_rows = _aggregate_intervention_effects(
            intervention_effect_rows
        )
        _write_csv(
            destination / "prior_intervention_summary.csv",
            intervention_summary_rows,
        )
        _write_csv(
            destination / "prior_intervention_assignments.csv",
            intervention_assignment_rows,
        )
        _write_csv(
            destination / "irf_intervention_effects.csv", irf_effect_rows
        )
        irf_summary_rows = _aggregate_intervention_effects(irf_effect_rows)
        _write_csv(
            destination / "irf_intervention_summary.csv", irf_summary_rows
        )
        _write_csv(
            destination / "irf_intervention_assignments.csv",
            irf_assignment_rows,
        )
        _write_csv(destination / "irf_case_metadata.csv", irf_case_metadata_rows)
        _write_csv(
            destination / "irf_difficulty_metrics.csv", irf_difficulty_rows
        )
        _write_csv(destination / "thickness_associations.csv", thickness_associations)
        _write_csv(destination / "multimodality.csv", multimodality_rows)
        _write_csv(destination / "domain_diagnostics.csv", domain_diagnostic_rows)
        numerical_failure_cases = [
            {
                "condition": "exact_prior",
                "scenario_id": scenario_id,
                "sample_id": sample_id,
            }
            for scenario_id, value in scenario_reports.items()
            for sample_id in value["native_diagnostics"]["numerical_failure_sample_ids"]
        ]
        numerical_failure_cases.extend(
            {
                "condition": mode,
                "scenario_id": scenario_id,
                "sample_id": sample_id,
            }
            for mode, mode_reports in intervention_reports.items()
            for scenario_id, value in mode_reports.items()
            for sample_id in value.get("native_diagnostics", {}).get(
                "numerical_failure_sample_ids", []
            )
        )
        numerical_failure_cases.extend(
            {
                "condition": mode,
                "scenario_id": scenario_id,
                "sample_id": sample_id,
            }
            for mode, mode_reports in irf_reports.items()
            for scenario_id, value in mode_reports.items()
            for sample_id in value.get("native_diagnostics", {}).get(
                "numerical_failure_sample_ids", []
            )
        )
        numerical_failures = len(numerical_failure_cases)
        gmm_failure_cases = [
            {
                "scenario_id": row["scenario_id"],
                "sample_id": row["sample_id"],
                "status": row.get("status", "unknown"),
                "failed_candidate_count": int(
                    row.get("gmm_failed_candidate_count", 0)
                ),
                "bootstrap_failed_replicates": int(
                    row.get("bootstrap_failed_replicates", 0)
                ),
            }
            for row in multimodality_rows
            if row.get("status") != "completed"
            or int(row.get("gmm_failed_candidate_count", 0)) > 0
            or int(row.get("bootstrap_failed_replicates", 0)) > 0
        ]
        gmm_retried_cases = sum(
            int(row.get("gmm_retried_candidate_count", 0)) > 0
            or int(row.get("bootstrap_retried_replicates", 0)) > 0
            for row in multimodality_rows
        )
        bootstrap_failed_replicates = sum(
            int(row.get("bootstrap_failed_replicates", 0))
            for row in multimodality_rows
        )
        bootstrap_skipped_replicates = sum(
            int(row.get("bootstrap_skipped_replicates", 0))
            for row in multimodality_rows
        )
        no_prior_headline: dict[str, Any] = {}
        named_headline: list[dict[str, Any]] = []
        for scenario in scenarios:
            entry = scenario_reports[scenario.scenario_id]
            aggregate = entry["metrics"]["all_unknown_parameters"].get("aggregate", {})
            joint = entry["metrics"]["all_unknown_parameters"].get("joint", {})
            if scenario.known_count == 0:
                headline_entry = {
                    "cases": entry["cases"],
                    "mean_normalized_mae": aggregate.get("mean_normalized_mae", float("nan")),
                    "mean_normalized_crps": aggregate.get("mean_normalized_crps", float("nan")),
                    "tarp_error": joint.get("tarp_mean_absolute_coverage_error", float("nan")),
                    "outside_fraction": entry["native_diagnostics"]["outside_training_support_fraction"],
                }
                if scenario.layer_count == 3:
                    headline_entry["surface_depth"] = entry["metrics"][
                        "all_unknown_parameters"
                    ]["parameters"].get(N3_SURFACE_DEPTH_NAME, {})
                no_prior_headline[str(scenario.layer_count)] = headline_entry
            if scenario.aliases and scenario.known_count:
                paired_parameters = [
                    row for row in prior_rows
                    if row["scenario_id"] == scenario.scenario_id and row["parameter"] != "__joint__"
                ]
                paired_joint = next(
                    (row for row in prior_rows if row["scenario_id"] == scenario.scenario_id and row["parameter"] == "__joint__"),
                    {},
                )
                named_headline.append({
                    "scenario_id": scenario.scenario_id,
                    "aliases": ";".join(scenario.aliases),
                    "layer_count": scenario.layer_count,
                    "known_count": scenario.known_count,
                    "mean_normalized_crps": aggregate.get("mean_normalized_crps", float("nan")),
                    "outside_fraction": entry["native_diagnostics"]["outside_training_support_fraction"],
                    "delta_normalized_mae": float(np.mean([row["delta_normalized_mae"] for row in paired_parameters])) if paired_parameters else float("nan"),
                    "delta_normalized_crps": float(np.mean([row["delta_normalized_crps"] for row in paired_parameters])) if paired_parameters else float("nan"),
                    "delta_normalized_width_90": float(np.mean([row["delta_normalized_width_90"] for row in paired_parameters])) if paired_parameters else float("nan"),
                    "delta_energy_score": paired_joint.get("delta_energy_score", float("nan")),
                    "relative_improvement_normalized_mae": float(np.mean([
                        row["relative_improvement_normalized_mae"] for row in paired_parameters
                    ])) if paired_parameters else float("nan"),
                    "relative_improvement_normalized_crps": float(np.mean([
                        row["relative_improvement_normalized_crps"] for row in paired_parameters
                    ])) if paired_parameters else float("nan"),
                    "relative_improvement_energy_score": paired_joint.get(
                        "relative_improvement_energy_score", float("nan")
                    ),
                    "improved_case_fraction": float(np.mean([row["improved_case_fraction"] for row in paired_parameters])) if paired_parameters else float("nan"),
                })
        intervention_evidence: list[dict[str, Any]] = []
        for scenario in intervention_scenarios:
            comparisons: dict[str, Any] = {}
            supported = True
            for comparison_id in (
                "exact_prior_vs_shuffled_prior",
                "exact_prior_vs_mask_only",
            ):
                macro = next(
                    (
                        row for row in intervention_effect_rows
                        if row["scenario_id"] == scenario.scenario_id
                        and row["comparison_id"] == comparison_id
                        and row["parameter"] == "__macro__"
                    ),
                    None,
                )
                comparisons[comparison_id] = macro
                if macro is None:
                    supported = False
                    continue
                crps_ci = macro.get("delta_normalized_crps_ci95", [float("nan")] * 2)
                energy_ci = macro.get("delta_energy_score_ci95", [float("nan")] * 2)
                supported = supported and bool(
                    len(crps_ci) == 2
                    and len(energy_ci) == 2
                    and np.isfinite(crps_ci[1])
                    and np.isfinite(energy_ci[1])
                    and crps_ci[1] < 0.0
                    and energy_ci[1] < 0.0
                )
            intervention_evidence.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "aliases": ";".join(scenario.aliases),
                    "layer_count": scenario.layer_count,
                    "known_count": scenario.known_count,
                    "numeric_value_use_supported": supported,
                    "comparisons": comparisons,
                }
            )
        supported_interventions = sum(
            bool(row["numeric_value_use_supported"])
            for row in intervention_evidence
        )
        if not intervention_evidence:
            intervention_conclusion = "not_evaluated"
        elif supported_interventions == len(intervention_evidence):
            intervention_conclusion = "consistently_supported"
        elif supported_interventions:
            intervention_conclusion = "mixed_evidence"
        else:
            intervention_conclusion = "not_supported"
        from .channel_evaluation import posterior_subsets
        channel_results = posterior_subsets(model, store, dtof_all, irf_all, truth_all,
            active_all, layers_all, sample_ids, scenarios, sds, device, cfg, posterior_seed, destination)
        numerical_failure_cases.extend(channel_results["numerical_failures"])
        numerical_failures = len(numerical_failure_cases)
        report = {
            "channel_subsets": channel_results,
            "status": (
                "completed_with_numerical_failures"
                if numerical_failures
                else "completed_with_diagnostic_warnings"
                if gmm_failure_cases
                else "completed"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_hash, "weights": selected_weights, "epoch": checkpoint_payload.get("epoch")},
            "scope": {"split": "test", "test_cases": len(sample_ids), "scenario_count": len(scenarios), "prior_intervention_scenario_count": len(intervention_scenarios) * len(intervention_modes), "irf_intervention_scenario_count": len(irf_scenarios) * len(irf_modes), "posterior_samples": sample_count, "solver_steps": solver_steps, "prior_protocol": "exhaustive_exact_with_named_counterfactuals", "time_alignment_enabled": time_alignment_enabled, "model_variant": model_variant, "model_parameter_count": sum(parameter.numel() for parameter in getattr(model, "parameters", lambda: ())())},
            "reproducibility": {"evaluation_seed": seed, "observation_seed": observation_seed, "posterior_seed": posterior_seed, "prior_intervention_seed": intervention_seed, "irf_intervention_seed": irf_seed, "fixed_irf_seed": fixed_irf_seed, "fixed_irf_sha256": fixed_irf_metadata.get("sha256"), "time_alignment_enabled": time_alignment_enabled, "dataset_fingerprint": fingerprint, "evaluation_signature": signature},
            "headline": {"no_prior_by_layer": no_prior_headline, "named_scenarios": named_headline, "prior_value_interventions": {"conclusion": intervention_conclusion, "supported_scenarios": supported_interventions, "evaluated_scenarios": len(intervention_evidence), "scenarios": intervention_evidence}},
            "scenario_reports": scenario_reports,
            "prior_intervention_reports": intervention_reports,
            "prior_intervention_summary": intervention_summary_rows,
            "prior_intervention_assignments": intervention_assignment_rows,
            "irf_intervention_reports": irf_reports,
            "irf_intervention_summary": irf_summary_rows,
            "irf_intervention_assignments": irf_assignment_rows,
            "irf_case_metadata": irf_case_metadata_rows,
            "irf_difficulty_metrics": irf_difficulty_rows,
            "fixed_irf_metadata": fixed_irf_metadata,
            "thickness_associations": thickness_associations,
            "prior_summary": prior_summary_rows,
            "multimodality_summary": {
                "initialization": GMM_INITIALIZATION,
                "evaluated_cases": len(multimodality_rows),
                "completed_cases": sum(
                    row.get("status") == "completed" for row in multimodality_rows
                ),
                "stable_candidate_multimodal_cases": sum(bool(row.get("stable_multimodal")) for row in multimodality_rows),
                "gmm_nonconverged_cases": sum(
                    row.get("status") == "gmm_nonconverged"
                    for row in multimodality_rows
                ),
                "cases_with_fit_failures": len(gmm_failure_cases),
                "gmm_retried_cases": gmm_retried_cases,
                "bootstrap_failed_replicates": bootstrap_failed_replicates,
                "bootstrap_skipped_replicates": bootstrap_skipped_replicates,
            },
            "gmm_failure_cases": gmm_failure_cases,
            "representative_mode_plots": representative_mode_plots,
            "numerical_failure_evaluations": numerical_failures,
            "numerical_failure_cases": numerical_failure_cases,
            "runtime": {"elapsed_seconds": time.perf_counter() - started, "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(), "torch_version": torch.__version__, "numpy_version": np.__version__, "python_version": platform.python_version()},
            "metric_definitions": {"point_estimate": "posterior median", "native_posterior": "one fixed draw set without rejection, top-up, or clipping", "normalized_mae": "physical MAE divided by the configured physical training-support width", "transformed_normalized_mae": "absolute error in the model's linear/log transformed coordinates divided by their normalized width of two; diagnostic only and distinct from headline nMAE", "normalized_crps": "CRPS in normalized model coordinates divided by two", "normalized_interval_width": "posterior interval width in normalized model coordinates divided by two", "inactive_parameters": "parameters not defined for the selected layer count are excluded from all losses and evaluation metrics", "paired_absolute_change": "candidate minus reference; negative means improvement", "paired_relative_improvement": "(reference minus candidate) divided by reference; positive means improvement", "paired_common_random_numbers": "paired conditions reuse the same DTOF, prior state, and posterior initial random draws for each case", "shuffled_prior": "same prior mask with the complete parameter vector supplied by a deterministic different case of the same layer count", "mask_only": "same prior mask with every known value fixed to zero in normalized coordinates; this is a neutral-value counterfactual, not a deployable prior", "shuffled_irf": "the paired four-channel IRF bundle is replaced by a deterministic different test case bundle of the same layer count", "fixed_irf": "all cases receive one deterministic four-channel measured-IRF medoid sampled only from the configured training acquisition distribution", "zero_irf": "all four IRF tensors are set to zero after preprocessing; this is an out-of-distribution destructive stress test, not a DTOF-only architecture", "channel_permuted_irf": "the four measured IRFs from the same case are cyclically permuted without fixed points, breaking the SDS-to-IRF pairing while preserving the bundle", "d1_plus_d2_mm": "N=3 third-layer starting depth, calculated by summing d1 and d2 within every joint posterior draw; normalized by the [1,35] mm study support and excluded only when both thicknesses are known", "overconfident_case": "truth lies outside the nominal 90% posterior interval", "tarp": "joint expected coverage from 32 deterministic random reference points per case", "candidate_multimodality": "stable GMM components; not proof of physical equivalence", "gmm_convergence": "unconverged fits are retried with stronger settings, recorded, and excluded from BIC selection; bootstrap failures remain in the stability denominator", "oracle_nearest_mode": "secondary diagnostic selected using test truth; it never replaces full-posterior accuracy or discards another mode"},
            "limitations": ["MCX-simulated test data are not external real-instrument validation.", "Exact priors are an idealized condition.", "Shuffled priors are deliberately incorrect and test sensitivity to numeric values; degradation alone does not prove beneficial use of correct priors.", "Mask-Only fixes known values to the transformed-domain midpoint and is not an uncertainty-aware or deployable prior.", ("IRF interventions occur after independent channel-paired time alignment and therefore evaluate explicit shape conditioning, not learned raw system-delay correction." if time_alignment_enabled else "IRF interventions occur after preprocessing without time alignment; the experiment therefore retains system-level and channel-specific delay variation."), "Channel permutation in all_shared cases primarily permutes independent measured-IRF noise because the underlying true IRFs are identical; interpret it separately from heterogeneous modes.", "Zero IRF is an out-of-distribution destructive input and must not be described as the DTOF-only architecture ablation.", "Candidate modes have not been forward-validated.", "Nearest-truth mode results are oracle diagnostics that use the test truth and therefore are not deployable accuracy estimates.", "Publication claims require multiple training seeds, baselines, ablations, and external validation."],
        }
        _atomic_json(destination / "report.json", report)
        html_path = destination / "report.html"
        temporary = html_path.with_name(f"{html_path.name}.tmp")
        temporary.write_text(
            render_test_report(
                report,
                parameter_rows=parameter_rows,
                thickness_rows=thickness_rows,
                prior_rows=prior_rows,
                prior_summary_rows=prior_summary_rows,
                prior_intervention_rows=intervention_effect_rows,
                prior_intervention_summary_rows=intervention_summary_rows,
                irf_intervention_rows=irf_effect_rows,
                irf_intervention_summary_rows=irf_summary_rows,
                thickness_associations=thickness_associations,
                multimodality_rows=multimodality_rows,
                asset_directory=destination,
            ),
            encoding="utf-8",
        )
        temporary.replace(html_path)
        print(json.dumps({"event": "test_complete", "report": str(html_path), "status": report["status"], "elapsed_seconds": report["runtime"]["elapsed_seconds"]}), flush=True)
        return html_path
    finally:
        store.close()
