from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .irf import GeneratedIRF, GeneratedIRFSet, fft_convolve_full, generate_channel_irfs


def _log_uniform_int(
    rng: np.random.Generator, bounds: list[int] | tuple[int, int]
) -> int:
    low, high = map(float, bounds)
    return int(round(np.exp(rng.uniform(np.log(low), np.log(high)))))


def _unit_sum(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Cannot normalize an empty waveform")
    return values / total


def _background_expectation(
    signal_total: float,
    length: int,
    fraction_bounds: list[float],
    drift_limit: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    fraction = float(
        np.exp(rng.uniform(np.log(fraction_bounds[0]), np.log(fraction_bounds[1])))
    )
    mean = max(signal_total * fraction / max(length, 1), 1e-12)
    slope = float(rng.uniform(-drift_limit, drift_limit))
    ramp = np.linspace(-1.0, 1.0, length, dtype=np.float64)
    return np.maximum(mean * (1.0 + slope * ramp), 0.0), fraction


def subtract_baseline(values: np.ndarray, baseline_bins: int) -> tuple[np.ndarray, float]:
    if baseline_bins <= 0 or baseline_bins >= len(values):
        raise ValueError("Invalid baseline window")
    baseline = float(np.mean(values[:baseline_bins]))
    return np.maximum(np.asarray(values, dtype=np.float64) - baseline, 0.0), baseline


def estimate_peak_bin(values: np.ndarray, smooth_bins: int = 7) -> float:
    values = np.asarray(values, dtype=np.float64)
    window = max(3, int(smooth_bins) | 1)
    if window >= len(values):
        window = (len(values) - 1) | 1
    kernel = np.hanning(window + 2)[1:-1]
    kernel /= kernel.sum()
    padded = np.pad(values, window // 2, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    peak = int(np.argmax(smoothed))
    if peak == 0 or peak == len(smoothed) - 1:
        return float(peak)
    left, center, right = smoothed[peak - 1 : peak + 2]
    denominator = left - 2.0 * center + right
    offset = 0.0 if abs(denominator) < 1e-20 else 0.5 * (left - right) / denominator
    return float(peak + np.clip(offset, -0.5, 0.5))


def _aligned_crop(
    values: np.ndarray,
    peak_bin: float,
    target_peak_bin: float,
    output_bins: int,
) -> np.ndarray:
    shift_bins = target_peak_bin - peak_bin
    source_positions = np.arange(output_bins, dtype=np.float64) - shift_bins
    source_indices = np.arange(len(values), dtype=np.float64)
    shifted = np.interp(source_positions, source_indices, values, left=0.0, right=0.0)
    return np.maximum(shifted, 0.0)


def _leading_crop(values: np.ndarray, output_bins: int) -> np.ndarray:
    """Keep the acquisition time origin unchanged and crop/pad at the end."""
    result = np.zeros(int(output_bins), dtype=np.float64)
    count = min(len(values), int(output_bins))
    result[:count] = np.asarray(values, dtype=np.float64)[:count]
    return np.maximum(result, 0.0)


def _preprocess_time_axis(
    values: np.ndarray,
    peak_bin: float,
    target_peak_bin: float,
    output_bins: int,
    alignment_enabled: bool,
) -> np.ndarray:
    if alignment_enabled:
        return _aligned_crop(values, peak_bin, target_peak_bin, output_bins)
    return _leading_crop(values, output_bins)


@dataclass(frozen=True)
class Measurement:
    dtof: np.ndarray
    irf: np.ndarray
    time_ps: np.ndarray
    metadata: dict[str, Any]


def _convolve_paired_channels(
    tpsf: np.ndarray, irf_set: GeneratedIRFSet
) -> np.ndarray:
    """Convolve each SDS channel only with its same-index true IRF."""
    tpsf = np.asarray(tpsf, dtype=np.float64)
    if tpsf.ndim != 2 or tpsf.shape[0] != len(irf_set.channels):
        raise ValueError("TPSF and IRF channel counts must match")
    return np.stack(
        [
            fft_convolve_full(tpsf_channel, irf.values)
            for tpsf_channel, irf in zip(tpsf, irf_set.channels)
        ],
        axis=0,
    )


def _artifact_summary(channel: GeneratedIRF) -> dict[str, Any]:
    kinds = [item.kind for item in channel.artifacts]
    return {
        "artifacts": [
            {
                "kind": item.kind,
                "fraction": item.fraction,
                "delay_ps": item.delay_ps,
                "width_scale": item.width_scale,
                "count": item.count,
                "spacing_ps": item.spacing_ps,
                "decay": item.decay,
            }
            for item in channel.artifacts
        ],
        "has_shoulder": "shoulder" in kinds,
        "has_satellite": "satellite" in kinds,
        "has_prepulse": "prepulse" in kinds,
        "has_echo_train": "echo_train" in kinds,
    }


def _irf_set_metadata(irf_set: GeneratedIRFSet) -> dict[str, Any]:
    channels = []
    for index, channel in enumerate(irf_set.channels):
        channels.append(
            {
                "channel_index": index,
                "family": channel.family,
                "system_delay_ps": channel.system_delay_ps,
                "true_peak_ps": channel.prompt_peak_ps,
                "actual_fwhm_ps": channel.actual_fwhm_ps,
                "core_fwhm_ps": channel.core_fwhm_ps,
                "beta": channel.beta,
                "tail_fraction": channel.tail_fraction,
                "tail_tau_ps": channel.tail_tau_ps,
                "second_tail_tau_ps": channel.second_tail_tau_ps,
                "skew_ratio": channel.skew_ratio,
                "stretch_exponent": channel.stretch_exponent,
                "true_irf_sha256": hashlib.sha256(
                    np.asarray(channel.values, dtype=np.float64).tobytes()
                ).hexdigest(),
                **_artifact_summary(channel),
            }
        )
    fwhm = np.asarray([item.actual_fwhm_ps for item in irf_set.channels])
    delays = np.asarray([item.system_delay_ps for item in irf_set.channels])
    return {
        "irf_channel_mode": irf_set.mode,
        "irf_group_ids": list(irf_set.group_ids),
        "irf_anomaly_mask": list(irf_set.anomaly_mask),
        "irf_anomaly_count": int(sum(irf_set.anomaly_mask)),
        "irf_delay_range_ps": float(np.ptp(delays)),
        "irf_fwhm_cv": float(np.std(fwhm) / max(float(np.mean(fwhm)), 1e-12)),
        "irf_channels": channels,
    }


def _pairwise_waveform_statistics(irf: np.ndarray) -> tuple[float, float]:
    correlations: list[float] = []
    distances: list[float] = []
    for left in range(len(irf)):
        for right in range(left + 1, len(irf)):
            a = np.asarray(irf[left], dtype=np.float64)
            b = np.asarray(irf[right], dtype=np.float64)
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            correlations.append(float(np.dot(a, b) / denominator) if denominator else 0.0)
            distances.append(float(np.linalg.norm(a - b)))
    return (float(np.mean(correlations)), float(np.mean(distances))) if correlations else (1.0, 0.0)


def simulate_measured_irf(
    dt_ps: float,
    output_bins: int,
    irf_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw and preprocess one four-channel measured-IRF bundle."""
    generated_set = generate_channel_irfs(dt_ps, irf_config, rng)
    baseline_bins = max(
        1,
        int(round(float(preprocessing_config["baseline_window_ps"]) / dt_ps)),
    )
    target_peak_bin = (
        float(preprocessing_config["alignment_anchor_ns"]) * 1000.0 / dt_ps
    )
    alignment_enabled = bool(preprocessing_config["time_alignment_enabled"])
    prepared_channels: list[np.ndarray] = []
    metadata = _irf_set_metadata(generated_set)
    for channel_index, generated in enumerate(generated_set.channels):
        irf_counts = _log_uniform_int(rng, irf_config["measured_counts"])
        signal = irf_counts * generated.values
        background, background_fraction = _background_expectation(
            float(signal.sum()), len(signal), [1e-4, 1e-1], 0.1, rng
        )
        measured = rng.poisson(signal + background).astype(np.float64)
        corrected, baseline = subtract_baseline(measured, baseline_bins)
        peak_bin = estimate_peak_bin(
            corrected, int(preprocessing_config["alignment_smooth_bins"])
        )
        estimated_shift_ps = (target_peak_bin - peak_bin) * dt_ps
        prepared = _unit_sum(
            _preprocess_time_axis(
                corrected,
                peak_bin,
                target_peak_bin,
                int(output_bins),
                alignment_enabled,
            )
        )
        prepared_channels.append(prepared)
        metadata["irf_channels"][channel_index].update(
            {
                "measured_counts": irf_counts,
                "background_fraction": background_fraction,
                "estimated_baseline": baseline,
                "measured_peak_before_alignment_ps": peak_bin * dt_ps,
                "peak_after_preprocessing_ps": estimate_peak_bin(
                    prepared, int(preprocessing_config["alignment_smooth_bins"])
                )
                * dt_ps,
                "estimated_alignment_shift_ps": estimated_shift_ps,
                "applied_alignment_shift_ps": (
                    estimated_shift_ps if alignment_enabled else 0.0
                ),
            }
        )
    result = np.stack(prepared_channels).astype(np.float32)
    correlation, distance = _pairwise_waveform_statistics(result)
    metadata.update(
        {
            "time_alignment_enabled": alignment_enabled,
            "irf_pairwise_correlation_mean": correlation,
            "irf_pairwise_distance_mean": distance,
        }
    )
    return result, metadata


def preprocess_observed(
    dtof_counts: np.ndarray,
    irf_counts: np.ndarray,
    dt_ps: float,
    config: dict[str, Any],
) -> Measurement:
    """Apply the deterministic real-data preprocessing contract.

    A single measured IRF may be broadcast for a shared-chain instrument.
    Otherwise each IRF channel is paired with, and optionally aligns, only its
    corresponding DTOF channel.
    """
    dtof_counts = np.asarray(dtof_counts, dtype=np.float64)
    irf_counts = np.asarray(irf_counts, dtype=np.float64)
    if dtof_counts.ndim != 2 or not 1 <= dtof_counts.shape[0] <= 4:
        raise ValueError("dtof_counts must have shape [C,T], C=1..4")
    broadcast = irf_counts.ndim == 1
    if broadcast:
        irf_counts = np.broadcast_to(irf_counts[None, :], dtof_counts.shape).copy()
    if irf_counts.ndim != 2 or irf_counts.shape[0] != dtof_counts.shape[0]:
        raise ValueError("irf_counts must have shape [C,T] or [T]")
    if dtof_counts.shape[1] != irf_counts.shape[1]:
        raise ValueError("DTOF and IRF must share the same time grid")
    if np.any(~np.isfinite(dtof_counts)) or np.any(~np.isfinite(irf_counts)):
        raise ValueError("Observed waveforms contain NaN or Inf")
    if np.any(dtof_counts < 0) or np.any(irf_counts < 0):
        raise ValueError("Raw observed counts must be nonnegative")
    baseline_bins = max(1, int(round(float(config["baseline_window_ps"]) / dt_ps)))
    target_peak_bin = float(config["alignment_anchor_ns"]) * 1000.0 / dt_ps
    alignment_enabled = bool(config["time_alignment_enabled"])
    output_bins = irf_counts.shape[1]
    prepared_irf: list[np.ndarray] = []
    prepared_dtof: list[np.ndarray] = []
    irf_baselines: list[float] = []
    dtof_baselines: list[float] = []
    peak_before: list[float] = []
    peak_after: list[float] = []
    estimated_shifts: list[float] = []
    applied_shifts: list[float] = []
    for dtof_channel, irf_channel in zip(dtof_counts, irf_counts):
        irf_corrected, irf_baseline = subtract_baseline(irf_channel, baseline_bins)
        peak_bin = estimate_peak_bin(
            irf_corrected, int(config["alignment_smooth_bins"])
        )
        prepared_irf_channel = _unit_sum(
            _preprocess_time_axis(
                irf_corrected,
                peak_bin,
                target_peak_bin,
                output_bins,
                alignment_enabled,
            )
        )
        corrected, baseline = subtract_baseline(dtof_channel, baseline_bins)
        prepared_dtof.append(
            _unit_sum(
                _preprocess_time_axis(
                    corrected,
                    peak_bin,
                    target_peak_bin,
                    output_bins,
                    alignment_enabled,
                )
            )
        )
        prepared_irf.append(prepared_irf_channel)
        irf_baselines.append(irf_baseline)
        dtof_baselines.append(baseline)
        estimated = (target_peak_bin - peak_bin) * dt_ps
        peak_before.append(peak_bin * dt_ps)
        peak_after.append(
            estimate_peak_bin(
                prepared_irf_channel, int(config["alignment_smooth_bins"])
            )
            * dt_ps
        )
        estimated_shifts.append(estimated)
        applied_shifts.append(estimated if alignment_enabled else 0.0)
    irf_array = np.stack(prepared_irf).astype(np.float32)
    correlation, distance = _pairwise_waveform_statistics(irf_array)
    return Measurement(
        dtof=np.stack(prepared_dtof).astype(np.float32),
        irf=irf_array,
        time_ps=np.arange(output_bins, dtype=np.float32) * float(dt_ps),
        metadata={
            "irf_measured_peak_before_alignment_ps": peak_before,
            "irf_peak_after_preprocessing_ps": peak_after,
            "irf_estimated_baseline": irf_baselines,
            "dtof_estimated_baseline": dtof_baselines,
            "time_alignment_enabled": alignment_enabled,
            "estimated_alignment_shift_ps": estimated_shifts,
            "applied_alignment_shift_ps": applied_shifts,
            "irf_broadcast_from_single_channel": broadcast,
            "irf_pairwise_correlation_mean": correlation,
            "irf_pairwise_distance_mean": distance,
        },
    )


def simulate_measurement(
    tpsf: np.ndarray,
    dt_ps: float,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> Measurement:
    tpsf = np.asarray(tpsf, dtype=np.float64)
    if tpsf.ndim != 2 or tpsf.shape[0] != 4:
        raise ValueError(f"Expected TPSF [4,T], got {tpsf.shape}")
    if np.any(~np.isfinite(tpsf)) or np.any(tpsf < 0):
        raise ValueError("TPSF must be finite and non-negative")
    output_bins = int(tpsf.shape[1])
    irf_cfg = config["irf"]
    dtof_cfg = config["dtof"]
    prep_cfg = config["preprocessing"]
    generated_set = generate_channel_irfs(dt_ps, irf_cfg, rng)

    ideal = _convolve_paired_channels(tpsf, generated_set)
    count_40 = _log_uniform_int(rng, dtof_cfg["count_40mm"])
    sum_40 = float(ideal[-1].sum())
    if sum_40 <= 0:
        raise ValueError("40 mm TPSF has no detected weight")
    exposure = count_40 / sum_40

    noisy_dtof: list[np.ndarray] = []
    dtof_background_fractions: list[float] = []
    for channel in ideal:
        signal = exposure * channel
        background, fraction = _background_expectation(
            float(signal.sum()),
            len(signal),
            list(map(float, dtof_cfg["background_fraction"])),
            float(dtof_cfg["background_drift_fraction"]),
            rng,
        )
        noisy_dtof.append(rng.poisson(signal + background).astype(np.float64))
        dtof_background_fractions.append(fraction)

    baseline_bins = max(1, int(round(float(prep_cfg["baseline_window_ps"]) / dt_ps)))
    target_peak_bin = float(prep_cfg["alignment_anchor_ns"]) * 1000.0 / dt_ps
    alignment_enabled = bool(prep_cfg["time_alignment_enabled"])
    prepared_irf: list[np.ndarray] = []
    prepared_dtof: list[np.ndarray] = []
    dtof_baselines: list[float] = []
    metadata = _irf_set_metadata(generated_set)
    for channel_index, (dtof_channel, generated) in enumerate(
        zip(noisy_dtof, generated_set.channels)
    ):
        irf_counts = _log_uniform_int(rng, irf_cfg["measured_counts"])
        irf_signal = irf_counts * generated.values
        irf_background, irf_background_fraction = _background_expectation(
            float(irf_signal.sum()), len(irf_signal), [1e-4, 1e-1], 0.1, rng
        )
        measured_irf = rng.poisson(irf_signal + irf_background).astype(np.float64)
        irf_corrected, irf_baseline = subtract_baseline(measured_irf, baseline_bins)
        peak_bin = estimate_peak_bin(
            irf_corrected, int(prep_cfg["alignment_smooth_bins"])
        )
        irf_channel = _unit_sum(
            _preprocess_time_axis(
                irf_corrected,
                peak_bin,
                target_peak_bin,
                output_bins,
                alignment_enabled,
            )
        )
        corrected, baseline = subtract_baseline(dtof_channel, baseline_bins)
        prepared = _preprocess_time_axis(
            corrected,
            peak_bin,
            target_peak_bin,
            output_bins,
            alignment_enabled,
        )
        prepared_dtof.append(_unit_sum(prepared))
        prepared_irf.append(irf_channel)
        dtof_baselines.append(baseline)
        estimated_shift_ps = (target_peak_bin - peak_bin) * dt_ps
        metadata["irf_channels"][channel_index].update(
            {
                "measured_counts": irf_counts,
                "background_fraction": irf_background_fraction,
                "estimated_baseline": irf_baseline,
                "measured_peak_before_alignment_ps": peak_bin * dt_ps,
                "peak_after_preprocessing_ps": estimate_peak_bin(
                    irf_channel, int(prep_cfg["alignment_smooth_bins"])
                )
                * dt_ps,
                "estimated_alignment_shift_ps": estimated_shift_ps,
                "applied_alignment_shift_ps": (
                    estimated_shift_ps if alignment_enabled else 0.0
                ),
            }
        )
    dtof = np.stack(prepared_dtof, axis=0).astype(np.float32)
    irf = np.stack(prepared_irf, axis=0).astype(np.float32)
    correlation, distance = _pairwise_waveform_statistics(irf)
    metadata.update({
        "dtof_count_40mm": count_40,
        "dtof_exposure": exposure,
        "dtof_background_fraction": dtof_background_fractions,
        "dtof_background_fraction_reference": "full_linear_convolution_window",
        "dtof_estimated_baseline": dtof_baselines,
        "time_alignment_enabled": alignment_enabled,
        "irf_pairwise_correlation_mean": correlation,
        "irf_pairwise_distance_mean": distance,
        "irf_broadcast_from_single_channel": False,
    })
    return Measurement(
        dtof=dtof,
        irf=irf,
        time_ps=np.arange(output_bins, dtype=np.float32) * float(dt_ps),
        metadata=metadata,
    )
