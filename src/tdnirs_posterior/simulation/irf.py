from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


IRF_FAMILIES = (
    "generalized_gaussian",
    "ex_gaussian",
    "single_tail",
    "double_tail",
    "stretched_tail",
    "skewed_peak",
)
IRF_CHANNEL_MODES = (
    "all_shared",
    "four_similar",
    "grouped_2_2",
    "grouped_3_1",
    "partial_outlier",
    "heterogeneous",
)


def fft_convolve_full(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Linear convolution implemented with NumPy FFTs."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    output_length = left.size + right.size - 1
    fft_length = 1 << (output_length - 1).bit_length()
    result = np.fft.irfft(
        np.fft.rfft(left, fft_length) * np.fft.rfft(right, fft_length),
        fft_length,
    )[:output_length]
    return np.maximum(result, 0.0)


def _log_uniform(
    rng: np.random.Generator, bounds: list[float] | tuple[float, float]
) -> float:
    low, high = map(float, bounds)
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _unit_sum(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("IRF has zero or non-finite area")
    return values / total


def shift_curve(values: np.ndarray, shift_bins: float) -> np.ndarray:
    indices = np.arange(len(values), dtype=np.float64)
    return np.interp(indices - shift_bins, indices, values, left=0.0, right=0.0)


def _fwhm_ps(values: np.ndarray, dt_ps: float) -> float:
    if values.size == 0 or float(values.max()) <= 0:
        return float("nan")
    above = np.flatnonzero(values >= 0.5 * float(values.max()))
    if above.size < 2:
        return dt_ps
    return float((above[-1] - above[0] + 1) * dt_ps)


def _weighted_choice(
    rng: np.random.Generator, probabilities: dict[str, float], allowed: tuple[str, ...]
) -> str:
    if set(probabilities) != set(allowed):
        raise ValueError(f"Probability keys must be exactly {allowed}")
    weights = np.asarray([float(probabilities[name]) for name in allowed])
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError("Probabilities must be non-negative and sum to one")
    return str(rng.choice(allowed, p=weights))


@dataclass(frozen=True)
class IRFArtifact:
    kind: str
    fraction: float
    delay_ps: float
    width_scale: float
    count: int = 1
    spacing_ps: float = 0.0
    decay: float = 1.0


@dataclass(frozen=True)
class IRFSpec:
    family: str
    core_fwhm_ps: float
    beta: float
    tail_fraction: float
    tail_tau_ps: float
    second_tail_tau_ps: float
    skew_ratio: float
    stretch_exponent: float
    system_delay_ps: float
    artifacts: tuple[IRFArtifact, ...]


@dataclass(frozen=True)
class GeneratedIRF:
    values: np.ndarray
    dt_ps: float
    family: str
    system_delay_ps: float
    prompt_peak_ps: float
    actual_fwhm_ps: float
    core_fwhm_ps: float
    beta: float
    tail_fraction: float
    tail_tau_ps: float
    second_tail_tau_ps: float
    skew_ratio: float
    stretch_exponent: float
    artifacts: tuple[IRFArtifact, ...]

    @property
    def satellite_fraction(self) -> float:
        return float(
            sum(item.fraction for item in self.artifacts if item.kind == "satellite")
        )

    @property
    def satellite_delay_ps(self) -> float:
        values = [item.delay_ps for item in self.artifacts if item.kind == "satellite"]
        return float(max(values, default=0.0))


@dataclass(frozen=True)
class GeneratedIRFSet:
    channels: tuple[GeneratedIRF, ...]
    mode: str
    group_ids: tuple[int, ...]
    anomaly_mask: tuple[bool, ...]

    @property
    def values(self) -> np.ndarray:
        return np.stack([channel.values for channel in self.channels], axis=0)


def _sample_artifacts(config: dict[str, Any], rng: np.random.Generator) -> tuple[IRFArtifact, ...]:
    artifacts: list[IRFArtifact] = []
    cfg = config["artifacts"]
    for kind in ("shoulder", "satellite", "prepulse", "echo_train"):
        item = cfg[kind]
        if rng.random() >= float(item["probability"]):
            continue
        count = int(rng.integers(int(item.get("count", [1, 1])[0]), int(item.get("count", [1, 1])[1]) + 1))
        delay = float(rng.uniform(*map(float, item["delay_ps"])))
        if kind == "prepulse":
            delay = -delay
        artifacts.append(
            IRFArtifact(
                kind=kind,
                fraction=float(rng.uniform(*map(float, item["fraction"]))),
                delay_ps=delay,
                width_scale=float(rng.uniform(*map(float, item["width_scale"]))),
                count=count,
                spacing_ps=float(rng.uniform(*map(float, item.get("spacing_ps", [0.0, 0.0])))),
                decay=float(rng.uniform(*map(float, item.get("decay", [1.0, 1.0])))),
            )
        )
    return tuple(artifacts)


def _cap_noncore(spec: IRFSpec, maximum: float) -> IRFSpec:
    total = spec.tail_fraction + sum(item.fraction for item in spec.artifacts)
    if total <= maximum or total <= 0.0:
        return spec
    scale = float(maximum) / total
    return replace(
        spec,
        tail_fraction=spec.tail_fraction * scale,
        artifacts=tuple(replace(item, fraction=item.fraction * scale) for item in spec.artifacts),
    )


def sample_irf_spec(config: dict[str, Any], rng: np.random.Generator) -> IRFSpec:
    family = _weighted_choice(rng, config["family_probabilities"], IRF_FAMILIES)
    tail_fraction = float(rng.uniform(*map(float, config["tail_fraction"])))
    if family in {"generalized_gaussian", "skewed_peak"}:
        tail_fraction = 0.0
    tau = _log_uniform(rng, config["tail_tau_ps"])
    second_factor = float(rng.uniform(*map(float, config["second_tail_multiplier"])))
    spec = IRFSpec(
        family=family,
        core_fwhm_ps=_log_uniform(rng, config["fwhm_ps"]),
        beta=float(rng.uniform(*map(float, config["beta"]))),
        tail_fraction=tail_fraction,
        tail_tau_ps=tau,
        second_tail_tau_ps=min(tau * second_factor, float(config["max_second_tail_tau_ps"])),
        skew_ratio=float(rng.uniform(*map(float, config["skew_ratio"]))),
        stretch_exponent=float(rng.uniform(*map(float, config["stretch_exponent"]))),
        system_delay_ps=float(rng.uniform(*map(float, config["system_delay_ns"]))) * 1000.0,
        artifacts=_sample_artifacts(config, rng),
    )
    return _cap_noncore(spec, float(config["max_noncore_fraction"]))


def _perturb_artifacts(
    artifacts: tuple[IRFArtifact, ...], amplitude_fraction: float, delay_ps: float,
    rng: np.random.Generator,
) -> tuple[IRFArtifact, ...]:
    return tuple(
        replace(
            item,
            fraction=max(0.0, item.fraction * (1.0 + rng.uniform(-amplitude_fraction, amplitude_fraction))),
            delay_ps=item.delay_ps + rng.uniform(-delay_ps, delay_ps),
        )
        for item in artifacts
    )


def _truncated_normal(rng: np.random.Generator, sigma: float, limit: float) -> float:
    return float(np.clip(rng.normal(0.0, sigma), -limit, limit))


def _perturb_spec(
    spec: IRFSpec, config: dict[str, Any], perturb: dict[str, Any],
    rng: np.random.Generator,
) -> IRFSpec:
    fwhm_sigma = float(perturb["fwhm_log_sigma"])
    fwhm_bounds = list(map(float, perturb["fwhm_scale_clip"]))
    fwhm_scale = float(np.clip(np.exp(rng.normal(0.0, fwhm_sigma)), *fwhm_bounds))
    tau_bounds = list(map(float, perturb["tail_tau_scale"]))
    tau_scale = float(np.exp(rng.uniform(np.log(tau_bounds[0]), np.log(tau_bounds[1]))))
    tail_delta = float(rng.uniform(-float(perturb["tail_fraction_delta"]), float(perturb["tail_fraction_delta"])))
    artifacts = spec.artifacts
    if rng.random() < float(perturb.get("artifact_toggle_probability", 0.0)):
        artifacts = _sample_artifacts(config, rng)
    else:
        artifacts = _perturb_artifacts(
            artifacts,
            float(perturb["artifact_amplitude_relative"]),
            float(perturb["artifact_delay_jitter_ps"]),
            rng,
        )
    updated = replace(
        spec,
        core_fwhm_ps=float(np.clip(spec.core_fwhm_ps * fwhm_scale, 30.0, 2000.0)),
        beta=float(np.clip(spec.beta + rng.uniform(-float(perturb["beta_delta"]), float(perturb["beta_delta"])), 1.1, 5.0)),
        tail_fraction=float(np.clip(spec.tail_fraction + tail_delta, 0.0, 0.5)),
        tail_tau_ps=float(np.clip(spec.tail_tau_ps * tau_scale, 20.0, 4000.0)),
        second_tail_tau_ps=float(np.clip(spec.second_tail_tau_ps * tau_scale, 20.0, 4000.0)),
        system_delay_ps=spec.system_delay_ps + _truncated_normal(
            rng, float(perturb["delay_sigma_ps"]), float(perturb["delay_clip_ps"])
        ),
        artifacts=artifacts,
    )
    return _cap_noncore(updated, float(config["max_noncore_fraction"]))


def _outlier_spec(spec: IRFSpec, config: dict[str, Any], rng: np.random.Generator) -> IRFSpec:
    outlier = config["channel_model"]["outlier"]
    fresh = sample_irf_spec(config, rng)
    sign = -1.0 if rng.random() < 0.5 else 1.0
    delay = spec.system_delay_ps + sign * float(rng.uniform(*map(float, outlier["delay_magnitude_ps"])))
    scale = float(rng.uniform(*map(float, outlier["fwhm_scale"])))
    updated = replace(
        fresh,
        core_fwhm_ps=float(np.clip(spec.core_fwhm_ps * scale, *map(float, outlier["fwhm_clip_ps"]))),
        tail_fraction=float(rng.uniform(0.0, float(outlier["max_tail_fraction"]))),
        tail_tau_ps=_log_uniform(rng, [50.0, float(outlier["max_tail_tau_ps"])]),
        system_delay_ps=delay,
    )
    return _cap_noncore(updated, float(config["max_noncore_fraction"]))


def render_irf_spec(dt_ps: float, config: dict[str, Any], spec: IRFSpec) -> GeneratedIRF:
    work_bins = int(round(float(config["work_window_ns"]) * 1000.0 / dt_ps))
    if work_bins < 16:
        raise ValueError("IRF work grid is too short")
    t_ps = np.arange(work_bins, dtype=np.float64) * dt_ps
    anchor_ps = float(config["canonical_peak_ns"]) * 1000.0
    left_fwhm = spec.core_fwhm_ps
    right_fwhm = spec.core_fwhm_ps * (spec.skew_ratio if spec.family == "skewed_peak" else 1.0)
    left_alpha = left_fwhm / (2.0 * np.log(2.0) ** (1.0 / spec.beta))
    right_alpha = right_fwhm / (2.0 * np.log(2.0) ** (1.0 / spec.beta))
    alpha = np.where(t_ps < anchor_ps, left_alpha, right_alpha)
    core = _unit_sum(np.exp(-np.abs((t_ps - anchor_ps) / np.maximum(alpha, dt_ps / 4.0)) ** spec.beta))
    causal = _unit_sum(np.where(t_ps >= anchor_ps, np.exp(-(t_ps - anchor_ps) / spec.tail_tau_ps), 0.0))
    if spec.family == "ex_gaussian":
        kernel = _unit_sum(np.exp(-t_ps / spec.tail_tau_ps))
        tail_shape = _unit_sum(fft_convolve_full(core, kernel)[:work_bins])
    elif spec.family == "double_tail":
        second = _unit_sum(np.where(t_ps >= anchor_ps, np.exp(-(t_ps - anchor_ps) / spec.second_tail_tau_ps), 0.0))
        tail_shape = 0.65 * causal + 0.35 * second
    elif spec.family == "stretched_tail":
        elapsed = np.maximum(t_ps - anchor_ps, 0.0)
        stretched = np.exp(
            -(elapsed / spec.tail_tau_ps) ** spec.stretch_exponent
        )
        stretched[t_ps < anchor_ps] = 0.0
        tail_shape = _unit_sum(stretched)
    else:
        tail_shape = causal

    components: list[tuple[float, np.ndarray]] = []
    for artifact in spec.artifacts:
        peaks = np.zeros(work_bins, dtype=np.float64)
        for index in range(artifact.count):
            delay = artifact.delay_ps + index * artifact.spacing_ps
            weight = artifact.decay**index
            width = max(spec.core_fwhm_ps * artifact.width_scale, dt_ps)
            sigma = width / 2.354820045
            peaks += weight * np.exp(-0.5 * ((t_ps - anchor_ps - delay) / sigma) ** 2)
        components.append((artifact.fraction, _unit_sum(peaks)))
    noncore = spec.tail_fraction + sum(weight for weight, _ in components)
    shape = (1.0 - noncore) * core + spec.tail_fraction * tail_shape
    for weight, component in components:
        shape += weight * component
    shape = _unit_sum(shape)
    detected_peak = int(np.argmax(shape))
    target_peak = int(round(anchor_ps / dt_ps))
    shape = _unit_sum(shift_curve(shape, target_peak - detected_peak))
    shape = _unit_sum(shift_curve(shape, spec.system_delay_ps / dt_ps))
    peak_ps = float(np.argmax(shape) * dt_ps)
    return GeneratedIRF(
        values=shape,
        dt_ps=float(dt_ps),
        family=spec.family,
        system_delay_ps=spec.system_delay_ps,
        prompt_peak_ps=peak_ps,
        actual_fwhm_ps=_fwhm_ps(shape, dt_ps),
        core_fwhm_ps=spec.core_fwhm_ps,
        beta=spec.beta,
        tail_fraction=spec.tail_fraction,
        tail_tau_ps=spec.tail_tau_ps,
        second_tail_tau_ps=spec.second_tail_tau_ps,
        skew_ratio=spec.skew_ratio,
        stretch_exponent=spec.stretch_exponent,
        artifacts=spec.artifacts,
    )


def generate_true_irf(
    dt_ps: float, config: dict[str, Any], rng: np.random.Generator
) -> GeneratedIRF:
    return render_irf_spec(dt_ps, config, sample_irf_spec(config, rng))


def generate_channel_irfs(
    dt_ps: float, config: dict[str, Any], rng: np.random.Generator,
    channel_count: int = 4,
) -> GeneratedIRFSet:
    if channel_count != 4:
        raise ValueError("The scientific protocol requires exactly four IRF channels")
    model = config["channel_model"]
    mode = _weighted_choice(rng, model["mode_probabilities"], IRF_CHANNEL_MODES)
    base = sample_irf_spec(config, rng)
    small = model["slight_perturbation"]
    moderate = model["moderate_perturbation"]
    anomaly = np.zeros(channel_count, dtype=bool)

    if mode == "all_shared":
        specs = [base] * channel_count
        groups = np.zeros(channel_count, dtype=int)
    elif mode == "four_similar":
        specs = [_perturb_spec(base, config, small, rng) for _ in range(channel_count)]
        groups = np.zeros(channel_count, dtype=int)
    elif mode in {"grouped_2_2", "grouped_3_1"}:
        permutation = rng.permutation(channel_count)
        split = 2 if mode == "grouped_2_2" else 3
        groups = np.ones(channel_count, dtype=int)
        groups[permutation[:split]] = 0
        centers = [_perturb_spec(base, config, moderate, rng) for _ in range(2)]
        half_small = dict(small)
        for key in ("delay_sigma_ps", "delay_clip_ps", "beta_delta", "tail_fraction_delta", "artifact_amplitude_relative", "artifact_delay_jitter_ps", "fwhm_log_sigma"):
            half_small[key] = 0.5 * float(half_small[key])
        half_small["fwhm_scale_clip"] = [
            float(np.sqrt(value)) for value in small["fwhm_scale_clip"]
        ]
        half_small["tail_tau_scale"] = [
            float(np.sqrt(value)) for value in small["tail_tau_scale"]
        ]
        specs = [_perturb_spec(centers[int(groups[index])], config, half_small, rng) for index in range(channel_count)]
    elif mode == "partial_outlier":
        specs = [_perturb_spec(base, config, small, rng) for _ in range(channel_count)]
        outlier_count = 1 if rng.random() < float(model["outlier"]["one_channel_probability"]) else 2
        selected = rng.choice(channel_count, size=outlier_count, replace=False)
        anomaly[selected] = True
        for index in selected:
            specs[int(index)] = _outlier_spec(base, config, rng)
        groups = np.where(anomaly, np.arange(channel_count) + 1, 0)
    else:
        specs = [sample_irf_spec(config, rng) for _ in range(channel_count)]
        groups = np.arange(channel_count, dtype=int)

    generated = tuple(render_irf_spec(dt_ps, config, spec) for spec in specs)
    return GeneratedIRFSet(
        channels=generated,
        mode=mode,
        group_ids=tuple(map(int, groups)),
        anomaly_mask=tuple(map(bool, anomaly)),
    )
