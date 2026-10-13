from __future__ import annotations

from typing import Any

import numpy as np

from .parameters import CANONICAL_PARAMETERS, ParameterSchema


def _crps_rows(draws: np.ndarray, target: np.ndarray) -> np.ndarray:
    ordered = np.sort(draws, axis=1)
    count = draws.shape[1]
    coefficients = 2.0 * np.arange(count) - count + 1.0
    pairwise = 2.0 * np.sum(ordered * coefficients[None, :], axis=1) / count**2
    return np.mean(np.abs(draws - target[:, None]), axis=1) - 0.5 * pairwise


def evaluate_casewise_scalar_posterior(
    draws: np.ndarray,
    truth: np.ndarray,
    support_bounds: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Return per-case scores for a scalar posterior-derived quantity."""

    draws = np.asarray(draws, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    low, high = map(float, support_bounds)
    width = high - low
    if draws.ndim != 2 or truth.shape != (draws.shape[0],):
        raise ValueError("Expected scalar draws [cases,draws] and truth [cases]")
    if draws.shape[1] < 2:
        raise ValueError("At least two posterior draws are required")
    if width <= 0.0:
        raise ValueError("Scalar support bounds must have positive width")
    if not np.all(np.isfinite(draws)) or not np.all(np.isfinite(truth)):
        raise ValueError("Scalar posterior draws and truth must be finite")
    medians = np.median(draws, axis=1)
    lower_90 = np.quantile(draws, 0.05, axis=1)
    upper_90 = np.quantile(draws, 0.95, axis=1)
    return {
        "median": medians,
        "absolute_error": np.abs(medians - truth),
        "normalized_absolute_error": np.abs(medians - truth) / width,
        "crps": _crps_rows(draws, truth),
        "normalized_crps": _crps_rows(draws, truth) / width,
        "width_90": upper_90 - lower_90,
        "normalized_width_90": (upper_90 - lower_90) / width,
        "covered_90": ((truth >= lower_90) & (truth <= upper_90)).astype(np.float64),
    }


def evaluate_scalar_posterior(
    draws: np.ndarray,
    truth: np.ndarray,
    support_bounds: tuple[float, float],
) -> dict[str, Any]:
    """Evaluate a scalar quantity derived jointly from posterior draws."""

    draws = np.asarray(draws, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    low, high = map(float, support_bounds)
    width = high - low
    casewise = evaluate_casewise_scalar_posterior(draws, truth, support_bounds)
    estimate = casewise["median"]
    error = estimate - truth
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    ranks = np.sum(draws < truth[:, None], axis=1)
    normalized_ranks = (ranks + 0.5) / (draws.shape[1] + 1.0)
    rank_histogram, _ = np.histogram(normalized_ranks, bins=10, range=(0.0, 1.0))
    deviations = np.abs(draws - np.median(draws, axis=1, keepdims=True))
    result: dict[str, Any] = {
        "count": int(len(truth)),
        "bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_mae": float(np.mean(np.abs(error)) / width),
        "r2": float(1.0 - np.sum(error**2) / denominator)
        if denominator > 0.0
        else float("nan"),
        "crps": float(np.mean(casewise["crps"])),
        "normalized_crps": float(np.mean(casewise["normalized_crps"])),
        "posterior_std": float(np.mean(np.std(draws, axis=1, ddof=1))),
        "posterior_iqr": float(
            np.mean(
                np.quantile(draws, 0.75, axis=1)
                - np.quantile(draws, 0.25, axis=1)
            )
        ),
        "posterior_mad": float(np.mean(np.median(deviations, axis=1))),
        "sbc_rank_mean": float(np.mean(normalized_ranks)),
        "sbc_rank_histogram_10": rank_histogram.tolist(),
    }
    for level in (0.5, 0.8, 0.9, 0.95):
        tail = (1.0 - level) / 2.0
        lower_interval = np.quantile(draws, tail, axis=1)
        upper_interval = np.quantile(draws, 1.0 - tail, axis=1)
        result[f"coverage_{int(level * 100)}"] = float(
            np.mean((truth >= lower_interval) & (truth <= upper_interval))
        )
        result[f"width_{int(level * 100)}"] = float(
            np.mean(upper_interval - lower_interval)
        )
    return result


def evaluate_casewise_posterior(
    samples: np.ndarray, truth: np.ndarray, evaluation_mask: np.ndarray
) -> dict[str, np.ndarray]:
    """Return paired per-case/per-parameter scores for audit and prior deltas."""
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    evaluation_mask = np.asarray(evaluation_mask, dtype=bool)
    schema = ParameterSchema()
    normalized_samples = schema.normalize(samples)
    normalized_truth = schema.normalize(truth)
    medians = np.median(samples, axis=1)
    normalized_medians = np.median(normalized_samples, axis=1)
    low, high, _ = schema._bounds()
    physical_widths = (high - low).astype(np.float64)
    shape = truth.shape
    absolute_error = np.full(shape, np.nan, dtype=np.float64)
    normalized_absolute_error = np.full(shape, np.nan, dtype=np.float64)
    crps = np.full(shape, np.nan, dtype=np.float64)
    normalized_crps = np.full(shape, np.nan, dtype=np.float64)
    width_90 = np.full(shape, np.nan, dtype=np.float64)
    normalized_width_90 = np.full(shape, np.nan, dtype=np.float64)
    covered_90 = np.full(shape, np.nan, dtype=np.float64)
    for index in range(len(CANONICAL_PARAMETERS)):
        selected = evaluation_mask[:, index]
        if not np.any(selected):
            continue
        physical_draws = samples[selected, :, index]
        normalized_draws = normalized_samples[selected, :, index]
        absolute_error[selected, index] = np.abs(
            medians[selected, index] - truth[selected, index]
        )
        normalized_absolute_error[selected, index] = (
            absolute_error[selected, index] / physical_widths[index]
        )
        crps[selected, index] = _crps_rows(
            physical_draws, truth[selected, index]
        )
        normalized_crps[selected, index] = (
            _crps_rows(normalized_draws, normalized_truth[selected, index]) / 2.0
        )
        width_90[selected, index] = np.quantile(
            physical_draws, 0.95, axis=1
        ) - np.quantile(physical_draws, 0.05, axis=1)
        normalized_width_90[selected, index] = (
            np.quantile(normalized_draws, 0.95, axis=1)
            - np.quantile(normalized_draws, 0.05, axis=1)
        ) / 2.0
        lower_90 = np.quantile(physical_draws, 0.05, axis=1)
        upper_90 = np.quantile(physical_draws, 0.95, axis=1)
        covered_90[selected, index] = (
            (truth[selected, index] >= lower_90)
            & (truth[selected, index] <= upper_90)
        ).astype(np.float64)
    return {
        "median": medians,
        "absolute_error": absolute_error,
        "normalized_absolute_error": normalized_absolute_error,
        "transformed_normalized_absolute_error": np.where(
            evaluation_mask,
            np.abs(normalized_medians - normalized_truth) / 2.0,
            np.nan,
        ),
        "crps": crps,
        "normalized_crps": normalized_crps,
        "width_90": width_90,
        "normalized_width_90": normalized_width_90,
        "covered_90": covered_90,
    }


def _joint_scores(
    normalized_samples: np.ndarray,
    normalized_truth: np.ndarray,
    evaluation_mask: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    energy: list[float] = []
    concentration: list[float] = []
    volume: list[float] = []
    covariance_volume: list[float] = []
    coverages: list[float] = []
    correlation_sums = np.zeros((len(CANONICAL_PARAMETERS), len(CANONICAL_PARAMETERS)))
    correlation_counts = np.zeros_like(correlation_sums)
    for case in range(normalized_samples.shape[0]):
        mask = evaluation_mask[case]
        dimension = int(mask.sum())
        if not dimension:
            continue
        draws = normalized_samples[case][:, mask] / 2.0
        target = normalized_truth[case, mask] / 2.0
        scale = np.sqrt(dimension)
        target_distance = np.linalg.norm(draws - target[None, :], axis=1) / scale
        paired_distance = np.linalg.norm(draws - np.roll(draws, 1, axis=0), axis=1) / scale
        energy.append(float(target_distance.mean() - 0.5 * paired_distance.mean()))
        concentration.append(float(paired_distance.mean()))
        centered = draws - np.mean(draws, axis=0, keepdims=True)
        covariance = np.zeros((dimension, dimension), dtype=np.float64)
        denominator = max(len(draws) - 1, 1)
        for row in range(dimension):
            for column in range(dimension):
                covariance[row, column] = float(
                    np.sum(centered[:, row] * centered[:, column]) / denominator
                )
        # Cholesky in scalar loops avoids platform BLAS instability while
        # retaining a covariance-volume measure in at most eight dimensions.
        cholesky = np.zeros_like(covariance)
        for row in range(dimension):
            for column in range(row + 1):
                value = covariance[row, column]
                for inner in range(column):
                    value -= cholesky[row, inner] * cholesky[column, inner]
                if row == column:
                    cholesky[row, column] = np.sqrt(max(value, 1e-12))
                else:
                    cholesky[row, column] = value / max(cholesky[column, column], 1e-12)
        log_determinant = 2.0 * float(
            np.sum(np.log(np.maximum(np.diag(cholesky), 1e-12)))
        )
        volume.append(float(np.exp(log_determinant / (2.0 * dimension))))
        covariance_volume.append(float(np.exp(np.clip(log_determinant / 2.0, -745.0, 700.0))))
        standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        active_indices = np.flatnonzero(mask)
        for row in range(dimension):
            for column in range(dimension):
                denominator = standard_deviation[row] * standard_deviation[column]
                value = covariance[row, column] / denominator if denominator > 0.0 else (1.0 if row == column else 0.0)
                correlation_sums[active_indices[row], active_indices[column]] += value
                correlation_counts[active_indices[row], active_indices[column]] += 1
        references = rng.uniform(-0.5, 0.5, size=(32, dimension))
        for reference in references:
            draw_distance = np.linalg.norm(draws - reference[None, :], axis=1) / scale
            truth_distance = np.linalg.norm(target - reference) / scale
            coverages.append(float(np.mean(draw_distance < truth_distance)))
    alpha = np.linspace(0.0, 1.0, 11)
    coverage_array = np.asarray(coverages, dtype=np.float64)
    ecp = (
        np.asarray([np.mean(coverage_array <= value) for value in alpha])
        if coverage_array.size
        else np.full(alpha.shape, np.nan)
    )
    return {
        "case_count": len(energy),
        "mean_energy_score": float(np.mean(energy)) if energy else float("nan"),
        "mean_pairwise_distance": float(np.mean(concentration)) if concentration else float("nan"),
        "mean_geometric_sd": float(np.mean(volume)) if volume else float("nan"),
        "mean_covariance_volume": float(np.mean(covariance_volume)) if covariance_volume else float("nan"),
        "mean_parameter_correlation_matrix": np.divide(
            correlation_sums,
            correlation_counts,
            out=np.full_like(correlation_sums, np.nan),
            where=correlation_counts > 0,
        ).tolist(),
        "tarp_alpha": alpha.tolist(),
        "tarp_ecp": ecp.tolist(),
        "tarp_mean_absolute_coverage_error": float(np.nanmean(np.abs(ecp - alpha)))
        if coverage_array.size
        else float("nan"),
        "case_energy_score": energy,
        "case_pairwise_distance": concentration,
        "case_geometric_sd": volume,
    }


def evaluate_posterior(
    samples: np.ndarray,
    truth: np.ndarray,
    active_mask: np.ndarray,
    *,
    joint_seed: int = 0,
) -> dict[str, Any]:
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    active_mask = np.asarray(active_mask, dtype=bool)
    if (
        samples.ndim != 3
        or samples.shape[-1] != len(CANONICAL_PARAMETERS)
        or truth.shape != (samples.shape[0], len(CANONICAL_PARAMETERS))
        or active_mask.shape != truth.shape
    ):
        raise ValueError("Expected samples [cases,draws,8] and truth [cases,8]")
    if samples.shape[1] < 2:
        raise ValueError("At least two posterior draws are required")
    if not np.all(np.isfinite(samples)) or not np.all(np.isfinite(truth)):
        raise ValueError("Posterior samples and truth must be finite")
    schema = ParameterSchema()
    low, high, _ = schema._bounds()
    normalized_samples = schema.normalize(samples)
    normalized_truth = schema.normalize(truth)
    median = np.median(samples, axis=1)
    report: dict[str, Any] = {
        "cases": int(samples.shape[0]),
        "draws_per_case": int(samples.shape[1]),
        "parameters": {},
    }
    for index, name in enumerate(CANONICAL_PARAMETERS):
        selected = active_mask[:, index]
        if not np.any(selected):
            continue
        estimate = median[selected, index]
        target = truth[selected, index]
        error = estimate - target
        denominator = float(np.sum((target - target.mean()) ** 2))
        draws = samples[selected, :, index]
        normalized_draws = normalized_samples[selected, :, index]
        normalized_target = normalized_truth[selected, index]
        crps = _crps_rows(draws, target)
        normalized_crps = _crps_rows(normalized_draws, normalized_target) / 2.0
        ranks = np.sum(draws < target[:, None], axis=1)
        normalized_ranks = (ranks + 0.5) / (draws.shape[1] + 1.0)
        rank_histogram, _ = np.histogram(normalized_ranks, bins=10, range=(0.0, 1.0))
        deviations = np.abs(draws - np.median(draws, axis=1, keepdims=True))
        parameter = {
            "count": int(selected.sum()),
            "bias": float(np.mean(error)),
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "normalized_mae": float(np.mean(np.abs(error)) / (high[index] - low[index])),
            "r2": float(1.0 - np.sum(error**2) / denominator)
            if denominator > 0
            else float("nan"),
            "crps": float(np.mean(crps)),
            "normalized_crps": float(np.mean(normalized_crps)),
            "posterior_std": float(np.mean(np.std(draws, axis=1, ddof=1))),
            "posterior_iqr": float(
                np.mean(np.quantile(draws, 0.75, axis=1) - np.quantile(draws, 0.25, axis=1))
            ),
            "posterior_mad": float(np.mean(np.median(deviations, axis=1))),
            "sbc_rank_mean": float(np.mean(normalized_ranks)),
            "sbc_rank_histogram_10": rank_histogram.tolist(),
        }
        for level in (0.5, 0.8, 0.9, 0.95):
            tail = (1.0 - level) / 2.0
            lower = np.quantile(draws, tail, axis=1)
            upper = np.quantile(draws, 1.0 - tail, axis=1)
            parameter[f"coverage_{int(level * 100)}"] = float(
                np.mean((target >= lower) & (target <= upper))
            )
            parameter[f"width_{int(level * 100)}"] = float(np.mean(upper - lower))
        report["parameters"][name] = parameter
    parameters = list(report["parameters"].values())
    report["aggregate"] = {
        "evaluated_parameter_count": len(parameters),
        "mean_normalized_mae": float(np.mean([v["normalized_mae"] for v in parameters])) if parameters else float("nan"),
        "mean_crps": float(np.mean([v["crps"] for v in parameters])) if parameters else float("nan"),
        "mean_normalized_crps": float(np.mean([v["normalized_crps"] for v in parameters])) if parameters else float("nan"),
        "mean_coverage_error_90": float(np.mean([abs(v["coverage_90"] - 0.9) for v in parameters])) if parameters else float("nan"),
        "mean_coverage_error_95": float(np.mean([abs(v["coverage_95"] - 0.95) for v in parameters])) if parameters else float("nan"),
    }
    report["joint"] = _joint_scores(
        normalized_samples, normalized_truth, active_mask, int(joint_seed)
    )
    return report
