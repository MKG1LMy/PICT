from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Any
import warnings

import numpy as np

from .parameters import ACTIVE_BY_LAYER, CANONICAL_PARAMETERS, ParameterSchema

GMM_INITIALIZATION = "random_from_data"

@dataclass(frozen=True)
class PriorScenario:
    scenario_id: str
    layer_count: int
    known_indices: tuple[int, ...]
    known_parameters: tuple[str, ...]
    unknown_parameters: tuple[str, ...]
    known_count: int
    prior_type: str
    optical_type: str
    aliases: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _prior_type(indices: tuple[int, ...]) -> tuple[str, str]:
    if not indices:
        return "no_prior", "none"
    thickness = [index for index in indices if index < 2]
    optical = [index for index in indices if index >= 2]
    if thickness and optical:
        prior_type = "mixed"
    elif thickness:
        prior_type = "thickness_only"
    else:
        prior_type = "optical_only"
    mua = [index for index in optical if 2 <= index <= 4]
    musp = [index for index in optical if 5 <= index <= 7]
    if not optical:
        optical_type = "none"
    elif mua and not musp:
        optical_type = "absorption_only"
    elif musp and not mua:
        optical_type = "scattering_only"
    else:
        paired_layers = {
            layer for layer in range(3) if (2 + layer in optical and 5 + layer in optical)
        }
        optical_type = "whole_layer" if 2 * len(paired_layers) == len(optical) else "mixed_optical"
    return prior_type, optical_type


def _named_aliases(layer: int, indices: tuple[int, ...]) -> tuple[str, ...]:
    names = set(indices)
    aliases: list[str] = []
    if not names:
        aliases.append("no_prior")
    definitions: dict[int, dict[str, set[int]]] = {
        1: {"known_mua1": {2}, "known_musp1": {5}},
        2: {
            "known_d1": {0},
            "known_layer1_optical": {2, 5},
            "known_layer2_optical": {3, 6},
            "known_all_optical": {2, 3, 5, 6},
            "known_d1_layer1_optical": {0, 2, 5},
            "known_d1_layer2_optical": {0, 3, 6},
        },
        3: {
            "known_d1": {0},
            "known_d2": {1},
            "known_all_thickness": {0, 1},
            "known_layer1_optical": {2, 5},
            "known_layer2_optical": {3, 6},
            "known_layer3_optical": {4, 7},
            "known_all_optical": {2, 3, 4, 5, 6, 7},
            "known_layer1_complete": {0, 2, 5},
            "known_thickness_layer1_optical": {0, 1, 2, 5},
            "known_surface_two_layers": {0, 1, 2, 3, 5, 6},
        },
    }
    aliases.extend(
        alias for alias, expected in definitions[layer].items() if names == expected
    )
    return tuple(aliases)


def enumerate_prior_scenarios(layer_count: int) -> list[PriorScenario]:
    active = tuple(ACTIVE_BY_LAYER[layer_count])
    scenarios: list[PriorScenario] = []
    for count in range(len(active)):
        for known in combinations(active, count):
            mask_bits = "".join("1" if index in known else "0" for index in active)
            prior_type, optical_type = _prior_type(known)
            scenarios.append(
                PriorScenario(
                    scenario_id=f"N{layer_count}_k{count}_{mask_bits}",
                    layer_count=layer_count,
                    known_indices=known,
                    known_parameters=tuple(CANONICAL_PARAMETERS[i] for i in known),
                    unknown_parameters=tuple(
                        CANONICAL_PARAMETERS[i] for i in active if i not in known
                    ),
                    known_count=count,
                    prior_type=prior_type,
                    optical_type=optical_type,
                    aliases=_named_aliases(layer_count, known),
                )
            )
    return scenarios


def scenario_mask(scenario: PriorScenario, cases: int) -> np.ndarray:
    result = np.zeros((cases, len(CANONICAL_PARAMETERS)), dtype=bool)
    if scenario.known_indices:
        result[:, list(scenario.known_indices)] = True
    return result


def is_deep_multimodality_scenario(scenario: PriorScenario) -> bool:
    return bool(scenario.aliases)


def thickness_group_labels(truth: np.ndarray, layers: np.ndarray) -> np.ndarray:
    labels = np.full(len(layers), "not_applicable", dtype=object)
    n2 = layers == 2
    n3 = layers == 3
    labels[n2] = np.select(
        [truth[n2, 0] < 5.0, truth[n2, 0] < 10.0, truth[n2, 0] < 15.0],
        ["N2_d1_[0.5,5)", "N2_d1_[5,10)", "N2_d1_[10,15)"],
        default="N2_d1_[15,20]",
    )
    total = truth[n3, 0] + truth[n3, 1]
    labels[n3] = np.select(
        [total < 10.0, total < 15.0, total < 20.0, total <= 25.0, total <= 30.0, total <= 35.0],
        ["N3_d1+d2_[1,10)", "N3_d1+d2_[10,15)", "N3_d1+d2_[15,20)",
         "N3_d1+d2_[20,25]", "N3_d1+d2_(25,30]", "N3_d1+d2_(30,35]"],
        default="N3_d1+d2_outside_study",
    )
    return labels.astype(str)


def spearman_with_bootstrap(
    x: np.ndarray, y: np.ndarray, seed: int, replicates: int = 1000
) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return {"count": int(len(x)), "rho": float("nan"), "ci95": [float("nan")] * 2}

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(len(values), dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            result[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return result

    def rho(a: np.ndarray, b: np.ndarray) -> float:
        ar, br = ranks(a), ranks(b)
        ar = ar - np.mean(ar)
        br = br - np.mean(br)
        sum_ar2 = float(np.sum(ar * ar))
        sum_br2 = float(np.sum(br * br))
        if sum_ar2 == 0.0 or sum_br2 == 0.0:
            return float("nan")
        # This scalar form is mathematically Pearson's correlation of the
        # ranks, but avoids dispatching a tiny calculation to an external
        # BLAS/OpenMP runtime after PyTorch has initialized its own runtime.
        return float(np.sum(ar * br) / math.sqrt(sum_ar2 * sum_br2))

    observed = rho(x, y)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(replicates)):
        selected = rng.integers(0, len(x), size=len(x))
        value = rho(x[selected], y[selected])
        if np.isfinite(value):
            estimates.append(value)
    interval = np.quantile(estimates, [0.025, 0.975]).tolist() if estimates else [float("nan")] * 2
    return {"count": int(len(x)), "rho": observed, "ci95": interval}


def _oracle_nearest_mode_diagnostics(
    values: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    """Describe the truth-nearest fitted component without changing the posterior."""

    distances = np.sqrt(np.mean((centers - target[None, :]) ** 2, axis=1))
    nearest = int(np.argmin(distances))
    selected = values[labels == nearest]
    median_error = float("nan")
    if len(selected):
        median_error = float(
            np.sqrt(np.mean((np.median(selected, axis=0) - target) ** 2))
        )
    return {
        "truth_rms_distance_normalized": distances.tolist(),
        "oracle_nearest_truth_component": nearest + 1,
        "oracle_nearest_truth_component_weight": float(weights[nearest]),
        "oracle_nearest_truth_empirical_fraction": float(np.mean(labels == nearest)),
        "oracle_nearest_truth_center_rms_distance_normalized": float(
            distances[nearest]
        ),
        "oracle_nearest_truth_mode_median_rms_error_normalized": median_error,
        "oracle_nearest_truth_mode_draw_count": int(len(selected)),
    }


def detect_candidate_modes(
    normalized_samples: np.ndarray,
    unknown_mask: np.ndarray,
    *,
    seed: int,
    normalized_truth: np.ndarray | None = None,
    max_components: int = 4,
    bootstrap_replicates: int = 20,
    minimum_weight: float = 0.05,
    minimum_separation: float = 2.0,
    max_iter: int = 300,
    retry_max_iter: int = 1000,
) -> dict[str, Any]:
    """Identify stable candidate components; this is not a physical equivalence test."""
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.exceptions import ConvergenceWarning
    except ImportError as exc:  # pragma: no cover - diagnosed during deployment
        raise RuntimeError("Multimodality analysis requires scikit-learn") from exc
    mask = np.asarray(unknown_mask, dtype=bool)
    values = np.asarray(normalized_samples, dtype=np.float64)[:, mask]
    if values.ndim != 2 or not values.shape[1] or not np.all(np.isfinite(values)):
        return {"status": "numerical_failure", "stable_multimodal": False}
    upper = min(int(max_components), max(1, values.shape[0] // 20))

    def fit(data: np.ndarray, local_seed: int) -> tuple[Any | None, list[dict[str, Any]]]:
        candidates: list[Any] = []
        diagnostics: list[dict[str, Any]] = []
        for components in range(1, upper + 1):
            attempts: list[dict[str, Any]] = []
            model = None
            settings = (
                (int(max_iter), 5, 1e-5, int(local_seed)),
                (int(retry_max_iter), 10, 1e-4, int(local_seed) + 7919),
            )
            for attempt_index, (iterations, initializations, regularization, attempt_seed) in enumerate(settings):
                warning_messages: list[str] = []
                error_message = ""
                candidate = None
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        candidate = GaussianMixture(
                            n_components=components,
                            covariance_type="full",
                            reg_covar=regularization,
                            n_init=initializations,
                            # Start at observed draws, not nearly identical soft
                            # responsibilities that can hide separated modes.
                            # This also avoids KMeans initialization.
                            init_params=GMM_INITIALIZATION,
                            max_iter=iterations,
                            random_state=attempt_seed,
                        ).fit(data)
                    warning_messages = [
                        str(item.message)
                        for item in caught
                        if issubclass(item.category, ConvergenceWarning)
                    ]
                except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
                    error_message = f"{type(exc).__name__}: {exc}"
                converged = bool(candidate is not None and getattr(candidate, "converged_", False))
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "converged": converged,
                        "max_iter": iterations,
                        "n_init": initializations,
                        "reg_covar": regularization,
                        "n_iter": int(getattr(candidate, "n_iter_", 0)) if candidate is not None else 0,
                        "convergence_warning_count": len(warning_messages),
                        "convergence_warnings": warning_messages,
                        "error": error_message,
                    }
                )
                if converged:
                    model = candidate
                    break
            diagnostics.append(
                {
                    "component_count": components,
                    "converged": model is not None,
                    "retried": len(attempts) > 1,
                    "attempts": attempts,
                }
            )
            if model is not None:
                candidates.append(model)
        if not candidates:
            return None, diagnostics
        return min(candidates, key=lambda candidate: candidate.bic(data)), diagnostics

    model, fit_diagnostics = fit(values, int(seed))
    failed_candidates = sum(not item["converged"] for item in fit_diagnostics)
    retried_candidates = sum(item["retried"] for item in fit_diagnostics)
    if model is None:
        return {
            "status": "gmm_nonconverged",
            "stable_multimodal": False,
            "gmm_converged": False,
            "gmm_retried_candidate_count": retried_candidates,
            "gmm_failed_candidate_count": failed_candidates,
            "gmm_fit_diagnostics": fit_diagnostics,
            "bootstrap_requested_replicates": int(bootstrap_replicates),
            "bootstrap_successful_replicates": 0,
            "bootstrap_failed_replicates": 0,
            "bootstrap_skipped_replicates": int(bootstrap_replicates),
            "bootstrap_retried_replicates": 0,
            "bootstrap_failure_replicate_indices": [],
            "unknown_parameters": [CANONICAL_PARAMETERS[i] for i in np.flatnonzero(mask)],
        }
    rng = np.random.default_rng(seed)
    selected_counts: list[int] = []
    bootstrap_failures: list[int] = []
    bootstrap_retried = 0
    for replicate in range(int(bootstrap_replicates)):
        sampled = values[rng.integers(0, len(values), size=len(values))]
        bootstrap_model, bootstrap_diagnostics = fit(sampled, int(seed) + replicate + 1)
        if any(item["retried"] for item in bootstrap_diagnostics):
            bootstrap_retried += 1
        if bootstrap_model is None:
            bootstrap_failures.append(replicate)
        else:
            selected_counts.append(int(bootstrap_model.n_components))
    matching = sum(count == model.n_components for count in selected_counts)
    stability = float(matching / max(int(bootstrap_replicates), 1))
    separations: list[float] = []
    for left in range(model.n_components):
        for right in range(left + 1, model.n_components):
            pooled = 0.5 * (
                np.diag(model.covariances_[left])
                + np.diag(model.covariances_[right])
            )
            delta = model.means_[left] - model.means_[right]
            separations.append(
                float(np.sqrt(np.sum(delta * delta / np.maximum(pooled, 1e-12))))
            )
    minimum_observed = min(separations) if separations else 0.0
    stable = bool(
        model.n_components > 1
        and stability >= 0.8
        and np.min(model.weights_) >= minimum_weight
        and minimum_observed >= minimum_separation
    )
    labels = model.predict(values)
    outside = np.any((values < -1.0) | (values > 1.0), axis=1)
    component_outside = [
        float(np.mean(outside[labels == component]))
        if np.any(labels == component)
        else float("nan")
        for component in range(model.n_components)
    ]
    oracle = {
        "truth_rms_distance_normalized": [],
        "oracle_nearest_truth_component": None,
        "oracle_nearest_truth_component_weight": float("nan"),
        "oracle_nearest_truth_empirical_fraction": float("nan"),
        "oracle_nearest_truth_center_rms_distance_normalized": float("nan"),
        "oracle_nearest_truth_mode_median_rms_error_normalized": float("nan"),
        "oracle_nearest_truth_mode_draw_count": 0,
    }
    if normalized_truth is not None:
        target = np.asarray(normalized_truth, dtype=np.float64)[mask]
        oracle = _oracle_nearest_mode_diagnostics(
            values, labels, model.means_, model.weights_, target
        )
    return {
        "status": "completed",
        "gmm_converged": True,
        "gmm_retried_candidate_count": retried_candidates,
        "gmm_failed_candidate_count": failed_candidates,
        "gmm_fit_diagnostics": fit_diagnostics,
        "bootstrap_requested_replicates": int(bootstrap_replicates),
        "bootstrap_successful_replicates": len(selected_counts),
        "bootstrap_failed_replicates": len(bootstrap_failures),
        "bootstrap_skipped_replicates": 0,
        "bootstrap_retried_replicates": bootstrap_retried,
        "bootstrap_failure_replicate_indices": bootstrap_failures,
        "component_count": int(model.n_components),
        "stable_multimodal": stable,
        "bootstrap_stability": stability,
        "weights": model.weights_.tolist(),
        "centers_normalized": model.means_.tolist(),
        "within_component_sd_normalized": [
            np.sqrt(np.maximum(np.diag(covariance), 0.0)).tolist()
            for covariance in model.covariances_
        ],
        "component_outside_training_support_fraction": component_outside,
        # This component is selected with the known test truth. It is an
        # oracle diagnostic, not a deployable point estimate or primary score.
        **oracle,
        "minimum_standardized_separation": minimum_observed,
        "unknown_parameters": [CANONICAL_PARAMETERS[i] for i in np.flatnonzero(mask)],
    }
