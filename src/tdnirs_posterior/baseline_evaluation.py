from __future__ import annotations

from typing import Any

import numpy as np

from .evaluation_protocol import spearman_with_bootstrap, thickness_group_labels
from .parameters import CANONICAL_PARAMETERS


PHYSICAL_WIDTHS = np.asarray(
    [19.5, 19.5, 0.049, 0.049, 0.049, 2.8, 2.8, 2.8], dtype=np.float64
)


def _metrics(prediction: np.ndarray, truth: np.ndarray, width: float) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    finite = np.isfinite(prediction) & np.isfinite(truth)
    counts = {
        "total_count": int(len(truth)),
        "finite_count": int(np.sum(finite)),
        "numerical_failure_count": int(np.sum(~finite)),
    }
    # A partially finite result must not look artificially better because its
    # failed cases were silently removed from the denominator.
    if not np.all(finite) or not len(truth):
        return {
            **counts,
            **{key: float("nan") for key in ("bias", "mae", "rmse", "normalized_mae", "r2")},
        }
    error = prediction - truth
    denominator = float(np.sum((truth - np.mean(truth)) ** 2))
    return {
        **counts,
        "bias": float(np.mean(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_mae": float(np.mean(np.abs(error)) / width),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0.0 else float("nan"),
    }


def evaluate_point_predictions(
    prediction: np.ndarray,
    truth: np.ndarray,
    active_mask: np.ndarray,
    layer_count: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int = 1000,
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    layers = np.asarray(layer_count, dtype=np.int64)
    if prediction.shape != truth.shape or active.shape != truth.shape:
        raise ValueError("prediction, truth, and active_mask must have identical shapes")
    if truth.shape[1] != len(CANONICAL_PARAMETERS) or layers.shape != (len(truth),):
        raise ValueError("Unexpected deterministic prediction dimensions")

    thickness = thickness_group_labels(truth, layers)
    rows: list[dict[str, Any]] = []
    thickness_rows: list[dict[str, Any]] = []
    associations: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    for case_index in range(len(truth)):
        for parameter_index in np.flatnonzero(active[case_index]):
            case_rows.append(
                {
                    "case_index": case_index,
                    "layer_count": int(layers[case_index]),
                    "parameter": CANONICAL_PARAMETERS[parameter_index],
                    "truth": float(truth[case_index, parameter_index]),
                    "prediction": float(prediction[case_index, parameter_index]),
                    "absolute_error": float(abs(prediction[case_index, parameter_index] - truth[case_index, parameter_index])),
                    "normalized_absolute_error": float(abs(prediction[case_index, parameter_index] - truth[case_index, parameter_index]) / PHYSICAL_WIDTHS[parameter_index]),
                    "thickness_group": str(thickness[case_index]),
                }
            )

    for layer in (1, 2, 3):
        selected_layer = layers == layer
        for parameter_index, parameter in enumerate(CANONICAL_PARAMETERS):
            selected = selected_layer & active[:, parameter_index]
            if not np.any(selected):
                continue
            row = {
                "layer_count": layer,
                "group": "all",
                "parameter": parameter,
                "count": int(np.sum(selected)),
                **_metrics(prediction[selected, parameter_index], truth[selected, parameter_index], float(PHYSICAL_WIDTHS[parameter_index])),
            }
            rows.append(row)
            if layer > 1:
                for group in sorted(set(thickness[selected])):
                    grouped = selected & (thickness == group)
                    thickness_rows.append(
                        {
                            "layer_count": layer,
                            "group": str(group),
                            "parameter": parameter,
                            "count": int(np.sum(grouped)),
                            **_metrics(prediction[grouped, parameter_index], truth[grouped, parameter_index], float(PHYSICAL_WIDTHS[parameter_index])),
                        }
                    )

        if layer == 3:
            predicted_total = prediction[selected_layer, 0] + prediction[selected_layer, 1]
            true_total = truth[selected_layer, 0] + truth[selected_layer, 1]
            rows.append(
                {
                    "layer_count": 3,
                    "group": "all",
                    "parameter": "d1_plus_d2_mm",
                    "count": int(np.sum(selected_layer)),
                    **_metrics(predicted_total, true_total, 34.0),
                }
            )
            for group in sorted(set(thickness[selected_layer])):
                grouped_local = thickness[selected_layer] == group
                thickness_rows.append(
                    {
                        "layer_count": 3,
                        "group": str(group),
                        "parameter": "d1_plus_d2_mm",
                        "count": int(np.sum(grouped_local)),
                        **_metrics(predicted_total[grouped_local], true_total[grouped_local], 34.0),
                    }
                )

        if layer > 1:
            local_truth = truth[selected_layer]
            thickness_variables = [("d1_mm", local_truth[:, 0])]
            if layer == 3:
                thickness_variables.extend(
                    [("d2_mm", local_truth[:, 1]), ("d1_plus_d2_mm", local_truth[:, 0] + local_truth[:, 1])]
                )
            for parameter_index in np.flatnonzero(np.any(active[selected_layer], axis=0)):
                error = np.abs(prediction[selected_layer, parameter_index] - truth[selected_layer, parameter_index]) / PHYSICAL_WIDTHS[parameter_index]
                for offset, (name, values) in enumerate(thickness_variables):
                    result = spearman_with_bootstrap(
                        values,
                        error,
                        bootstrap_seed + layer * 1000 + parameter_index * 10 + offset,
                        bootstrap_replicates,
                    )
                    associations.append(
                        {
                            "layer_count": layer,
                            "thickness_variable": name,
                            "parameter": CANONICAL_PARAMETERS[parameter_index],
                            "total_count": int(len(values)),
                            **result,
                        }
                    )

    layer_scores: dict[str, float] = {}
    for layer in (1, 2, 3):
        values = [row["normalized_mae"] for row in rows if row["layer_count"] == layer and row["parameter"] != "d1_plus_d2_mm"]
        if values:
            layer_scores[f"N{layer}"] = float(np.mean(values))
    if not layer_scores:
        raise ValueError("No active deterministic prediction metrics were produced")
    return {
        "parameter_metrics": rows,
        "thickness_metrics": thickness_rows,
        "thickness_associations": associations,
        "casewise_metrics": case_rows,
        "layer_macro_nmae": layer_scores,
        "macro_nmae": float(np.mean(list(layer_scores.values()))),
        "numerical_failure_cases": int(
            np.sum(np.any(active & ~np.isfinite(prediction), axis=1))
        ),
    }
