from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .baseline_evaluation import evaluate_point_predictions
from .evaluation_protocol import enumerate_prior_scenarios
from .testing import _bootstrap_mean


def _dataset_sha256(dataset: h5py.Dataset) -> str:
    digest = hashlib.sha256()
    for start in range(0, dataset.shape[0], max(1, min(16, dataset.shape[0]))):
        digest.update(np.asarray(dataset[start : start + 16]).tobytes())
    return digest.hexdigest()


def _strings(dataset: h5py.Dataset) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in dataset[:]]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        if not columns:
            return
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def compare_deterministic_baselines(
    explicit_irf_directory: str | Path,
    cnn_directory: str | Path,
    ann_directory: str | Path,
    output_directory: str | Path,
) -> Path:
    explicit_dir = Path(explicit_irf_directory).resolve()
    baseline_dirs = {"CNN": Path(cnn_directory).resolve(), "ANN": Path(ann_directory).resolve()}
    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    explicit_report = json.loads((explicit_dir / "report.json").read_text(encoding="utf-8"))
    if explicit_report.get("scope", {}).get("model_variant") != "explicit_irf":
        raise ValueError("The reference test must be an explicit_irf formal report")
    rows: list[dict[str, Any]] = []
    casewise_rows: list[dict[str, Any]] = []
    input_statuses: dict[str, str] = {
        "PICT Model": str(explicit_report.get("status", "unknown"))
    }
    with h5py.File(explicit_dir / "posterior.h5", "r") as full_model:
        truth = full_model["cases"]["truth"][:]
        active = full_model["cases"]["active_mask"][:].astype(bool)
        layers = full_model["cases"]["layer_count"][:]
        sample_ids = _strings(full_model["cases"]["sample_id"])
        dtof_hash = _dataset_sha256(full_model["cases"]["dtof_input"])
        full_model_prediction = np.zeros_like(truth)
        for layer in (1, 2, 3):
            scenario = next(item for item in enumerate_prior_scenarios(layer) if item.known_count == 0)
            group = full_model["scenarios"][scenario.scenario_id]
            indices = group["global_case_index"][:]
            full_model_prediction[indices] = np.median(group["samples"][:], axis=1)
        full_model_metrics = evaluate_point_predictions(full_model_prediction, truth, active, layers, bootstrap_seed=int(explicit_report["reproducibility"]["observation_seed"]) + 810000)
        reports: dict[str, Any] = {"PICT Model": full_model_metrics}
        model_profiles: dict[str, Any] = {
            "PICT Model": {
                "parameter_count": explicit_report.get("scope", {}).get("model_parameter_count"),
                "test_elapsed_seconds": explicit_report.get("runtime", {}).get("elapsed_seconds"),
                "probabilistic_output": True,
            }
        }
        for model_index, (name, directory) in enumerate(baseline_dirs.items()):
            report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
            input_statuses[name] = str(report.get("status", "unknown"))
            if report.get("model", {}).get("architecture", "").upper() != name:
                raise ValueError(f"Expected {name} report at {directory}")
            with h5py.File(directory / "predictions.h5", "r") as baseline:
                checks = {
                    "dataset_fingerprint": explicit_report["reproducibility"]["dataset_fingerprint"] == report["reproducibility"]["dataset_fingerprint"],
                    "observation_seed": explicit_report["reproducibility"]["observation_seed"] == report["reproducibility"]["observation_seed"],
                    "sample_ids": sample_ids == _strings(baseline["cases"]["sample_id"]),
                    "dtof_sha256": dtof_hash == _dataset_sha256(baseline["cases"]["dtof_input"]),
                    "truth": np.array_equal(truth, baseline["cases"]["truth"][:]),
                    "active_mask": np.array_equal(active, baseline["cases"]["active_mask"][:].astype(bool)),
                    "weights": explicit_report["checkpoint"]["weights"] == report["checkpoint"]["weights"],
                }
                mismatches = [key for key, passed in checks.items() if not passed]
                if mismatches:
                    raise ValueError(f"{name} is not strictly paired with the PICT Model test: {', '.join(mismatches)}")
                prediction = baseline["cases"]["prediction"][:]
            metrics = evaluate_point_predictions(prediction, truth, active, layers, bootstrap_seed=int(report["reproducibility"]["observation_seed"]) + 820000 + model_index)
            reports[name] = metrics
            runtime = report.get("runtime", {})
            model_profiles[name] = {
                "parameter_count": report["model"].get("parameter_count"),
                "approximate_flops_per_case": report["model"].get("approximate_flops_per_case"),
                "test_elapsed_seconds": runtime.get("elapsed_seconds"),
                "inference_milliseconds_per_case": runtime.get("milliseconds_per_case"),
                "peak_cuda_allocated_gib": runtime.get("peak_cuda_allocated_gib"),
                "probabilistic_output": False,
                "finite_fraction": report.get("native_output", {}).get("finite_fraction"),
            }
            full_model_case = {(row["case_index"], row["parameter"]): row for row in full_model_metrics["casewise_metrics"]}
            for item in metrics["casewise_metrics"]:
                reference = full_model_case[(item["case_index"], item["parameter"])]
                delta = item["normalized_absolute_error"] - reference["normalized_absolute_error"]
                casewise_rows.append({"model": name, "sample_id": sample_ids[item["case_index"]], "layer_count": item["layer_count"], "parameter": item["parameter"], "baseline_normalized_absolute_error": item["normalized_absolute_error"], "full_model_normalized_absolute_error": reference["normalized_absolute_error"], "delta_baseline_minus_full_model": delta})

        for model_name, metrics in reports.items():
            for item in metrics["parameter_metrics"]:
                rows.append({"model": model_name, **item})

    paired_rows: list[dict[str, Any]] = []
    for model in ("CNN", "ANN"):
        selected_model = [row for row in casewise_rows if row["model"] == model]
        for layer in (1, 2, 3):
            parameters = sorted({row["parameter"] for row in selected_model if row["layer_count"] == layer})
            for parameter in parameters:
                selected = [row for row in selected_model if row["layer_count"] == layer and row["parameter"] == parameter]
                delta = np.asarray([row["delta_baseline_minus_full_model"] for row in selected])
                baseline_error = float(np.mean([row["baseline_normalized_absolute_error"] for row in selected]))
                full_model_error = float(np.mean([row["full_model_normalized_absolute_error"] for row in selected]))
                paired_rows.append({"baseline": model, "layer_count": layer, "parameter": parameter, "baseline_nmae": baseline_error, "full_model_nmae": full_model_error, "delta_baseline_minus_full_model": float(np.mean(delta)), "full_model_relative_improvement": (baseline_error - full_model_error) / baseline_error if baseline_error > 0 else float("nan"), "delta_ci95": _bootstrap_mean(delta, 900000 + layer * 100 + len(paired_rows), 1000)})

    _write_csv(destination / "model_metrics.csv", rows)
    _write_csv(destination / "paired_effects.csv", paired_rows)
    _write_csv(destination / "casewise_effects.csv", casewise_rows)
    noncomplete_models = [name for name, status in input_statuses.items() if status != "completed"]
    numerical_failure_models = [
        name for name, status in input_statuses.items() if "numerical_failures" in status
    ]
    comparison_status = (
        "completed_with_numerical_failures" if numerical_failure_models
        else "completed_with_input_warnings" if noncomplete_models
        else "completed"
    )
    summary = {
        "status": comparison_status,
        "comparison": "explicit_irf_conditional_flow_median_vs_dtof_only_deterministic_baselines",
        "input_statuses": input_statuses,
        "noncomplete_input_models": noncomplete_models,
        "numerical_failure_models": numerical_failure_models,
        "models": {name: {"macro_nmae": metrics["macro_nmae"], "layer_macro_nmae": metrics["layer_macro_nmae"]} for name, metrics in reports.items()},
        "model_profiles": model_profiles,
        "paired_effects": paired_rows,
        "limitations": ["The deterministic CNN and ANN do not provide posterior calibration or multimodality metrics.", "The baselines omit neural IRF and optional-prior inputs by design, so this is a whole-framework comparison rather than an isolated encoder ablation.", "The DTOF-only Flow experiment remains the dedicated IRF architecture ablation."],
    }
    (destination / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    def pct(value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "—"
        return f"{numeric:.2%}" if np.isfinite(numeric) else "Numerical failure"
    table = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{pct(value['macro_nmae'])}</td>"
        f"<td>{pct(value['layer_macro_nmae']['N1'])}</td><td>{pct(value['layer_macro_nmae']['N2'])}</td>"
        f"<td>{pct(value['layer_macro_nmae']['N3'])}</td><td>{html.escape(input_statuses[name])}</td></tr>"
        for name, value in summary["models"].items()
    )
    profile_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value.get('parameter_count') or '—'}</td>"
        f"<td>{value.get('approximate_flops_per_case') or '—'}</td>"
        f"<td>{value.get('inference_milliseconds_per_case') or '—'}</td>"
        f"<td>{'Yes' if value.get('probabilistic_output') else 'No'}</td></tr>"
        for name, value in model_profiles.items()
    )
    effect_rows = "".join(
        f"<tr><td>{html.escape(row['baseline'])}</td><td>N={row['layer_count']}</td>"
        f"<td>{html.escape(row['parameter'])}</td><td>{pct(row['baseline_nmae'])}</td>"
        f"<td>{pct(row['full_model_nmae'])}</td><td>{pct(row['full_model_relative_improvement'])}</td>"
        f"<td>{html.escape(str(row['delta_ci95']))}</td></tr>" for row in paired_rows
    )
    warning = "" if not noncomplete_models else (
        "<div class='warning'><b>Not all input reports completed without warnings: </b>" + html.escape(", ".join(noncomplete_models))
        + ". Interpret results alongside each input report status and diagnostics.</div>"
    )
    page = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><title>TD-NIRS baseline comparison</title><style>body{{background:#eef3f8;color:#183047;font:15px/1.6 system-ui,sans-serif;margin:0}}main{{max-width:1240px;margin:auto;padding:38px 24px}}section{{background:white;border:1px solid #dce5ed;border-radius:14px;padding:20px;margin:18px 0;overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:10px;border-bottom:1px solid #e5ebf0;text-align:right}}th:first-child,td:first-child{{text-align:left}}h1{{font-size:34px}}.sub{{color:#63788d}}.warning{{background:#fff2f0;border:1px solid #ffb5ad;border-radius:12px;padding:14px;color:#8d231d}}a{{margin-right:16px}}</style></head><body><main><p class='sub'>Strictly paired fixed test set · Status: {html.escape(summary['status'])}</p><h1>Full model and deterministic baselines</h1><p>Same cases, DTOF, truth and active-parameter masks. The full model uses the no-prior posterior median; CNN/ANN use deterministic predictions. Overall macro nMAE weights N=1, N=2 and N=3 equally.</p>{warning}<section><h2>Prediction accuracy</h2><table><thead><tr><th>Model</th><th>Overall macro nMAE</th><th>N=1</th><th>N=2</th><th>N=3</th><th>Input report status</th></tr></thead><tbody>{table}</tbody></table></section><section><h2>Model size and computation</h2><table><thead><tr><th>Model</th><th>Parameters</th><th>Estimated FLOPs/case</th><th>Inference ms/case</th><th>Probabilistic output</th></tr></thead><tbody>{profile_rows}</tbody></table></section><section><h2>Paired parameter effects</h2><p class='sub'>Positive relative improvement favors the full model; intervals use case-level paired bootstrap. </p><table><thead><tr><th>Baseline</th><th>N</th><th>Parameter</th><th>Baseline nMAE</th><th>Full-model nMAE</th><th> relative improvement</th><th>Difference 95% CI</th></tr></thead><tbody>{effect_rows}</tbody></table></section><section><h2>Data files</h2><p><a href='model_metrics.csv'>Parameters CSV</a><a href='paired_effects.csv'>Paired effects CSV</a><a href='casewise_effects.csv'>Cases CSV</a><a href='comparison.json'>Complete JSON</a></p></section><section><h2>Interpretation limits</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in summary['limitations'])}</ul></section></main></body></html>"""
    report = destination / "comparison.html"
    report.write_text(page, encoding="utf-8")
    return report
