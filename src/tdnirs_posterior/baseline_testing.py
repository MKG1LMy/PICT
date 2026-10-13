from __future__ import annotations

import csv
import hashlib
import html
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from .baseline_evaluation import PHYSICAL_WIDTHS, _metrics, evaluate_point_predictions
from .config import config_seed, project_path
from .data.dataset import H5TPSFDataset
from .model import BaselineConfig, build_baseline, estimate_baseline_macs
from .parameters import CANONICAL_PARAMETERS, ParameterSchema
from .precision import AUTOCAST_DTYPE, use_bf16_amp
from .training import _dataset_fingerprint, _manifest, _sha256
from .testing import _irf_strata
from .channel_evaluation import render_channel_section


def _irf_stratified_rows(
    prediction: np.ndarray,
    truth: np.ndarray,
    active: np.ndarray,
    layers: np.ndarray,
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stratum_type, labels in _irf_strata(metadata).items():
        for stratum_value in sorted(set(map(str, labels))):
            in_stratum = labels == stratum_value
            for layer in (1, 2, 3):
                for parameter_index, parameter in enumerate(CANONICAL_PARAMETERS):
                    selected = in_stratum & (layers == layer) & active[:, parameter_index]
                    if not np.any(selected):
                        continue
                    rows.append(
                        {
                            "stratum_type": stratum_type,
                            "stratum_value": stratum_value,
                            "layer_count": layer,
                            "parameter": parameter,
                            "count": int(selected.sum()),
                            **_metrics(
                                prediction[selected, parameter_index],
                                truth[selected, parameter_index],
                                float(PHYSICAL_WIDTHS[parameter_index]),
                            ),
                        }
                    )
                if layer == 3:
                    selected = in_stratum & (layers == 3)
                    if np.any(selected):
                        rows.append(
                            {
                                "stratum_type": stratum_type,
                                "stratum_value": stratum_value,
                                "layer_count": 3,
                                "parameter": "d1_plus_d2_mm",
                                "count": int(selected.sum()),
                                **_metrics(
                                    prediction[selected, 0] + prediction[selected, 1],
                                    truth[selected, 0] + truth[selected, 1],
                                    34.0,
                                ),
                            }
                        )
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
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _native_flags(prediction: np.ndarray, layers: np.ndarray) -> dict[str, np.ndarray]:
    schema = ParameterSchema()
    low, high, _ = schema._bounds()
    active = np.stack([schema.active_mask(int(layer)) for layer in layers])
    finite = np.all((~active) | np.isfinite(prediction), axis=1)
    outside = np.any(active & ((prediction < low) | (prediction > high)), axis=1)
    study = (layers == 3) & (prediction[:, 0] + prediction[:, 1] > schema.n3_total_thickness_max_mm)
    impossible = np.zeros(len(prediction), dtype=bool)
    for index, layer in enumerate(layers):
        thickness_indices = [0] if layer == 2 else [0, 1] if layer == 3 else []
        optical_indices = np.flatnonzero(active[index] & np.asarray([False, False, True, True, True, True, True, True]))
        impossible[index] = bool(
            any(prediction[index, item] < 0.0 for item in thickness_indices)
            or any(prediction[index, item] <= 0.0 for item in optical_indices)
        )
    return {"finite": finite, "outside_training_support": outside, "outside_study_constraint": study, "physically_impossible": impossible}


def _report_html(report: dict[str, Any]) -> str:
    def number(value: Any, digits: int = 4) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "—"
        return f"{numeric:.{digits}f}" if np.isfinite(numeric) else "Numerical failure"

    def percent(value: Any) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return "—"
        return f"{numeric:.2%}" if np.isfinite(numeric) else "Numerical failure"

    parameter_rows = "".join(
        "<tr>"
        f"<td>N={item['layer_count']}</td><td>{html.escape(item['parameter'])}</td>"
        f"<td>{item['count']}</td><td>{percent(item['normalized_mae'])}</td>"
        f"<td>{number(item['mae'], 6)}</td><td>{number(item['rmse'], 6)}</td>"
        f"<td>{number(item['r2'], 3)}</td><td>{item.get('numerical_failure_count', 0)}</td>"
        "</tr>" for item in report["metrics"]["parameter_metrics"]
    )
    thickness_rows = "".join(
        "<tr>"
        f"<td>N={item['layer_count']}</td><td>{html.escape(item['group'])}</td>"
        f"<td>{html.escape(item['parameter'])}</td><td>{item['count']}</td>"
        f"<td>{percent(item['normalized_mae'])}</td><td>{number(item['r2'], 3)}</td>"
        "</tr>" for item in report["metrics"]["thickness_metrics"]
    )
    association_rows = "".join(
        "<tr>"
        f"<td>N={item['layer_count']}</td><td>{html.escape(item['thickness_variable'])}</td>"
        f"<td>{html.escape(item['parameter'])}</td><td>{number(item.get('rho'), 3)}</td>"
        f"<td>{html.escape(str(item.get('ci95', '—')))}</td>"
        f"<td>{item.get('count', 0)}/{item.get('total_count', item.get('count', 0))}</td>"
        "</tr>" for item in report["metrics"]["thickness_associations"]
    )
    irf_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(item['stratum_type']))}</td>"
        f"<td>{html.escape(str(item['stratum_value']))}</td>"
        f"<td>N={item['layer_count']}</td>"
        f"<td>{html.escape(str(item['parameter']))}</td>"
        f"<td>{item['count']}</td>"
        f"<td>{percent(item['normalized_mae'])}</td>"
        f"<td>{number(item['r2'], 3)}</td>"
        "</tr>"
        for item in report["metrics"].get("irf_stratified_metrics", [])
    )
    layer_cards = "".join(
        f"<div class='metric'><span>{name}</span><b>{percent(value)}</b><small>Equal-weight parameter macro nMAE</small></div>"
        for name, value in report["metrics"]["layer_macro_nmae"].items()
    )
    profile, runtime, native = report["model"], report["runtime"], report["native_output"]
    warning = "" if report["status"] == "completed" else (
        "<div class='warning'><b>Numerical failure warning: </b>At least one active prediction contains NaN or Inf; "
        "Affected aggregates are invalidated rather than silently excluding failed cases.</div>"
    )
    limitations = "".join(f"<li>{html.escape(value)}</li>" for value in report["limitations"])
    return f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><title>TD-NIRS {profile['architecture'].upper()} baseline report</title>
<style>body{{margin:0;background:#eef3f8;color:#183047;font:15px/1.6 system-ui,sans-serif}}main{{max-width:1240px;margin:auto;padding:36px 24px}}h1{{font-size:34px;margin:.2em 0}}h2{{margin-top:0}}.sub{{color:#63788d}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:22px 0}}.metric,section{{background:white;border:1px solid #dbe4ec;border-radius:14px;padding:18px;box-shadow:0 5px 18px #17324d0a}}.metric span,.metric small{{display:block;color:#6c8092}}.metric b{{font-size:25px;color:#155f9b}}section{{margin:18px 0;overflow:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{padding:9px;border-bottom:1px solid #e3e9ef;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}.warning{{background:#fff2f0;border:1px solid #ffb5ad;border-radius:12px;padding:14px;color:#8d231d}}.links a{{margin-right:16px}}code,pre{{font-family:Consolas,monospace}}pre{{white-space:pre-wrap}}</style></head>
<body><main><p class='sub'>Fixed test set · Deterministic DTOF-only baseline · {html.escape(report['status'])}</p><h1>{profile['architecture'].upper()} baseline evaluation report</h1>
<p class='sub'>The model receives preprocessed DTOF, channel availability and layer count N; no IRF tensors or parameter priors. Accuracy includes only parameters active for the specified layer count.</p>{warning}
<div class='grid'>{layer_cards}<div class='metric'><span>Overall</span><b>{percent(report['metrics']['macro_nmae'])}</b><small>N=1/2/3 equal-weight macro average</small></div><div class='metric'><span>Parameters</span><b>{int(profile['parameter_count']):,}</b><small>Trainable parameters</small></div><div class='metric'><span>Inference time</span><b>{number(runtime['milliseconds_per_case'],2)} ms</b><small>per case</small></div><div class='metric'><span>Finite predictions</span><b>{percent(native['finite_fraction'])}</b><small>Cases with all active predictions finite</small></div></div>
<section><h2>Parameter accuracy</h2><table><thead><tr><th>Layers</th><th>Parameter</th><th>Cases</th><th>nMAE</th><th>MAE</th><th>RMSE</th><th>R²</th><th>Numerical failure</th></tr></thead><tbody>{parameter_rows}</tbody></table></section>
<section><h2>Surface-thickness groups</h2><table><thead><tr><th>Layers</th><th>Thickness interval</th><th>Parameter</th><th>Cases</th><th>nMAE</th><th>R²</th></tr></thead><tbody>{thickness_rows}</tbody></table></section>
<section><h2>Thickness-error associations</h2><p class='sub'>Spearman describes association, not causality; finite/total cases shows losses due to numerical failures; small-group R² is auxiliary.</p><table><thead><tr><th>Layers</th><th>Thickness variable</th><th>Target parameter</th><th>Spearman ρ</th><th>Bootstrap 95% CI</th><th>Finite/total cases</th></tr></thead><tbody>{association_rows}</tbody></table></section>
<section><h2>Four-channel IRF difficulty strata</h2><p class='sub'>Stratified by channel relationship, delay difference, FWHM variability, waveform heterogeneity, outlier channels and response components; only active parameters are evaluated.</p><table><thead><tr><th>Stratification</th><th>Stratum</th><th>Layers</th><th>Parameter</th><th>Cases</th><th>nMAE</th><th>R²</th></tr></thead><tbody>{irf_rows}</tbody></table></section>
<section><h2>Native-output diagnostics</h2><pre>{html.escape(json.dumps(native, ensure_ascii=False, indent=2))}</pre></section>
<section><h2>Files and limitations</h2><p class='links'><a href='parameter_metrics.csv'>Parameters CSV</a><a href='thickness_metrics.csv'>Thickness groups CSV</a><a href='thickness_associations.csv'>Associations CSV</a><a href='irf_stratified_metrics.csv'>IRF strata CSV</a><a href='casewise_metrics.csv'>Cases CSV</a><a href='predictions.h5'>Raw predictions HDF5</a><a href='report.json'>Complete JSON</a></p><ul>{limitations}</ul></section>
{render_channel_section(report)}
</main></body></html>"""


@torch.no_grad()
def run_baseline_test(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    weights: str = "ema",
    max_test_samples: int | None = None,
) -> Path:
    started = time.perf_counter()
    checkpoint = project_path(config, checkpoint_path).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = payload["config"]
    if checkpoint_config["baseline"]["architecture"] != config["baseline"]["architecture"]:
        raise ValueError("Current baseline architecture differs from the checkpoint")
    if checkpoint_config["simulation"] != config["simulation"]:
        raise ValueError("Current simulation configuration differs from the checkpoint")
    for key in ("sds_mm", "time_bins", "time_bin_ps", "time_start_ps"):
        if checkpoint_config["data"][key] != config["data"][key]:
            raise ValueError(f"Current data.{key} differs from the checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = BaselineConfig.from_mapping(checkpoint_config["baseline"])
    model = build_baseline(model_config, int(config["data"]["time_bins"])).to(device)
    state = payload["ema"] if weights == "ema" else payload["model"]
    model.load_state_dict(state)
    model.eval()
    index = _manifest(config)
    fingerprint = _dataset_fingerprint(index)
    if payload.get("dataset_fingerprint") != fingerprint:
        raise ValueError("Current manifest or HDF5 data differs from the baseline checkpoint")
    evaluation = config["evaluation"]
    time_alignment_enabled = bool(
        config["simulation"]["preprocessing"]["time_alignment_enabled"]
    )
    evaluation_seed = config_seed(config, "evaluation")
    observation_seed = evaluation_seed + int(evaluation.get("observation_seed_offset", 200000))
    requested = max_test_samples if max_test_samples is not None else evaluation.get("max_test_samples")
    dataset = H5TPSFDataset(index, "test", config["simulation"], config["data"]["sds_mm"], observation_seed, requested, paired_views=False)
    dataset.set_epoch(0)
    loader = DataLoader(dataset, batch_size=int(evaluation.get("batch_size", 3)), shuffle=False, num_workers=int(evaluation.get("num_workers", 0)), pin_memory=device.type == "cuda")
    schema = ParameterSchema()
    predictions_normalized: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    active_masks: list[np.ndarray] = []
    layers: list[np.ndarray] = []
    dtofs: list[np.ndarray] = []
    sample_ids: list[str] = []
    irf_metadata: list[dict[str, Any]] = []
    inference_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for batch_index, raw in enumerate(loader):
            dtof = raw["view_a"]["dtof"].to(device, non_blocking=True)
            layer = raw["layer_count"].to(device, non_blocking=True)
            tick = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=use_bf16_amp(evaluation, device), dtype=AUTOCAST_DTYPE):
                prediction = model(dtof, layer)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - tick
            predictions_normalized.append(prediction.float().cpu().numpy())
            truths.append(raw["theta_physical"].numpy())
            active_masks.append(raw["active_mask"].numpy().astype(bool))
            layers.append(raw["layer_count"].numpy())
            dtofs.append(raw["view_a"]["dtof"].numpy())
            irf_metadata.extend(
                json.loads(str(value))
                for value in raw["view_a"]["irf_metadata_json"]
            )
            sample_ids.extend(map(str, raw["sample_id"]))
            if (batch_index + 1) % 10 == 0 or batch_index + 1 == len(loader):
                print(
                    json.dumps(
                        {
                            "event": "baseline_test_progress",
                            "architecture": model_config.architecture,
                            "completed_cases": len(sample_ids),
                            "total_cases": len(dataset),
                        }
                    ),
                    flush=True,
                )
    finally:
        dataset.close()
    normalized = np.concatenate(predictions_normalized)
    truth = np.concatenate(truths)
    active = np.concatenate(active_masks)
    normalized = np.where(active, normalized, 0.0)
    prediction = schema.denormalize(normalized)
    prediction = np.where(active, prediction, 0.0)
    layer_count = np.concatenate(layers)
    dtof_input = np.concatenate(dtofs)
    metrics = evaluate_point_predictions(prediction, truth, active, layer_count, bootstrap_seed=observation_seed + 700000, bootstrap_replicates=int(evaluation.get("bootstrap_replicates", 1000)))
    metrics["irf_stratified_metrics"] = _irf_stratified_rows(
        prediction, truth, active, layer_count, irf_metadata
    )
    flags = _native_flags(prediction, layer_count)
    native = {f"{name}_fraction": float(np.mean(values)) for name, values in flags.items()}
    native["numerical_failure_count"] = int(np.sum(~flags["finite"]))
    native["numerical_failure_sample_ids"] = [
        sample_ids[index] for index in np.flatnonzero(~flags["finite"])
    ]
    destination = project_path(config, output_dir) if output_dir is not None else checkpoint.parent / "test"
    destination.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination / "predictions.h5", "w") as store:
        cases = store.create_group("cases")
        cases.create_dataset("prediction", data=prediction)
        cases.create_dataset("prediction_normalized", data=normalized)
        cases.create_dataset("truth", data=truth)
        cases.create_dataset("active_mask", data=active)
        cases.create_dataset("layer_count", data=layer_count)
        cases.create_dataset("dtof_input", data=dtof_input, compression="gzip", chunks=(1, *dtof_input.shape[1:]))
        cases.create_dataset("sample_id", data=np.asarray(sample_ids, dtype=object), dtype=h5py.string_dtype("utf-8"))
        cases.create_dataset(
            "irf_metadata_json",
            data=np.asarray(
                [json.dumps(item, sort_keys=True) for item in irf_metadata],
                dtype=object,
            ),
            dtype=h5py.string_dtype("utf-8"),
        )
        for name, values in flags.items():
            cases.create_dataset(name, data=values)
        store.attrs["evaluation_metadata"] = json.dumps({"observation_seed": observation_seed, "sample_ids": sample_ids, "weights": weights, "time_alignment_enabled": time_alignment_enabled})
        from .channel_evaluation import baseline_subsets
        channel_results = baseline_subsets(model, store, dtof_input, prediction, truth, active,
            layer_count, sample_ids, device, evaluation, observation_seed, destination)
    for case, sample_id in zip(metrics["casewise_metrics"], np.repeat(sample_ids, [int(np.sum(row)) for row in active])):
        case["sample_id"] = sample_id
    _write_csv(destination / "parameter_metrics.csv", metrics["parameter_metrics"])
    _write_csv(destination / "thickness_metrics.csv", metrics["thickness_metrics"])
    _write_csv(destination / "thickness_associations.csv", metrics["thickness_associations"])
    _write_csv(destination / "casewise_metrics.csv", metrics["casewise_metrics"])
    _write_csv(destination / "irf_stratified_metrics.csv", metrics["irf_stratified_metrics"])
    training_summary_path = checkpoint.parent / "training_summary.json"
    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8")) if training_summary_path.exists() else {}
    report = {
        "status": "completed" if native["finite_fraction"] == 1.0 and not channel_results["numerical_failures"] else "completed_with_numerical_failures",
        "channel_subsets": channel_results,
        "model": {"architecture": model_config.architecture, "parameter_count": sum(parameter.numel() for parameter in model.parameters()), "approximate_macs_per_case": estimate_baseline_macs(model_config, int(config["data"]["time_bins"])), "approximate_flops_per_case": 2 * estimate_baseline_macs(model_config, int(config["data"]["time_bins"])), "input": "DTOF and layer count N", "output": "one deterministic eight-slot parameter vector with layer masking"},
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint), "epoch": int(payload["epoch"]) + 1, "weights": weights},
        "scope": {"split": "test", "test_cases": len(sample_ids), "probabilistic_metrics": "not_applicable", "time_alignment_enabled": time_alignment_enabled},
        "reproducibility": {"evaluation_seed": evaluation_seed, "observation_seed": observation_seed, "time_alignment_enabled": time_alignment_enabled, "dataset_fingerprint": fingerprint},
        "metrics": metrics,
        "metric_definitions": {
            "normalized_mae": "physical MAE divided by the configured physical training-support width",
            "layer_macro_nmae": "equal mean across active parameters within each N",
            "macro_nmae": "equal mean of N=1, N=2, and N=3 layer macro nMAE",
            "inactive_parameters": "excluded from every loss and evaluation metric",
        },
        "native_output": native,
        "runtime": {"training_elapsed_seconds": training_summary.get("training_elapsed_seconds"), "training_completed_epochs": training_summary.get("completed_epochs"), "elapsed_seconds": time.perf_counter() - started, "inference_seconds": inference_seconds, "milliseconds_per_case": 1000.0 * inference_seconds / max(len(sample_ids), 1), "device": str(device), "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None},
        "limitations": ["The baseline is deterministic and does not define CRPS, coverage, SBC, TARP, energy score, or multimodality.", ("Each measured IRF is used only to align its paired DTOF during deterministic preprocessing and is not a neural-network input." if time_alignment_enabled else "The measured IRF bundle is not a neural-network input and no IRF-based time alignment is applied."), "The baseline has no optional-prior interface; comparisons with PICT Model are restricted to no-prior cases."],
    }
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "report.html").write_text(_report_html(report), encoding="utf-8")
    print(json.dumps({"event": "baseline_test_complete", "report": str(destination / "report.html"), "status": report["status"]}), flush=True)
    return destination / "report.html"
