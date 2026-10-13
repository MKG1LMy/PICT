from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .evaluation import evaluate_casewise_posterior
from .evaluation_protocol import enumerate_prior_scenarios, scenario_mask
from .parameters import CANONICAL_PARAMETERS
from .testing import (
    _aggregate_intervention_effects,
    _prior_intervention_comparison_rows,
)


def _dataset_sha256(dataset: h5py.Dataset) -> str:
    digest = hashlib.sha256()
    step = max(1, min(16, dataset.shape[0]))
    for start in range(0, dataset.shape[0], step):
        digest.update(np.asarray(dataset[start : start + step]).tobytes())
    return digest.hexdigest()


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
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, tuple, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _load_report(directory: Path) -> dict[str, Any]:
    path = directory / "report.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed test report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def compare_irf_ablation(
    explicit_directory: str | Path,
    dtof_only_directory: str | Path,
    output_directory: str | Path,
) -> Path:
    explicit_dir = Path(explicit_directory).resolve()
    dtof_dir = Path(dtof_only_directory).resolve()
    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    explicit_report = _load_report(explicit_dir)
    dtof_report = _load_report(dtof_dir)
    variants = (
        explicit_report.get("scope", {}).get("model_variant"),
        dtof_report.get("scope", {}).get("model_variant"),
    )
    if variants != ("explicit_irf", "dtof_only"):
        raise ValueError(
            "Comparison requires explicit_irf first and dtof_only second; "
            f"received {variants}"
        )
    explicit_h5 = h5py.File(explicit_dir / "posterior.h5", "r")
    dtof_h5 = h5py.File(dtof_dir / "posterior.h5", "r")
    try:
        explicit_meta = json.loads(explicit_h5.attrs["evaluation_metadata"])
        dtof_meta = json.loads(dtof_h5.attrs["evaluation_metadata"])
        checks = {
            "dataset_fingerprint": (
                explicit_report["reproducibility"]["dataset_fingerprint"],
                dtof_report["reproducibility"]["dataset_fingerprint"],
            ),
            "observation_seed": (
                explicit_meta["observation_seed"], dtof_meta["observation_seed"]
            ),
            "posterior_seed": (
                explicit_meta["posterior_seed"], dtof_meta["posterior_seed"]
            ),
            "posterior_samples": (
                explicit_meta["posterior_samples"], dtof_meta["posterior_samples"]
            ),
            "solver_steps": (
                explicit_meta["solver_steps"], dtof_meta["solver_steps"]
            ),
            "weights": (
                explicit_report["checkpoint"]["weights"],
                dtof_report["checkpoint"]["weights"],
            ),
            "sample_ids": (explicit_meta["sample_ids"], dtof_meta["sample_ids"]),
            "dtof_sha256": (
                _dataset_sha256(explicit_h5["cases"]["dtof_input"]),
                _dataset_sha256(dtof_h5["cases"]["dtof_input"]),
            ),
        }
        mismatches = [key for key, values in checks.items() if values[0] != values[1]]
        if mismatches:
            raise ValueError(
                "The two tests are not a strict paired experiment; mismatched: "
                + ", ".join(mismatches)
            )
        truth = explicit_h5["cases"]["truth"][:]
        active = explicit_h5["cases"]["active_mask"][:].astype(bool)
        if not np.array_equal(truth, dtof_h5["cases"]["truth"][:]) or not np.array_equal(
            active, dtof_h5["cases"]["active_mask"][:].astype(bool)
        ):
            raise ValueError("Truth or active-parameter masks differ between tests")
        sample_ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in explicit_h5["cases"]["sample_id"][:]
        ]
        scenario_map = {
            scenario.scenario_id: scenario
            for layer in (1, 2, 3)
            for scenario in enumerate_prior_scenarios(layer)
        }
        common = sorted(
            set(explicit_h5["scenarios"]) & set(dtof_h5["scenarios"])
        )
        rows: list[dict[str, Any]] = []
        casewise_rows: list[dict[str, Any]] = []
        excluded_numerical_pairs: list[dict[str, Any]] = []
        replicates = 1000
        for ordinal, scenario_id in enumerate(common):
            scenario = scenario_map[scenario_id]
            explicit_group = explicit_h5["scenarios"][scenario_id]
            dtof_group = dtof_h5["scenarios"][scenario_id]
            explicit_indices = explicit_group["global_case_index"][:]
            dtof_indices = dtof_group["global_case_index"][:]
            if not np.array_equal(explicit_indices, dtof_indices):
                raise ValueError(f"Case order differs for scenario {scenario_id}")
            explicit_diagnostics = explicit_group["diagnostics"][:]
            dtof_diagnostics = dtof_group["diagnostics"][:]
            valid = (
                (explicit_diagnostics[:, 4] == 0.0)
                & (dtof_diagnostics[:, 4] == 0.0)
            )
            if np.any(~valid):
                excluded_numerical_pairs.append(
                    {
                        "scenario_id": scenario_id,
                        "excluded_case_count": int(np.sum(~valid)),
                        "sample_ids": [
                            sample_ids[int(explicit_indices[index])]
                            for index in np.flatnonzero(~valid)
                        ],
                    }
                )
            if not np.any(valid):
                continue
            compared = _prior_intervention_comparison_rows(
                scenario,
                explicit_group["samples"][:][valid],
                dtof_group["samples"][:][valid],
                truth[explicit_indices][valid],
                active[explicit_indices][valid],
                explicit_diagnostics[valid],
                dtof_diagnostics[valid],
                "explicit_irf",
                "dtof_only",
                int(explicit_meta["posterior_seed"]) + ordinal,
                replicates,
            )
            rows.extend(compared)
            known = scenario_mask(scenario, len(explicit_indices))
            unknown = active[explicit_indices] & ~known
            explicit_casewise = evaluate_casewise_posterior(
                explicit_group["samples"][:], truth[explicit_indices], unknown
            )
            dtof_casewise = evaluate_casewise_posterior(
                dtof_group["samples"][:], truth[explicit_indices], unknown
            )
            for local, global_index in enumerate(explicit_indices):
                for parameter_index in np.flatnonzero(unknown[local]):
                    explicit_nmae = float(
                        explicit_casewise["normalized_absolute_error"][
                            local, parameter_index
                        ]
                    )
                    dtof_nmae = float(
                        dtof_casewise["normalized_absolute_error"][
                            local, parameter_index
                        ]
                    )
                    explicit_ncrps = float(
                        explicit_casewise["normalized_crps"][
                            local, parameter_index
                        ]
                    )
                    dtof_ncrps = float(
                        dtof_casewise["normalized_crps"][
                            local, parameter_index
                        ]
                    )
                    casewise_rows.append(
                        {
                            "scenario_id": scenario_id,
                            "sample_id": sample_ids[int(global_index)],
                            "layer_count": scenario.layer_count,
                            "known_count": scenario.known_count,
                            "parameter": CANONICAL_PARAMETERS[parameter_index],
                            "explicit_irf_normalized_mae": explicit_nmae,
                            "dtof_only_normalized_mae": dtof_nmae,
                            "delta_normalized_mae": explicit_nmae - dtof_nmae,
                            "explicit_irf_normalized_crps": explicit_ncrps,
                            "dtof_only_normalized_crps": dtof_ncrps,
                            "delta_normalized_crps": explicit_ncrps - dtof_ncrps,
                        }
                    )
        summary = _aggregate_intervention_effects(rows)
        input_statuses = {
            "explicit_irf": str(explicit_report.get("status", "unknown")),
            "dtof_only": str(dtof_report.get("status", "unknown")),
        }
        has_numerical_failures = bool(excluded_numerical_pairs) or any(
            "numerical_failures" in status for status in input_statuses.values()
        )
        has_other_warnings = any(status != "completed" for status in input_statuses.values())
        payload = {
            "status": (
                "completed_with_numerical_failures" if has_numerical_failures
                else "completed_with_input_warnings" if has_other_warnings
                else "completed"
            ),
            "comparison": "explicit_irf_vs_dtof_only",
            "input_statuses": input_statuses,
            "excluded_numerical_pairs": excluded_numerical_pairs,
            "strict_pairing_checks": {
                key: values[0] for key, values in checks.items()
            },
            "explicit_irf": explicit_report["checkpoint"],
            "dtof_only": dtof_report["checkpoint"],
            "scenario_count": len(common),
            "effect_rows": rows,
            "summary": summary,
            "interpretation": (
                "candidate minus reference: negative delta is better; relative "
                "improvement is positive when explicit IRF is better"
            ),
        }
        (destination / "irf_ablation_comparison.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _write_csv(destination / "irf_ablation_metrics.csv", rows)
        _write_csv(destination / "irf_ablation_summary.csv", summary)
        _write_csv(destination / "irf_ablation_casewise.csv", casewise_rows)
        macro = [
            row
            for row in rows
            if row["parameter"] == "__macro__" and row["known_count"] == 0
        ]
        table_rows = "".join(
            "<tr>"
            f"<td>N={int(row['layer_count'])}</td>"
            f"<td>{float(row['delta_normalized_mae']):.5f}</td>"
            f"<td>{100*float(row['relative_improvement_normalized_mae']):.2f}%</td>"
            f"<td>{float(row['delta_normalized_crps']):.5f}</td>"
            f"<td>{100*float(row['relative_improvement_normalized_crps']):.2f}%</td>"
            f"<td>{float(row['delta_energy_score']):.5f}</td>"
            "</tr>"
            for row in sorted(macro, key=lambda value: value["layer_count"])
        )
        warning = "" if payload["status"] == "completed" else (
            "<p class='warning'><b>Input reports or paired cases contain diagnostic warnings.</b> "
            f"Case-scenario pairs excluded due to numerical failure: {sum(item['excluded_case_count'] for item in excluded_numerical_pairs)}. "
            "Exclusions are recorded; this comparison must not be described as free of numerical failures.</p>"
        )
        document = f"""<!doctype html><html lang='en'><meta charset='utf-8'>
<title>Explicit-IRF and DTOF-only paired ablation report</title>
<style>body{{font:15px/1.6 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#17202c}}h1{{color:#245b91}}.note{{background:#edf4fb;padding:14px;border-left:4px solid #2463a7}}.warning{{background:#fff2f0;border:1px solid #ffb5ad;padding:14px;color:#8d231d}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #dce3ed;padding:9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style>
<h1>Explicit-IRF and DTOF-only paired ablation report</h1>
<p class='note'>Comparisons reuse identical test cases, DTOF, observation seeds, prior masks and initial posterior noise. Δ = explicit IRF − DTOF-only: negative error differences and positive relative improvements favor explicit IRF. Status: {html.escape(payload['status'])}</p>{warning}
<h2>No-prior results by layer count</h2><table><thead><tr><th>Layers</th><th>ΔnMAE</th><th>nMAE relative improvement</th><th>ΔnCRPS</th><th>nCRPS relative improvement</th><th>Δenergy</th></tr></thead><tbody>{table_rows}</tbody></table>
<h2>Audit information</h2><p>Paired scenarios: {len(common)}; Explicit IRF checkpoint: {html.escape(str(explicit_report['checkpoint']['path']))}; DTOF-only checkpoint: {html.escape(str(dtof_report['checkpoint']['path']))}</p>
<p>Full scenario-level and unknown-parameter results:  <a href='irf_ablation_metrics.csv'>irf_ablation_metrics.csv</a>  and  <a href='irf_ablation_comparison.json'>irf_ablation_comparison.json</a>. </p></html>"""
        output = destination / "irf_ablation_comparison.html"
        output.write_text(document, encoding="utf-8")
        return output
    finally:
        explicit_h5.close()
        dtof_h5.close()
