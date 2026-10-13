from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _scenario_summaries(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scenario_id, entry in report.get("scenario_reports", {}).items():
        metadata = entry.get("metadata", {})
        metrics = entry.get("metrics", {}).get("all_unknown_parameters", {})
        result.append(
            {
                "scenario_id": scenario_id,
                "metadata": metadata,
                "cases": entry.get("cases"),
                "evaluated_cases": entry.get("evaluated_cases"),
                "aggregate": metrics.get("aggregate", {}),
                "joint": {
                    key: value
                    for key, value in metrics.get("joint", {}).items()
                    if key
                    in {
                        "case_count",
                        "mean_energy_score",
                        "mean_pairwise_distance",
                        "mean_geometric_sd",
                        "mean_covariance_volume",
                        "tarp_alpha",
                        "tarp_ecp",
                        "tarp_mean_absolute_coverage_error",
                    }
                },
                "native_diagnostics": entry.get("native_diagnostics", {}),
            }
        )
    return result


def _selected_parameter_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # The all-parameter group is the audit table for every mask. Thickness rows
    # are exposed separately, so the redundant layer/group copies stay out of
    # the self-contained report payload.
    return [dict(row) for row in rows if row.get("group") == "all"]


def _selected_thickness_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # Thickness evidence is most useful for the no-prior baseline and the
    # named paper scenarios. Full per-mask rows remain in the companion CSV.
    return [
        dict(row)
        for row in rows
        if int(row.get("known_count", 0)) == 0 or bool(row.get("aliases"))
    ]


def render_test_report(
    report: Mapping[str, Any],
    *,
    parameter_rows: Iterable[Mapping[str, Any]] = (),
    thickness_rows: Iterable[Mapping[str, Any]] = (),
    prior_rows: Iterable[Mapping[str, Any]] = (),
    prior_summary_rows: Iterable[Mapping[str, Any]] = (),
    prior_intervention_rows: Iterable[Mapping[str, Any]] = (),
    prior_intervention_summary_rows: Iterable[Mapping[str, Any]] = (),
    irf_intervention_rows: Iterable[Mapping[str, Any]] = (),
    irf_intervention_summary_rows: Iterable[Mapping[str, Any]] = (),
    thickness_associations: Iterable[Mapping[str, Any]] = (),
    multimodality_rows: Iterable[Mapping[str, Any]] = (),
    asset_directory: str | Path | None = None,
) -> str:
    """Create the complete, self-contained, offline technical test report.

    The report embeds reviewed aggregate/detail tables generated during the
    test itself. Opening report.html never reparses the HDF5 posterior or CSV
    sidecars and therefore cannot silently produce a second evaluation.
    """

    mode_rows = [dict(row) for row in multimodality_rows]
    if asset_directory is not None:
        asset_root = Path(asset_directory).resolve()
        for row in mode_rows:
            relative = row.get("plot_file")
            if not relative:
                continue
            candidate = (asset_root / str(relative)).resolve()
            try:
                candidate.relative_to(asset_root)
            except ValueError:
                continue
            if candidate.is_file() and candidate.suffix.lower() == ".svg":
                encoded_svg = base64.b64encode(candidate.read_bytes()).decode("ascii")
                row["plot_data_uri"] = f"data:image/svg+xml;base64,{encoded_svg}"
    payload = {
        "report": {
            key: report.get(key)
            for key in (
                "status",
                "created_at_utc",
                "checkpoint",
                "scope",
                "reproducibility",
                "headline",
                "prior_intervention_reports",
                "prior_intervention_summary",
                "irf_intervention_reports",
                "irf_intervention_summary",
                "irf_case_metadata",
                "irf_difficulty_metrics",
                "fixed_irf_metadata",
                "multimodality_summary",
                "gmm_failure_cases",
                "representative_mode_plots",
                "numerical_failure_evaluations",
                "numerical_failure_cases",
                "runtime",
                "metric_definitions",
                "limitations",
            )
        },
        "scenario_summaries": _scenario_summaries(report),
        "parameter_metrics": _selected_parameter_rows(parameter_rows),
        "thickness_metrics": _selected_thickness_rows(thickness_rows),
        "prior_paired_effects": [dict(row) for row in prior_rows],
        "prior_summary": [dict(row) for row in prior_summary_rows],
        "prior_intervention_effects": [
            dict(row) for row in prior_intervention_rows
        ],
        "prior_intervention_summary": [
            dict(row) for row in prior_intervention_summary_rows
        ],
        "irf_intervention_effects": [dict(row) for row in irf_intervention_rows],
        "irf_intervention_summary": [
            dict(row) for row in irf_intervention_summary_rows
        ],
        "thickness_associations": [dict(row) for row in thickness_associations],
        "multimodality": mode_rows,
    }
    encoded = json.dumps(
        _json_safe(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).replace("</", "<\\/")
    from .channel_evaluation import render_channel_section
    return _REPORT_TEMPLATE.replace("__REPORT_DATA__", encoded).replace("</body>", render_channel_section(report) + "</body>")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def render_report_directory(
    test_directory: str | Path, output: str | Path | None = None
) -> Path:
    """Rebuild only the presentation from an already completed test folder."""

    directory = Path(test_directory).resolve()
    report_path = directory / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing completed report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    destination = Path(output).resolve() if output else directory / "report.html"
    html_text = render_test_report(
        report,
        parameter_rows=_read_csv(directory / "parameter_metrics.csv"),
        thickness_rows=_read_csv(directory / "thickness_metrics.csv"),
        prior_rows=_read_csv(directory / "prior_paired_effects.csv"),
        prior_summary_rows=_read_csv(directory / "prior_summary.csv"),
        prior_intervention_rows=_read_csv(
            directory / "prior_intervention_effects.csv"
        ),
        prior_intervention_summary_rows=_read_csv(
            directory / "prior_intervention_summary.csv"
        ),
        irf_intervention_rows=_read_csv(
            directory / "irf_intervention_effects.csv"
        ),
        irf_intervention_summary_rows=_read_csv(
            directory / "irf_intervention_summary.csv"
        ),
        thickness_associations=_read_csv(directory / "thickness_associations.csv"),
        multimodality_rows=_read_csv(directory / "multimodality.csv"),
        asset_directory=directory,
    )
    temporary = destination.with_name(f"{destination.name}.tmp")
    temporary.write_text(html_text, encoding="utf-8")
    temporary.replace(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the detailed HTML presentation of a completed test"
    )
    parser.add_argument("test_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(render_report_directory(args.test_directory, args.output))


_REPORT_TEMPLATE = Path(__file__).with_name("report_template.html").read_text(encoding="utf-8")



if __name__ == "__main__":
    main()
