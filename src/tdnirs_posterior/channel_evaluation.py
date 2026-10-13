"""Paired tests of all nonempty subsets of the four fixed SDS channels."""
from __future__ import annotations

import hashlib
import html
import json

import numpy as np
import torch

from .channels import SDS, channel_id, subsets
from .evaluation_protocol import scenario_mask, thickness_group_labels
from .inference import sample_native_posterior_batch
from .parameters import ParameterSchema
from .precision import AUTOCAST_DTYPE, use_bf16_amp


def posterior_subsets(model, store, dtof, irf, truth, active, layers, ids, scenarios,
                      sds, device, cfg, posterior_seed, destination):
    from .testing import (_diagnostic_row, _sample_status_flags, _group_metrics,
                          _prior_intervention_comparison_rows, _write_csv, DIAGNOSTIC_COLUMNS)
    reports, effects, failures = {}, [], []
    schema = ParameterSchema()
    normalized = schema.normalize(truth)
    batch_size = int(cfg.get("batch_size", 3))
    count = int(cfg["posterior_samples"])
    selected_scenarios = [s for s in scenarios if not s.known_count or s.aliases]
    root = store.require_group("channel_subsets")
    for indices in subsets():
        name = channel_id(indices)
        is_full = len(indices) == 4
        branch = root.require_group(name)
        branch.attrs["sds_mm"] = [SDS[i] for i in indices]
        branch.attrs["dtof_sha256"] = hashlib.sha256(np.ascontiguousarray(dtof[:, indices]).tobytes()).hexdigest()
        encoded = []
        if not is_full:
            for start in range(0, len(truth), batch_size):
                x = torch.from_numpy(dtof[start:start+batch_size]).to(device)
                h = torch.from_numpy(irf[start:start+batch_size]).to(device)
                mask = torch.zeros(x.shape[:2], device=device, dtype=torch.bool)
                mask[:, list(indices)] = True
                with torch.no_grad(), torch.autocast(device_type=device.type,
                        enabled=use_bf16_amp(cfg, device), dtype=AUTOCAST_DTYPE):
                    signal = model.encode_view({"dtof": x, "irf": h, "channel_mask": mask}, sds)
                encoded.append(signal.float().cpu().numpy())
            encoded = np.concatenate(encoded)
        result = {}
        for scenario in selected_scenarios:
            selected = np.flatnonzero(layers == scenario.layer_count)
            reference = store["scenarios"][scenario.scenario_id]
            if is_full:
                group = reference
            else:
                group = branch.require_group(scenario.scenario_id)
                if "completed_cases" not in group.attrs:
                    # Incomplete initialization is safely rebuilt before any committed chunk.
                    for key in list(group.keys()):
                        del group[key]
                    group.create_dataset("samples", shape=(len(selected), count, 8), dtype="f4",
                                         chunks=(1, min(count, 128), 8), compression="gzip")
                    group.create_dataset("diagnostics", shape=(len(selected), len(DIAGNOSTIC_COLUMNS)), dtype="f4")
                    group.create_dataset("sample_status_flags", shape=(len(selected), count), dtype="u1")
                    group.create_dataset("global_case_index", data=selected)
                    group.attrs["completed_cases"] = 0
                    store.flush()
                for start in range(int(group.attrs["completed_cases"]), len(selected), batch_size):
                    stop = min(start + batch_size, len(selected))
                    ix = selected[start:stop]
                    known = scenario_mask(scenario, len(ix))
                    torch.manual_seed(posterior_seed + scenario.layer_count * 1_000_003 + start)
                    samples, diagnostics = sample_native_posterior_batch(
                        model, torch.from_numpy(encoded[ix]).to(device), torch.from_numpy(layers[ix]).to(device),
                        torch.from_numpy(active[ix]).to(device), torch.from_numpy(known).to(device),
                        torch.from_numpy(np.where(known, normalized[ix], 0).astype(np.float32)).to(device),
                        np.where(known, truth[ix], 0).astype(np.float32), count, int(cfg["solver_steps"]))
                    group["samples"][start:stop] = samples
                    group["diagnostics"][start:stop] = np.asarray([_diagnostic_row(d) for d in diagnostics])
                    group["sample_status_flags"][start:stop] = _sample_status_flags(samples, active[ix], layers[ix])
                    store.flush()
                    group.attrs["completed_cases"] = stop
                    store.flush()
            samples, diag = group["samples"][:], group["diagnostics"][:]
            finite = np.all(np.isfinite(samples) | ~active[selected, None, :], axis=(1, 2))
            for case in selected[~finite]:
                failures.append({"channel_subset": name, "scenario_id": scenario.scenario_id, "sample_id": ids[case]})
            known = scenario_mask(scenario, len(selected))
            metrics = _group_metrics(samples[finite], truth[selected][finite], active[selected][finite], known[finite],
                         layers[selected][finite], thickness_group_labels(truth[selected][finite], layers[selected][finite]),
                         joint_seed=posterior_seed + scenario.layer_count) if finite.any() else {}
            result[scenario.scenario_id] = {"metadata": scenario.to_dict(), "metrics": metrics,
                "total_cases": len(selected), "finite_cases": int(finite.sum()),
                "diagnostics": dict(zip(DIAGNOSTIC_COLUMNS, diag.mean(axis=0).tolist()))}
            if scenario.known_count:
                no_prior = next(s for s in selected_scenarios if s.layer_count == scenario.layer_count and not s.known_count)
                baseline = (store["scenarios"] if is_full else branch)[no_prior.scenario_id]
                base_samples = baseline["samples"][:]
                paired = finite & np.all(np.isfinite(base_samples) | ~active[selected, None, :], axis=(1, 2))
                if paired.any():
                    rows = _prior_intervention_comparison_rows(scenario, samples[paired], base_samples[paired],
                        truth[selected][paired], active[selected][paired], diag[paired], baseline["diagnostics"][:][paired],
                        "exact_prior", "no_prior", posterior_seed, int(cfg.get("bootstrap_replicates", 1000)))
                    for row in rows:
                        row.update(channel_subset=name, comparison_axis="prior", paired_finite_cases=int(paired.sum()))
                    effects.extend(rows)
            if not is_full:
                reference_samples = reference["samples"][:]
                paired = finite & np.all(np.isfinite(reference_samples) | ~active[selected, None, :], axis=(1, 2))
                if paired.any():
                    rows = _prior_intervention_comparison_rows(scenario, samples[paired], reference_samples[paired],
                        truth[selected][paired], active[selected][paired], diag[paired], reference["diagnostics"][:][paired],
                        name, "sds_10_20_30_40", posterior_seed, int(cfg.get("bootstrap_replicates", 1000)))
                    for row in rows:
                        row.update(channel_subset=name, comparison_axis="channels", channel_count=len(indices), total_cases=len(selected), paired_finite_cases=int(paired.sum()))
                    effects.extend(rows)
        reports[name] = {"sds_mm": [SDS[i] for i in indices], "channel_count": len(indices), "scenarios": result,
                         "dtof_sha256": str(branch.attrs["dtof_sha256"])}
        print(json.dumps({"event": "channel_test_progress", "channel_subset": name,
                          "completed_subsets": len(reports), "total_subsets": 15}), flush=True)
    _write_csv(destination / "channel_subset_effects.csv", effects)
    rows = []
    for name, entry in reports.items():
        for sid, scenario in entry["scenarios"].items():
            for parameter, metric in scenario["metrics"].get("all_unknown_parameters", {}).get("parameters", {}).items():
                rows.append({"channel_subset": name, "channel_count": entry["channel_count"], "scenario_id": sid,
                             "parameter": parameter, **metric})
    _write_csv(destination / "channel_subset_metrics.csv", rows)
    return {"subsets": reports, "paired_effects": effects, "numerical_failures": failures,
            "protocol": "Full acquisition is shared; only visible channels change. Missing-subset priors are no-prior and named exact priors."}


@torch.no_grad()
def baseline_subsets(model, store, dtof, prediction, truth, active, layers, ids, device, cfg, seed, destination):
    from .baseline_evaluation import evaluate_point_predictions, PHYSICAL_WIDTHS
    from .baseline_testing import _native_flags
    from .testing import _write_csv, _bootstrap_mean
    from .parameters import CANONICAL_PARAMETERS
    schema = ParameterSchema()
    reports, effects, failures = {}, [], []
    root = store.create_group("channel_subsets")
    size = int(cfg.get("batch_size", 3))
    replicates = int(cfg.get("bootstrap_replicates", 1000))
    metric_rows = []
    for indices in subsets():
        name = channel_id(indices)
        values = prediction
        if len(indices) != 4:
            parts = []
            for start in range(0, len(truth), size):
                x = torch.from_numpy(dtof[start:start+size]).to(device)
                n = torch.from_numpy(layers[start:start+size]).to(device)
                mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=device)
                mask[:, list(indices)] = True
                with torch.autocast(device_type=device.type, enabled=use_bf16_amp(cfg, device), dtype=AUTOCAST_DTYPE):
                    out = model(x, n, mask)
                parts.append(out.float().cpu().numpy())
            values = np.where(active, schema.denormalize(np.where(active, np.concatenate(parts), 0)), 0)
        group = root.create_group(name)
        group.create_dataset("prediction", data=values)
        group.attrs["sds_mm"] = [SDS[i] for i in indices]
        group.attrs["dtof_sha256"] = hashlib.sha256(np.ascontiguousarray(dtof[:, indices]).tobytes()).hexdigest()
        flags = _native_flags(values, layers)
        for key, flag in flags.items():
            group.create_dataset(key, data=flag)
        failed = [str(ids[i]) for i in np.flatnonzero(~flags["finite"])]
        failures.extend({"channel_subset": name, "sample_id": i} for i in failed)
        metrics = evaluate_point_predictions(values, truth, active, layers, bootstrap_seed=seed, bootstrap_replicates=replicates)
        reports[name] = {"sds_mm": [SDS[i] for i in indices], "channel_count": len(indices), "metrics": metrics,
                         "diagnostics": {k: float(np.mean(v)) for k, v in flags.items()}, "numerical_failure_sample_ids": failed,
                         "dtof_sha256": str(group.attrs["dtof_sha256"])}
        metric_rows.extend({"channel_subset": name, **r} for r in metrics["parameter_metrics"])
        for n in (1, 2, 3):
            selected = layers == n
            if not selected.any():
                continue
            for j in np.flatnonzero(active[selected][0]):
                ref = np.abs(prediction[selected, j] - truth[selected, j]) / PHYSICAL_WIDTHS[j]
                current = np.abs(values[selected, j] - truth[selected, j]) / PHYSICAL_WIDTHS[j]
                good = np.isfinite(ref) & np.isfinite(current)
                delta = current[good] - ref[good]
                effects.append({"channel_subset": name, "layer_count": n, "parameter": CANONICAL_PARAMETERS[j],
                    "total_cases": int(selected.sum()), "paired_finite_cases": int(good.sum()),
                    "current_nmae": float(current[good].mean()) if good.any() else None,
                    "reference_nmae": float(ref[good].mean()) if good.any() else None,
                    "delta_nmae": float(delta.mean()) if good.any() else None,
                    "relative_improvement_percent": float(-delta.mean()/ref[good].mean()*100) if good.any() and ref[good].mean()>0 else None,
                    "delta_ci95": _bootstrap_mean(delta, seed + n*10+j, replicates) if good.any() else None})
    _write_csv(destination / "channel_subset_metrics.csv", metric_rows)
    _write_csv(destination / "channel_subset_effects.csv", effects)
    return {"subsets": reports, "paired_effects": effects, "numerical_failures": failures}


def render_channel_section(report):
    data = report.get("channel_subsets")
    if not data:
        return ""
    from .reporting import _json_safe
    payload = json.dumps(_json_safe(data), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return """<section id="channel-subsets"><h2>Available SDS channel counts and subsets</h2>
<p>Cases and complete observations are paired; only visible channels change. Four channels are the reference. Missing-channel prior analysis covers no-prior and named exact-prior scenarios. Negative error changes and positive relative improvements favor the candidate. Numerical failures are recorded explicitly.</p>
<label>Filter subset, N or parameter <input id="channel-filter" placeholder="e.g. sds_20_40 / mua / N3"></label>
<div style="overflow:auto;max-height:650px"><table><thead><tr><th>Subset / Scenario</th><th>Parameter</th><th>Metrics</th></tr></thead><tbody id="channel-rows"></tbody></table></div>
<details><summary>Full paired results and 95% confidence intervals</summary><pre id="channel-effects" style="max-height:500px;overflow:auto"></pre></details>
<script>(()=>{const data=""" + payload + """;const rows=[];
for(const [name,entry] of Object.entries(data.subsets||{})){
 if(entry.scenarios){for(const [sid,s] of Object.entries(entry.scenarios)){
 for(const [p,m] of Object.entries((s.metrics?.all_unknown_parameters||{}).parameters||{})) rows.push([name+' / '+sid,p,JSON.stringify(m)]);}}
 else {for(const m of entry.metrics?.parameter_metrics||[])rows.push([name+' / N'+m.layer_count,m.parameter,JSON.stringify(m)]);}}
const body=document.getElementById('channel-rows');const draw=()=>{body.replaceChildren();const q=document.getElementById('channel-filter').value.toLowerCase();for(const r of rows){if(!r.join(' ').toLowerCase().includes(q))continue;const tr=document.createElement('tr');for(const v of r){const td=document.createElement('td');td.textContent=v;tr.appendChild(td);}body.appendChild(tr);}};
document.getElementById('channel-filter').addEventListener('input',draw);draw();
document.getElementById('channel-effects').textContent=JSON.stringify(data.paired_effects||[],null,2);})();</script></section>"""
