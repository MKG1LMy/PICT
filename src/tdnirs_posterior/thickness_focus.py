"""Shared, truth-defined thickness objective; never alters the fixed data split."""
from __future__ import annotations

import numpy as np


def focus_policy(values=None):
    values = values or {}
    result = {
        "enabled": bool(values.get("enabled", False)),
        "n2_d1_max_mm": float(values.get("n2_d1_max_mm", 15.0)),
        "n3_depth_mm": list(map(float, values.get("n3_depth_mm", [10.0, 20.0]))),
        "core_weight": float(values.get("core_weight", 0.7)),
        "other_weight": float(values.get("other_weight", 0.3)),
    }
    lohi = result["n3_depth_mm"]
    if (len(lohi) != 2 or not np.all(np.isfinite(lohi)) or not 0 <= lohi[0] < lohi[1]
            or not np.isfinite(result["n2_d1_max_mm"]) or result["n2_d1_max_mm"] <= 0
            or not all(np.isfinite(result[k]) and result[k] > 0 for k in ("core_weight", "other_weight"))):
        raise ValueError("Invalid training.thickness_focus policy")
    total = result["core_weight"] + result["other_weight"]
    result["core_weight"] /= total
    result["other_weight"] /= total
    return result


def core_mask(truth, layers, policy):
    truth, layers = np.asarray(truth), np.asarray(layers)
    lo, hi = policy["n3_depth_mm"]
    depth = truth[:, 0] + truth[:, 1]
    return ((layers == 1) | ((layers == 2) & (truth[:, 0] >= 0) & (truth[:, 0] <= policy["n2_d1_max_mm"]))
            | ((layers == 3) & (depth >= lo) & (depth <= hi)))


def strata(truth, layer, policy):
    """Return nonempty strata with normalized target mass, including tiny smokes."""
    if not policy["enabled"] or layer == 1:
        return {"all": (np.ones(len(truth), dtype=bool), 1.0)}
    core = core_mask(truth, np.full(len(truth), layer), policy)
    groups = {name: (mask, policy[name + "_weight"]) for name, mask in
              (("core", core), ("other", ~core)) if np.any(mask)}
    total = sum(weight for _, weight in groups.values())
    return {name: (mask, weight / total) for name, (mask, weight) in groups.items()}


def case_weights(truth, layers, policy):
    """Global weights sum to case count, preserving unbiased minibatch means.

    Equal N mass, then 70/30 thickness mass within N; do not normalize by the
    random minibatch's sum of weights (that would change the objective).
    """
    layers = np.asarray(layers)
    weights = np.ones(len(layers), dtype=np.float32)
    report = {}
    if not policy["enabled"]:
        return weights, report
    present = np.unique(layers)
    for layer in present:
        indices = np.flatnonzero(layers == layer)
        report[f"N{layer}"] = {}
        for name, (mask, mass) in strata(np.asarray(truth)[indices], layer, policy).items():
            selected = indices[mask]
            value = len(layers) / len(present) * mass / len(selected)
            weights[selected] = value
            report[f"N{layer}"][name] = {"cases": len(selected), "target_mass_within_layer": mass, "case_weight": value}
    return weights, report


def dataset_truth(dataset):
    """Read only thickness metadata, grouped by HDF5 source, without simulation."""
    import h5py
    truth = np.zeros((len(dataset.records), 8), dtype=np.float32)
    layers = np.asarray([r.layer_count for r in dataset.records])
    for file in sorted({r.file for r in dataset.records}):
        with h5py.File(dataset.index.root / file, "r") as handle:
            thickness = np.asarray(handle["thickness_mm"])
        for i, record in enumerate(dataset.records):
            if record.file == file:
                truth[i, :max(record.layer_count - 1, 0)] = thickness[record.source_index, :max(record.layer_count - 1, 0)]
    return truth, layers


def configure_training_focus(dataset, policy):
    if not policy["enabled"]:
        return {}
    truth, layers = dataset_truth(dataset)
    dataset.loss_weights, report = case_weights(truth, layers, policy)
    return report


class FocusValidationLoader:
    def __init__(self, loader, policy, core):
        self.loader, self.policy, self.core = loader, policy, core

    def __iter__(self):
        import torch
        for raw in self.loader:
            selected = core_mask(raw["theta_physical"].numpy(), raw["layer_count"].numpy(), self.policy)
            selected = selected if self.core else ~selected
            if not np.any(selected):
                continue
            def select(value):
                if isinstance(value, torch.Tensor):
                    return value[torch.from_numpy(selected)]
                if isinstance(value, dict):
                    return {k: select(v) for k, v in value.items()}
                if isinstance(value, (tuple, list)):
                    return [v for v, keep in zip(value, selected) if keep]
                return value
            yield select(raw)
