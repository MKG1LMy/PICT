from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tdnirs_posterior.thickness_focus import focus_policy, core_mask, case_weights, dataset_truth
from tdnirs_posterior.log_time import timestamp
from tdnirs_posterior.model import masked_smooth_l1
from tdnirs_posterior.parameters import ParameterSchema
from tdnirs_posterior.training import (
    _run_validation, _run_posterior_validation, _posterior_validation_indices,
    _posterior_group_regression_check,
)
from tdnirs_posterior.baseline_training import _validate
from tdnirs_posterior.baseline_evaluation import PHYSICAL_WIDTHS


FOCUS = focus_policy({"enabled": True})


def batch():
    schema = ParameterSchema()
    layers = np.array([1, 2, 2, 3, 3])
    active = np.stack([schema.active_mask(int(n)) for n in layers])
    truth = np.tile([8., 7., .01, .01, .01, 1., 1., 1.], (5, 1)).astype(np.float32)
    truth[2, 0], truth[4, 0], truth[4, 1] = 18., 15., 12.
    truth[~active] = 0
    wave = torch.from_numpy(truth)
    return {"theta": torch.from_numpy(schema.normalize(truth)), "theta_physical": wave,
            "layer_count": torch.from_numpy(layers), "active_mask": torch.from_numpy(active),
            "known_mask": torch.zeros((5, 8), dtype=torch.bool), "known_values": torch.zeros(5, 8),
            "view_a": {"dtof": wave, "irf": wave}, "view_b": {"dtof": wave, "irf": wave}}


def test_boundaries_and_global_training_mass():
    truth = np.zeros((9, 8))
    layers = np.array([1, 2, 2, 2, 3, 3, 3, 3, 3])
    truth[:, 0] = [0, .5, 15, 15.01, 4, 5, 10, 10, 18]
    truth[:, 1] = [0, 0, 0, 0, 5, 5, 10, 10.01, 10]
    core = core_mask(truth, layers, FOCUS)
    assert core.tolist() == [True, True, True, False, False, True, True, False, False]
    weights, _ = case_weights(truth, layers, FOCUS)
    assert weights.sum() == pytest.approx(len(layers))
    for n in (1, 2, 3):
        assert weights[layers == n].sum() == pytest.approx(len(layers) / 3)
        if n > 1:
            assert weights[(layers == n) & core].sum() == pytest.approx(len(layers) / 3 * .7)
    # Splitting weighted loss into accumulated microbatches gives the same loss.
    pred = torch.arange(9.).reshape(9, 1).repeat(1, 8)
    active = torch.ones_like(pred, dtype=torch.bool)
    w = torch.from_numpy(weights)
    whole = masked_smooth_l1(pred, torch.zeros_like(pred), active, .1, w)
    chunks = sum(masked_smooth_l1(pred[s], torch.zeros_like(pred[s]), active[s], .1, w[s]) * len(pred[s]) / 9
                 for s in (slice(0, 4), slice(4, 8), slice(8, 9)))
    torch.testing.assert_close(whole, chunks)


def test_flow_and_baseline_validation_use_seventy_thirty():
    raw = batch()
    core = core_mask(raw["theta_physical"].numpy(), raw["layer_count"].numpy(), FOCUS)
    errors = np.where(core, .01, .2)

    class Flow:
        def flow_matching_loss(self, theta, *args):
            return theta[:, 0].mean()

    class Model:
        posterior = Flow()
        def eval(self): return self
        def encode_view(self, view, sds): return view["dtof"]
        def __call__(self, dtof, layers, channel_mask=None): return dtof

    raw["theta"][:, 0] = torch.from_numpy(errors)
    flow = _run_validation(Model(), [raw], torch.device("cpu"), torch.tensor([10.]), .1, False, 1, FOCUS)
    expected = (.01 + 2 * (.7 * .01 + .3 * .2)) / 3
    assert flow["flow"] == pytest.approx(expected)
    schema = ParameterSchema()
    truth = raw["theta_physical"].numpy()
    pred = truth + errors[:, None] * PHYSICAL_WIDTHS[None, :]
    raw["view_a"]["dtof"] = torch.from_numpy(schema.normalize(pred).astype(np.float32))
    baseline = _validate(Model(), [raw], torch.device("cpu"), False, FOCUS)
    assert baseline["macro_nmae"] == pytest.approx(expected, abs=1e-6)
    assert baseline["core_layer_nmae"]["N3"] == pytest.approx(.01, abs=1e-6)


def test_posterior_objective_includes_other_but_guards_protect_core():
    raw = batch()
    class Model:
        def eval(self): return self
        def encode_view(self, view, sds): return view["dtof"]

    def samples(**kwargs):
        truth = kwargs["signal"].numpy()
        layers = kwargs["layer_count"].numpy()
        core = core_mask(truth, layers, FOCUS)
        unknown = (kwargs["active_mask"] & ~kwargs["known_mask"]).numpy()
        prediction = truth + (~core)[:, None] * unknown * PHYSICAL_WIDTHS[None, :] * .1
        draws = np.repeat(prediction[:, None, :], kwargs["sample_count"], axis=1)
        diagnostics = [{"finite_fraction": 1., "numerical_failure": False,
                        "outside_training_support_fraction": 0., "physically_impossible_fraction": 0.}
                       for _ in truth]
        return draws, diagnostics

    with patch("tdnirs_posterior.training.sample_native_posterior_batch", side_effect=samples):
        result = _run_posterior_validation(Model(), [raw], torch.device("cpu"), torch.tensor([10.]),
            False, 1, 8, 2, .15, focus=FOCUS)
    assert result["score"] > 0
    assert result["eligible"]
    assert all(v == pytest.approx(0) for v in result["group_scores"].values())
    assert _posterior_group_regression_check(result["group_scores"], result["group_scores"], .05)["passed"]
    detail = result["scenarios"]["N3/no_prior"]
    assert detail["by_thickness"]["core"]["weight"] == .7
    assert detail["by_thickness"]["other"]["metrics"]["aggregate"]["mean_normalized_crps"] > 0


def test_fixed_subset_stratifies_real_validation_without_changing_split():
    from tdnirs_posterior.config import load_config
    from tdnirs_posterior.training import _manifest
    from tdnirs_posterior.data.dataset import H5TPSFDataset
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/research.yaml")
    data = H5TPSFDataset(_manifest(cfg), "validation", cfg["simulation"], cfg["data"]["sds_mm"], 2026)
    try:
        indices = _posterior_validation_indices(data, 64, 2026, FOCUS)
        assert indices == _posterior_validation_indices(data, 64, 2026, FOCUS)
        assert len(indices) == len(set(indices)) == 192
        truth, layers = dataset_truth(data)
        core = core_mask(truth[indices], layers[indices], FOCUS)
        for n in (2, 3):
            assert int(core[layers[indices] == n].sum()) == 45
        assert all(data.records[i].split == "validation" for i in indices)
    finally:
        data.close()


def test_timestamps_are_timezone_aware():
    assert datetime.fromisoformat(timestamp()).utcoffset() is not None


def test_flow_loss_weights_cases_without_weighting_known_or_inactive_parameters():
    from tdnirs_posterior.model.posterior import MaskedPosteriorFlow
    flow = MaskedPosteriorFlow(signal_dim=8, d_model=8, n_heads=2, n_layers=1, dropout=0.)
    theta = torch.tensor([[1.] * 8, [2.] * 8])
    active = torch.tensor([[True, True, False, False, False, False, False, False], [True] * 8])
    known = torch.zeros_like(active)
    known[0, 0] = True
    weights = torch.tensor([1.4, .6])
    with patch.object(flow, "forward", return_value=torch.zeros_like(theta)):
        value = flow.flow_matching_loss(theta, torch.zeros(2, 8), torch.tensor([1, 3]),
            active, known, torch.zeros_like(theta), noise=torch.zeros_like(theta),
            time=torch.zeros(2), case_weights=weights)
    assert value.item() == pytest.approx((1.4 * 1 + .6 * 4) / 2)
