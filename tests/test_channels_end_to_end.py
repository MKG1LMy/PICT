"""Small forward/backward, checkpoint and report-generation checks."""
import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from tdnirs_posterior.config import load_config
from tdnirs_posterior.training import train
from tdnirs_posterior.testing import run_test
from tdnirs_posterior.baseline_training import train_baseline
from tdnirs_posterior.baseline_testing import run_baseline_test


@pytest.mark.parametrize("variant", ["explicit_irf", "dtof_only", "cnn", "ann"])
def test_small_real_training_and_fifteen_subset_reports(tmp_path, variant):
    config = load_config("configs/smoke.yaml")
    config["model"] = dict(variant=variant if variant in {"explicit_irf", "dtof_only"} else "explicit_irf",
        d_model=16,n_heads=2,waveform_latents=2,waveform_layers=1,signal_layers=1,posterior_layers=1,frequency_bins=4,dropout=0)
    config["training"].update(epochs=1,micro_batch_size=2,effective_batch_size=2,num_workers=0,amp=False,
                              max_train_samples=3,max_val_samples=3,warmup_epochs=0,output_dir=str(tmp_path/variant))
    config["training"]["posterior_validation"]["enabled"] = False
    config["training"]["baseline_early_stopping"] = {"start_after_warmup":False,"patience_epochs":2}
    config["evaluation"].update(auto_run=False,device="cpu",num_workers=0,amp=False,max_test_samples=2,
                               posterior_samples=2,solver_steps=1,bootstrap_replicates=2,batch_size=2)
    config["evaluation"]["multimodality"]["enabled"] = False
    for key in ("irf_interventions", "prior_interventions"):
        config["evaluation"][key]["enabled"] = False
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with patch("torch.cuda.is_available", return_value=False):
            if variant in {"cnn", "ann"}:
                config["baseline"] = dict(architecture=variant,cnn_channels=[8],cnn_pool_bins=2,
                                          cnn_head_widths=[8],ann_widths=[8],dropout=0,loss_beta=.1)
                checkpoint = train_baseline(config, auto_test=False)
                page = run_baseline_test(config,checkpoint)
            else:
                checkpoint = train(config)
                page = run_test(config,checkpoint)
        report = json.loads((page.parent/"report.json").read_text(encoding="utf-8"))
        assert len(report["channel_subsets"]["subsets"]) == 15
        assert "channel-subsets" in page.read_text(encoding="utf-8")
        assert (page.parent/"channel_subset_metrics.csv").is_file()
        name = "predictions.h5" if variant in {"cnn", "ann"} else "posterior.h5"
        with h5py.File(page.parent/name) as handle:
            assert len(handle["channel_subsets"]) == 15
    finally:
        torch.set_num_threads(old_threads)
