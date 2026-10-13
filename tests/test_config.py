from pathlib import Path

import pytest

from tdnirs_posterior.config import load_config


def test_dtof_only_config_inherits_scientific_protocol() -> None:
    root = Path(__file__).resolve().parents[1]
    explicit = load_config(root / "configs" / "research.yaml")
    dtof_only = load_config(root / "configs" / "research_dtof_only.yaml")
    cnn = load_config(root / "configs" / "research_cnn.yaml")
    ann = load_config(root / "configs" / "research_ann.yaml")
    assert explicit["model"]["variant"] == "explicit_irf"
    assert explicit["seeds"] == {
        "data_split": 2026,
        "training": [20736, 78631, 26605],
        "validation": 2026,
        "evaluation": 2026,
    }
    assert dtof_only["model"]["variant"] == "dtof_only"
    assert dtof_only["data"] == explicit["data"]
    assert dtof_only["simulation"] == explicit["simulation"]
    assert dtof_only["training"]["epochs"] == explicit["training"]["epochs"]
    assert explicit["training"]["epochs"] == 10000
    assert explicit["training"]["warmup_epochs"] == 100
    assert explicit["training"]["early_stopping"]["patience_epochs"] == 50
    assert explicit["training"]["posterior_validation"]["patience_evaluations"] == 7
    for derived in (dtof_only, cnn, ann):
        assert derived["seeds"] == explicit["seeds"]
        assert derived["training"]["epochs"] == 10000
        assert derived["training"]["warmup_epochs"] == 100
        assert derived["training"]["lr_schedule"] == explicit["training"]["lr_schedule"]
    for baseline in (cnn, ann):
        assert baseline["training"]["baseline_early_stopping"]["patience_epochs"] == 50
    assert explicit["data"]["time_bins"] == 2500
    assert explicit["data"]["time_bin_ps"] == 4.0
    assert explicit["data"]["time_start_ps"] == 0.0
    smoke = load_config(root / "configs" / "smoke.yaml")
    assert smoke["seeds"]["data_split"] == explicit["seeds"]["data_split"]
    assert smoke["seeds"]["validation"] == smoke["seeds"]["evaluation"] == 2026
    assert smoke["seeds"]["training"] == explicit["seeds"]["training"][0]
    assert dtof_only["evaluation"]["irf_interventions"]["enabled"] is False
    assert dtof_only["training"]["output_dir"] != explicit["training"]["output_dir"]
    assert explicit["simulation"]["preprocessing"]["time_alignment_enabled"] is False
    assert explicit["simulation"]["irf"]["channel_model"]["mode_probabilities"] == {
        "all_shared": 0.20,
        "four_similar": 0.35,
        "grouped_2_2": 0.15,
        "grouped_3_1": 0.10,
        "partial_outlier": 0.15,
        "heterogeneous": 0.05,
    }
    assert explicit["evaluation"]["irf_interventions"]["modes"] == [
        "shuffled_irf",
        "fixed_irf",
        "zero_irf",
        "channel_permuted_irf",
    ]
    assert cnn["simulation"] == explicit["simulation"]
    assert ann["simulation"] == explicit["simulation"]


def test_configuration_rejects_a_noncanonical_time_grid(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "configs" / "research.yaml").read_text(encoding="utf-8")
    invalid = tmp_path / "invalid-grid.yaml"
    invalid.write_text(
        text.replace("time_bin_ps: 4.0", "time_bin_ps: 8.0", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires data.time_bins=2500"):
        load_config(invalid)
