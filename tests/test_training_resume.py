from __future__ import annotations

import tempfile
import unittest
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tdnirs_posterior.config import load_config
from tdnirs_posterior.training import train


def _assert_nested_equal(test: unittest.TestCase, left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        test.assertIsInstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        test.assertIsInstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        test.assertIsInstance(right, dict)
        test.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertIsInstance(right, type(left))
        test.assertEqual(len(left), len(right))
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(test, left_value, right_value)
    else:
        test.assertEqual(left, right)


class TrainingResumeTests(unittest.TestCase):
    def _config(self, output: Path) -> dict:
        project = Path(__file__).resolve().parents[1]
        config = load_config(project / "configs" / "smoke.yaml")
        config["model"] = {
            "d_model": 16,
            "n_heads": 4,
            "waveform_latents": 4,
            "waveform_layers": 1,
            "signal_layers": 1,
            "posterior_layers": 1,
            "frequency_bins": 8,
            "dropout": 0.1,
        }
        config["training"].update(
            {
                "epochs": 2,
                "micro_batch_size": 2,
                "effective_batch_size": 4,
                "num_workers": 0,
                "amp": False,
                "max_train_samples": 4,
                "max_val_samples": 2,
                "output_dir": str(output),
            }
        )
        config["training"]["early_stopping"]["patience_epochs"] = 2
        return config

    def test_resumed_run_matches_uninterrupted_run_exactly(self) -> None:
        project = Path(__file__).resolve().parents[1]
        runs = project / "runs"
        runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runs) as temporary:
            root = Path(temporary)
            continuous_config = self._config(root / "continuous")
            resumed_config = self._config(root / "resumed")
            fingerprint = {"records_sha256": "test", "source_files_sha256": {}}
            with (
                patch("tdnirs_posterior.training.torch.cuda.is_available", return_value=False),
                patch(
                    "tdnirs_posterior.training._dataset_fingerprint",
                    return_value=fingerprint,
                ),
            ):
                train(continuous_config)

                first_invocation = deepcopy(resumed_config)
                first_invocation["training"]["run_epochs"] = 1
                train(first_invocation)
                first_checkpoint = torch.load(
                    root / "resumed" / "last.pt", map_location="cpu", weights_only=False
                )
                self.assertEqual(first_checkpoint["next_epoch"], 1)
                self.assertTrue((root / "resumed" / "best_loss.pt").exists())
                with (root / "resumed" / "metrics.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle:
                    handle.write('{"epoch":')

                second_invocation = deepcopy(resumed_config)
                second_invocation["training"]["run_epochs"] = 1
                train(second_invocation, resume="auto")

                incompatible = deepcopy(resumed_config)
                incompatible["model"]["d_model"] = 32
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    train(incompatible, resume="auto")
                with self.assertRaisesRegex(FileExistsError, "already exist"):
                    train(resumed_config)

            continuous = torch.load(
                root / "continuous" / "last.pt", map_location="cpu", weights_only=False
            )
            resumed = torch.load(
                root / "resumed" / "last.pt", map_location="cpu", weights_only=False
            )
            for key in (
                "model",
                "ema",
                "optimizer",
                "scheduler",
                "scaler",
                "epoch",
                "next_epoch",
                "global_step",
                "best_validation",
                "stale_epochs",
                "skipped_optimizer_steps",
                "rng_state",
            ):
                _assert_nested_equal(self, continuous[key], resumed[key])
            self.assertEqual(
                continuous["metrics_record"]["train_loss"],
                resumed["metrics_record"]["train_loss"],
            )
            self.assertEqual(
                continuous["metrics_record"]["validation_loss"],
                resumed["metrics_record"]["validation_loss"],
            )
            lines = (root / "resumed" / "metrics.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(lines), 2)

    def test_warm_start_from_best_resets_training_state_and_records_provenance(self) -> None:
        project = Path(__file__).resolve().parents[1]
        runs = project / "runs"
        runs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runs) as temporary:
            root = Path(temporary)
            source_config = self._config(root / "source")
            destination_config = self._config(root / "destination")
            destination_config["training"]["epochs"] = 1
            destination_config["training"]["early_stopping"]["patience_epochs"] = 1
            fingerprint = {"records_sha256": "test", "source_files_sha256": {}}
            with (
                patch("tdnirs_posterior.training.torch.cuda.is_available", return_value=False),
                patch(
                    "tdnirs_posterior.training._dataset_fingerprint",
                    return_value=fingerprint,
                ),
            ):
                source = train(source_config)
                destination = train(
                    destination_config,
                    init_checkpoint=source,
                    init_weights="raw",
                )
            source_payload = torch.load(source, map_location="cpu", weights_only=False)
            destination_payload = torch.load(
                destination, map_location="cpu", weights_only=False
            )
            self.assertEqual(destination_payload["global_step"], 1)
            self.assertEqual(
                destination_payload["initialization"]["source_epoch"],
                source_payload["epoch"],
            )
            self.assertEqual(destination_payload["initialization"]["weights"], "raw")
            summary = json.loads(
                (root / "destination" / "training_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["stop_reason"], "max_epochs")
            self.assertEqual(summary["status"], "completed")


if __name__ == "__main__":
    unittest.main()
