import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from tdnirs_posterior.parameters import ParameterSchema
from tdnirs_posterior.config import load_config
from tdnirs_posterior.training import (
    ExponentialMovingAverage,
    _epoch_progress_line,
    _materialize_fixed_dataset,
    _posterior_group_regression_check,
    _run_posterior_validation,
    _run_validation,
    train,
)


class _FlowByFirstValue:
    def flow_matching_loss(
        self,
        theta,
        signal,
        layer_count,
        active_mask,
        known_mask,
        known_values,
    ):
        return theta[:, 0].mean()


class _ValidationModel:
    def __init__(self):
        self.posterior = _FlowByFirstValue()

    def eval(self):
        return self

    def encode_view(self, view, sds):
        return view["dtof"]


class _PosteriorValidationModel:
    def eval(self):
        return self

    def encode_view(self, view, sds):
        return torch.zeros(
            (view["dtof"].shape[0], 4), dtype=view["dtof"].dtype
        )


def _raw_batch(layers, first_values):
    schema = ParameterSchema()
    active = np.stack([schema.active_mask(layer) for layer in layers])
    theta = np.zeros((len(layers), 8), dtype=np.float32)
    theta[:, 0] = first_values
    zeros = np.zeros_like(theta)
    signal = torch.ones((len(layers), 4), dtype=torch.float32)
    return {
        "theta": torch.from_numpy(theta),
        "theta_physical": torch.from_numpy(theta),
        "active_mask": torch.from_numpy(active),
        "known_mask": torch.from_numpy(zeros.astype(bool)),
        "known_values": torch.from_numpy(zeros),
        "layer_count": torch.tensor(layers, dtype=torch.long),
        "view_a": {"dtof": signal, "irf": signal},
        "view_b": {"dtof": signal, "irf": signal},
    }


class TrainingValidationTests(unittest.TestCase):
    def test_fixed_validation_cache_materializes_once_and_closes_source(self):
        class Source:
            def __init__(self):
                self.calls = []
                self.closed = False

            def __len__(self):
                return 2

            def __getitem__(self, index):
                self.calls.append(index)
                return {
                    "view_a": {"dtof": torch.ones(2 + index)},
                    "sample_id": f"case-{index}",
                }

            def close(self):
                self.closed = True

        source = Source()
        cached, tensor_bytes = _materialize_fixed_dataset(source)
        self.assertEqual(source.calls, [0, 1])
        self.assertTrue(source.closed)
        self.assertIsInstance(cached, tuple)
        self.assertEqual([item["sample_id"] for item in cached], ["case-0", "case-1"])
        self.assertEqual(tensor_bytes, (2 + 3) * torch.tensor(0.0).element_size())

    def test_epoch_progress_line_is_compact_and_keeps_key_metrics(self):
        record = {
            "epoch": 49,
            "train_loss": 0.42135,
            "validation_loss": 0.38216,
            "validation_loss_smoothed": 0.38402,
            "posterior_validation": {
                "status": "completed",
                "score": 0.1284,
                "eligible": True,
                "group_regression_check": {"passed": True},
            },
            "best_posterior_score": 0.1284,
            "stale_epochs": 0,
            "posterior_stale_evaluations": 2,
            "posterior_evaluations": 5,
            "posterior_patience_evaluations": 4,
            "posterior_validation_enabled": True,
            "learning_rate": 0.0001822,
            "epoch_seconds": 18.6,
        }
        line = _epoch_progress_line(record, total_epochs=200, patience=40)
        self.assertEqual(line.count("\n"), 0)
        self.assertIn("Epoch 050/200", line)
        self.assertIn("ema_flow=0.38216", line)
        self.assertIn("ncrps=0.12840(ok)", line)
        self.assertIn("ema_stale=0/40", line)
        self.assertIn("post_stale=2/4(eval=5)", line)
        self.assertIn("early_stop=ema_wait:40", line)
        self.assertNotIn("by_scenario", line)

        record["stale_epochs"] = 41
        line = _epoch_progress_line(record, total_epochs=200, patience=40)
        self.assertIn("ema_stale=41/40", line)
        self.assertIn("early_stop=posterior_wait:2", line)

        record["posterior_validation"].update(
            {
                "eligible": False,
                "group_regression_check": {
                    "passed": False,
                    "maximum_allowed_relative": 0.05,
                    "relative_changes_from_selected_best": {
                        "N1/known_musp1": 0.1098,
                        "N3/no_prior": 0.061,
                    },
                },
            }
        )
        line = _epoch_progress_line(record, total_epochs=200, patience=40)
        self.assertIn(
            "ncrps=0.12840(regressed:N1/known_musp1,+11.0%;+1)", line
        )

        record["posterior_validation"].update(
            {
                "group_regression_check": {"passed": True},
                "finite_fraction": 1.0,
                "prior_guardrail": {
                    "passed": False,
                    "maximum_allowed_relative": 0.05,
                    "relative_changes_from_no_prior": {
                        "N2/k1_known_d1": 0.083,
                    },
                },
            }
        )
        line = _epoch_progress_line(record, total_epochs=200, patience=40)
        self.assertIn(
            "ncrps=0.12840(guarded:prior:N2/k1_known_d1,+8.3%)", line
        )

    def test_training_prefers_posterior_checkpoint_when_eligible(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=project / "runs") as temporary:
            output = Path(temporary) / "posterior-selection"
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
                    "epochs": 1,
                    "micro_batch_size": 2,
                    "effective_batch_size": 4,
                    "num_workers": 0,
                    "amp": False,
                    "max_train_samples": 4,
                    "max_val_samples": 2,
                    "output_dir": str(output),
                    "posterior_validation": {
                        "enabled": True,
                        "start_epoch": 1,
                        "interval_epochs": 1,
                        "cases_per_layer": 1,
                        "batch_size": 1,
                        "patience_evaluations": 2,
                    },
                }
            )
            posterior_result = {
                "status": "completed",
                "score": 0.1,
                "eligible": True,
                "group_scores": {
                    "N1/no_prior": 0.1,
                    "N2/no_prior": 0.1,
                    "N3/no_prior": 0.1,
                },
            }
            with (
                patch(
                    "tdnirs_posterior.training._dataset_fingerprint",
                    return_value={"records_sha256": "test", "source_files_sha256": {}},
                ),
                patch(
                    "tdnirs_posterior.training._run_posterior_validation",
                    return_value=posterior_result,
                ),
                patch(
                    "tdnirs_posterior.training.torch.cuda.is_available",
                    return_value=False,
                ),
            ):
                selected = train(config)
            self.assertEqual(selected, output / "best_posterior.pt")
            self.assertTrue((output / "best_loss.pt").exists())
            self.assertTrue((output / "best_posterior.pt").exists())

    def test_group_guardrail_rejects_sacrificing_n1_for_n3(self):
        reference = {
            "no_prior/N1": 0.10,
            "no_prior/N2": 0.20,
            "no_prior/N3": 0.30,
        }
        candidate = {
            "no_prior/N1": 0.12,
            "no_prior/N2": 0.20,
            "no_prior/N3": 0.20,
        }
        result = _posterior_group_regression_check(
            candidate, reference, maximum_relative=0.05
        )
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(
            result["relative_changes_from_selected_best"]["no_prior/N1"], 0.2
        )

    def test_validation_weights_partial_batch_by_terms_not_batch_count(self):
        loader = [
            _raw_batch([3, 3], [1.0, 1.0]),
            _raw_batch([3], [3.0]),
        ]
        result = _run_validation(
            _ValidationModel(),
            loader,
            torch.device("cpu"),
            torch.tensor([10.0]),
            consistency_weight=0.1,
            amp_enabled=False,
            seed=123,
        )
        self.assertAlmostEqual(
            result["by_scenario"]["N3/no_prior"]["by_layer_count"]["3"]["flow"],
            5.0 / 3.0,
        )

    def test_ema_context_restores_raw_and_shadow_weights(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.0)
        ema = ExponentialMovingAverage(model)
        ema.shadow["weight"].fill_(1.0)
        with ema.apply_to(model):
            self.assertEqual(float(model.weight.item()), 1.0)
        self.assertEqual(float(model.weight.item()), 2.0)
        self.assertEqual(float(ema.shadow["weight"].item()), 1.0)

    def test_posterior_validation_uses_normalized_crps_and_guardrails(self):
        schema = ParameterSchema()
        layers = [1, 2, 3]
        active = np.stack([schema.active_mask(layer) for layer in layers])
        truth = np.zeros((3, 8), dtype=np.float32)
        truth[:, 0:2] = 4.0
        truth[:, 2:5] = 0.01
        truth[:, 5:8] = 1.0
        truth[~active] = 0.0
        normalized = schema.normalize(truth)
        raw = {
            "theta": torch.from_numpy(normalized),
            "theta_physical": torch.from_numpy(truth),
            "active_mask": torch.from_numpy(active),
            "known_mask": torch.zeros((3, 8), dtype=torch.bool),
            "known_values": torch.zeros((3, 8)),
            "layer_count": torch.tensor(layers, dtype=torch.long),
            "view_a": {
                "dtof": torch.ones((3, 2)),
                "irf": torch.ones((3, 2)),
            },
        }

        def exact_samples(**kwargs):
            selected_layers = kwargs["layer_count"].cpu().numpy().tolist()
            physical = np.stack([truth[layers.index(int(layer))] for layer in selected_layers])
            samples = np.repeat(physical[:, None, :], kwargs["sample_count"], axis=1)
            diagnostics = [
                {
                    "generated_samples": kwargs["sample_count"],
                    "finite_fraction": 1.0,
                    "outside_training_support_fraction": 0.0,
                    "physically_impossible_fraction": 0.0,
                    "numerical_failure": False,
                }
                for _ in selected_layers
            ]
            return samples, diagnostics

        with patch(
            "tdnirs_posterior.training.sample_native_posterior_batch",
            side_effect=exact_samples,
        ):
            result = _run_posterior_validation(
                _PosteriorValidationModel(),
                [raw],
                torch.device("cpu"),
                torch.tensor([10.0]),
                amp_enabled=False,
                seed=123,
                sample_count=8,
                solver_steps=2,
                max_coverage_error_90=0.15,
            )
        self.assertAlmostEqual(result["score"], 0.0)
        self.assertAlmostEqual(result["finite_fraction"], 1.0)
        self.assertTrue(result["eligible"])


if __name__ == "__main__":
    unittest.main()
