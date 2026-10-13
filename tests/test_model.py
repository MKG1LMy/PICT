import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch not installed")
class ModelTests(unittest.TestCase):
    def test_resolution_profiler_uses_four_paired_irf_channels(self) -> None:
        import torch

        from tdnirs_posterior.cli.profile_resolution import _profile

        config = {
            "model": {
                "d_model": 16,
                "n_heads": 4,
                "waveform_latents": 4,
                "waveform_layers": 1,
                "signal_layers": 1,
                "posterior_layers": 1,
                "frequency_bins": 4,
                "dropout": 0.0,
            },
            "training": {"amp": False, "consistency_weight": 0.1},
        }
        result = _profile(config, length=32, batch_size=1, device=torch.device("cpu"))
        self.assertEqual(result["time_bins"], 32)
        self.assertEqual(result["batch_size"], 1)
        self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))

    def test_forward_and_flow_loss(self) -> None:
        import torch

        from tdnirs_posterior.model import ICAPModel, ModelConfig

        model = ICAPModel(
            ModelConfig(
                d_model=32,
                n_heads=4,
                waveform_latents=8,
                waveform_layers=1,
                signal_layers=1,
                posterior_layers=1,
                frequency_bins=8,
            )
        )
        batch = 2
        length = 128
        dtof = torch.rand(batch, 4, 2, length)
        irf = torch.rand(batch, 4, 2, length)
        signal = model.signal_encoder(dtof, irf, torch.tensor([10.0, 20.0, 30.0, 40.0]))
        theta = torch.zeros(batch, 8)
        active = torch.tensor(
            [[False, False, True, False, False, True, False, False], [True] * 8]
        )
        known = torch.zeros_like(active)
        known[1, 0] = True
        known_values = torch.zeros_like(theta)
        loss = model.posterior.flow_matching_loss(
            theta,
            signal,
            torch.tensor([1, 3]),
            active,
            known,
            known_values,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_explicit_irf_encoder_rejects_single_model_channel(self) -> None:
        import torch

        from tdnirs_posterior.model import ICAPModel, ModelConfig

        model = ICAPModel(
            ModelConfig(
                d_model=32,
                n_heads=4,
                waveform_latents=8,
                waveform_layers=1,
                signal_layers=1,
                posterior_layers=1,
                frequency_bins=8,
            )
        )
        with self.assertRaisesRegex(ValueError, r"\[B,4,2,T\]"):
            model.signal_encoder(
                torch.rand(2, 4, 2, 128),
                torch.rand(2, 1, 2, 128),
                torch.tensor([10.0, 20.0, 30.0, 40.0]),
            )

    def test_frequency_features_use_same_index_irf_only(self) -> None:
        import torch

        from tdnirs_posterior.model.waveform import SignalEncoder

        encoder = SignalEncoder(32, 4, 8, 1, 1, 8, 0.0).eval()
        dtof = torch.rand(2, 4, 2, 128)
        irf = torch.rand(2, 4, 2, 128)
        baseline = encoder._frequency_features(dtof, irf)
        changed = irf.clone()
        changed[:, 0] = torch.rand_like(changed[:, 0])
        updated = encoder._frequency_features(dtof, changed)
        self.assertFalse(torch.allclose(baseline[:, 0], updated[:, 0]))
        torch.testing.assert_close(baseline[:, 1:], updated[:, 1:])

    def test_posterior_samples_hard_clamp_known_values(self) -> None:
        import torch

        from tdnirs_posterior.model.posterior import MaskedPosteriorFlow

        flow = MaskedPosteriorFlow(
            signal_dim=16,
            d_model=16,
            n_heads=4,
            n_layers=1,
            dropout=0.0,
        ).eval()
        active = torch.tensor([[False, False, True, False, False, True, False, False]])
        known_mask = torch.zeros_like(active)
        known_mask[0, 2] = True
        known_values = torch.zeros(1, 8)
        known_values[0, 2] = 0.375
        samples = flow.sample(
            signal=torch.zeros(1, 16),
            layer_count=torch.tensor([1]),
            active_mask=active,
            known_mask=known_mask,
            known_values=known_values,
            sample_count=5,
            steps=2,
        )
        torch.testing.assert_close(samples[0, :, 2], torch.full((5,), 0.375))
        torch.testing.assert_close(samples[0, :, ~active[0]], torch.zeros(5, 6))

    def test_flow_loss_weights_cases_not_parameter_slots(self) -> None:
        import torch

        from tdnirs_posterior.model.posterior import MaskedPosteriorFlow

        flow = MaskedPosteriorFlow(4, 16, 4, 1, 0.0)
        for parameter in flow.parameters():
            torch.nn.init.zeros_(parameter)
        theta = torch.ones(2, 8)
        theta[0] = 2.0
        active = torch.zeros(2, 8, dtype=torch.bool)
        active[0, :2] = True
        active[1, :] = True
        loss = flow.flow_matching_loss(
            theta, torch.zeros(2, 4), torch.tensor([1, 3]), active,
            torch.zeros_like(active), torch.zeros_like(theta),
            noise=torch.zeros_like(theta), time=torch.ones(2),
        )
        self.assertAlmostEqual(float(loss.detach()), 2.5, places=6)

    def test_dtof_only_model_has_no_irf_modules_or_input_dependency(self) -> None:
        import torch

        from tdnirs_posterior.model import ICAPModel, ModelConfig

        model = ICAPModel(
            ModelConfig(
                variant="dtof_only",
                d_model=32,
                n_heads=4,
                waveform_latents=8,
                waveform_layers=1,
                signal_layers=1,
                posterior_layers=1,
                frequency_bins=8,
            )
        ).eval()
        names = set(dict(model.named_modules()))
        self.assertFalse(any("irf_encoder" in name for name in names))
        self.assertFalse(any("irf_condition" in name for name in names))
        dtof = torch.rand(2, 4, 2, 128)
        first = model.encode_view(
            {"dtof": dtof, "irf": torch.zeros(2, 2, 128)},
            torch.tensor([10.0, 20.0, 30.0, 40.0]),
        )
        second = model.encode_view(
            {"dtof": dtof, "irf": torch.rand(2, 4, 2, 128)},
            torch.tensor([10.0, 20.0, 30.0, 40.0]),
        )
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
