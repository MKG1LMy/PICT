from __future__ import annotations

import math

import torch
from torch import nn
from ..channels import tensor_mask
from .stable_attention import StableCrossAttention


def _time_features(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    position = torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)
    frequencies = torch.tensor(
        [1.0, 2.0, 4.0, 8.0], device=device, dtype=dtype
    )
    phase = 2.0 * math.pi * position[:, None] * frequencies[None, :]
    return torch.cat([position[:, None], torch.sin(phase), torch.cos(phase)], dim=-1)


class WaveformPerceiver(nn.Module):
    """Linear-complexity latent reader that preserves the full input time grid."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_latents: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.point_projection = nn.Linear(11, d_model)
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.read = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.refine = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        # A persistent marker verifies the normalized-output protocol during
        # strict checkpoint loading.
        self.register_buffer("normalized_output_protocol", torch.tensor(1))

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Encode [B,C,2,T] waveforms into [B,C,M,D] latent tokens."""
        if waveforms.ndim != 4 or waveforms.shape[2] != 2:
            raise ValueError(f"Expected [B,C,2,T], got {tuple(waveforms.shape)}")
        batch, channels, _, length = waveforms.shape
        values = waveforms.permute(0, 1, 3, 2).reshape(batch * channels, length, 2)
        time = _time_features(length, waveforms.device, waveforms.dtype)
        time = time.unsqueeze(0).expand(batch * channels, -1, -1)
        points = self.point_projection(torch.cat([values, time], dim=-1))
        latent = self.latents.unsqueeze(0).expand(batch * channels, -1, -1)
        read, _ = self.read(latent, points, points, need_weights=False)
        latent = self.refine(self.norm(latent + read))
        latent = torch.nn.functional.layer_norm(latent, (latent.shape[-1],))
        return latent.reshape(batch, channels, latent.shape[1], latent.shape[2])


class SignalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        waveform_latents: int,
        waveform_layers: int,
        signal_layers: int,
        frequency_bins: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.frequency_bins = int(frequency_bins)
        self.dtof_encoder = WaveformPerceiver(
            d_model, n_heads, waveform_latents, waveform_layers, dropout
        )
        self.irf_encoder = WaveformPerceiver(
            d_model, n_heads, waveform_latents, waveform_layers, dropout
        )
        self.irf_condition = StableCrossAttention(d_model, n_heads, dropout=dropout)
        self.condition_norm = nn.LayerNorm(d_model)
        self.frequency_projection = nn.Sequential(
            nn.Linear(3 * self.frequency_bins, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.sds_projection = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sds_fusion = nn.TransformerEncoder(
            layer, num_layers=signal_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.availability = nn.Linear(4, d_model, bias=False)

    def _frequency_features(self, dtof: torch.Tensor, irf: torch.Tensor) -> torch.Tensor:
        dtof_linear = dtof[:, :, 0, :]
        irf_linear = irf[:, :, 0, :]
        y = torch.fft.rfft(dtof_linear.float(), dim=-1)
        r = torch.fft.rfft(irf_linear.float(), dim=-1)
        max_bins = min(self.frequency_bins, y.shape[-1] - 1)
        y = y[..., 1 : max_bins + 1]
        r = r[..., 1 : max_bins + 1]
        denominator = r.abs().square() + 1e-6 * r.abs().square().mean(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        transfer = y * r.conj() / denominator
        bounded = transfer / (1.0 + transfer.abs())
        features = torch.cat(
            [torch.log1p(transfer.abs()), bounded.real, bounded.imag], dim=-1
        )
        if max_bins < self.frequency_bins:
            features = torch.nn.functional.pad(
                features, (0, 3 * (self.frequency_bins - max_bins))
            )
        return self.frequency_projection(features.to(dtof.dtype))

    def forward(
        self, dtof: torch.Tensor, irf: torch.Tensor, sds_mm: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            dtof.ndim != 4
            or irf.ndim != 4
            or dtof.shape[0] != irf.shape[0]
            or dtof.shape[1] != 4
            or irf.shape[1] != 4
            or dtof.shape[2] != 2
            or irf.shape[2] != 2
            or dtof.shape[-1] != irf.shape[-1]
        ):
            raise ValueError(
                "DTOF and IRF must both have matching shape [B,4,2,T]"
            )
        if sds_mm.shape[-1] != 4:
            raise ValueError("sds_mm must contain four entries in channel order")
        mask = tensor_mask(dtof, channel_mask)
        if not torch.isfinite(irf[mask]).all():
            raise ValueError("Valid IRF contains NaN/Inf")
        valid_dtof, valid_irf = dtof[mask][:, None], irf[mask][:, None]
        dtof_flat = self.dtof_encoder(valid_dtof)[:, 0]
        irf_flat = self.irf_encoder(valid_irf)[:, 0]
        batch, channels = mask.shape
        conditioned, _ = self.irf_condition(
            dtof_flat, irf_flat, irf_flat, need_weights=False
        )
        conditioned = self.condition_norm(dtof_flat + conditioned)
        valid_tokens = conditioned.mean(dim=1) + self._frequency_features(valid_dtof, valid_irf)[:, 0]
        channel_tokens = valid_tokens.new_zeros(batch, channels, valid_tokens.shape[-1])
        channel_tokens[mask] = valid_tokens
        if sds_mm.ndim == 1:
            sds_mm = sds_mm[None, :].expand(batch, -1)
        channel_tokens = channel_tokens + self.sds_projection(
            (sds_mm / 40.0).unsqueeze(-1).to(channel_tokens.dtype)
        )
        fused = self.sds_fusion(channel_tokens, src_key_padding_mask=~mask)
        pooled = fused.masked_fill(~mask[..., None], 0).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.output_norm(pooled + self.availability(mask.to(pooled.dtype)))


class DTOFOnlySignalEncoder(nn.Module):
    """Signal encoder for the strict DTOF-only architecture ablation.

    The measured IRF is used by preprocessing only when optional algorithmic
    alignment is enabled. No IRF tensor or IRF-derived feature enters this
    module under either preprocessing protocol.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        waveform_latents: int,
        waveform_layers: int,
        signal_layers: int,
        frequency_bins: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.frequency_bins = int(frequency_bins)
        self.dtof_encoder = WaveformPerceiver(
            d_model, n_heads, waveform_latents, waveform_layers, dropout
        )
        self.frequency_projection = nn.Sequential(
            nn.Linear(3 * self.frequency_bins, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.sds_projection = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sds_fusion = nn.TransformerEncoder(
            layer, num_layers=signal_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.availability = nn.Linear(4, d_model, bias=False)

    def _frequency_features(self, dtof: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(dtof[:, :, 0, :].float(), dim=-1)
        max_bins = min(self.frequency_bins, spectrum.shape[-1] - 1)
        spectrum = spectrum[..., 1 : max_bins + 1]
        bounded = spectrum / (1.0 + spectrum.abs())
        features = torch.cat(
            [torch.log1p(spectrum.abs()), bounded.real, bounded.imag], dim=-1
        )
        if max_bins < self.frequency_bins:
            features = torch.nn.functional.pad(
                features, (0, 3 * (self.frequency_bins - max_bins))
            )
        return self.frequency_projection(features.to(dtof.dtype))

    def forward(self, dtof: torch.Tensor, sds_mm: torch.Tensor, channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        mask = tensor_mask(dtof, channel_mask)
        valid = dtof[mask][:, None]
        valid_tokens = self.dtof_encoder(valid).mean(dim=2)[:, 0] + self._frequency_features(valid)[:, 0]
        channel_tokens = valid_tokens.new_zeros(dtof.shape[0], 4, valid_tokens.shape[-1])
        channel_tokens[mask] = valid_tokens
        batch = dtof.shape[0]
        if sds_mm.ndim == 1:
            sds_mm = sds_mm[None, :].expand(batch, -1)
        channel_tokens = channel_tokens + self.sds_projection(
            (sds_mm / 40.0).unsqueeze(-1).to(channel_tokens.dtype)
        )
        fused = self.sds_fusion(channel_tokens, src_key_padding_mask=~mask)
        pooled = fused.masked_fill(~mask[..., None], 0).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        return self.output_norm(pooled + self.availability(mask.to(pooled.dtype)))
