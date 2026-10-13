from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .posterior import MaskedPosteriorFlow
from .waveform import DTOFOnlySignalEncoder, SignalEncoder


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "explicit_irf"
    d_model: int = 256
    n_heads: int = 8
    waveform_latents: int = 128
    waveform_layers: int = 2
    signal_layers: int = 6
    posterior_layers: int = 8
    frequency_bins: int = 128
    dropout: float = 0.1

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "ModelConfig":
        return cls(**{field: values[field] for field in cls.__dataclass_fields__ if field in values})


class ICAPModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.variant not in {"explicit_irf", "dtof_only"}:
            raise ValueError(
                "model.variant must be 'explicit_irf' or 'dtof_only'"
            )
        encoder_type = (
            SignalEncoder
            if config.variant == "explicit_irf"
            else DTOFOnlySignalEncoder
        )
        self.signal_encoder = encoder_type(
            d_model=config.d_model,
            n_heads=config.n_heads,
            waveform_latents=config.waveform_latents,
            waveform_layers=config.waveform_layers,
            signal_layers=config.signal_layers,
            frequency_bins=config.frequency_bins,
            dropout=config.dropout,
        )
        self.posterior = MaskedPosteriorFlow(
            signal_dim=config.d_model,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.posterior_layers,
            dropout=config.dropout,
        )

    def encode_view(self, view: dict[str, torch.Tensor], sds_mm: torch.Tensor) -> torch.Tensor:
        if self.config.variant == "explicit_irf":
            return self.signal_encoder(view["dtof"], view["irf"], sds_mm, view.get("channel_mask"))
        return self.signal_encoder(view["dtof"], sds_mm, view.get("channel_mask"))

    def training_loss(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        sds_mm: torch.Tensor,
        consistency_weight: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        signal_a = self.encode_view(batch["view_a"], sds_mm)  # type: ignore[arg-type]
        signal_b = self.encode_view(batch["view_b"], sds_mm)  # type: ignore[arg-type]
        theta = batch["theta"]  # type: ignore[assignment]
        layer_count = batch["layer_count"]  # type: ignore[assignment]
        active_mask = batch["active_mask"]  # type: ignore[assignment]
        prior_mask = batch["known_mask"]  # type: ignore[assignment]
        prior_values = batch["known_values"]  # type: ignore[assignment]
        no_prior_mask = torch.zeros_like(prior_mask)
        no_prior_values = torch.zeros_like(prior_values)
        shared_noise = torch.randn_like(theta)
        shared_time = torch.rand(
            theta.shape[0], device=theta.device, dtype=theta.dtype
        )

        # Two measurement views crossed with two prior states:
        # A = view A / no prior, B = view B / no prior,
        # C = view A / non-empty prior, D = view B / the same prior.
        flow_a = self.posterior.flow_matching_loss(
            theta,
            signal_a,
            layer_count,
            active_mask,
            no_prior_mask,
            no_prior_values,
            noise=shared_noise,
            time=shared_time,
            case_weights=batch.get("loss_weight"),
        )
        flow_b = self.posterior.flow_matching_loss(
            theta,
            signal_b,
            layer_count,
            active_mask,
            no_prior_mask,
            no_prior_values,
            noise=shared_noise,
            time=shared_time,
            case_weights=batch.get("loss_weight"),
        )
        flow_c = self.posterior.flow_matching_loss(
            theta,
            signal_a,
            layer_count,
            active_mask,
            prior_mask,
            prior_values,
            noise=shared_noise,
            time=shared_time,
            case_weights=batch.get("loss_weight"),
        )
        flow_d = self.posterior.flow_matching_loss(
            theta,
            signal_b,
            layer_count,
            active_mask,
            prior_mask,
            prior_values,
            noise=shared_noise,
            time=shared_time,
            case_weights=batch.get("loss_weight"),
        )
        no_prior_flow = 0.5 * (flow_a + flow_b)
        prior_flow = 0.5 * (flow_c + flow_d)
        flow = 0.5 * (no_prior_flow + prior_flow)
        per_case_consistency = 1.0 - F.cosine_similarity(signal_a, signal_b, dim=-1)
        if "loss_weight" in batch:
            per_case_consistency = per_case_consistency * batch["loss_weight"]
        consistency = per_case_consistency.mean()
        total = flow + float(consistency_weight) * consistency
        return total, {
            "flow": flow.detach(),
            "flow_no_prior": no_prior_flow.detach(),
            "flow_prior": prior_flow.detach(),
            "flow_a": flow_a.detach(),
            "flow_b": flow_b.detach(),
            "flow_c": flow_c.detach(),
            "flow_d": flow_d.detach(),
            "consistency": consistency.detach(),
        }
