from __future__ import annotations

import math

import torch
from torch import nn


def _diffusion_time_features(t: torch.Tensor, width: int = 16) -> torch.Tensor:
    frequencies = torch.exp(
        torch.linspace(math.log(1.0), math.log(1000.0), width // 2, device=t.device)
    )
    phase = 2.0 * math.pi * t[:, None] * frequencies[None, :]
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class MaskedPosteriorFlow(nn.Module):
    def __init__(
        self,
        signal_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        parameter_count: int = 8,
    ) -> None:
        super().__init__()
        self.parameter_count = parameter_count
        self.parameter_embedding = nn.Embedding(parameter_count, d_model)
        self.layer_embedding = nn.Embedding(4, d_model)
        self.state_projection = nn.Linear(4, d_model)
        self.time_projection = nn.Linear(16, d_model)
        self.signal_projection = nn.Linear(signal_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.velocity = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        signal: torch.Tensor,
        layer_count: torch.Tensor,
        active_mask: torch.Tensor,
        known_mask: torch.Tensor,
        known_values: torch.Tensor,
    ) -> torch.Tensor:
        batch = state.shape[0]
        ids = torch.arange(self.parameter_count, device=state.device)
        token_state = torch.stack(
            [
                state,
                known_values,
                active_mask.to(state.dtype),
                known_mask.to(state.dtype),
            ],
            dim=-1,
        )
        tokens = self.state_projection(token_state)
        tokens = tokens + self.parameter_embedding(ids)[None, :, :]
        tokens = tokens + self.layer_embedding(layer_count)[:, None, :]
        tokens = tokens + self.signal_projection(signal)[:, None, :]
        tokens = tokens + self.time_projection(_diffusion_time_features(time))[:, None, :]
        encoded = self.transformer(tokens, src_key_padding_mask=~active_mask.bool())
        velocity = self.velocity(encoded).squeeze(-1)
        return velocity * active_mask.to(velocity.dtype) * (~known_mask).to(velocity.dtype)

    def flow_matching_loss(
        self,
        theta: torch.Tensor,
        signal: torch.Tensor,
        layer_count: torch.Tensor,
        active_mask: torch.Tensor,
        known_mask: torch.Tensor,
        known_values: torch.Tensor,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        case_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(theta)
        if time is None:
            time = torch.rand(theta.shape[0], device=theta.device, dtype=theta.dtype)
        unknown = active_mask.bool() & ~known_mask.bool()
        state = (1.0 - time[:, None]) * noise + time[:, None] * theta
        state = torch.where(known_mask.bool(), known_values, state)
        state = torch.where(active_mask.bool(), state, torch.zeros_like(state))
        target = theta - noise
        predicted = self(
            state, time, signal, layer_count, active_mask, known_mask, known_values
        )
        squared = (predicted - target).square() * unknown.to(theta.dtype)
        unknown_per_case = unknown.sum(dim=1)
        if torch.any(unknown_per_case == 0):
            raise ValueError("Every training case must retain at least one unknown parameter")
        # Normalize each phantom by its unknown-parameter count before applying
        # explicit case weights. N=3 must not receive a larger implicit weight
        # solely because it has 8 slots.
        per_case = squared.sum(dim=1) / unknown_per_case
        if case_weights is not None:
            per_case = per_case * case_weights
        return per_case.mean()

    @torch.no_grad()
    def sample(
        self,
        signal: torch.Tensor,
        layer_count: torch.Tensor,
        active_mask: torch.Tensor,
        known_mask: torch.Tensor,
        known_values: torch.Tensor,
        sample_count: int,
        steps: int,
    ) -> torch.Tensor:
        batch = signal.shape[0]
        repeat = int(sample_count)
        signal = signal.repeat_interleave(repeat, dim=0)
        layer_count = layer_count.repeat_interleave(repeat, dim=0)
        active_mask = active_mask.repeat_interleave(repeat, dim=0)
        known_mask = known_mask.repeat_interleave(repeat, dim=0)
        known_values = known_values.repeat_interleave(repeat, dim=0)
        state = torch.randn(
            batch * repeat,
            self.parameter_count,
            device=signal.device,
            dtype=signal.dtype,
        )
        state = torch.where(known_mask.bool(), known_values, state)
        state = torch.where(active_mask.bool(), state, torch.zeros_like(state))
        dt = 1.0 / max(int(steps), 1)
        for step in range(max(int(steps), 1)):
            t0 = torch.full(
                (state.shape[0],), step * dt, device=state.device, dtype=state.dtype
            )
            velocity0 = self(
                state,
                t0,
                signal,
                layer_count,
                active_mask,
                known_mask,
                known_values,
            )
            proposal = state + dt * velocity0
            t1 = torch.full_like(t0, min((step + 1) * dt, 1.0))
            velocity1 = self(
                proposal,
                t1,
                signal,
                layer_count,
                active_mask,
                known_mask,
                known_values,
            )
            state = state + 0.5 * dt * (velocity0 + velocity1)
            state = torch.where(known_mask.bool(), known_values, state)
            state = torch.where(active_mask.bool(), state, torch.zeros_like(state))
        return state.reshape(batch, repeat, self.parameter_count)
