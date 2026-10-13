from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from ..channels import tensor_mask


@dataclass(frozen=True)
class BaselineConfig:
    architecture: str
    dropout: float = 0.1
    cnn_channels: tuple[int, ...] = (32, 64, 128, 256)
    cnn_pool_bins: int = 16
    cnn_head_widths: tuple[int, ...] = (512, 256)
    ann_widths: tuple[int, ...] = (384, 256, 128)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "BaselineConfig":
        def integer_tuple(name: str, default: Sequence[int]) -> tuple[int, ...]:
            return tuple(int(value) for value in values.get(name, default))

        architecture = str(values["architecture"])
        if architecture not in {"cnn", "ann"}:
            raise ValueError("baseline.architecture must be 'cnn' or 'ann'")
        return cls(
            architecture=architecture,
            dropout=float(values.get("dropout", 0.1)),
            cnn_channels=integer_tuple("cnn_channels", (32, 64, 128, 256)),
            cnn_pool_bins=int(values.get("cnn_pool_bins", 16)),
            cnn_head_widths=integer_tuple("cnn_head_widths", (512, 256)),
            ann_widths=integer_tuple("ann_widths", (384, 256, 128)),
        )


def _layer_one_hot(layer_count: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if layer_count.ndim != 1:
        raise ValueError("layer_count must have shape [batch]")
    if torch.any((layer_count < 1) | (layer_count > 3)):
        raise ValueError("layer_count must contain only 1, 2, or 3")
    return F.one_hot(layer_count.long() - 1, num_classes=3).to(dtype=dtype)


class CNNRegressor(nn.Module):
    """Compact conventional 1-D CNN over the four DTOF channels.

    The linear and log-shape representations are folded into eight input
    channels. The measured IRF and optional-prior tensors are intentionally not
    accepted by this interface.
    """

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        if not config.cnn_channels:
            raise ValueError("baseline.cnn_channels cannot be empty")
        blocks: list[nn.Module] = []
        input_channels = 8
        kernels = (15, 9, 7, 5)
        for index, output_channels in enumerate(config.cnn_channels):
            kernel = kernels[min(index, len(kernels) - 1)]
            blocks.extend(
                [
                    nn.Conv1d(
                        input_channels,
                        output_channels,
                        kernel_size=kernel,
                        stride=2,
                        padding=kernel // 2,
                    ),
                    nn.BatchNorm1d(output_channels),
                    nn.GELU(),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(config.cnn_pool_bins)
        widths = (
            input_channels * config.cnn_pool_bins + 7,
            *config.cnn_head_widths,
            8,
        )
        head: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(widths, widths[1:])):
            head.append(nn.Linear(left, right))
            if index < len(widths) - 2:
                head.extend([nn.LayerNorm(right), nn.GELU(), nn.Dropout(config.dropout)])
        self.head = nn.Sequential(*head)

    def forward(self, dtof: torch.Tensor, layer_count: torch.Tensor, channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        if dtof.ndim != 4 or dtof.shape[1] != 4 or dtof.shape[2] != 2:
            raise ValueError(f"Expected DTOF [B,4,2,T], got {tuple(dtof.shape)}")
        batch, sds, representations, length = dtof.shape
        mask = tensor_mask(dtof, channel_mask)
        dtof = dtof.masked_fill(~mask[..., None, None], 0)
        signal = dtof.reshape(batch, sds * representations, length)
        encoded = self.pool(self.features(signal)).flatten(1)
        return self.head(torch.cat([encoded, _layer_one_hot(layer_count, encoded.dtype), mask.to(encoded.dtype)], dim=1))


class ANNRegressor(nn.Module):
    """Fully connected baseline that retains every DTOF time bin."""

    def __init__(self, config: BaselineConfig, time_bins: int) -> None:
        super().__init__()
        if not config.ann_widths:
            raise ValueError("baseline.ann_widths cannot be empty")
        widths = (4 * 2 * int(time_bins) + 7, *config.ann_widths, 8)
        layers: list[nn.Module] = []
        for index, (left, right) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(left, right))
            if index < len(widths) - 2:
                layers.extend([nn.LayerNorm(right), nn.GELU(), nn.Dropout(config.dropout)])
        self.network = nn.Sequential(*layers)
        self.time_bins = int(time_bins)

    def forward(self, dtof: torch.Tensor, layer_count: torch.Tensor, channel_mask: torch.Tensor | None = None) -> torch.Tensor:
        expected = (4, 2, self.time_bins)
        if dtof.ndim != 4 or tuple(dtof.shape[1:]) != expected:
            raise ValueError(f"Expected DTOF [B,{expected[0]},{expected[1]},{expected[2]}], got {tuple(dtof.shape)}")
        mask = tensor_mask(dtof, channel_mask)
        flattened = dtof.masked_fill(~mask[..., None, None], 0).flatten(1)
        return self.network(
            torch.cat([flattened, _layer_one_hot(layer_count, flattened.dtype), mask.to(flattened.dtype)], dim=1)
        )


def build_baseline(config: BaselineConfig, time_bins: int) -> nn.Module:
    if config.architecture == "cnn":
        return CNNRegressor(config)
    return ANNRegressor(config, time_bins)


def estimate_baseline_macs(config: BaselineConfig, time_bins: int) -> int:
    """Approximate multiply-accumulates for one forward pass, excluding norms."""

    if config.architecture == "ann":
        widths = (4 * 2 * int(time_bins) + 7, *config.ann_widths, 8)
        return int(sum(left * right for left, right in zip(widths, widths[1:])))
    length = int(time_bins)
    input_channels = 8
    kernels = (15, 9, 7, 5)
    macs = 0
    for index, output_channels in enumerate(config.cnn_channels):
        kernel = kernels[min(index, len(kernels) - 1)]
        length = (length + 1) // 2
        macs += length * output_channels * input_channels * kernel
        input_channels = output_channels
    widths = (input_channels * config.cnn_pool_bins + 7, *config.cnn_head_widths, 8)
    macs += sum(left * right for left, right in zip(widths, widths[1:]))
    return int(macs)


def masked_smooth_l1(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    active_mask: torch.Tensor,
    beta: float,
    case_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.shape != truth.shape or active_mask.shape != truth.shape:
        raise ValueError("prediction, truth, and active_mask must have identical shapes")
    per_parameter = F.smooth_l1_loss(prediction, truth, reduction="none", beta=float(beta))
    active = active_mask.to(per_parameter.dtype)
    per_case = (per_parameter * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
    if case_weights is not None:
        per_case = per_case * case_weights
    return per_case.mean()
