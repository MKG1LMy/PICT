"""Fixed SDS identities, channel availability and reproducible subset protocols."""
from __future__ import annotations

import hashlib
import itertools
from typing import Any

import numpy as np
import torch

SDS = (10.0, 20.0, 30.0, 40.0)
COUNT_WEIGHTS = {1: 0.15, 2: 0.15, 3: 0.30, 4: 0.40}
PROTOCOL = "paired_sds_subsets_40_30_15_15"


def subsets(count: int | None = None) -> list[tuple[int, ...]]:
    return [v for k in range(1, 5) if count is None or k == count
            for v in itertools.combinations(range(4), k)]


def tensor_mask(dtof: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if dtof.ndim != 4 or dtof.shape[1:3] != (4, 2):
        raise ValueError("DTOF must have shape [B,4,2,T]")
    if mask is None:
        mask = torch.ones(dtof.shape[:2], device=dtof.device, dtype=torch.bool)
    if mask.shape != dtof.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("channel_mask must be boolean [B,4]")
    mask = mask.to(dtof.device)
    if not mask.any(dim=1).all():
        raise ValueError("Every case requires at least one valid channel")
    if not torch.isfinite(dtof[mask]).all():
        raise ValueError("Valid channels contain NaN/Inf")
    return mask


def random_mask(rng: np.random.Generator) -> np.ndarray:
    count = int(rng.choice([1, 2, 3, 4], p=list(COUNT_WEIGHTS.values())))
    result = np.zeros(4, dtype=bool)
    result[rng.choice(4, count, replace=False)] = True
    return result


class ChannelValidationLoader:
    """Reiterate fixed observations with balanced subsets within each tissue N."""
    def __init__(self, loader: Any, count: int):
        self.loader, self.count = loader, count

    def __iter__(self):
        counters: dict[int, int] = {}
        combinations = subsets(self.count)
        for raw in self.loader:
            mask = torch.zeros((len(raw["layer_count"]), 4), dtype=torch.bool)
            for row, layer in enumerate(raw["layer_count"].tolist()):
                ordinal = counters.get(layer, 0)
                mask[row, list(combinations[ordinal % len(combinations)])] = True
                counters[layer] = ordinal + 1
            batch = dict(raw)
            for name in ("view_a", "view_b"):
                if name in raw:
                    batch[name] = {**raw[name], "channel_mask": mask}
            yield batch


def channel_id(indices: tuple[int, ...]) -> str:
    return "sds_" + "_".join(str(int(SDS[i])) for i in indices)
