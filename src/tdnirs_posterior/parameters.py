from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


CANONICAL_PARAMETERS = (
    "d1_mm",
    "d2_mm",
    "mua1_mm-1",
    "mua2_mm-1",
    "mua3_mm-1",
    "musp1_mm-1",
    "musp2_mm-1",
    "musp3_mm-1",
)

ACTIVE_BY_LAYER = {
    1: (2, 5),
    2: (0, 2, 3, 5, 6),
    3: tuple(range(8)),
}


@dataclass(frozen=True)
class ParameterSchema:
    thickness_bounds_mm: tuple[float, float] = (0.5, 20.0)
    mua_bounds_mm1: tuple[float, float] = (0.001, 0.05)
    musp_bounds_mm1: tuple[float, float] = (0.2, 3.0)
    n3_total_thickness_max_mm: float = 35.0

    @property
    def names(self) -> tuple[str, ...]:
        return CANONICAL_PARAMETERS

    def active_mask(self, layer_count: int) -> np.ndarray:
        if layer_count not in ACTIVE_BY_LAYER:
            raise ValueError(f"layer_count must be 1, 2, or 3; got {layer_count}")
        result = np.zeros(len(CANONICAL_PARAMETERS), dtype=bool)
        result[list(ACTIVE_BY_LAYER[layer_count])] = True
        return result

    def vector_from_arrays(
        self,
        layer_count: int,
        thickness_mm: Sequence[float],
        mua_mm: Sequence[float],
        musp_mm: Sequence[float],
    ) -> np.ndarray:
        vector = np.zeros(8, dtype=np.float32)
        if layer_count >= 2:
            vector[0] = float(thickness_mm[0])
        if layer_count >= 3:
            vector[1] = float(thickness_mm[1])
        vector[2:5] = np.asarray(mua_mm, dtype=np.float32)[:3]
        vector[5:8] = np.asarray(musp_mm, dtype=np.float32)[:3]
        vector[~self.active_mask(layer_count)] = 0.0
        return vector

    def known_arrays(
        self, layer_count: int, known_values: Mapping[str, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        active = self.active_mask(layer_count)
        known_mask = np.zeros(8, dtype=bool)
        known_vector = np.zeros(8, dtype=np.float32)
        for name, value in known_values.items():
            if name not in CANONICAL_PARAMETERS:
                raise ValueError(f"Unknown parameter name: {name}")
            index = CANONICAL_PARAMETERS.index(name)
            if not active[index]:
                raise ValueError(f"{name} is inactive for N={layer_count}")
            known_mask[index] = True
            known_vector[index] = float(value)
        if known_mask.sum() >= active.sum():
            raise ValueError("At least one active parameter must remain unknown")
        self.validate_physical(layer_count, known_vector, known_mask)
        return known_mask, known_vector

    def random_known_mask(
        self, layer_count: int, rng: np.random.Generator
    ) -> np.ndarray:
        active_indices = np.flatnonzero(self.active_mask(layer_count))
        count = int(rng.integers(0, len(active_indices)))
        result = np.zeros(8, dtype=bool)
        if count:
            result[rng.choice(active_indices, size=count, replace=False)] = True
        return result

    def random_nonempty_known_mask(
        self, layer_count: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Sample a prior mask while keeping at least one parameter unknown."""

        active_indices = np.flatnonzero(self.active_mask(layer_count))
        count = int(rng.integers(1, len(active_indices)))
        result = np.zeros(8, dtype=bool)
        result[rng.choice(active_indices, size=count, replace=False)] = True
        return result

    def _bounds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        low = np.array(
            [
                self.thickness_bounds_mm[0],
                self.thickness_bounds_mm[0],
                *([self.mua_bounds_mm1[0]] * 3),
                *([self.musp_bounds_mm1[0]] * 3),
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                self.thickness_bounds_mm[1],
                self.thickness_bounds_mm[1],
                *([self.mua_bounds_mm1[1]] * 3),
                *([self.musp_bounds_mm1[1]] * 3),
            ],
            dtype=np.float32,
        )
        log_mask = np.array([False, False, True, True, True, True, True, True])
        return low, high, log_mask

    def normalize(self, values: np.ndarray) -> np.ndarray:
        """Map physical values to model coordinates without clipping positives.

        Positive optical values below the training support deliberately map
        below -1 so native posterior extrapolation remains observable.  A
        non-positive optical placeholder maps to the lower support edge; such
        values are diagnosed independently as physically impossible.
        """
        low, high, log_mask = self._bounds()
        values = np.asarray(values, dtype=np.float32)
        work = values.copy()
        optical = work[..., log_mask]
        work[..., log_mask] = np.log(
            np.where(optical > 0.0, optical, low[log_mask])
        )
        low_t = low.copy()
        high_t = high.copy()
        low_t[log_mask] = np.log(low_t[log_mask])
        high_t[log_mask] = np.log(high_t[log_mask])
        return (2.0 * (work - low_t) / (high_t - low_t) - 1.0).astype(np.float32)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        """Map normalized values to physical units without clipping.

        Values outside [-1, 1] are deliberately extrapolated so callers can
        retain and diagnose the model's native posterior support excursions.
        Nonfinite values remain nonfinite and must be reported explicitly.
        """
        low, high, log_mask = self._bounds()
        values = np.asarray(values, dtype=np.float32)
        low_t = low.copy()
        high_t = high.copy()
        low_t[log_mask] = np.log(low_t[log_mask])
        high_t[log_mask] = np.log(high_t[log_mask])
        work = low_t + (values + 1.0) * 0.5 * (high_t - low_t)
        work[..., log_mask] = np.exp(work[..., log_mask])
        return work.astype(np.float32)

    def normalized_valid_mask(
        self, layer_count: int, values: np.ndarray
    ) -> np.ndarray:
        """Return rows whose active values are finite and inside [-1, 1]."""
        values = np.asarray(values)
        if values.shape[-1] != len(CANONICAL_PARAMETERS):
            raise ValueError("Normalized parameter vectors must have length 8")
        active = self.active_mask(layer_count)
        valid_dimensions = np.isfinite(values) & (values >= -1.0) & (values <= 1.0)
        return np.all((~active) | valid_dimensions, axis=-1)

    def validate_physical(
        self,
        layer_count: int,
        values: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        active = self.active_mask(layer_count)
        check = active if mask is None else active & np.asarray(mask, dtype=bool)
        low, high, _ = self._bounds()
        values = np.asarray(values)
        if np.any(values[check] < low[check]) or np.any(values[check] > high[check]):
            raise ValueError("Parameter value is outside the training support")
        if layer_count == 3 and (mask is None or (mask[0] and mask[1])):
            if float(values[0] + values[1]) > self.n3_total_thickness_max_mm:
                raise ValueError("N=3 requires d1_mm + d2_mm <= 35 mm")

    def physical_valid_mask(self, layer_count: int, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        low, high, _ = self._bounds()
        active = self.active_mask(layer_count)
        valid = np.all(
            (~active) | ((values >= low) & (values <= high)), axis=-1
        )
        if layer_count == 3:
            valid &= values[..., 0] + values[..., 1] <= self.n3_total_thickness_max_mm
        return valid
