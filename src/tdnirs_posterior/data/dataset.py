from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ..parameters import ParameterSchema
from ..simulation import simulate_measurement
from .index import DatasetIndex, SampleRecord
from ..channels import random_mask

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Data audit and simulation remain usable without PyTorch.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


def waveform_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    length = values.shape[-1]
    log_shape = np.log1p(values * float(length)).astype(np.float32)
    return np.stack([values, log_shape], axis=-2)


class H5TPSFDataset(Dataset):
    def __init__(
        self,
        index: DatasetIndex,
        split: str,
        simulation_config: dict[str, Any],
        sds_mm: list[float],
        seed: int,
        max_samples: int | None = None,
        paired_views: bool = True,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to construct H5TPSFDataset")
        self.index = index
        self.records = index.subset(split)
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        self.simulation_config = simulation_config
        self.requested_sds = np.asarray(sds_mm, dtype=np.float32)
        self.seed = int(seed)
        self.split = split
        self.epoch = 0
        self.paired_views = paired_views
        self.schema = ParameterSchema()
        self.loss_weights = np.ones(len(self.records), dtype=np.float32)
        self._handles: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _handle(self, record: SampleRecord) -> h5py.File:
        if record.file not in self._handles:
            self._handles[record.file] = h5py.File(self.index.root / record.file, "r")
        return self._handles[record.file]

    def _rng(self, record: SampleRecord, view: int) -> np.random.Generator:
        payload = f"{self.seed}|{self.epoch}|{record.sample_id}|{view}".encode()
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        return np.random.default_rng(value)

    def _read(self, record: SampleRecord) -> tuple[np.ndarray, float, np.ndarray, int]:
        handle = self._handle(record)
        index = record.source_index
        available_sds = np.asarray(handle["sds_mm"][index], dtype=np.float32)
        channel_indices: list[int] = []
        for target in self.requested_sds:
            matches = np.flatnonzero(np.isclose(available_sds, target, atol=1e-5))
            if matches.size != 1:
                raise ValueError(f"SDS {target:g} mm is missing or duplicated")
            channel_indices.append(int(matches[0]))
        tpsf = np.asarray(handle["tpsf"][index, channel_indices, :], dtype=np.float64)
        edges = np.asarray(handle["time_edges_s"], dtype=np.float64)
        dt_ps = float(np.median(np.diff(edges)) * 1e12)
        theta = self.schema.vector_from_arrays(
            record.layer_count,
            handle["thickness_mm"][index],
            handle["mua_mm"][index],
            handle["musp_mm"][index],
        )
        return tpsf, dt_ps, theta, record.layer_count

    def _simulate_view(
        self, tpsf: np.ndarray, dt_ps: float, rng: np.random.Generator
    ) -> dict[str, Any]:
        measurement = simulate_measurement(
            tpsf, dt_ps, self.simulation_config, rng
        )
        return {
            "dtof": torch.from_numpy(waveform_features(measurement.dtof)),
            "irf": torch.from_numpy(waveform_features(measurement.irf)),
            "time_ps": torch.from_numpy(measurement.time_ps),
            "irf_metadata_json": json.dumps(measurement.metadata, sort_keys=True),
        }

    def __getitem__(self, item: int) -> dict[str, Any]:
        record = self.records[item]
        tpsf, dt_ps, theta, layer_count = self._read(record)
        view_a = self._simulate_view(tpsf, dt_ps, self._rng(record, 0))
        view_b = (
            self._simulate_view(tpsf, dt_ps, self._rng(record, 1))
            if self.paired_views
            else view_a
        )
        channel_mask = random_mask(self._rng(record, 3)) if self.split == "train" else np.ones(4, dtype=bool)
        view_a["channel_mask"] = torch.from_numpy(channel_mask.copy())
        view_b["channel_mask"] = torch.from_numpy(channel_mask.copy())
        # The training objective always pairs a no-prior condition with this
        # non-empty prior condition. Both measurement views reuse the same
        # mask so observation and prior perturbations form a controlled 2x2
        # crossing instead of being confounded.
        known_mask = self.schema.random_nonempty_known_mask(
            layer_count, self._rng(record, 2)
        )
        active_mask = self.schema.active_mask(layer_count)
        normalized = self.schema.normalize(theta)
        return {
            "view_a": view_a,
            "view_b": view_b,
            "theta": torch.from_numpy(normalized),
            "loss_weight": torch.tensor(self.loss_weights[item], dtype=torch.float32),
            "theta_physical": torch.from_numpy(theta),
            "active_mask": torch.from_numpy(active_mask),
            "known_mask": torch.from_numpy(known_mask),
            "known_values": torch.from_numpy(np.where(known_mask, normalized, 0.0).astype(np.float32)),
            "layer_count": torch.tensor(layer_count, dtype=torch.long),
            "sample_id": record.sample_id,
        }

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        self.close()
