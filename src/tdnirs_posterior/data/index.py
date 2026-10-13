from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np


EXPECTED_SDS_MM = np.asarray([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
EXPECTED_TIME_BINS = 2500
EXPECTED_TIME_BIN_PS = 4.0


def _ratio_targets(total: int, ratios: Sequence[float]) -> tuple[int, int, int]:
    values = np.asarray(ratios, dtype=np.float64)
    if values.shape != (3,) or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("data.split must contain three finite non-negative ratios")
    if float(values.sum()) <= 0:
        raise ValueError("data.split ratios must have a positive sum")
    values /= values.sum()
    ideal = total * values
    allocation = np.floor(ideal).astype(int)
    remainder = total - int(allocation.sum())
    order = np.argsort(-(ideal - allocation), kind="stable")
    for split_index in order[:remainder]:
        allocation[int(split_index)] += 1
    return tuple(int(value) for value in allocation)  # type: ignore[return-value]


def _layer_split_targets(
    layer_counts: Mapping[int, int], ratios: Sequence[float]
) -> dict[int, tuple[int, int, int]]:
    """Allocate exact global split totals while preserving layer proportions."""
    layers = (1, 2, 3)
    if set(layer_counts) != set(layers) or any(
        layer_counts[layer] <= 0 for layer in layers
    ):
        raise ValueError("Layer-aware splitting requires non-empty N=1, N=2, and N=3")
    global_targets = _ratio_targets(sum(layer_counts.values()), ratios)
    values = np.asarray(ratios, dtype=np.float64)
    values /= values.sum()
    ideals = np.asarray(
        [[layer_counts[layer] * ratio for ratio in values] for layer in layers],
        dtype=np.float64,
    )
    allocation = np.floor(ideals).astype(int)
    row_remaining = np.asarray(
        [layer_counts[layer] for layer in layers], dtype=int
    ) - allocation.sum(axis=1)
    column_remaining = (
        np.asarray(global_targets, dtype=int) - allocation.sum(axis=0)
    )

    while int(row_remaining.sum()) > 0:
        candidates: list[tuple[float, int, int]] = []
        for row in np.flatnonzero(row_remaining > 0):
            for column in np.flatnonzero(column_remaining > 0):
                fractional = (
                    ideals[int(row), int(column)]
                    - allocation[int(row), int(column)]
                )
                candidates.append((-float(fractional), int(row), int(column)))
        if not candidates:
            raise RuntimeError("Unable to preserve layer proportions in exact split")
        _, row, column = min(candidates)
        allocation[row, column] += 1
        row_remaining[row] -= 1
        column_remaining[column] -= 1

    return {
        layer: tuple(int(value) for value in allocation[row])  # type: ignore[misc]
        for row, layer in enumerate(layers)
    }


def dataset_policy(data_config: Mapping[str, Any], split_seed: int) -> dict[str, Any]:
    """Return the normalized policy that uniquely defines a manifest."""
    return {
        "split_seed": int(split_seed),
        "split": [float(value) for value in data_config["split"]],
    }


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


@dataclass(frozen=True)
class SampleRecord:
    file: str
    source_index: int
    sample_id: str
    layer_count: int
    status: int
    category: str
    split: str = ""

    @property
    def stratum(self) -> tuple[str, int, int, str]:
        return (self.file, self.layer_count, self.status, self.category)


class DatasetIndex:
    def __init__(
        self,
        records: Iterable[SampleRecord],
        root: Path,
        metadata: Mapping[str, Any] | None = None,
    ):
        self.records = list(records)
        self.root = Path(root).resolve()
        self.metadata = dict(metadata or {})

    @classmethod
    def discover(cls, root: str | Path) -> "DatasetIndex":
        root_path = Path(root).resolve()
        files = sorted(root_path.glob("*/mixed-dataset.h5"))
        if not files:
            raise ValueError(f"No mixed-dataset.h5 files under {root_path}")
        records: list[SampleRecord] = []
        seen_ids: set[str] = set()
        for path in files:
            relative = path.relative_to(root_path).as_posix()
            with h5py.File(path, "r") as handle:
                cls._audit_file(handle, path)
                count = int(handle["tpsf"].shape[0])
                for index in range(count):
                    sample_id = _decode(handle["sample_id"][index])
                    if sample_id in seen_ids:
                        raise ValueError(f"Duplicate sample_id: {sample_id}")
                    seen_ids.add(sample_id)
                    records.append(
                        SampleRecord(
                            file=relative,
                            source_index=index,
                            sample_id=sample_id,
                            layer_count=int(handle["layer_count"][index]),
                            status=int(handle["status"][index]),
                            category=_decode(handle["category"][index]),
                        )
                    )
        return cls(records, root_path)

    @staticmethod
    def _audit_file(handle: h5py.File, path: Path) -> None:
        required = {
            "tpsf",
            "time_edges_s",
            "sds_mm",
            "sample_id",
            "layer_count",
            "status",
            "category",
            "thickness_mm",
            "mua_mm",
            "musp_mm",
        }
        missing = sorted(required - set(handle.keys()))
        if missing:
            raise ValueError(f"{path} is missing datasets: {missing}")
        tpsf = handle["tpsf"]
        if tpsf.ndim != 3 or tpsf.shape[1:] != (16, EXPECTED_TIME_BINS):
            raise ValueError(f"Unexpected TPSF shape in {path}: {tpsf.shape}")
        if handle["time_edges_s"].shape != (EXPECTED_TIME_BINS + 1,):
            raise ValueError(f"Unexpected time grid in {path}")
        time_edges_s = np.asarray(handle["time_edges_s"][:], dtype=np.float64)
        expected_step_s = EXPECTED_TIME_BIN_PS * 1e-12
        if not np.isclose(time_edges_s[0], 0.0, rtol=0.0, atol=1e-18) or not np.allclose(
            np.diff(time_edges_s), expected_step_s, rtol=1e-9, atol=1e-18
        ):
            raise ValueError(
                f"Expected a 0 ps origin and {EXPECTED_TIME_BIN_PS:g} ps/bin in {path}"
            )
        if handle["thickness_mm"].shape != (tpsf.shape[0], 2):
            raise ValueError(f"Unexpected thickness shape in {path}")
        sample_sds = np.asarray(handle["sds_mm"][0], dtype=np.float32)
        for value in EXPECTED_SDS_MM:
            if not np.any(np.isclose(sample_sds, value, atol=1e-5)):
                raise ValueError(f"SDS {value:g} mm is absent from {path}")
        if handle.attrs.get("tpsf_unit", "") != "detected_weight_per_launched_photon_per_bin":
            raise ValueError(f"Unexpected TPSF unit in {path}")
        from ..parameters import ParameterSchema
        schema = ParameterSchema()
        for start in range(0, len(tpsf), 32):
            stop = min(start + 32, len(tpsf))
            curves = np.asarray(tpsf[start:stop])
            if not np.all(np.isfinite(curves)) or np.any(curves < 0):
                raise ValueError(f"Nonfinite or negative TPSF in {path}, rows {start}:{stop}")
            sds = np.asarray(handle["sds_mm"][start:stop])
            if not np.allclose(sds, sample_sds[None], atol=1e-5):
                raise ValueError(f"Inconsistent SDS rows in {path}")
            selected = [int(np.flatnonzero(np.isclose(sample_sds, s))[0]) for s in EXPECTED_SDS_MM]
            if np.any(curves[:, selected].sum(axis=-1) <= 0):
                raise ValueError(f"Empty selected TPSF in {path}, rows {start}:{stop}")
            if not np.all(np.isin(handle["status"][start:stop], [1, 3])):
                raise ValueError(f"Unusable sample status in {path}")
            layers = np.asarray(handle["layer_count"][start:stop])
            if not np.all(np.isin(layers, [1, 2, 3])):
                raise ValueError(f"Invalid layer count in {path}")
            # Validate only parameters physically present in each phantom.
            for j, n in enumerate(layers):
                row = start + j
                d = np.asarray(handle["thickness_mm"][row])[:int(n)-1]
                a = np.asarray(handle["mua_mm"][row])[:int(n)]
                s = np.asarray(handle["musp_mm"][row])[:int(n)]
                values = np.concatenate([d, a, s])
                if not np.all(np.isfinite(values)) or np.any(values <= 0):
                    raise ValueError(f"Invalid active physical parameters in {path}, row {row}")
                if n == 3 and d.sum() > schema.n3_total_thickness_max_mm + 1e-5:
                    raise ValueError(f"N=3 cumulative thickness exceeds 35 mm in {path}, row {row}")

    def stratified_split(
        self, seed: int, target: tuple[int, int, int]
    ) -> "DatasetIndex":
        if sum(target) != len(self.records):
            raise ValueError("Split targets must sum to the dataset size")
        split_names = ("train", "validation", "test")
        ratios = np.asarray(target, dtype=np.float64) / sum(target)
        groups: dict[tuple[str, int, int, str], list[SampleRecord]] = defaultdict(list)
        for record in self.records:
            groups[record.stratum].append(record)

        assigned: dict[str, list[SampleRecord]] = {name: [] for name in split_names}
        group_allocations: dict[tuple[str, int, int, str], list[int]] = {}
        ideals: dict[tuple[str, int, int, str], np.ndarray] = {}
        shuffled_groups: dict[tuple[str, int, int, str], list[SampleRecord]] = {}
        for key in sorted(groups):
            stable = int.from_bytes(
                hashlib.sha256(f"{seed}|{key}".encode()).digest()[:8], "little"
            )
            rng = np.random.default_rng(stable)
            values = list(groups[key])
            rng.shuffle(values)
            shuffled_groups[key] = values
            ideal = len(values) * ratios
            ideals[key] = ideal
            allocation = np.floor(ideal).astype(int).tolist()
            remainder = len(values) - sum(allocation)
            fractional_order = np.argsort(-(ideal - np.floor(ideal)))
            for split_index in fractional_order[:remainder]:
                allocation[int(split_index)] += 1
            group_allocations[key] = allocation

        totals = np.sum(np.asarray(list(group_allocations.values())), axis=0)
        desired = np.asarray(target)
        while not np.array_equal(totals, desired):
            sources = np.flatnonzero(totals > desired)
            destinations = np.flatnonzero(totals < desired)
            best: tuple[float, tuple[str, int, int, str], int, int] | None = None
            for key, allocation in group_allocations.items():
                for source in sources:
                    if allocation[int(source)] == 0:
                        continue
                    for destination in destinations:
                        before = np.sum((np.asarray(allocation) - ideals[key]) ** 2)
                        changed = np.asarray(allocation).copy()
                        changed[int(source)] -= 1
                        changed[int(destination)] += 1
                        cost = float(np.sum((changed - ideals[key]) ** 2) - before)
                        candidate = (cost, key, int(source), int(destination))
                        if best is None or candidate < best:
                            best = candidate
            if best is None:
                raise RuntimeError("Unable to satisfy exact split targets")
            _, key, source, destination = best
            group_allocations[key][source] -= 1
            group_allocations[key][destination] += 1
            totals[source] -= 1
            totals[destination] += 1

        for key in sorted(shuffled_groups):
            values = shuffled_groups[key]
            cursor = 0
            for split_name, count in zip(split_names, group_allocations[key], strict=True):
                for record in values[cursor : cursor + count]:
                    assigned[split_name].append(
                        SampleRecord(**{**asdict(record), "split": split_name})
                    )
                cursor += count

        merged = [record for name in split_names for record in assigned[name]]
        return DatasetIndex(merged, self.root, self.metadata)

    def stratified_split_by_layer(
        self, seed: int, ratios: Sequence[float]
    ) -> "DatasetIndex":
        """Keep every sample and preserve the near-equal layer mix in each split."""
        layer_counts = Counter(record.layer_count for record in self.records)
        targets = _layer_split_targets(layer_counts, ratios)
        assigned: dict[str, list[SampleRecord]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        for layer in (1, 2, 3):
            layer_records = [
                record for record in self.records if record.layer_count == layer
            ]
            split = DatasetIndex(layer_records, self.root).stratified_split(
                seed=seed, target=targets[layer]
            )
            for split_name in assigned:
                assigned[split_name].extend(split.subset(split_name))
        merged = [record for name in assigned for record in assigned[name]]
        return DatasetIndex(merged, self.root, self.metadata)

    def write(self, destination: str | Path) -> tuple[Path, Path]:
        output = Path(destination)
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "manifest.jsonl"
        summary_path = output / "summary.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        summary = {
            "root": str(self.root),
            "total": len(self.records),
            "split": Counter(record.split for record in self.records),
            "layer_count": Counter(record.layer_count for record in self.records),
            "split_layer_count": {
                split: Counter(
                    record.layer_count for record in self.records if record.split == split
                )
                for split in ("train", "validation", "test")
            },
            "status": Counter(record.status for record in self.records),
            "source_files": Counter(record.file for record in self.records),
            **self.metadata,
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, default=dict)
        return manifest_path, summary_path

    @classmethod
    def read(cls, manifest: str | Path, root: str | Path) -> "DatasetIndex":
        records: list[SampleRecord] = []
        with Path(manifest).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(SampleRecord(**json.loads(line)))
        return cls(records, Path(root))

    def subset(self, split: str) -> list[SampleRecord]:
        return [record for record in self.records if record.split == split]


def prepare_dataset_index(
    root: str | Path, data_config: Mapping[str, Any], seed: int
) -> DatasetIndex:
    """Discover all raw samples and create the configured stratified split."""
    index = DatasetIndex.discover(root)
    policy = dataset_policy(data_config, seed)
    index = index.stratified_split_by_layer(seed=seed, ratios=policy["split"])
    return DatasetIndex(
        index.records,
        index.root,
        {**index.metadata, "dataset_policy": policy},
    )


def manifest_policy_errors(
    raw_index: DatasetIndex,
    manifest: DatasetIndex,
    data_config: Mapping[str, Any],
) -> list[str]:
    """Validate a fixed manifest without replaying version-sensitive RNG details.

    The manifest is the reproducible split source of truth. Validation checks
    complete raw-data membership, raw-record identity, and exact split counts
    without replaying NumPy's version-sensitive shuffle choices.
    """

    errors: list[str] = []
    raw_by_id = {record.sample_id: record for record in raw_index.records}
    manifest_ids = [record.sample_id for record in manifest.records]
    if len(set(manifest_ids)) != len(manifest_ids):
        errors.append("manifest contains duplicate sample_id values")

    unknown = sorted(set(manifest_ids) - set(raw_by_id))
    if unknown:
        errors.append(f"manifest references {len(unknown)} unknown raw samples")

    identity_fields = (
        "file",
        "source_index",
        "layer_count",
        "status",
        "category",
    )
    mismatched_identity = 0
    for record in manifest.records:
        source = raw_by_id.get(record.sample_id)
        if source is None:
            continue
        if any(getattr(record, field) != getattr(source, field) for field in identity_fields):
            mismatched_identity += 1
    if mismatched_identity:
        errors.append(
            f"manifest metadata differs from raw data for {mismatched_identity} samples"
        )

    policy = dataset_policy(data_config, split_seed=0)
    raw_ids = set(raw_by_id)
    if set(manifest_ids) != raw_ids:
        errors.append("manifest membership does not match the complete raw dataset")

    ratios = policy["split"]
    split_names = ("train", "validation", "test")
    expected_splits = Counter(
        dict(
            zip(
                split_names,
                _ratio_targets(len(raw_index.records), ratios),
                strict=True,
            )
        )
    )
    actual_splits = Counter(record.split for record in manifest.records)
    if actual_splits != expected_splits:
        errors.append(
            f"manifest split counts are {dict(actual_splits)}, expected {dict(expected_splits)}"
        )
    expected_layer_splits = _layer_split_targets(
        Counter(record.layer_count for record in raw_index.records), ratios
    )
    actual_split_layers = Counter(
        (record.split, record.layer_count) for record in manifest.records
    )
    expected_split_layers: Counter[tuple[str, int]] = Counter()
    for layer, allocations in expected_layer_splits.items():
        for split, count in zip(split_names, allocations, strict=True):
            expected_split_layers[(split, layer)] = count
    if actual_split_layers != expected_split_layers:
        errors.append("manifest does not preserve the configured layer proportions")
    return errors
