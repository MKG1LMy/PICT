from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ..inference import (
    infer_raw_counts,
    load_checkpoint,
    save_inference,
    validate_model_input_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run posterior inference on raw counts")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="NPZ with dtof_counts, irf_counts, time_ps")
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer-count", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--known-json", default="{}")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(args.checkpoint, device)
    payload = np.load(args.input)
    time_ps = np.asarray(payload["time_ps"], dtype=np.float64)
    dtof_counts = np.asarray(payload["dtof_counts"])
    irf_counts = np.asarray(payload["irf_counts"])
    if time_ps.ndim != 1 or len(time_ps) < 2 or np.any(~np.isfinite(time_ps)):
        parser.error("time_ps must be a finite one-dimensional time axis")
    differences = np.diff(time_ps)
    dt_ps = float(np.median(differences))
    if dt_ps <= 0.0 or not np.allclose(differences, dt_ps, rtol=1e-5, atol=1e-8):
        parser.error("time_ps must be strictly increasing and uniformly spaced")
    supplied_sds = payload["sds_mm"].tolist() if "sds_mm" in payload else [10, 20, 30, 40]
    if dtof_counts.shape != (len(supplied_sds), len(time_ps)):
        parser.error("dtof_counts must have shape [C,T] matching sds_mm and time_ps")
    if irf_counts.shape not in {(len(time_ps),), (len(supplied_sds), len(time_ps))}:
        parser.error("irf_counts must have shape [T] or [C,T] matching DTOF")
    expected_start_ps = float(config["data"].get("time_start_ps", 0.0))
    if not np.isclose(time_ps[0], expected_start_ps, rtol=0.0, atol=1e-6):
        parser.error(
            f"time_ps must start at {expected_start_ps:g} ps after external cropping"
        )
    try:
        validate_model_input_grid(config, dtof_counts, irf_counts, dt_ps)
    except ValueError as error:
        parser.error(str(error))
    samples, metadata = infer_raw_counts(
        model,
        config,
        dtof_counts,
        irf_counts,
        dt_ps,
        args.layer_count,
        json.loads(args.known_json),
        args.samples,
        args.seed,
        sds_mm=supplied_sds,
    )
    paths = save_inference(args.output, samples, metadata)
    print(json.dumps({"samples": str(paths[0]), "summary": str(paths[1])}, indent=2))


if __name__ == "__main__":
    main()
