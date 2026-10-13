"""Sequential, process-isolated repetitions with per-seed provenance."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from .config import project_path
from .log_time import timestamp


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_repetitions(config, module, arguments):
    seeds = config["seeds"]["training"]
    seeds = seeds if isinstance(seeds, list) else [seeds]
    root = project_path(config, config["training"]["output_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    records = []
    from .training import _manifest, _dataset_fingerprint
    fingerprint = _dataset_fingerprint(_manifest(config))
    from .channels import PROTOCOL
    experiment_config = {k: deepcopy(v) for k, v in config.items() if not k.startswith("_")}
    experiment_config["seeds"]["training"] = None
    experiment_config["training"].pop("run_epochs", None)
    experiment_config["training"]["output_dir"] = str(root)
    experiment = {"config": experiment_config, "dataset": fingerprint, "channel_protocol": PROTOCOL, "module": module}
    experiment_signature = hashlib.sha256(json.dumps(experiment, sort_keys=True).encode()).hexdigest()
    experiment_path = root / "repeat_experiment.json"
    if experiment_path.exists() and json.loads(experiment_path.read_text(encoding="utf-8"))["signature"] != experiment_signature:
        raise ValueError(f"Existing seed run has a different configuration or dataset: {root}")
    _write(experiment_path, {"signature": experiment_signature, **experiment})
    wants_test = "--no-test" not in arguments and ("--test" in arguments or config.get("evaluation", {}).get("auto_run", False))
    if len(seeds) > 1 and "--resume" in arguments:
        raise ValueError("Use the queue's automatic per-seed resume, or --seed for a single explicit checkpoint")
    for ordinal, seed in enumerate(seeds, 1):
        destination = root / f"seed_{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        resolved = deepcopy(config)
        resolved["seeds"]["training"] = seed
        resolved["training"]["output_dir"] = str(destination)
        payload = {k: v for k, v in resolved.items() if not k.startswith("_")}
        from .channels import PROTOCOL
        scientific = deepcopy(payload)
        scientific["training"].pop("run_epochs", None)
        signature = hashlib.sha256(json.dumps({"config": scientific, "dataset": fingerprint, "channel_protocol": PROTOCOL}, sort_keys=True).encode()).hexdigest()
        record_path = destination / "repeat_state.json"
        if record_path.exists():
            previous = json.loads(record_path.read_text(encoding="utf-8"))
            if previous["signature"] != signature:
                raise ValueError(f"Existing seed run has a different configuration: {destination}")
        elif (destination / "last.pt").exists():
            raise ValueError(f"Existing run lacks repeat provenance: {destination}")
        record = {"seed": seed, "module": module, "signature": signature,
                  "output": str(destination), "status": "running", "session_started_at": timestamp()}
        _write(record_path, record)
        _write(destination / "resolved_config.json", payload)
        summary_path = destination / "training_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        report_path = destination / "test" / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        print(json.dumps({"event": "repeat_progress", "timestamp": record["session_started_at"], "seed": seed, "task": ordinal,
                          "total": len(seeds), "model": module, "output": str(destination)}), flush=True)
        try:
            if summary.get("status") == "completed":
                if wants_test and (not report or not (report_path.parent / "report.html").is_file()):
                    checkpoint = Path(summary["best_checkpoint"])
                    if not checkpoint.exists():
                        checkpoint = destination / checkpoint.name
                    test_module = module.replace(".train", ".test")
                    subprocess.run([sys.executable, "-m", test_module, "--config", config["_config_path"],
                                    "--checkpoint", str(checkpoint)], check=True)
            else:
                command = [sys.executable, "-m", module, *arguments,
                           "--seed", str(seed), "--output-dir", str(destination)]
                if (destination / "last.pt").exists() and "--resume" not in arguments:
                    command += ["--resume", str(destination / "last.pt")]
                subprocess.run(command, check=True)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") != "completed":
                record["status"] = "paused"
                record["session_finished_at"] = timestamp()
                _write(record_path, record)
                print(json.dumps({"event": "repeat_paused", "timestamp": record["session_finished_at"], "seed": seed}), flush=True)
                return
            if wants_test:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if not (report_path.parent / "report.html").is_file():
                    raise RuntimeError(f"Formal HTML report is missing: {report_path.parent}")
                if report.get("status") not in {"completed", "completed_with_diagnostic_warnings"}:
                    raise RuntimeError(f"Formal test requires attention: {report.get('status')}")
            record["status"] = "completed"
            record["session_finished_at"] = timestamp()
            _write(record_path, record)
            print(json.dumps({"event": "repeat_complete", "timestamp": record["session_finished_at"], "seed": seed}), flush=True)
        except BaseException as error:
            record.update(status="failed", error=str(error), session_finished_at=timestamp())
            _write(record_path, record)
            print(json.dumps({"event": "repeat_failed", "timestamp": record["session_finished_at"], "seed": seed, "error": str(error)}), flush=True)
            raise
        records = []
        for path in sorted(root.glob("seed_*/repeat_state.json")):
            other = json.loads(path.read_text(encoding="utf-8"))
            if other.get("status") == "completed":
                saved_report = path.parent / "test" / "report.json"
                records.append({**other, "report": json.loads(saved_report.read_text(encoding="utf-8")) if saved_report.exists() else {}})
        _write(root / "repeat_summary.json", summarize(records))


def summarize(records):
    metrics = {}
    for record in records:
        report = record.get("report", {})
        values = {}
        if "metrics" in report:
            values = {"macro_nmae": report["metrics"].get("macro_nmae")}
        else:
            for layer, row in report.get("headline", {}).get("no_prior_by_layer", {}).items():
                for key, value in row.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        values[f"N{layer}/{key}"] = value
        for key, value in values.items():
            if value is not None and np.isfinite(value):
                metrics.setdefault(key, []).append({"seed": record["seed"], "value": float(value)})
    return {"runs": [{k: v for k, v in r.items() if k != "report"} for r in records],
            "metrics": {key: {"by_seed": rows, "mean": float(np.mean([r['value'] for r in rows])),
                              "std_between_training_seeds": float(np.std([r['value'] for r in rows], ddof=1)) if len(rows)>1 else None}
                        for key, rows in metrics.items()},
            "interpretation": "Between-training-seed variation; not a case bootstrap confidence interval."}


