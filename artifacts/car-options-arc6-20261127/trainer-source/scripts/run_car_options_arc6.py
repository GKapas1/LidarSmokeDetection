"""Run the complete car-options matrix and evaluate each bundle on ARC-6."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "car-options-arc6-20261127"
OUT = ROOT / "runs" / PREFIX
SETUP = OUT / "setup"
DATA = ROOT / "data" / "training" / "car_training_options_v1"
ARC = ROOT / "data" / "training" / "unified_v1"
SPLIT = ROOT / "experiments" / "splits" / "grandtour_all_external_test_v1.json"
PY = sys.executable


def command(args, log):
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + subprocess.list2cmdline([str(x) for x in args]) + "\n")
        stream.flush()
        subprocess.run([str(x) for x in args], cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})


def summarize(runs):
    rows = []
    def flatten(value, prefix, row):
        if isinstance(value, dict):
            for key, item in value.items():
                flatten(item, f"{prefix}.{key}" if prefix else key, row)
        elif not isinstance(value, list):
            row[prefix] = value
    for run in runs:
        directory = OUT / run["run_name"]
        row = {"model": run["model"], "recipe": run["recipe"], "run_name": run["run_name"]}
        for name, prefix in [("car_test_metrics.json", "car"), ("arc6_test_metrics.json", "arc6")]:
            path = directory / name
            if path.exists():
                flatten(json.loads(path.read_text(encoding="utf-8")), prefix, row)
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    log = OUT / "batch.log"
    try:
        (OUT / "README.txt").write_text(
            "Car options / ARC-6 experiment. status.json shows progress; batch.log records commands and errors.\n"
            "Each run has checkpoints, bundle, history, normalization, calibration and car and ARC-6 test metrics.\n"
            "comparison.csv is updated after each finished run. setup contains configurations and splits.\n"
            "The original car-options and unified_v1 datasets are not in this package. Transfer both directories\n"
            "byte-for-byte to reproduce evaluations or make per-point predictions; manifests and schemas are included.\n"
            "Install trainer-source using trainer-source/README.md. Absolute paths in configurations reflect this machine.\n"
            "ARC-6 uses the pre-existing all-GrandTour external split and the car-selected operating threshold.\n"
            "GT needs five-frame context and scores fewer frames. Car labels are surface-v2 pilot labels; see\n"
            "artifacts/car-training-options-v1/README.md for dataset and augmentation limitations.\n", encoding="utf-8")
        (OUT / "status.json").write_text(json.dumps({"status": "preparing"}, indent=2))
        command([PY, "-m", "pip", "freeze"], OUT / "environment.txt")
        command(["git", "rev-parse", "HEAD"], OUT / "git-commit.txt")
        command(["git", "status", "--short"], OUT / "git-status.txt")
        shutil.copytree(ROOT / "trainer", OUT / "trainer-source", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc", "*.egg-info"))
        for name, source in [("car_dataset_manifest.json", DATA / "dataset_manifest.json"),
                             ("car_dataset_schema.json", DATA / "dataset_schema.json"),
                             ("arc6_dataset_manifest.json", ARC / "dataset_manifest.json"),
                             ("arc6_dataset_schema.json", ARC / "dataset_schema.json"),
                             ("arc6_split.json", SPLIT)]:
            shutil.copy2(source, OUT / name)
        command([PY, "-m", "smoke_trainer.cli", "inspect", "--manifest", DATA / "dataset_manifest.json", "--verify-checksums"], log)
        command([PY, "-m", "smoke_trainer.cli", "inspect", "--manifest", ARC / "dataset_manifest.json", "--verify-checksums"], log)
        command([PY, ROOT / "trainer/scripts/prepare_all_car_options.py", "--dataset", DATA,
                 "--output", SETUP, "--run-prefix", PREFIX, "--models", "l1", "g1", "gt", "voxel"], log)
        runs = json.loads((SETUP / "runs.json").read_text(encoding="utf-8"))
        # Get all four baseline models early, while keeping the full matrix fixed.
        runs.sort(key=lambda row: (row["recipe"], ["l1", "g1", "gt", "voxel"].index(row["model"])))
        summarize(runs)
        for index, run in enumerate(runs):
            run_dir = OUT / run["run_name"]
            (OUT / "status.json").write_text(json.dumps({"status": "running", "completed": index, "total": len(runs), "current": run["run_name"]}, indent=2))
            command([PY, "-m", "smoke_trainer.cli", "train", "--config", run["config"], "--run-name", run["run_name"], "--device", "cuda"], log)
            for label, manifest, split in [("car_test_metrics.json", DATA / "dataset_manifest.json", run["split"]),
                                           ("arc6_test_metrics.json", ARC / "dataset_manifest.json", SPLIT)]:
                command([PY, "-m", "smoke_trainer.cli", "evaluate", "--bundle", run_dir / "bundle.pt", "--manifest", manifest,
                         "--split", split, "--partition", "test", "--output", run_dir / label, "--device", "cuda"], log)
            summarize(runs)
        (OUT / "status.json").write_text(json.dumps({"status": "complete", "completed": len(runs), "total": len(runs)}, indent=2))
        with zipfile.ZipFile(OUT.with_suffix(".zip"), "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path in OUT.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(OUT.parent))
    except Exception:
        with log.open("a", encoding="utf-8") as stream:
            traceback.print_exc(file=stream)
        (OUT / "status.json").write_text(json.dumps({"status": "failed", "error": traceback.format_exc()}, indent=2))
        raise


if __name__ == "__main__":
    main()
