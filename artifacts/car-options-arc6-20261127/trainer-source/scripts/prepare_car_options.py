"""Assemble car data and the ARC-5 outdoor core with eight G1 experiment recipes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from smoke_trainer.config import load_config
from smoke_trainer.data import UnifiedDataset, _compact_ranges, load_split_plan, split_summary
from smoke_trainer.schema import sha256_file


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


SMOKE = """
[augmentation.smoke_relocation]
probability = 0.5
target_min_m = 3.0
target_max_m = 8.0
patch_radius_m = 0.35
clearance_m = 0.25
ray_tolerance_rad = 0.02
minimum_points = 8
attempts = 16
"""
MASK = """
[augmentation.background_mask]
probability = 0.5
range_min_m = 8.0
range_max_m = 12.0
width_min_deg = 90.0
width_max_deg = 180.0
"""


def prepare(args: argparse.Namespace) -> dict:
    car = UnifiedDataset(args.car / "dataset_manifest.json", verify_checksums=True)
    car_plan = load_split_plan(car, args.car_split)
    arc = UnifiedDataset(args.arc5 / "dataset_manifest.json", verify_checksums=True)
    arc_plan = load_split_plan(arc, args.arc5 / "split.json")
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Output exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="car-options-", dir=output.parent) as temporary:
        root = Path(temporary) / "dataset"
        (root / "chunks").mkdir(parents=True)
        (root / "configs").mkdir()
        rows = []
        sources = [(car, chunk.index) for chunk in car.chunks]
        sources += [(arc, row["chunk_index"]) for row in arc_plan["roles"]["train"]]
        for source, index in sources:
            chunk = source.chunks[index]
            arrays = source.load_chunk(index)
            if source is arc:
                start = chunk.source_start_s
                seconds = arrays["frame_time_s"] + start
                if not (np.all((seconds >= 95) & (seconds < 195)) and np.all(arrays["label"] != 1)):
                    raise ValueError("ARC-5 chunk is outside the clean 95–195 s core")
            destination = root / "chunks" / chunk.path.name
            try:
                os.link(chunk.path, destination)
            except OSError:
                shutil.copy2(chunk.path, destination)
            row = deepcopy(source.raw_manifest["chunks"][index])
            row["path"] = str(destination.relative_to(root))
            row["source_manifest_sha256"] = source.manifest_sha256
            rows.append(row)
        schema = json.loads((args.car / "dataset_schema.json").read_text())
        schema["arrays"]["source_domain"] = "scalar string: car_ros2 or grandtour_ros1"
        schema["arc5_clean_policy"] = json.loads((args.arc5 / "dataset_schema.json").read_text())["known_clean_policy"]
        write_json(root / "dataset_schema.json", schema)
        manifest = {
            "schema_version": "1.0", "status": "complete", "dataset_schema": "dataset_schema.json",
            "split_policy": "Use the supplied car_only or with_arc5 split. Whole car recordings retain their roles.",
            "chunks": rows, "frames": sum(row["frames"] for row in rows),
            "points": sum(row["points"] for row in rows),
            "labels": {key: sum(row["labels"][key] for row in rows) for key in ("unimpacted", "smoke_impacted", "ignore")},
            "sources": {"car_surface_v2": car.manifest_sha256, "arc5_outdoor_clean_v1": arc.manifest_sha256},
            "label_review_caveat": "Car surface v2 includes 2,440 smoke-to-background changes from surface matches and reviewed overrides. Automatic changes require further annotation review.",
        }
        write_json(root / "dataset_manifest.json", manifest)
        combined = UnifiedDataset(root / "dataset_manifest.json", verify_checksums=True)
        for with_arc in (False, True):
            roles = deepcopy(car_plan["roles"])
            excluded = deepcopy(car_plan.get("excluded", []))
            for index in range(len(car.chunks), len(rows)):
                frames = rows[index]["frames"]
                selected = list(range(0, frames, 4)) if with_arc else []
                if selected:
                    roles["train"].append({"chunk_index": index, "ranges": _compact_ranges(selected)})
                omitted = sorted(set(range(frames)) - set(selected))
                if omitted:
                    excluded.append({"chunk_index": index, "ranges": _compact_ranges(omitted)})
            name = "with_arc5" if with_arc else "car_only"
            plan = {
                "version": 1, "protocol": f"car_options_v1_{name}", "manifest": "dataset_manifest.json",
                "manifest_sha256": combined.manifest_sha256, "purge_frames_each_side": 0,
                "assignment": deepcopy(car_plan["assignment"]),
                "arc5_training_policy": "Every fourth core scan, approximately 2.5 Hz" if with_arc else "Excluded",
                "chunks": [{"index": c.index, "path": str(c.path.relative_to(root)), "sha256": c.sha256,
                            "frames": c.frames, "stream_id": c.stream_id} for c in combined.chunks],
                "roles": roles, "excluded": excluded,
            }
            plan["summary"] = split_summary(combined, plan)
            canonical = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
            plan["split_sha256"] = hashlib.sha256(canonical).hexdigest()
            write_json(root / f"split_{name}.json", plan)
            load_split_plan(combined, root / f"split_{name}.json")

        recipes = []
        for key, extra in (("a_baseline", ""), ("b_distant_smoke", SMOKE), ("c_background_mask", MASK), ("d_both", SMOKE + MASK)):
            for source in ("car_only", "with_arc5"):
                name = f"{key}_{source}"
                text = f'''[data]
manifest = "../dataset_manifest.json"
split = "../split_{source}.json"
verify_checksums = true

[model]
name = "global_pointnet"
adapter = "global_points"
point_widths = [64, 128]
head_widths = [64]
input_alignment = "tnet"
alignment_widths = [32, 64]
alignment_regularization = 0.001

[training]
seed = 20260920
device = "auto"
epochs = 30
learning_rate = 0.001
weight_decay = 0.0001
positive_weight = 8.0
patience = 5
max_hours = 4.0
max_optimizer_steps_per_epoch = 1400
yaw_augmentation_deg = 180.0
pitch_augmentation_deg = 0.0
roll_augmentation_deg = 0.0

[output]
root = "../../../../runs"
{extra}'''
                (root / "configs" / f"{name}.toml").write_text(text, encoding="utf-8")
                recipes.append(name)
        write_json(root / "experiments.json", {"recipes": recipes, "model": "G1", "seed": 20260920})
        # Each source chunk is unchanged. The narrower ARC-5 interval is selected by chunk identity.
        write_json(root / "preparation.json", {
            "car_chunks_unchanged": len(car.chunks), "arc5_core_chunks": len(rows) - len(car.chunks),
            "arc5_core_frames": 1000, "arc5_training_frames_per_epoch_pool": 250,
            "arc5_training_span_s": [95, 195], "exporter_sha256": sha256_file(Path(__file__)),
        })
        root.rename(output)
    for config in (output / "configs").glob("*.toml"):
        resolved = load_config(config)
        assert Path(resolved["data"]["manifest"]).is_file()
        assert Path(resolved["data"]["split"]).is_file()
    return {"dataset": str(output), "frames": manifest["frames"], "recipes": recipes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--car", type=Path, default=Path("data/training/car_surface_v2"))
    parser.add_argument("--car-split", type=Path, default=Path("experiments/splits/car_whole_recording_surface_v2.json"))
    parser.add_argument("--arc5", type=Path, default=Path("data/training/arc5_outdoor_clean_v1"))
    parser.add_argument("--output", type=Path, default=Path("data/training/car_training_options_v1"))
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
