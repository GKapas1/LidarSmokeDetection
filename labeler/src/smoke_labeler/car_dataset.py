"""Export reviewed car smoke and known-clean drives for binary training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from mcap.reader import make_reader

from .bag import _custom_msg_cdr_arrays
from .car_label import CONFIG
from .unified_dataset import SCHEMA_VERSION, _metadata, _schema, _sha256, _write_chunk


def _read_recording(prepared, record, smoke_labels, topic):
    recording_id = record["id"]
    with np.load(prepared / "recordings" / recording_id / "motion.npz") as motion:
        timestamps = motion["frame_time_ns"]
        ptr = np.r_[0, np.cumsum(motion["frame_point_count"], dtype=np.int64)]
    if len(timestamps) != record["timing"]["lidar"]["count"] or ptr[-1] != record["points"]:
        raise ValueError(f"{recording_id}: prepared frame or point count mismatch")
    if not len(timestamps):
        raise ValueError(f"{recording_id}: missing LiDAR frames")

    size = int(ptr[-1])
    if smoke_labels is None:
        labels = np.full(size, 255, dtype=np.uint8)
    else:
        with np.load(smoke_labels, allow_pickle=False) as frozen:
            labels = frozen["label"]
            if not (np.array_equal(frozen["timestamp_ns"], timestamps)
                    and np.array_equal(frozen["frame_ptr"], ptr)
                    and labels.shape == (size,)
                    and labels.dtype == np.uint8
                    and ((labels == 1) | (labels == 255)).all()):
                raise ValueError(f"{smoke_labels}: invalid or mismatched frozen labels")

    xyz = np.empty((size, 3), dtype=np.float32)
    intensity = np.empty(size, dtype=np.float32)
    tag = np.empty(size, dtype=np.uint8)
    line = np.empty(size, dtype=np.uint8)
    offsets = np.empty(size, dtype=np.float32)
    frame = 0
    min_range, max_range = CONFIG["range_m"]
    for source in record["source_files"]:
        path = Path(source["path"])
        if _sha256(path) != source["sha256"]:
            raise ValueError(f"{path}: MCAP changed since preparation")
        with path.open("rb") as stream:
            reader = make_reader(stream, validate_crcs=True)
            for schema, _, message in reader.iter_messages(topics=[topic]):
                if schema is None or schema.name != "livox_ros_driver2/msg/CustomMsg":
                    raise ValueError(f"{path}: unexpected LiDAR message schema")
                points, reflectivity, tags, lines, times, stamp = _custom_msg_cdr_arrays(message.data)
                if (frame >= len(timestamps) or stamp != timestamps[frame]
                        or len(points) != ptr[frame + 1] - ptr[frame]):
                    raise ValueError(f"{recording_id}: MCAP mismatch at frame {frame}")
                sl = slice(ptr[frame], ptr[frame + 1])
                finite = np.isfinite(points).all(axis=1)
                if smoke_labels is None:
                    ranges = np.linalg.norm(points, axis=1)
                    labels[sl][finite & (ranges >= min_range) & (ranges <= max_range)] = 0
                elif np.any((labels[sl] == 1) & (~finite | ~np.any(points != 0, axis=1))):
                    raise ValueError(f"{recording_id}: frozen smoke on an invalid return in frame {frame}")
                # Trainer features must be finite even when the return is ignored.
                points[~finite] = 0
                xyz[sl], intensity[sl], tag[sl], line[sl], offsets[sl] = points, reflectivity, tags, lines, times
                frame += 1
    if frame != len(timestamps):
        raise ValueError(f"{recording_id}: missing MCAP frames")

    return {
        **_metadata(prepared.name, "car_ros2", recording_id, record["role"]),
        "xyz": xyz,
        "intensity": intensity,
        "tag": tag,
        "line": line,
        "point_offset_s": offsets,
        "label": labels,
        "frame_ptr": ptr,
        "frame_index": np.repeat(np.arange(frame, dtype=np.int32), np.diff(ptr)),
        "frame_time_s": (timestamps - timestamps[0]).astype(np.float64) * 1e-9,
    }


def export_car_dataset(prepared_dir: Path, frozen_manifest: Path, output: Path) -> dict:
    prepared = prepared_dir.resolve()
    inventory_path = prepared / "manifest.json"
    inventory = json.loads(inventory_path.read_text())
    frozen = json.loads(frozen_manifest.read_text())
    inventory_hash = _sha256(inventory_path)
    if frozen["inventory_sha256"] != inventory_hash:
        raise ValueError("Frozen smoke labels were generated from a different prepared inventory")
    records = [r for r in inventory["recordings"] if r["role"] in {"clean_control", "smoke"}]
    expected_smoke = {r["id"] for r in records if r["role"] == "smoke"}
    smoke = {row["id"]: row for row in frozen["recordings"]}
    if smoke.keys() != expected_smoke or len(smoke) != len(frozen["recordings"]):
        raise ValueError("Frozen smoke labels must cover each prepared smoke recording exactly once")
    if not smoke or not any(r["role"] == "clean_control" for r in records):
        raise ValueError("Binary car export requires smoky recordings and clean controls")
    if output.exists():
        raise FileExistsError(output)

    schema = _schema()
    schema["arrays"]["source_domain"] = "scalar string: car_ros2"
    schema["known_clean_policy"] = (
        f"Clean controls use label 0 for finite returns within {CONFIG['range_m']} metres inclusive, "
        "regardless of alignment. Other clean returns are ignored. Smoke recordings preserve frozen labels."
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "dataset"
        chunks = []
        for record in records:
            name = record["id"]
            labels_path = None
            policy = "known_clean_valid_points_are_unimpacted"
            if name in smoke:
                row = smoke[name]
                labels_path = frozen_manifest.parent / row["file"]
                if row["source_files"] != record["source_files"] or _sha256(labels_path) != row["sha256"]:
                    raise ValueError(f"{name}: frozen label or MCAP provenance mismatch")
                policy = "preserve_frozen_smoke_and_ignore"
            destination = Path("chunks") / f"{name}.npz"
            written = _write_chunk(stage / destination, _read_recording(prepared, record, labels_path, frozen["topic"]))
            chunks.append({
                **written,
                "path": str(destination),
                "source_domain": "car_ros2",
                "session_id": prepared.name,
                "recording_id": name,
                "condition": record["role"],
                "layout": record["layout"],
                "source_files": record["source_files"],
                "label_policy": policy,
                "split": "unassigned",
            })
            print(f"{name}: {written['frames']} frames, {written['labels']}", flush=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "dataset_schema": "dataset_schema.json",
            "split_policy": "unassigned; keep whole recordings together",
            "prepared_inventory": {"path": str(inventory_path), "sha256": inventory_hash},
            "frozen_smoke_manifest": {"path": str(frozen_manifest.resolve()), "sha256": _sha256(frozen_manifest)},
            "chunks": chunks,
            "frames": sum(row["frames"] for row in chunks),
            "points": sum(row["points"] for row in chunks),
            "labels": {
                key: sum(row["labels"][key] for row in chunks)
                for key in ("unimpacted", "smoke_impacted", "ignore")
            },
        }
        for name, document in (("dataset_schema.json", schema), ("dataset_manifest.json", manifest)):
            (stage / name).write_text(json.dumps(document, indent=2) + "\n")
        stage.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("frozen_smoke_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export_car_dataset(args.prepared, args.frozen_smoke_manifest, args.output)


if __name__ == "__main__":
    main()
