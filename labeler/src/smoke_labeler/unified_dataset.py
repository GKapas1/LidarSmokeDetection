from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "1.0"
LABEL_VALUES = np.asarray([0, 1, 255], dtype=np.uint8)
ARRAY_NAMES = (
    "schema_version",
    "session_id",
    "source_domain",
    "recording_id",
    "condition",
    "xyz",
    "intensity",
    "tag",
    "line",
    "point_offset_s",
    "label",
    "frame_index",
    "frame_ptr",
    "frame_time_s",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_" for character in value
    )


def _label_counts(label: np.ndarray) -> dict[str, int]:
    return {
        "unimpacted": int(np.count_nonzero(label == 0)),
        "smoke_impacted": int(np.count_nonzero(label == 1)),
        "ignore": int(np.count_nonzero(label == 255)),
    }


def _validate_arrays(arrays: dict[str, np.ndarray], source: str | Path) -> None:
    names = tuple(arrays)
    if set(names) != set(ARRAY_NAMES):
        missing = sorted(set(ARRAY_NAMES) - set(names))
        extra = sorted(set(names) - set(ARRAY_NAMES))
        raise ValueError(f"{source}: schema fields differ; missing={missing}, extra={extra}")
    for name in ARRAY_NAMES[:5]:
        value = arrays[name]
        if value.ndim != 0 or value.dtype.kind not in "US":
            raise ValueError(f"{source}: {name} must be a scalar string")
    if str(arrays["schema_version"].item()) != SCHEMA_VERSION:
        raise ValueError(f"{source}: unsupported unified schema version")

    xyz = arrays["xyz"]
    point_count = len(xyz)
    expected = {
        "xyz": (np.dtype(np.float32), (point_count, 3)),
        "intensity": (np.dtype(np.float32), (point_count,)),
        "tag": (np.dtype(np.uint8), (point_count,)),
        "line": (np.dtype(np.uint8), (point_count,)),
        "point_offset_s": (np.dtype(np.float32), (point_count,)),
        "label": (np.dtype(np.uint8), (point_count,)),
        "frame_index": (np.dtype(np.int32), (point_count,)),
    }
    for name, (dtype, shape) in expected.items():
        value = arrays[name]
        if value.dtype != dtype or value.shape != shape:
            raise ValueError(
                f"{source}: {name} must be {dtype} {shape}, got {value.dtype} {value.shape}"
            )

    frame_ptr = arrays["frame_ptr"]
    frame_time_s = arrays["frame_time_s"]
    if frame_ptr.dtype != np.int64 or frame_ptr.ndim != 1:
        raise ValueError(f"{source}: frame_ptr must be int64 [F+1]")
    if frame_time_s.dtype != np.float64 or frame_time_s.shape != (len(frame_ptr) - 1,):
        raise ValueError(f"{source}: frame_time_s must be float64 [F]")
    if len(frame_ptr) == 0 or frame_ptr[0] != 0 or frame_ptr[-1] != point_count:
        raise ValueError(f"{source}: frame_ptr does not span all points")
    if np.any(np.diff(frame_ptr) < 0):
        raise ValueError(f"{source}: frame_ptr is not monotonic")
    if len(frame_time_s) > 1 and np.any(np.diff(frame_time_s) <= 0):
        raise ValueError(f"{source}: frame_time_s is not strictly increasing")
    if not np.isfinite(frame_time_s).all():
        raise ValueError(f"{source}: frame_time_s contains non-finite values")
    for name in ("xyz", "intensity", "point_offset_s"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{source}: {name} contains non-finite values")
    if not np.isin(arrays["label"], LABEL_VALUES).all():
        raise ValueError(f"{source}: labels must be 0, 1, or 255")
    for frame in range(len(frame_time_s)):
        start, end = int(frame_ptr[frame]), int(frame_ptr[frame + 1])
        if not np.all(arrays["frame_index"][start:end] == frame):
            raise ValueError(f"{source}: frame_index disagrees with frame_ptr at frame {frame}")


def load_unified_chunk(path: str | Path, *, validate: bool = True) -> dict[str, np.ndarray]:
    """Load either source domain through the canonical trainer-facing contract."""
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    if validate:
        _validate_arrays(arrays, source)
    return arrays


def _write_chunk(output: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    _validate_arrays(arrays, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    labels = arrays["label"]
    return {
        "path": str(output),
        "sha256": _sha256(output),
        "frames": int(len(arrays["frame_time_s"])),
        "points": int(len(labels)),
        "labels": _label_counts(labels),
    }


def _metadata(
    session_id: str, source_domain: str, recording_id: str, condition: str
) -> dict[str, np.ndarray]:
    return {
        "schema_version": np.array(SCHEMA_VERSION),
        "session_id": np.array(session_id),
        "source_domain": np.array(source_domain),
        "recording_id": np.array(recording_id),
        "condition": np.array(condition),
    }


def _stationary_sources(
    dataset_dir: Path,
) -> Iterable[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    summary_path = dataset_dir / "dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    session_id = str(summary["session_id"])
    for recording in summary["recordings"]:
        source = dataset_dir / recording["data_file"]
        condition = str(recording["condition"])
        with np.load(source, allow_pickle=False) as data:
            frame_time_s = data["frame_time_s"].astype(np.float64, copy=False)
            if len(frame_time_s):
                frame_time_s = frame_time_s - frame_time_s[0]
            label = data["label"].astype(np.uint8, copy=False)
            label_policy = "preserve_final_label"
            if condition == "clean_control":
                # The clean control's class is known from the acquisition protocol.
                # Retain unsupported/invalid points as ignore, and convert measured
                # pseudo-label false positives to hard-negative targets.
                label = label.copy()
                label[label == 1] = 0
                label_policy = "known_clean_valid_points_are_unimpacted"
            arrays = {
                **_metadata(
                    session_id,
                    "stationary_ros2",
                    str(recording["recording"]),
                    condition,
                ),
                "xyz": data["xyz"].astype(np.float32, copy=False),
                "intensity": data["reflectivity"].astype(np.float32, copy=False),
                "tag": data["tag"].astype(np.uint8, copy=False),
                "line": data["line"].astype(np.uint8, copy=False),
                "point_offset_s": data["point_offset_s"].astype(np.float32, copy=False),
                "label": label,
                "frame_index": data["frame_index"].astype(np.int32, copy=False),
                "frame_ptr": data["frame_ptr"].astype(np.int64, copy=False),
                "frame_time_s": frame_time_s,
            }
        info = {
            "source_domain": "stationary_ros2",
            "session_id": session_id,
            "recording_id": str(recording["recording"]),
            "condition": condition,
            "label_policy": label_policy,
            "source_path": str(source.resolve()),
            "split": "unassigned",
        }
        yield info, arrays


def _grandtour_sources(
    manifest_path: Path,
) -> Iterable[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"{manifest_path}: GrandTour training manifest is not complete")
    session_id = str(manifest["session_id"])
    for chunk in manifest["chunks"]:
        chunk_index = int(chunk["chunk_index"])
        source = Path(chunk["path"])
        if not source.is_absolute():
            source = manifest_path.parent / source
        with np.load(source, allow_pickle=False) as data:
            frame_time_ns = data["frame_time_ns"].astype(np.int64, copy=False)
            start_ns = int(frame_time_ns[0]) if len(frame_time_ns) else 0
            arrays = {
                **_metadata(session_id, "grandtour_ros1", f"chunk_{chunk_index:03d}", "smoke"),
                "xyz": data["xyz"].astype(np.float32, copy=False),
                "intensity": data["intensity"].astype(np.float32, copy=False),
                "tag": data["tag"].astype(np.uint8, copy=False),
                "line": data["line"].astype(np.uint8, copy=False),
                "point_offset_s": (
                    data["relative_time_ns"].astype(np.float64) * 1.0e-9
                ).astype(np.float32),
                "label": data["label"].astype(np.uint8, copy=False),
                "frame_index": data["frame_index"].astype(np.int32, copy=False),
                "frame_ptr": data["frame_ptr"].astype(np.int64, copy=False),
                "frame_time_s": (frame_time_ns - start_ns).astype(np.float64) * 1.0e-9,
            }
        info = {
            "source_domain": "grandtour_ros1",
            "session_id": session_id,
            "recording_id": f"chunk_{chunk_index:03d}",
            "condition": "smoke",
            "source_path": str(source.resolve()),
            "source_time_selection_s": chunk.get("time_selection_s"),
            "split": "unassigned",
        }
        yield info, arrays


def _schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "description": "Canonical frame-preserving Livox MID-360 per-point training dataset",
        "coordinate_frame": "original LiDAR sensor frame for each recording",
        "arrays": {
            "schema_version": "scalar string",
            "session_id": "scalar string",
            "source_domain": "scalar string: stationary_ros2 or grandtour_ros1",
            "recording_id": "scalar string",
            "condition": "scalar string",
            "xyz": "float32 [P,3], metres",
            "intensity": "float32 [P], raw Livox reflectivity/intensity values",
            "tag": "uint8 [P], raw Livox tag",
            "line": "uint8 [P], raw Livox laser line",
            "point_offset_s": "float32 [P], point time relative to its frame timestamp",
            "label": "uint8 [P]: 0 unimpacted, 1 smoke impacted, 255 ignore",
            "frame_index": "int32 [P], zero-based frame within this chunk",
            "frame_ptr": "int64 [F+1], point slice for each frame",
            "frame_time_s": "float64 [F], frame timestamp relative to chunk/recording start",
        },
        "model_input_arrays": ["xyz", "intensity", "tag", "line", "point_offset_s"],
        "target_array": "label",
        "loss_mask": "label != 255",
        "known_clean_policy": (
            "For condition=clean_control, valid pseudo-label false positives are exported as "
            "unimpacted; existing ignore labels remain ignored."
        ),
        "normalization": "No normalization is applied; fit preprocessing on the training split only.",
        "forbidden_model_inputs": [
            "source_domain",
            "condition",
            "automatic_label",
            "world_xyz",
            "nearest_reference_distance_m",
            "label_confidence",
        ],
    }


def export_unified_dataset(
    stationary_datasets: Iterable[str | Path],
    grandtour_manifests: Iterable[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    chunks_dir = output / "chunks"
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    chunks_dir.mkdir(parents=True, exist_ok=True)

    sources = itertools.chain(
        itertools.chain.from_iterable(
            _stationary_sources(Path(value).expanduser().resolve())
            for value in stationary_datasets
        ),
        itertools.chain.from_iterable(
            _grandtour_sources(Path(value).expanduser().resolve())
            for value in grandtour_manifests
        ),
    )

    manifest_chunks: list[dict[str, Any]] = []
    totals = {"frames": 0, "points": 0, "unimpacted": 0, "smoke_impacted": 0, "ignore": 0}
    for info, arrays in sources:
        name = "__".join(
            _safe_name(str(info[key])) for key in ("source_domain", "session_id", "recording_id")
        )
        destination = chunks_dir / f"{name}.npz"
        written = _write_chunk(destination, arrays)
        written["path"] = str(destination.relative_to(output))
        row = {**info, **written}
        manifest_chunks.append(row)
        totals["frames"] += written["frames"]
        totals["points"] += written["points"]
        for key, value in written["labels"].items():
            totals[key] += value

    if not manifest_chunks:
        raise ValueError("at least one --stationary or --grandtour source is required")

    schema_path = output / "dataset_schema.json"
    schema_path.write_text(json.dumps(_schema(), indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "dataset_schema": schema_path.name,
        "split_policy": "unassigned; assign groups before training to prevent recording leakage",
        "chunks": manifest_chunks,
        "frames": totals.pop("frames"),
        "points": totals.pop("points"),
        "labels": totals,
    }
    manifest_path = output / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
