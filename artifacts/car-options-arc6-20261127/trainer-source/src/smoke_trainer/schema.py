from __future__ import annotations

import hashlib
from pathlib import Path

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_chunk(arrays: dict[str, np.ndarray], source: str | Path) -> None:
    """Validate the canonical unified-dataset contract used by the labeler."""
    names = set(arrays)
    expected_names = set(ARRAY_NAMES)
    if names != expected_names:
        missing = sorted(expected_names - names)
        extra = sorted(names - expected_names)
        raise ValueError(f"{source}: schema fields differ; missing={missing}, extra={extra}")

    for name in ARRAY_NAMES[:5]:
        value = arrays[name]
        if value.ndim != 0 or value.dtype.kind not in "US":
            raise ValueError(f"{source}: {name} must be a scalar string")
    if str(arrays["schema_version"].item()) != SCHEMA_VERSION:
        raise ValueError(f"{source}: unsupported schema version")

    xyz = arrays["xyz"]
    point_count = len(xyz)
    point_arrays = {
        "xyz": (np.dtype(np.float32), (point_count, 3)),
        "intensity": (np.dtype(np.float32), (point_count,)),
        "tag": (np.dtype(np.uint8), (point_count,)),
        "line": (np.dtype(np.uint8), (point_count,)),
        "point_offset_s": (np.dtype(np.float32), (point_count,)),
        "label": (np.dtype(np.uint8), (point_count,)),
        "frame_index": (np.dtype(np.int32), (point_count,)),
    }
    for name, (dtype, shape) in point_arrays.items():
        value = arrays[name]
        if value.dtype != dtype or value.shape != shape:
            raise ValueError(
                f"{source}: {name} must be {dtype} {shape}, got {value.dtype} {value.shape}"
            )

    frame_ptr = arrays["frame_ptr"]
    frame_time_s = arrays["frame_time_s"]
    if frame_ptr.dtype != np.int64 or frame_ptr.ndim != 1 or len(frame_ptr) == 0:
        raise ValueError(f"{source}: frame_ptr must be nonempty int64 [F+1]")
    if frame_time_s.dtype != np.float64 or frame_time_s.shape != (len(frame_ptr) - 1,):
        raise ValueError(f"{source}: frame_time_s must be float64 [F]")
    if frame_ptr[0] != 0 or frame_ptr[-1] != point_count or np.any(np.diff(frame_ptr) < 0):
        raise ValueError(f"{source}: frame_ptr does not monotonically span all points")
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


def load_chunk(path: Path, *, validate: bool = True) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if validate:
        validate_chunk(arrays, path)
    return arrays

