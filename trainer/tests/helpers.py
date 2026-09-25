from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def make_dataset(root: Path, *, frames: int = 40, points_per_frame: int = 4) -> Path:
    chunks = root / "chunks"
    chunks.mkdir(parents=True)
    point_count = frames * points_per_frame
    frame_index = np.repeat(np.arange(frames, dtype=np.int32), points_per_frame)
    point_in_frame = np.tile(np.arange(points_per_frame), frames)
    xyz = np.column_stack(
        (
            1.0 + point_in_frame * 0.2,
            frame_index.astype(np.float32) * 0.01,
            np.zeros(point_count, dtype=np.float32),
        )
    ).astype(np.float32)
    intensity = np.where(point_in_frame == 1, 120.0, 10.0).astype(np.float32)
    pattern = np.asarray([0, 1, 255, 0], dtype=np.uint8)
    label = np.tile(pattern, frames)
    chunk_path = chunks / "tiny.npz"
    np.savez_compressed(
        chunk_path,
        schema_version=np.array("1.0"),
        session_id=np.array("tiny_session"),
        source_domain=np.array("stationary_ros2"),
        recording_id=np.array("tiny_recording"),
        condition=np.array("smoke"),
        xyz=xyz,
        intensity=intensity,
        tag=np.zeros(point_count, dtype=np.uint8),
        line=np.zeros(point_count, dtype=np.uint8),
        point_offset_s=np.zeros(point_count, dtype=np.float32),
        label=label,
        frame_index=frame_index,
        frame_ptr=np.arange(0, point_count + 1, points_per_frame, dtype=np.int64),
        frame_time_s=np.arange(frames, dtype=np.float64) * 0.1,
    )
    labels = {
        "unimpacted": int(np.count_nonzero(label == 0)),
        "smoke_impacted": int(np.count_nonzero(label == 1)),
        "ignore": int(np.count_nonzero(label == 255)),
    }
    manifest = {
        "schema_version": "1.0",
        "status": "complete",
        "chunks": [
            {
                "source_domain": "stationary_ros2",
                "session_id": "tiny_session",
                "recording_id": "tiny_recording",
                "condition": "smoke",
                "path": "chunks/tiny.npz",
                "sha256": _sha256(chunk_path),
                "frames": frames,
                "points": point_count,
                "labels": labels,
            }
        ],
        "frames": frames,
        "points": point_count,
        "labels": labels,
    }
    manifest_path = root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path

