from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from .bag import read_bag_points, resolve_mcap_sources
from .core import IGNORE, SMOKE_IMPACTED, UNAFFECTED

DEFAULT_CONFIG = {'sampling': {'reference_point_stride': 2,
              'calibration_point_stride': 2,
              'target_point_stride': 1,
              'max_reference_points': 8000000,
              'max_calibration_points': 8000000,
              'max_target_points': 30000000},
 'reference': {'angular_resolution_deg': 0.3,
               'min_returns_per_cell': 3,
               'max_cell_depth_spread_m': 0.12},
 'calibration': {'false_positive_quantile': 0.999,
                 'normal_quantile': 0.99,
                 'minimum_normal_residual_m': 0.02,
                 'minimum_early_return_m': 0.05,
                 'maximum_usable_early_return_m': 0.25,
                 'range_band_edges_m': [0.0, 2.0, 4.0, 6.0, 10.0, 20.0, 100.0],
                 'minimum_calibration_samples_per_band': 500},
 'output': {'preview_max_points': 250000},
 'dataset': {'sequence_length': 5,
             'maximum_frame_gap_s': 0.25,
             'minimum_input_range_m': 0.1,
             'maximum_input_range_m': 30.0,
             'clean_control_calibration_fraction': 0.5}}

def load_config(path: str | Path) -> dict[str, dict[str, Any]]:
    import copy
    import tomllib

    config = copy.deepcopy(DEFAULT_CONFIG)
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        supplied = tomllib.load(stream)
    for section, values in supplied.items():
        if section not in config or not isinstance(values, dict):
            raise ValueError(f"Unknown or invalid config section [{section}]")
        config[section].update(values)

    return config

def _write_preview(path: Path, xyz: np.ndarray, labels: np.ndarray, max_points: int) -> None:
    if len(xyz) > max_points:
        indices = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
        xyz = xyz[indices]
        labels = labels[indices]
    colors = np.full((len(labels), 3), (115, 115, 115), dtype=np.uint8)
    colors[labels == UNAFFECTED] = (35, 180, 70)
    colors[labels == SMOKE_IMPACTED] = (230, 40, 40)
    colors[labels == IGNORE] = (115, 115, 115)
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(xyz)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("property uchar label\nend_header\n")
        for point, color, label in zip(xyz, colors, labels):
            stream.write(
                f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {int(label)}\n"
            )

def inspect_bag(value: str | Path, topic: str | None = None) -> dict[str, Any]:
    files = resolve_mcap_sources(value)
    points = read_bag_points(value, topic=topic, point_stride=20, max_points=100_000)
    result = {
        "source_files": [str(p) for p in files],
        "selected_topic": points.topic,
        "message_type": points.message_type,
        "reader_backend": points.reader_backend,
        "sampled_points": int(len(points.xyz)),
        "sampled_complete_frames": int(len(points.frame_time_s)) if points.frame_time_s is not None else 0,
        "inspection_hit_100000_point_cap": bool(points.truncated),
        "time_span_s": float(points.time_s[-1] - points.time_s[0]) if len(points.time_s) > 1 else 0.0,
        "range_m": {
            "min": float(np.nanmin(np.linalg.norm(points.xyz, axis=1))),
            "median": float(np.nanmedian(np.linalg.norm(points.xyz, axis=1))),
            "max": float(np.nanmax(np.linalg.norm(points.xyz, axis=1))),
        },
        "has_reflectivity": bool(np.isfinite(points.reflectivity).any()),
        "has_nonzero_tag": bool(np.any(points.tag)),
        "has_nonzero_line": bool(np.any(points.line)),
    }
    print(json.dumps(result, indent=2))
    return result

