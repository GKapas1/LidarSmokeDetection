from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tomllib
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .geometry import transform_points
from .grandtour import (
    GrandTourPaths,
    iter_paired_livox_frames,
    load_dlio_trajectory,
    load_static_transform,
    resolve_grandtour_paths,
)


STRUCTURE = np.uint8(0)
SMOKE_CANDIDATE = np.uint8(1)
EXCLUDED = np.uint8(254)
UNKNOWN = np.uint8(255)

REASON_LABELED = np.uint8(0)
REASON_INVALID_INPUT = np.uint8(1)
REASON_OUTSIDE_COVERAGE = np.uint8(2)
REASON_AMBIGUOUS_DISTANCE = np.uint8(3)
REASON_EXCLUDED_REGION = np.uint8(4)
REASON_OUTSIDE_CONFIRMED_SMOKE_REGION = np.uint8(5)
REASON_REVIEWED_BACKGROUND = np.uint8(6)
REASON_HARD_NEGATIVE = np.uint8(7)
REASON_ISOLATED_RETURN = np.uint8(8)


def _load_toml(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    with source.open("rb") as stream:
        return source, tomllib.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sensor_transform(paths: GrandTourPaths) -> np.ndarray:
    transform_box_hesai = load_static_transform(paths.tf_minimal, "box_base", "hesai_lidar")
    transform_box_livox = load_static_transform(paths.tf_minimal, "box_base", "livox_lidar")
    return np.linalg.inv(transform_box_hesai) @ transform_box_livox


def _inside_box(xyz: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    return np.all((xyz >= minimum) & (xyz <= maximum), axis=1)


def _isolated_return_mask(
    xyz: np.ndarray, valid: np.ndarray, maximum_neighbor_distance_m: float
) -> np.ndarray:
    isolated = np.zeros(len(xyz), dtype=bool)
    if maximum_neighbor_distance_m <= 0:
        return isolated
    valid_index = np.flatnonzero(valid)
    if len(valid_index) < 2:
        isolated[valid_index] = True
        return isolated
    nearest_other = cKDTree(xyz[valid_index]).query(
        xyz[valid_index], k=2, workers=-1
    )[0][:, 1]
    isolated[valid_index] = nearest_other > maximum_neighbor_distance_m
    return isolated


def _enabled_regions(
    config: dict[str, Any], key: str = "excluded_regions"
) -> list[dict[str, Any]]:
    regions = []
    for region in config.get(key, []):
        if not bool(region.get("enabled", True)):
            continue
        minimum = np.asarray(region["min_m"], dtype=np.float64)
        maximum = np.asarray(region["max_m"], dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,) or np.any(minimum >= maximum):
            raise ValueError(f"Invalid {key} entry {region.get('name', '<unnamed>')!r}")
        regions.append(region)
    return regions


def classify_reference_distances(
    world_xyz: np.ndarray,
    nearest_distance_m: np.ndarray,
    input_valid: np.ndarray,
    crop_min: np.ndarray,
    crop_max: np.ndarray,
    coverage_margin_m: float,
    structure_threshold_m: float,
    smoke_threshold_m: float,
    excluded_regions: list[dict[str, Any]],
    confirmed_smoke_regions: list[dict[str, Any]] | None = None,
    reviewed_background_unimpacted: bool = False,
    hard_negative_regions: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classify points and retain why any point was withheld from training."""
    if coverage_margin_m < 0:
        raise ValueError("coverage_margin_m must be non-negative")
    coverage_min = crop_min + coverage_margin_m
    coverage_max = crop_max - coverage_margin_m
    if np.any(coverage_min >= coverage_max):
        raise ValueError("coverage_margin_m leaves no usable reference volume")

    inside_coverage = _inside_box(world_xyz, coverage_min, coverage_max)
    coverage = input_valid & inside_coverage
    automatic = np.full(len(world_xyz), UNKNOWN, dtype=np.uint8)
    automatic[coverage & (nearest_distance_m <= structure_threshold_m)] = STRUCTURE
    automatic[coverage & (nearest_distance_m >= smoke_threshold_m)] = SMOKE_CANDIDATE

    reason = np.full(len(world_xyz), REASON_INVALID_INPUT, dtype=np.uint8)
    reason[input_valid & ~inside_coverage] = REASON_OUTSIDE_COVERAGE
    reason[coverage] = REASON_AMBIGUOUS_DISTANCE
    reason[(automatic == STRUCTURE) | (automatic == SMOKE_CANDIDATE)] = REASON_LABELED

    final = automatic.copy()
    human_verified = np.zeros(len(world_xyz), dtype=bool)
    if confirmed_smoke_regions:
        confirmed_smoke = np.zeros(len(world_xyz), dtype=bool)
        for region in confirmed_smoke_regions:
            inside = coverage & _inside_box(
                world_xyz,
                np.asarray(region["min_m"], dtype=np.float64),
                np.asarray(region["max_m"], dtype=np.float64),
            )
            confirmed_smoke |= inside
            reviewed_threshold = float(region.get("minimum_distance_m", smoke_threshold_m))
            reviewed_smoke = inside & (nearest_distance_m >= reviewed_threshold)
            final[reviewed_smoke] = SMOKE_CANDIDATE
            reason[reviewed_smoke] = REASON_LABELED
            human_verified[reviewed_smoke] = True
        outside_smoke = coverage & ~confirmed_smoke
        if reviewed_background_unimpacted:
            final[outside_smoke] = STRUCTURE
            reason[outside_smoke] = REASON_REVIEWED_BACKGROUND
            human_verified[outside_smoke] = True
        else:
            rejected = (final == SMOKE_CANDIDATE) & outside_smoke
            final[rejected] = UNKNOWN
            reason[rejected] = REASON_OUTSIDE_CONFIRMED_SMOKE_REGION

    hard_negative_id = np.full(len(world_xyz), -1, dtype=np.int16)
    for index, region in enumerate(hard_negative_regions or []):
        inside = coverage & _inside_box(
            world_xyz,
            np.asarray(region["min_m"], dtype=np.float64),
            np.asarray(region["max_m"], dtype=np.float64),
        )
        final[inside] = STRUCTURE
        reason[inside] = REASON_HARD_NEGATIVE
        human_verified[inside] = True
        hard_negative_id[inside] = index

    region_id = np.full(len(world_xyz), -1, dtype=np.int16)
    for index, region in enumerate(excluded_regions):
        inside = coverage & _inside_box(
            world_xyz,
            np.asarray(region["min_m"], dtype=np.float64),
            np.asarray(region["max_m"], dtype=np.float64),
        )
        final[inside] = EXCLUDED
        reason[inside] = REASON_EXCLUDED_REGION
        human_verified[inside] = True
        region_id[inside] = index
    return final, automatic, reason, region_id, human_verified, hard_negative_id


def _write_reference_ply(path: Path, xyz: np.ndarray, support: np.ndarray) -> None:
    maximum = max(int(support.max(initial=1)), 1)
    scaled = np.clip(np.log1p(support) / np.log1p(maximum), 0.0, 1.0)
    colors = np.column_stack(
        (60 + 50 * scaled, 100 + 150 * scaled, 220 - 100 * scaled)
    ).astype(np.uint8)
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(xyz)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("property uint support_scans\nend_header\n")
        for point, color, count in zip(xyz, colors, support):
            stream.write(
                f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {int(count)}\n"
            )


def build_grandtour_reference(
    session_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config_source, config = _load_toml(config_path)
    paths = resolve_grandtour_paths(session_dir)
    if paths.session_id != str(config["session_id"]):
        raise ValueError("Reference configuration session_id does not match the source bags")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    crop_min = np.asarray(config["crop"]["min_m"], dtype=np.float64)
    crop_max = np.asarray(config["crop"]["max_m"], dtype=np.float64)
    if crop_min.shape != (3,) or crop_max.shape != (3,) or np.any(crop_min >= crop_max):
        raise ValueError("Invalid reference crop")
    voxel_size = float(config["voxel"]["size_m"])
    minimum_support = int(config["voxel"]["minimum_distinct_scans"])
    if voxel_size <= 0 or minimum_support < 1:
        raise ValueError("Invalid voxel configuration")

    sample = config["sampling"]
    frame_stride = int(sample["frame_stride"])
    point_stride = int(sample["point_stride"])
    min_range = float(sample["minimum_range_m"])
    max_range = float(sample["maximum_range_m"])
    time_cfg = config["time"]
    start_s = float(time_cfg["reference_start_s"])
    end_s = float(time_cfg["reference_end_s"])
    trajectory = load_dlio_trajectory(paths.dlio)
    transform_hesai_livox = _sensor_transform(paths)
    session_origin_ns: int | None = None
    voxel_chunks: list[np.ndarray] = []
    used_frames = 0
    sampled_points = 0
    retained_points = 0
    regions = _enabled_regions(config)

    for pair in iter_paired_livox_frames(paths):
        if session_origin_ns is None:
            session_origin_ns = pair.raw.header_time_ns
        relative_s = (pair.raw.header_time_ns - session_origin_ns) / 1e9
        if relative_s < start_s or relative_s >= end_s or pair.raw.frame_index % frame_stride:
            continue
        if not trajectory.time_ns[0] <= pair.raw.header_time_ns <= trajectory.time_ns[-1]:
            continue
        raw_xyz = pair.raw.xyz[::point_stride]
        xyz = pair.undistorted_xyz[::point_stride]
        ranges = np.linalg.norm(raw_xyz, axis=1)
        valid = (
            np.isfinite(xyz).all(axis=1)
            & np.isfinite(ranges)
            & (ranges >= min_range)
            & (ranges <= max_range)
        )
        sampled_points += len(xyz)
        transform_map_livox = trajectory.at(pair.raw.header_time_ns) @ transform_hesai_livox
        world = transform_points(transform_map_livox, xyz[valid])
        keep = _inside_box(world, crop_min, crop_max)
        for region in regions:
            keep &= ~_inside_box(
                world,
                np.asarray(region["min_m"], dtype=np.float64),
                np.asarray(region["max_m"], dtype=np.float64),
            )
        world = world[keep]
        retained_points += len(world)
        if len(world):
            voxel = np.floor((world - crop_min) / voxel_size).astype(np.int32)
            voxel_chunks.append(np.unique(voxel, axis=0))
        used_frames += 1

    if not voxel_chunks:
        raise RuntimeError("Reference selection contains no points")
    all_voxels = np.concatenate(voxel_chunks)
    voxels, support = np.unique(all_voxels, axis=0, return_counts=True)
    stable = support >= minimum_support
    voxels = voxels[stable]
    support = support[stable].astype(np.uint32)
    xyz = (crop_min + (voxels.astype(np.float64) + 0.5) * voxel_size).astype(np.float32)

    npz_path = output / "arc5_clean_reference.npz"
    np.savez_compressed(
        npz_path,
        schema_version=np.array("1.0"),
        session_id=np.array(paths.session_id),
        frame=np.array("arc5_dlio_map"),
        xyz=xyz,
        support_scans=support,
        voxel_size_m=np.float64(voxel_size),
        crop_min_m=crop_min,
        crop_max_m=crop_max,
        source_config_sha256=np.array(_sha256(config_source)),
    )
    ply_path = output / "arc5_clean_reference.ply"
    _write_reference_ply(ply_path, xyz, support)
    result = {
        "schema_version": "1.0",
        "session_id": paths.session_id,
        "frame": "arc5_dlio_map",
        "configuration": str(config_source),
        "configuration_sha256": _sha256(config_source),
        "source_files": {
            "raw_livox": str(paths.raw_livox),
            "undistorted_livox": str(paths.undistorted_livox),
            "dlio": str(paths.dlio),
            "tf_minimal": str(paths.tf_minimal),
        },
        "time_selection_s": {"start": start_s, "end": end_s},
        "validation_start_s": float(time_cfg["validation_start_s"]),
        "crop_min_m": crop_min.tolist(),
        "crop_max_m": crop_max.tolist(),
        "voxel_size_m": voxel_size,
        "minimum_distinct_scans": minimum_support,
        "input_frames": used_frames,
        "sampled_points": sampled_points,
        "points_inside_crop_before_voxelization": retained_points,
        "candidate_voxels": int(len(stable)),
        "stable_reference_voxels": int(len(xyz)),
        "excluded_regions": regions,
        "reference_npz": str(npz_path),
        "reference_ply": str(ply_path),
    }
    (output / "reference_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _alignment_matrix(config_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    _, config = _load_toml(config_path)
    matrix = np.asarray(config["matrix_row_major"], dtype=np.float64)
    if matrix.size != 16:
        raise ValueError("Alignment matrix must contain 16 values")
    matrix = matrix.reshape(4, 4)
    if not bool(config.get("validated", False)):
        raise ValueError("Alignment configuration has not been validated")
    return matrix, config


def _write_label_preview(path: Path, xyz: np.ndarray, labels: np.ndarray, maximum: int) -> None:
    if len(xyz) > maximum:
        index = np.linspace(0, len(xyz) - 1, maximum, dtype=np.int64)
        xyz, labels = xyz[index], labels[index]
    colors = np.full((len(labels), 3), (120, 120, 120), dtype=np.uint8)
    colors[labels == STRUCTURE] = (40, 190, 70)
    colors[labels == SMOKE_CANDIDATE] = (235, 45, 45)
    colors[labels == EXCLUDED] = (145, 70, 200)
    colors[labels == UNKNOWN] = (130, 130, 130)
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


def label_grandtour_pilot(
    session_dir: str | Path,
    config_path: str | Path,
    reference_npz: str | Path,
    output_dir: str | Path,
    time_range_s: tuple[float, float] | None = None,
) -> dict[str, Any]:
    config_source, config = _load_toml(config_path)
    paths = resolve_grandtour_paths(session_dir)
    if paths.session_id != str(config["session_id"]):
        raise ValueError("Pilot configuration session_id does not match the source bags")
    config_dir = config_source.parent
    alignment_path = config_dir / str(config["alignment_config"])
    transform_arc5_arc6, alignment_config = _alignment_matrix(alignment_path)
    reference_path = Path(reference_npz).expanduser().resolve()
    with np.load(reference_path) as reference:
        reference_xyz = reference["xyz"].astype(np.float32)
        crop_min = reference["crop_min_m"].astype(np.float64)
        crop_max = reference["crop_max_m"].astype(np.float64)
        reference_session = str(reference["session_id"])
    if reference_session != str(config["reference_session_id"]):
        raise ValueError("Pilot and reference session IDs do not agree")
    tree = cKDTree(reference_xyz)

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectory = load_dlio_trajectory(paths.dlio)
    transform_hesai_livox = _sensor_transform(paths)
    if time_range_s is None:
        start_s = float(config["time"]["start_s"])
        end_s = float(config["time"]["end_s"])
    else:
        start_s, end_s = map(float, time_range_s)
    if not 0 <= start_s < end_s:
        raise ValueError("Require 0 <= start_s < end_s")
    min_range = float(config["input"]["minimum_range_m"])
    max_range = float(config["input"]["maximum_range_m"])
    coverage_margin = float(config["input"].get("coverage_margin_m", 0.0))
    isolation_threshold = float(
        config["input"].get("maximum_isolated_neighbor_distance_m", 0.0)
    )
    structure_threshold = float(config["labeling"]["structure_max_distance_m"])
    smoke_threshold = float(config["labeling"]["smoke_min_distance_m"])
    if not 0 < structure_threshold < smoke_threshold:
        raise ValueError("Require 0 < structure threshold < smoke threshold")
    regions = _enabled_regions(config)
    confirmed_smoke_regions = _enabled_regions(config, "confirmed_smoke_regions")
    hard_negative_regions = _enabled_regions(config, "hard_negative_regions")
    reviewed_background = bool(
        config.get("human_review", {}).get("label_background_unimpacted", False)
    )

    chunks: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "sensor_xyz", "world_xyz", "intensity", "tag", "line", "point_time_ns",
            "frame_index", "source_point_index", "nearest_reference_distance_m", "label",
            "automatic_label", "label_reason", "exclusion_region_id",
            "human_verified", "hard_negative_region_id",
        )
    }
    frame_ptr = [0]
    frame_time_ns: list[int] = []
    session_origin_ns: int | None = None
    skipped_pose_frames = 0
    isolated_returns = 0
    for pair in iter_paired_livox_frames(paths):
        if session_origin_ns is None:
            session_origin_ns = pair.raw.header_time_ns
        relative_s = (pair.raw.header_time_ns - session_origin_ns) / 1e9
        if relative_s < start_s or relative_s >= end_s:
            continue
        if not trajectory.time_ns[0] <= pair.raw.header_time_ns <= trajectory.time_ns[-1]:
            skipped_pose_frames += 1
            continue
        transform_arc5_livox = (
            transform_arc5_arc6 @ trajectory.at(pair.raw.header_time_ns) @ transform_hesai_livox
        )
        world = transform_points(transform_arc5_livox, pair.undistorted_xyz)
        ranges = np.linalg.norm(pair.raw.xyz, axis=1)
        input_valid = (
            np.isfinite(pair.raw.xyz).all(axis=1)
            & np.isfinite(pair.undistorted_xyz).all(axis=1)
            & np.isfinite(ranges)
            & (ranges >= min_range)
            & (ranges <= max_range)
        )
        isolated = _isolated_return_mask(
            pair.raw.xyz, input_valid, isolation_threshold
        )
        input_valid &= ~isolated
        isolated_returns += int(np.count_nonzero(isolated))
        coverage_min = crop_min + coverage_margin
        coverage_max = crop_max - coverage_margin
        if np.any(coverage_min >= coverage_max):
            raise ValueError("coverage_margin_m leaves no usable reference volume")
        coverage = input_valid & _inside_box(world, coverage_min, coverage_max)
        distance = np.full(len(world), np.nan, dtype=np.float32)
        if np.any(coverage):
            distance[coverage] = tree.query(world[coverage], workers=-1)[0].astype(np.float32)
        (
            labels,
            automatic,
            label_reason,
            region_id,
            human_verified,
            hard_negative_id,
        ) = classify_reference_distances(
            world,
            distance,
            input_valid,
            crop_min,
            crop_max,
            coverage_margin,
            structure_threshold,
            smoke_threshold,
            regions,
            confirmed_smoke_regions,
            reviewed_background,
            hard_negative_regions,
        )
        label_reason[isolated] = REASON_ISOLATED_RETURN

        chunks["sensor_xyz"].append(pair.raw.xyz.astype(np.float32, copy=False))
        chunks["world_xyz"].append(world)
        chunks["intensity"].append(pair.raw.intensity)
        chunks["tag"].append(pair.raw.tag)
        chunks["line"].append(pair.raw.line)
        chunks["point_time_ns"].append(pair.raw.point_time_ns)
        chunks["frame_index"].append(np.full(len(world), len(frame_time_ns), dtype=np.int32))
        chunks["source_point_index"].append(pair.raw.point_index)
        chunks["nearest_reference_distance_m"].append(distance)
        chunks["label"].append(labels)
        chunks["automatic_label"].append(automatic)
        chunks["label_reason"].append(label_reason)
        chunks["exclusion_region_id"].append(region_id)
        chunks["human_verified"].append(human_verified)
        chunks["hard_negative_region_id"].append(hard_negative_id)
        frame_time_ns.append(pair.raw.header_time_ns)
        frame_ptr.append(frame_ptr[-1] + len(world))

    if not frame_time_ns:
        raise RuntimeError("Pilot time selection contains no frames with pose coverage")
    arrays = {key: np.concatenate(values) for key, values in chunks.items()}
    arrays["frame_ptr"] = np.asarray(frame_ptr, dtype=np.int64)
    arrays["frame_time_ns"] = np.asarray(frame_time_ns, dtype=np.int64)
    arrays["schema_version"] = np.array("1.0")
    arrays["session_id"] = np.array(paths.session_id)
    arrays["reference_session_id"] = np.array(reference_session)
    arrays["reference_map_path"] = np.array(str(reference_path))
    arrays["alignment_config_sha256"] = np.array(_sha256(alignment_path))
    data_path = output / "arc6_pilot_labels.npz"
    np.savez_compressed(data_path, **arrays)
    preview_path = output / "arc6_pilot_labels_preview.ply"
    _write_label_preview(
        preview_path,
        arrays["world_xyz"],
        arrays["label"],
        int(config["output"]["preview_max_points"]),
    )
    smoke_path = output / "arc6_smoke_candidates.ply"
    smoke_mask = arrays["label"] == SMOKE_CANDIDATE
    _write_label_preview(
        smoke_path,
        arrays["world_xyz"][smoke_mask],
        arrays["label"][smoke_mask],
        int(np.count_nonzero(smoke_mask)),
    )
    excluded_path = output / "arc6_excluded_regions.ply"
    excluded_mask = arrays["label"] == EXCLUDED
    _write_label_preview(
        excluded_path,
        arrays["world_xyz"][excluded_mask],
        arrays["label"][excluded_mask],
        int(np.count_nonzero(excluded_mask)),
    )
    hard_negative_path = output / "arc6_hard_negatives.ply"
    hard_negative_mask = arrays["hard_negative_region_id"] >= 0
    _write_label_preview(
        hard_negative_path,
        arrays["world_xyz"][hard_negative_mask],
        arrays["label"][hard_negative_mask],
        int(np.count_nonzero(hard_negative_mask)),
    )
    isolated_path = output / "arc6_isolated_returns.ply"
    isolated_mask = arrays["label_reason"] == REASON_ISOLATED_RETURN
    _write_label_preview(
        isolated_path,
        arrays["world_xyz"][isolated_mask],
        arrays["label"][isolated_mask],
        int(np.count_nonzero(isolated_mask)),
    )
    labels = arrays["label"]
    counts = {
        "structure": int(np.count_nonzero(labels == STRUCTURE)),
        "smoke_candidate": int(np.count_nonzero(labels == SMOKE_CANDIDATE)),
        "excluded": int(np.count_nonzero(labels == EXCLUDED)),
        "unknown": int(np.count_nonzero(labels == UNKNOWN)),
    }
    finite_distance = arrays["nearest_reference_distance_m"]
    finite_distance = finite_distance[np.isfinite(finite_distance)]
    result = {
        "schema_version": "1.0",
        "session_id": paths.session_id,
        "reference_session_id": reference_session,
        "frames": len(frame_time_ns),
        "points": int(len(labels)),
        "time_selection_s": {"start": start_s, "end": end_s},
        "skipped_frames_outside_pose_coverage": skipped_pose_frames,
        "isolated_sensor_returns": isolated_returns,
        "maximum_isolated_neighbor_distance_m": isolation_threshold,
        "thresholds_m": {
            "structure_max": structure_threshold,
            "smoke_min": smoke_threshold,
        },
        "coverage_margin_m": coverage_margin,
        "coverage_min_m": (crop_min + coverage_margin).tolist(),
        "coverage_max_m": (crop_max - coverage_margin).tolist(),
        "labels": counts,
        "label_percent": {
            key: 100.0 * value / max(len(labels), 1) for key, value in counts.items()
        },
        "reference_distance_quantiles_m": {
            str(q): float(np.quantile(finite_distance, q))
            for q in (0.5, 0.9, 0.95, 0.99)
        },
        "excluded_regions": regions,
        "confirmed_smoke_regions": confirmed_smoke_regions,
        "hard_negative_regions": hard_negative_regions,
        "label_reviewed_background_unimpacted": reviewed_background,
        "configuration": str(config_source),
        "configuration_sha256": _sha256(config_source),
        "alignment_configuration": str(alignment_path),
        "alignment_method": alignment_config["method"],
        "reference_npz": str(reference_path),
        "labels_npz": str(data_path),
        "preview_ply": str(preview_path),
        "smoke_candidates_ply": str(smoke_path),
        "excluded_regions_ply": str(excluded_path),
        "hard_negatives_ply": str(hard_negative_path),
        "isolated_returns_ply": str(isolated_path),
    }
    (output / "pilot_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _time_chunks(start_s: float, end_s: float, duration_s: float) -> list[tuple[float, float]]:
    if not 0 <= start_s < end_s or duration_s <= 0:
        raise ValueError("Invalid full-session time or chunk duration")
    count = int(math.ceil((end_s - start_s) / duration_s))
    return [
        (start_s + index * duration_s, min(end_s, start_s + (index + 1) * duration_s))
        for index in range(count)
    ]


def label_grandtour_session(
    session_dir: str | Path,
    config_path: str | Path,
    reference_npz: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Label a full session in bounded time chunks."""
    config_source, config = _load_toml(config_path)
    start_s = float(config["time"]["start_s"])
    end_s = float(config["time"]["end_s"])
    chunk_duration_s = float(config["output"].get("chunk_duration_s", 60.0))
    chunks = _time_chunks(start_s, end_s, chunk_duration_s)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "arc6_full_manifest.json"
    summaries: list[dict[str, Any]] = []

    for index, (chunk_start, chunk_end) in enumerate(chunks):
        chunk_name = f"chunk_{index:03d}_{int(chunk_start):04d}_{int(chunk_end):04d}s"
        chunk_output = output / chunk_name
        summary = label_grandtour_pilot(
            session_dir,
            config_source,
            reference_npz,
            chunk_output,
            time_range_s=(chunk_start, chunk_end),
        )
        summaries.append(summary)
        partial = {
            "schema_version": "1.0",
            "status": "running",
            "session_id": str(config["session_id"]),
            "configuration": str(config_source),
            "configuration_sha256": _sha256(config_source),
            "reference_npz": str(Path(reference_npz).expanduser().resolve()),
            "chunk_duration_s": chunk_duration_s,
            "completed_chunks": index + 1,
            "planned_chunks": len(chunks),
            "chunks": summaries,
        }
        manifest_path.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")

    label_totals = {
        key: sum(chunk["labels"][key] for chunk in summaries)
        for key in ("structure", "smoke_candidate", "excluded", "unknown")
    }
    smoke_chunks: list[np.ndarray] = []
    isolated_chunks: list[np.ndarray] = []
    for chunk in summaries:
        with np.load(chunk["labels_npz"]) as arrays:
            smoke_chunks.append(arrays["world_xyz"][arrays["label"] == SMOKE_CANDIDATE])
            isolated_chunks.append(
                arrays["world_xyz"][arrays["label_reason"] == REASON_ISOLATED_RETURN]
            )
    all_smoke = np.concatenate(smoke_chunks) if smoke_chunks else np.empty((0, 3), np.float32)
    all_smoke_path = output / "arc6_all_smoke_candidates.ply"
    _write_label_preview(
        all_smoke_path,
        all_smoke,
        np.full(len(all_smoke), SMOKE_CANDIDATE, dtype=np.uint8),
        len(all_smoke),
    )
    all_isolated = (
        np.concatenate(isolated_chunks) if isolated_chunks else np.empty((0, 3), np.float32)
    )
    all_isolated_path = output / "arc6_all_isolated_returns.ply"
    _write_label_preview(
        all_isolated_path,
        all_isolated,
        np.full(len(all_isolated), UNKNOWN, dtype=np.uint8),
        len(all_isolated),
    )
    result = {
        "schema_version": "1.0",
        "status": "complete",
        "session_id": str(config["session_id"]),
        "configuration": str(config_source),
        "configuration_sha256": _sha256(config_source),
        "reference_npz": str(Path(reference_npz).expanduser().resolve()),
        "time_selection_s": {"start": start_s, "end": end_s},
        "chunk_duration_s": chunk_duration_s,
        "completed_chunks": len(summaries),
        "planned_chunks": len(chunks),
        "frames": sum(chunk["frames"] for chunk in summaries),
        "points": sum(chunk["points"] for chunk in summaries),
        "labels": label_totals,
        "all_smoke_candidates_ply": str(all_smoke_path),
        "all_isolated_returns_ply": str(all_isolated_path),
        "chunks": summaries,
    }
    result["label_percent"] = {
        key: 100.0 * value / max(result["points"], 1) for key, value in label_totals.items()
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def validate_grandtour_clean(
    session_dir: str | Path,
    reference_config_path: str | Path,
    label_config_path: str | Path,
    reference_npz: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Measure pseudo-label false positives on the reserved clean interval."""
    reference_config_source, reference_config = _load_toml(reference_config_path)
    label_config_source, label_config = _load_toml(label_config_path)
    paths = resolve_grandtour_paths(session_dir)
    if paths.session_id != str(reference_config["session_id"]):
        raise ValueError("Clean validation session does not match the reference config")
    reference_path = Path(reference_npz).expanduser().resolve()
    with np.load(reference_path) as reference:
        reference_xyz = reference["xyz"].astype(np.float32)
        crop_min = reference["crop_min_m"].astype(np.float64)
        crop_max = reference["crop_max_m"].astype(np.float64)
    tree = cKDTree(reference_xyz)
    trajectory = load_dlio_trajectory(paths.dlio)
    transform_hesai_livox = _sensor_transform(paths)

    start_s = float(reference_config["time"]["validation_start_s"])
    input_config = label_config["input"]
    min_range = float(input_config["minimum_range_m"])
    max_range = float(input_config["maximum_range_m"])
    isolation_threshold = float(
        input_config.get("maximum_isolated_neighbor_distance_m", 0.0)
    )
    structure_threshold = float(label_config["labeling"]["structure_max_distance_m"])
    global_smoke_threshold = float(label_config["labeling"]["smoke_min_distance_m"])
    smoke_regions = _enabled_regions(label_config, "confirmed_smoke_regions")

    session_origin_ns: int | None = None
    frames = points = covered_points = isolated_returns = skipped_pose_frames = 0
    structure_points = global_smoke_points = effective_smoke_points = 0
    distance_chunks: list[np.ndarray] = []
    global_false_xyz: list[np.ndarray] = []
    effective_false_xyz: list[np.ndarray] = []
    per_frame_effective: list[int] = []

    for pair in iter_paired_livox_frames(paths):
        if session_origin_ns is None:
            session_origin_ns = pair.raw.header_time_ns
        relative_s = (pair.raw.header_time_ns - session_origin_ns) / 1e9
        if relative_s < start_s:
            continue
        if not trajectory.time_ns[0] <= pair.raw.header_time_ns <= trajectory.time_ns[-1]:
            skipped_pose_frames += 1
            continue
        ranges = np.linalg.norm(pair.raw.xyz, axis=1)
        valid = (
            np.isfinite(pair.raw.xyz).all(axis=1)
            & np.isfinite(pair.undistorted_xyz).all(axis=1)
            & np.isfinite(ranges)
            & (ranges >= min_range)
            & (ranges <= max_range)
        )
        isolated = _isolated_return_mask(pair.raw.xyz, valid, isolation_threshold)
        valid &= ~isolated
        isolated_returns += int(np.count_nonzero(isolated))
        transform_map_livox = trajectory.at(pair.raw.header_time_ns) @ transform_hesai_livox
        world = transform_points(transform_map_livox, pair.undistorted_xyz)
        coverage = valid & _inside_box(world, crop_min, crop_max)
        distance = np.empty(0, dtype=np.float32)
        if np.any(coverage):
            distance = tree.query(world[coverage], workers=-1)[0].astype(np.float32)
            distance_chunks.append(distance)
        covered_world = world[coverage]
        structure = distance <= structure_threshold
        global_smoke = distance >= global_smoke_threshold
        effective_smoke = np.zeros(len(distance), dtype=bool)
        for region in smoke_regions:
            inside = _inside_box(
                covered_world,
                np.asarray(region["min_m"], dtype=np.float64),
                np.asarray(region["max_m"], dtype=np.float64),
            )
            threshold = float(region.get("minimum_distance_m", global_smoke_threshold))
            effective_smoke |= inside & (distance >= threshold)
        if np.any(global_smoke):
            global_false_xyz.append(covered_world[global_smoke])
        if np.any(effective_smoke):
            effective_false_xyz.append(covered_world[effective_smoke])
        frames += 1
        points += len(world)
        covered_points += len(distance)
        structure_points += int(np.count_nonzero(structure))
        global_smoke_points += int(np.count_nonzero(global_smoke))
        effective_count = int(np.count_nonzero(effective_smoke))
        effective_smoke_points += effective_count
        per_frame_effective.append(effective_count)

    if not frames or not distance_chunks:
        raise RuntimeError("Reserved clean validation interval contains no usable frames")
    distances = np.concatenate(distance_chunks)
    global_xyz = (
        np.concatenate(global_false_xyz) if global_false_xyz else np.empty((0, 3), np.float32)
    )
    effective_xyz = (
        np.concatenate(effective_false_xyz)
        if effective_false_xyz else np.empty((0, 3), np.float32)
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    global_ply = output / "arc5_clean_global_false_positives.ply"
    effective_ply = output / "arc5_clean_effective_false_positives.ply"
    _write_label_preview(
        global_ply, global_xyz,
        np.full(len(global_xyz), SMOKE_CANDIDATE, dtype=np.uint8), len(global_xyz)
    )
    _write_label_preview(
        effective_ply, effective_xyz,
        np.full(len(effective_xyz), SMOKE_CANDIDATE, dtype=np.uint8), len(effective_xyz)
    )
    result = {
        "schema_version": "1.0",
        "session_id": paths.session_id,
        "status": "complete",
        "validation_start_s": start_s,
        "frames": frames,
        "points": points,
        "covered_points": covered_points,
        "skipped_frames_outside_pose_coverage": skipped_pose_frames,
        "isolated_sensor_returns": isolated_returns,
        "thresholds_m": {
            "structure_max": structure_threshold,
            "global_smoke_min": global_smoke_threshold,
        },
        "structure_points": structure_points,
        "global_false_positive_points": global_smoke_points,
        "global_false_positive_percent": 100.0 * global_smoke_points / covered_points,
        "effective_reviewed_rule_false_positive_points": effective_smoke_points,
        "effective_reviewed_rule_false_positive_percent": (
            100.0 * effective_smoke_points / covered_points
        ),
        "frames_with_effective_false_positives": int(
            np.count_nonzero(np.asarray(per_frame_effective))
        ),
        "effective_false_positives_per_frame_quantiles": {
            str(q): float(np.quantile(per_frame_effective, q)) for q in (0.5, 0.9, 0.99)
        },
        "reference_distance_quantiles_m": {
            str(q): float(np.quantile(distances, q))
            for q in (0.5, 0.9, 0.95, 0.99, 0.999)
        },
        "reference_config": str(reference_config_source),
        "reference_config_sha256": _sha256(reference_config_source),
        "label_config": str(label_config_source),
        "label_config_sha256": _sha256(label_config_source),
        "reference_npz": str(reference_path),
        "reference_npz_sha256": _sha256(reference_path),
        "global_false_positive_ply": str(global_ply),
        "effective_false_positive_ply": str(effective_ply),
    }
    summary_path = output / "held_out_clean_validation.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def export_grandtour_training_dataset(
    source_manifest: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Export reviewed labels without any clean-map or world-frame teacher features."""
    manifest_path = Path(source_manifest).expanduser().resolve()
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source.get("status") != "complete":
        raise ValueError("Source labeling manifest is not complete")
    output = Path(output_dir).expanduser().resolve()
    chunks_dir = output / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    totals = {"unimpacted": 0, "smoke_impacted": 0, "ignore": 0}

    for index, chunk in enumerate(source["chunks"]):
        source_path = Path(chunk["labels_npz"]).resolve()
        with np.load(source_path) as arrays:
            labels = arrays["label"].astype(np.uint8, copy=True)
            labels[labels == EXCLUDED] = UNKNOWN
            allowed = np.isin(labels, [STRUCTURE, SMOKE_CANDIDATE, UNKNOWN])
            if not np.all(allowed):
                raise ValueError(f"Unsupported label value in {source_path}")
            frame_index = arrays["frame_index"].astype(np.int32, copy=False)
            frame_time_ns = arrays["frame_time_ns"].astype(np.int64, copy=False)
            point_time_ns = arrays["point_time_ns"].astype(np.int64, copy=False)
            relative_time_ns = point_time_ns - frame_time_ns[frame_index]
            payload = {
                "schema_version": np.array("1.0"),
                "session_id": np.array(str(arrays["session_id"])),
                "xyz": arrays["sensor_xyz"].astype(np.float32, copy=False),
                "intensity": arrays["intensity"].astype(np.float32, copy=False),
                "tag": arrays["tag"].astype(np.uint8, copy=False),
                "line": arrays["line"].astype(np.uint8, copy=False),
                "relative_time_ns": relative_time_ns,
                "label": labels,
                "frame_index": frame_index,
                "source_point_index": arrays["source_point_index"].astype(np.uint32, copy=False),
                "frame_ptr": arrays["frame_ptr"].astype(np.int64, copy=False),
                "frame_time_ns": frame_time_ns,
            }
            if int(payload["frame_ptr"][-1]) != len(labels):
                raise ValueError(f"Invalid frame_ptr in {source_path}")
            output_path = chunks_dir / f"chunk_{index:03d}.npz"
            np.savez_compressed(output_path, **payload)
        counts = {
            "unimpacted": int(np.count_nonzero(labels == STRUCTURE)),
            "smoke_impacted": int(np.count_nonzero(labels == SMOKE_CANDIDATE)),
            "ignore": int(np.count_nonzero(labels == UNKNOWN)),
        }
        for key, value in counts.items():
            totals[key] += value
        exported.append({
            "chunk_index": index,
            "time_selection_s": chunk["time_selection_s"],
            "frames": int(len(frame_time_ns)),
            "points": int(len(labels)),
            "labels": counts,
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "source_labels_npz": str(source_path),
        })

    schema = {
        "schema_version": "1.0",
        "purpose": "GrandTour MID-360 per-point smoke segmentation training",
        "model_input_arrays": ["xyz", "intensity", "tag", "line", "relative_time_ns"],
        "target_array": "label",
        "labels": {"0": "unimpacted", "1": "smoke_impacted", "255": "ignore"},
        "arrays": {
            "xyz": {"dtype": "float32", "shape": ["points", 3], "frame": "livox_lidar"},
            "intensity": {"dtype": "float32", "shape": ["points"]},
            "tag": {"dtype": "uint8", "shape": ["points"]},
            "line": {"dtype": "uint8", "shape": ["points"]},
            "relative_time_ns": {"dtype": "int64", "shape": ["points"]},
            "label": {"dtype": "uint8", "shape": ["points"]},
            "frame_index": {"dtype": "int32", "shape": ["points"]},
            "source_point_index": {"dtype": "uint32", "shape": ["points"]},
            "frame_ptr": {"dtype": "int64", "shape": ["frames_plus_one"]},
            "frame_time_ns": {"dtype": "int64", "shape": ["frames"]},
        },
        "forbidden_model_inputs": [
            "world_xyz", "nearest_reference_distance_m", "automatic_label",
            "label_reason", "human_verified", "reference_map_path",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    schema_path = output / "dataset_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    result = {
        "schema_version": "1.0",
        "status": "complete",
        "session_id": source["session_id"],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "dataset_schema": str(schema_path),
        "chunks": exported,
        "frames": sum(chunk["frames"] for chunk in exported),
        "points": sum(chunk["points"] for chunk in exported),
        "labels": totals,
    }
    result["label_percent"] = {
        key: 100.0 * value / max(result["points"], 1) for key, value in totals.items()
    }
    output_manifest = output / "training_manifest.json"
    output_manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
