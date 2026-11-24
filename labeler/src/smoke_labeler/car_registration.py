"""Conservative clean-control scan-to-map registration pilots for RC-car bags."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from .bag import _custom_msg_cdr_arrays
from .car import _fresh_directory, _sha256, validate_transform
from .geometry import transform_points
from .provenance import write_run_provenance


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _voxel_indices(xyz: np.ndarray, size_m: float) -> np.ndarray:
    if size_m <= 0:
        raise ValueError("voxel size must be positive")
    _, first = np.unique(np.floor(xyz / size_m).astype(np.int32), axis=0, return_index=True)
    return np.sort(first)


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("rigid fit requires corresponding Nx3 arrays with at least three points")
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = target_center - rotation @ source_center
    return result


def _trimmed_icp(
    scan_xyz: np.ndarray,
    initial: np.ndarray,
    reference_xyz: np.ndarray,
    reference_tree: cKDTree,
) -> tuple[np.ndarray, bool, int]:
    """Point-to-point ICP using only the closest map-supported fraction."""
    transform = initial.copy()
    total_iterations = 0
    for maximum_distance, retained_fraction, limit in (
        (1.20, 0.55, 12),
        (0.50, 0.50, 15),
        (0.25, 0.45, 12),
    ):
        for _ in range(limit):
            world = transform_points(transform, scan_xyz).astype(np.float64)
            distance, nearest = reference_tree.query(world, workers=-1)
            candidates = np.flatnonzero(distance < maximum_distance)
            retain = min(len(candidates), max(500, int(len(distance) * retained_fraction)))
            if retain < 500:
                return transform, False, total_iterations
            selected = candidates[np.argsort(distance[candidates])[:retain]]
            correction = _rigid_fit(world[selected], reference_xyz[nearest[selected]])
            transform = correction @ transform
            total_iterations += 1
            translation_step = float(np.linalg.norm(correction[:3, 3]))
            rotation_step = float(Rotation.from_matrix(correction[:3, :3]).magnitude())
            if translation_step < 2e-5 and rotation_step < 2e-5:
                break
    return transform, True, total_iterations


class _OdomInterpolator:
    def __init__(self, motion: dict[str, np.ndarray]):
        self.time_ns = motion["odom_time_ns"].astype(np.int64)
        self.origin_ns = int(self.time_ns[0])
        self.seconds = (self.time_ns - self.origin_ns).astype(np.float64) * 1e-9
        self.translation = motion["odom_translation"].astype(np.float64)
        self.slerp = Slerp(
            self.seconds,
            Rotation.from_quat(motion["odom_quaternion_xyzw"].astype(np.float64)),
        )
        self.mount = validate_transform(motion["T_base_lidar"])

    def poses(self, query_ns: np.ndarray) -> np.ndarray:
        query_ns = np.asarray(query_ns, dtype=np.int64)
        seconds = (query_ns - self.origin_ns).astype(np.float64) * 1e-9
        if seconds.min(initial=0) < self.seconds[0] or seconds.max(initial=0) > self.seconds[-1]:
            raise ValueError("point timestamps lie outside wheel-odometry coverage")
        translation = np.column_stack(
            [np.interp(seconds, self.seconds, self.translation[:, axis]) for axis in range(3)]
        )
        result = np.repeat(np.eye(4, dtype=np.float64)[None], len(seconds), axis=0)
        result[:, :3, :3] = self.slerp(seconds).as_matrix()
        result[:, :3, 3] = translation
        return result @ self.mount

    def pose(self, query_ns: int) -> np.ndarray:
        return self.poses(np.asarray([query_ns], dtype=np.int64))[0]


def _deskew_to_start(
    xyz: np.ndarray,
    offset_s: np.ndarray,
    frame_time_ns: int,
    odom: _OdomInterpolator,
) -> np.ndarray:
    point_time_ns = frame_time_ns + np.rint(offset_s * 1e9).astype(np.int64)
    poses = odom.poses(point_time_ns)
    in_odom = np.einsum("nij,nj->ni", poses[:, :3, :3], xyz) + poses[:, :3, 3]
    start_inverse = np.linalg.inv(odom.pose(frame_time_ns))
    return transform_points(start_inverse, in_odom)


def _write_overlay(
    path: Path,
    reference_xyz: np.ndarray,
    scan_xyz: np.ndarray,
    nearest_distance: np.ndarray,
) -> None:
    map_take = np.arange(0, len(reference_xyz), 2, dtype=np.int64)
    count = len(map_take) + len(scan_xyz)
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
             ("source", "u1"), ("nearest_map_distance_m", "<f4")]
    values = np.empty(count, dtype=dtype)
    xyz = np.vstack((reference_xyz[map_take], scan_xyz)).astype(np.float32)
    values["x"], values["y"], values["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    values["source"] = np.concatenate((np.zeros(len(map_take), np.uint8), np.ones(len(scan_xyz), np.uint8)))
    values["nearest_map_distance_m"] = np.concatenate(
        (np.zeros(len(map_take), np.float32), nearest_distance.astype(np.float32))
    )
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\nproperty float x\nproperty float y\nproperty float z\n"
        "property uchar source\nproperty float nearest_map_distance_m\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(values.tobytes())


def _load_initial_pose(path: Path) -> tuple[int, np.ndarray, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    frame_index = int(document["frame_index"])
    return frame_index, validate_transform(document["T_layout_lidar"]), document


def register_clean_pilot(
    prepared: str | Path,
    recording_id: str,
    reference: str | Path,
    initial_pose: str | Path,
    output: str | Path,
    frame_count: int = 150,
) -> dict:
    prepared = Path(prepared).expanduser().resolve()
    reference = Path(reference).expanduser().resolve()
    initial_pose = Path(initial_pose).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if frame_count < 3:
        raise ValueError("frame_count must be at least three")
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    matches = [row for row in manifest["recordings"] if row["id"] == recording_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate recording ID: {recording_id}")
    recording = matches[0]
    if recording["role"] != "clean_control":
        raise ValueError("clean pilot requires a clean_control recording")
    reference_manifest = json.loads((reference / "reference_manifest.json").read_text(encoding="utf-8"))
    if reference_manifest["layout"] != recording["layout"]:
        raise ValueError("reference and recording layouts do not match")
    reference_npz = reference / "reference.npz"
    with np.load(reference_npz, allow_pickle=False) as cloud:
        reference_xyz = cloud["xyz"].astype(np.float64)
    reference_keep = _voxel_indices(reference_xyz, 0.08)
    registration_reference = reference_xyz[reference_keep]
    reference_tree = cKDTree(registration_reference)
    pose_frame, reviewed_pose, pose_document = _load_initial_pose(initial_pose)
    motion_path = prepared / "recordings" / recording_id / "motion.npz"
    with np.load(motion_path, allow_pickle=False) as motion_file:
        motion = {name: motion_file[name] for name in motion_file.files}
    if pose_frame >= len(motion["frame_time_ns"]):
        raise ValueError("initial-pose frame is outside the recording")
    if not bool(motion["full_odom_coverage"][pose_frame]):
        raise ValueError("initial-pose frame lacks full odometry coverage")
    odom = _OdomInterpolator(motion)
    initial_odom_pose = odom.pose(int(motion["frame_time_ns"][pose_frame]))
    layout_from_odom = reviewed_pose @ np.linalg.inv(initial_odom_pose)
    files = [Path(item["path"]) for item in recording["source_files"]]
    if any(_sha256(path) != item["sha256"] for path, item in zip(files, recording["source_files"])):
        raise ValueError("recording MCAP changed since preparation")
    _fresh_directory(output)
    write_run_provenance(output, {
        "point_stride": 3, "scan_voxel_m": 0.10, "reference_voxel_m": 0.08,
        "range_m": [0.3, 25.0], "registration": "three-stage trimmed point-to-point ICP",
    }, {"command": "car-register-clean", "recording": recording_id,
        "reference": str(reference), "initial_pose": str(initial_pose), "frames": frame_count})
    frame_indices, frame_times, transforms, rows = [], [], [], []
    correction = np.eye(4, dtype=np.float64)
    overlay_targets = {pose_frame, pose_frame + frame_count // 2, pose_frame + frame_count - 1}
    processed = 0
    for file in files:
        with file.open("rb") as stream:
            for schema, channel, message in make_reader(stream, validate_crcs=True).iter_messages(
                topics=["/livox/lidar"]
            ):
                processed += 1
                source_index = processed - 1
                if source_index < pose_frame:
                    continue
                if len(frame_indices) >= frame_count:
                    break
                if source_index >= len(motion["frame_time_ns"]):
                    raise ValueError("MCAP contains more LiDAR frames than motion audit")
                if not bool(motion["full_odom_coverage"][source_index]):
                    continue
                xyz, _, _, _, offsets, timebase = _custom_msg_cdr_arrays(message.data, point_stride=3)
                expected_time = int(motion["frame_time_ns"][source_index])
                if int(timebase) != expected_time:
                    raise ValueError(f"LiDAR timestamp mismatch at frame {source_index}")
                ranges = np.linalg.norm(xyz, axis=1)
                valid = np.isfinite(xyz).all(axis=1) & (ranges >= 0.3) & (ranges <= 25.0)
                xyz, offsets = xyz[valid].astype(np.float64), offsets[valid]
                take = _voxel_indices(xyz, 0.10)
                deskewed = _deskew_to_start(xyz[take], offsets[take], expected_time, odom)
                odom_pose = odom.pose(expected_time)
                predicted = layout_from_odom @ odom_pose
                proposed = correction @ predicted
                refined, converged, iterations = _trimmed_icp(
                    deskewed, proposed, registration_reference, reference_tree
                )
                delta = refined @ np.linalg.inv(proposed)
                delta_translation = float(np.linalg.norm(delta[:3, 3]))
                delta_rotation = float(Rotation.from_matrix(delta[:3, :3]).magnitude())
                accepted = bool(converged and delta_translation <= (2.0 if not frame_indices else 0.5)
                                and delta_rotation <= np.deg2rad(15 if not frame_indices else 8))
                if accepted:
                    correction = refined @ np.linalg.inv(predicted)
                    selected_pose = refined
                else:
                    selected_pose = proposed
                world = transform_points(selected_pose, deskewed).astype(np.float64)
                distance = reference_tree.query(world, workers=-1)[0]
                metrics = {
                    "frame_index": source_index, "frame_time_ns": expected_time,
                    "accepted": accepted, "iterations": iterations, "sampled_points": len(world),
                    "median_m": float(np.median(distance)), "p75_m": float(np.percentile(distance, 75)),
                    "p90_m": float(np.percentile(distance, 90)), "p95_m": float(np.percentile(distance, 95)),
                    "within_0_10": float(np.mean(distance <= 0.10)),
                    "within_0_20": float(np.mean(distance <= 0.20)),
                    "within_0_30": float(np.mean(distance <= 0.30)),
                    "correction_translation_m": delta_translation,
                    "correction_rotation_deg": float(np.rad2deg(delta_rotation)),
                }
                rows.append(metrics)
                frame_indices.append(source_index)
                frame_times.append(expected_time)
                transforms.append(selected_pose)
                if source_index in overlay_targets:
                    _write_overlay(output / f"overlay_frame_{source_index:06d}.ply",
                                   reference_xyz, world, distance)
            if len(frame_indices) >= frame_count:
                break
    if len(frame_indices) < frame_count:
        raise ValueError(f"Only processed {len(frame_indices)} of {frame_count} requested frames")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "trajectory.npz",
        frame_index=np.asarray(frame_indices, np.int32),
        frame_time_ns=np.asarray(frame_times, np.int64),
        T_layout_lidar=np.asarray(transforms, np.float64))
    _json(output / "initial_pose_used.json", pose_document)
    medians = np.asarray([row["median_m"] for row in rows])
    within = np.asarray([row["within_0_20"] for row in rows])
    result = {
        "schema": "car_clean_registration_pilot_v1", "recording": recording_id,
        "layout": recording["layout"], "frames": len(rows),
        "frame_range": [frame_indices[0], frame_indices[-1]],
        "accepted_frames": int(sum(row["accepted"] for row in rows)),
        "median_distance_m": float(np.median(medians)),
        "p90_of_frame_medians_m": float(np.percentile(medians, 90)),
        "median_fraction_within_0_20_m": float(np.median(within)),
        "reference_sha256": _sha256(reference_npz),
        "source_sha256": [item["sha256"] for item in recording["source_files"]],
        "status": "pilot_requires_visual_overlay_review",
    }
    _json(output / "report.json", result)
    return result
