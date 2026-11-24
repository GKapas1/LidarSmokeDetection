"""Generate conservative car smoke pseudo-labels and score them on clean drives."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from mcap.reader import make_reader
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .bag import _custom_msg_cdr_arrays
from .car import _sha256, validate_transform
from .car_ray_review import CONFIG as RAY_CONFIG, query_directional_ranges
from .car_registration import _OdomInterpolator
from .provenance import write_run_provenance
from .unified_dataset import SCHEMA_VERSION, _metadata, _schema, _write_chunk


CONFIG = {
    "range_m": [0.3, 25.0],
    "pose_match_m": 0.20,
    "minimum_pose_match": 0.55,
    "minimum_pose_sectors": 3,
    "maximum_origin_error_m": 0.06,
    "negative_map_distance_m": 0.10,
    "negative_ray_error_m": 0.15,
    "positive_map_clearance_m": 0.35,
    "positive_ray_shortfall_m": 0.60,
    "maximum_positive_intensity": 5,
    "neighbor_radius_m": 0.30,
    "minimum_neighbor_returns": 3,
    "maximum_frame_gap_s": 0.15,
    "clean_veto_radius_m": 0.30,
}


def _evidence(reference, tree, odom, pose, timestamp, xyz, offsets, intensity):
    times = timestamp + np.rint(offsets * 1e9).astype(np.int64)
    point_poses = (pose @ np.linalg.inv(odom.pose(timestamp))) @ odom.poses(times)
    origins = point_poses[:, :3, 3]
    world = np.einsum("nij,nj->ni", point_poses[:, :3, :3], xyz) + origins
    distance = tree.query(world)[0]
    match = distance <= CONFIG["pose_match_m"]
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    sectors = np.floor((azimuth + np.pi) * 8 / (2 * np.pi)).astype(int).clip(0, 7)
    fraction = float(match.mean())
    supported = np.count_nonzero(np.bincount(sectors[match], minlength=8) >= 60)
    positive = np.zeros(len(xyz), dtype=bool)
    negative = positive.copy()
    if fraction < CONFIG["minimum_pose_match"] or supported < CONFIG["minimum_pose_sectors"]:
        return world, positive, negative, 0, fraction, False

    far = distance >= CONFIG["positive_map_clearance_m"]
    near = ((distance <= CONFIG["negative_map_distance_m"])
            & (intensity > CONFIG["maximum_positive_intensity"]))
    residual = np.full(len(xyz), np.nan)
    for half in np.array_split(np.argsort(offsets), 2):
        if not len(half):
            continue
        origin = origins[half].mean(axis=0)
        if np.linalg.norm(origins[half] - origin, axis=1).max() <= CONFIG["maximum_origin_error_m"]:
            selected = half[(far | near)[half]]
            residual[selected] = query_directional_ranges(reference, world[selected], origin, RAY_CONFIG)[1]
    geometry = (residual <= -CONFIG["positive_ray_shortfall_m"]) & far
    positive = geometry & (intensity <= CONFIG["maximum_positive_intensity"])
    negative = (np.abs(residual) <= CONFIG["negative_ray_error_m"]) & near
    return world, positive, negative, int(geometry.sum()), fraction, True


def _repeat_candidates(current, previous, following):
    """Require nearby weak, shortened returns in each adjacent scan."""
    xyz = current["candidates"]
    result = np.zeros(len(xyz), dtype=bool)
    if previous is None or following is None or not len(xyz):
        return result
    neighbors = CONFIG["minimum_neighbor_returns"]
    result[:] = True
    for other in (previous, following):
        if (abs(other["time"] - current["time"]) * 1e-9 > CONFIG["maximum_frame_gap_s"]
                or len(other["candidates"]) < neighbors):
            return np.zeros(len(xyz), dtype=bool)
        distance = cKDTree(other["candidates"]).query(xyz, k=[neighbors])[0][:, 0]
        result &= distance <= CONFIG["neighbor_radius_m"]
    return result


def _frames(record, motion, poses, reference, tree, odom):
    frame = 0
    for source in record["source_files"]:
        path = Path(source["path"])
        with path.open("rb") as stream:
            for schema, _, msg in make_reader(stream, validate_crcs=True).iter_messages(topics=["/livox/lidar"]):
                if schema.name != "livox_ros_driver2/msg/CustomMsg":
                    raise ValueError(f"Unexpected LiDAR schema in {path}")
                xyz, intensity, tag, line, offsets, timestamp = _custom_msg_cdr_arrays(msg.data)
                if frame >= len(motion["frame_time_ns"]) or timestamp != motion["frame_time_ns"][frame]:
                    raise ValueError(f"Prepared timestamp mismatch for {record['id']} frame {frame}")
                ranges = np.linalg.norm(xyz, axis=1)
                valid = np.isfinite(ranges) & (ranges >= CONFIG["range_m"][0]) & (ranges <= CONFIG["range_m"][1])
                labels = np.full(len(xyz), 255, dtype=np.uint8)
                candidate_indices = np.empty(0, dtype=np.int64)
                candidates = np.empty((0, 3))
                geometry, match, covered = 0, None, False
                if frame in poses and motion["full_odom_coverage"][frame] and valid.any():
                    indices = np.flatnonzero(valid)
                    world, positive, negative, geometry, match, covered = _evidence(
                        reference, tree, odom, poses[frame], timestamp, xyz[indices],
                        offsets[indices], intensity[indices],
                    )
                    candidate_indices = indices[positive]
                    candidates = world[positive]
                    labels[indices[negative]] = 0
                if record["role"] == "clean_control":
                    labels[valid] = 0
                yield {"index": frame, "time": timestamp, "label": labels, "xyz": xyz,
                       "intensity": intensity, "tag": tag, "line": line, "point_offset_s": offsets,
                       "candidate_indices": candidate_indices, "candidates": candidates,
                       "valid": int(valid.sum()), "pose_match": match,
                       "pose_passed": covered, "geometry_candidates": geometry}
                frame += 1
    if frame != len(motion["frame_time_ns"]):
        raise ValueError(f"Prepared frame count mismatch for {record['id']}")


def _write_preview(path, reference, candidates, indices):
    ref = reference[::3]
    xyz = np.concatenate((ref, candidates)).astype("<f4")
    fields = np.empty(len(xyz), dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3), ("source_point_index", "<i4")])
    fields["xyz"] = xyz
    fields["rgb"][:len(ref)] = [90, 120, 150]
    fields["rgb"][len(ref):] = [255, 135, 35]
    fields["source_point_index"][:len(ref)] = -1
    fields["source_point_index"][len(ref):] = indices
    header = (
        f"ply\nformat binary_little_endian 1.0\nelement vertex {len(fields)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property int source_point_index\nend_header\n"
    )
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        fields.tofile(stream)


def _load_trajectory(prepared, reference_dir, recording_id, registration, legacy_yaw):
    invocation = json.loads((registration / "run_provenance.json").read_text())["invocation"]
    if invocation["recording"] != recording_id or Path(invocation["reference"]).resolve() != reference_dir.resolve():
        raise ValueError(f"Registration recording/reference mismatch for {recording_id}")
    effective = json.loads((registration / "effective_config.json").read_text())
    reference_path = reference_dir / "reference.npz"
    motion_path = prepared / "recordings" / recording_id / "motion.npz"
    for key, path in (("reference_sha256", reference_path), ("motion_sha256", motion_path)):
        if key in invocation and _sha256(path) != invocation[key]:
            raise ValueError(f"{path} changed since registration")
    yaw = effective.get("wheel_mount_yaw_deg", legacy_yaw)
    with np.load(motion_path, allow_pickle=False) as archive:
        motion = dict(archive)
    mount = motion["T_base_lidar"]
    mount[:3, :3] = mount[:3, :3] @ Rotation.from_euler("z", yaw, degrees=True).as_matrix()
    with np.load(registration / "trajectory.npz", allow_pickle=False) as data:
        poses = {}
        for index, time, pose in zip(data["frame_index"], data["frame_time_ns"], data["T_layout_lidar"], strict=True):
            if (index in poses or not 0 <= index < len(motion["frame_time_ns"])
                    or time != motion["frame_time_ns"][index]):
                raise ValueError(f"Registration frame mismatch for {recording_id}")
            poses[index] = validate_transform(pose)
    return poses, motion, yaw


def _label_recording(prepared, record, registration, reference_dir, reference, tree, output, legacy_yaw,
                     clean_veto, collect_veto):
    recording_id = record["id"]
    poses, motion, yaw = _load_trajectory(prepared, reference_dir, recording_id, registration, legacy_yaw)
    if any(_sha256(Path(item["path"])) != item["sha256"] for item in record["source_files"]):
        raise ValueError(f"{recording_id}: MCAP changed since preparation")
    frames = _frames(record, motion, poses, reference, tree, _OdomInterpolator(motion))
    previous = None
    current = next(frames, None)
    following = next(frames, None)
    rows = []
    keys = ("xyz", "intensity", "tag", "line", "point_offset_s", "label")
    arrays = {key: [] for key in keys}
    offsets, times = [0], []
    best_count = 0
    best_candidates = None
    veto_samples = []
    while current is not None:
        repeated = _repeat_candidates(current, previous, following)
        before_veto = int(repeated.sum())
        if collect_veto and before_veto:
            veto_samples.append(current["candidates"][repeated])
        if clean_veto is not None and before_veto:
            repeated[repeated] = clean_veto.query(current["candidates"][repeated])[0] > CONFIG["clean_veto_radius_m"]
        labels = current["label"]
        if record["role"] == "smoke":
            labels[current["candidate_indices"][repeated]] = 1
        rows.append({
            "frame": current["index"], "valid": current["valid"], "pose_match": current["pose_match"],
            "pose_passed": current["pose_passed"], "geometry_candidates": current["geometry_candidates"],
            "single_frame_candidates": len(repeated),
            "repeated_before_clean_veto": before_veto,
            "repeated_candidates": int(repeated.sum()), "unimpacted": int((labels == 0).sum()),
            "smoke_impacted": int((labels == 1).sum()), "ignore": int((labels == 255).sum()),
        })
        if rows[-1]["repeated_candidates"] > best_count:
            best_count = rows[-1]["repeated_candidates"]
            best_frame = current["index"]
            best_candidates = current["candidates"][repeated]
            best_indices = current["candidate_indices"][repeated]
        for key in keys:
            arrays[key].append(current[key])
        times.append(current["time"])
        offsets.append(offsets[-1] + len(labels))
        previous, current, following = current, following, next(frames, None)
    if not rows:
        raise ValueError(f"No LiDAR frames for {recording_id}")

    data = {key: np.concatenate(values) for key, values in arrays.items()}
    data["point_offset_s"] = data["point_offset_s"].astype(np.float32)
    # The unified loader requires finite features, even for ignored points.
    data["xyz"][~np.isfinite(data["xyz"]).all(axis=1)] = 0
    data["frame_ptr"] = np.asarray(offsets, dtype=np.int64)
    data["frame_index"] = np.repeat(np.arange(len(rows), dtype=np.int32), np.diff(offsets))
    data["frame_time_s"] = (np.asarray(times, dtype=np.int64) - times[0]) * 1e-9
    condition = "clean_control" if record["role"] == "clean_control" else "smoke"
    metadata = _metadata(prepared.name, "car_ros2", recording_id, condition)
    path = output / "chunks" / f"{recording_id}.npz"
    chunk = _write_chunk(path, {**metadata, **data})
    chunk.update(path=str(path.relative_to(output)), recording_id=recording_id,
                 source_domain="car_ros2", session_id=prepared.name, condition=condition)
    with (output / f"{recording_id}_frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if best_candidates is not None:
        _write_preview(output / f"{recording_id}_frame_{best_frame:06d}_candidates.ply", reference,
                       best_candidates, best_indices)
    summary = {key: sum(row[key] for row in rows) for key in (
        "valid", "pose_passed", "geometry_candidates", "single_frame_candidates",
        "repeated_before_clean_veto", "repeated_candidates",
    )}
    summary.update(trajectory=str(registration.resolve()), wheel_mount_yaw_deg=yaw,
                   trajectory_sha256=_sha256(registration / "trajectory.npz"),
                   worst_or_peak_frame=best_frame if best_candidates is not None else None,
                   frames_with_candidates=sum(row["repeated_candidates"] > 0 for row in rows),
                   source_files=record["source_files"])
    return chunk, summary, np.concatenate(veto_samples) if veto_samples else np.empty((0, 3))


def label_dataset(prepared, reference_dir, registrations, output, legacy_yaw, veto_recording=None):
    manifest = json.loads((prepared / "manifest.json").read_text())
    reference_info = json.loads((reference_dir / "reference_manifest.json").read_text())
    alignment = json.loads((reference_dir / "alignment_used.json").read_text())
    reference_path = reference_dir / "reference.npz"
    if _sha256(reference_path) != reference_info["reference_sha256"]:
        raise ValueError("Reference changed after construction")
    records = {row["id"]: row for row in manifest["recordings"]}
    if not registrations or len(registrations) != len({recording for recording, _ in registrations}):
        raise ValueError("Supply at least one registration and each recording only once")
    for recording_id, _ in registrations:
        record = records.get(recording_id)
        if record is None or record["layout"] != reference_info["layout"] or record["role"] not in {"clean_control", "smoke"}:
            raise ValueError(f"{recording_id}: expected a moving recording for {reference_info['layout']}")
        if record["role"] == "smoke" and alignment.get("layout_matches_smoke_confirmed") is not True:
            raise ValueError(f"{recording_id}: clean reference layout has no confirmed match to smoke runs")
    if veto_recording is not None:
        if veto_recording not in dict(registrations) or records[veto_recording]["role"] != "clean_control":
            raise ValueError("Clean veto recording must be one of the provided clean controls")
        registrations = sorted(registrations, key=lambda row: row[0] != veto_recording)
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = archive["xyz"].astype(np.float64)
    tree = cKDTree(reference)
    if output.exists():
        raise ValueError(f"Choose a new output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    chunks, audit = [], {}
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "dataset"
        stage.mkdir()
        write_run_provenance(stage, {"labels": CONFIG, "directional": RAY_CONFIG}, {
            "command": "car-label", "prepared": str(prepared.resolve()), "reference": str(reference_dir.resolve()),
            "registrations": registrations, "legacy_wheel_mount_yaw_deg": legacy_yaw,
            "clean_veto_recording": veto_recording,
        })
        clean_veto = None
        for recording_id, trajectory in registrations:
            chunk, summary, veto_samples = _label_recording(
                prepared, records[recording_id], Path(trajectory), reference_dir, reference, tree, stage, legacy_yaw,
                clean_veto, recording_id == veto_recording,
            )
            if recording_id == veto_recording and len(veto_samples):
                clean_veto = cKDTree(veto_samples)
                np.savez_compressed(stage / "clean_veto_points.npz", world_xyz=veto_samples)
            chunks.append(chunk)
            audit[recording_id] = summary
            print(json.dumps({"recording": recording_id, "labels": chunk["labels"],
                              "repeated_candidates": summary["repeated_candidates"]}), flush=True)
        dataset = {
            "schema_version": SCHEMA_VERSION, "status": "complete", "dataset_schema": "dataset_schema.json",
            "label_quality": "experimental_pseudo_labels", "split_policy": "unassigned; split by recording",
            "chunks": chunks, "frames": sum(chunk["frames"] for chunk in chunks),
            "points": sum(chunk["points"] for chunk in chunks),
            "labels": {key: sum(chunk["labels"][key] for chunk in chunks) for key in chunks[0]["labels"]},
        }
        schema = _schema()
        schema["arrays"]["source_domain"] = "scalar string: car_ros2"
        for name, document in (
            ("dataset_manifest.json", dataset), ("dataset_schema.json", schema),
            ("label_audit.json", {"config": CONFIG, "ray_config": RAY_CONFIG,
                                 "reference_sha256": reference_info["reference_sha256"],
                                 "clean_veto_recording": veto_recording, "recordings": audit}),
        ):
            (stage / name).write_text(json.dumps(document, indent=2) + "\n")
        stage.rename(output)
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--registration", required=True, action="append", nargs=2, metavar=("RECORDING", "DIRECTORY"))
    parser.add_argument("--legacy-wheel-mount-yaw-deg", type=float, default=0.0,
                        help="Deskew correction for trajectories with no recorded mount yaw override")
    parser.add_argument("--clean-veto-recording", help="Use false candidates from this clean control as a spatial veto")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = label_dataset(args.prepared, args.reference, args.registration, args.output,
                           args.legacy_wheel_mount_yaw_deg, args.clean_veto_recording)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
