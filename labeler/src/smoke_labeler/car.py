"""Offline preparation of moving-car MCAPs; no training labels are inferred here."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import struct

import numpy as np
from mcap.reader import make_reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

from .bag import _custom_msg_cdr_arrays
from .geometry import pose_matrix, transform_points
from .provenance import write_run_provenance


NAME = re.compile(
    r"mid360_(?P<layout>[a-z0-9_]+)_"
    r"(?P<kind>clean_stationary|clean_staionary|clean_moving|smoke)"
    r"(?P<number>\d+)(?:ű)?$"
)
REQUIRED_TOPICS = ("/livox/lidar", "/livox/imu", "/odom", "/tf", "/tf_static")


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fresh_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output is not empty: {path}. Choose a new directory to preserve reviews.")
    path.mkdir(parents=True, exist_ok=True)


def _stamp(header) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def _xyz(vector) -> list[float]:
    return [float(vector.x), float(vector.y), float(vector.z)]


def _quat(q) -> list[float]:
    return [float(q.x), float(q.y), float(q.z), float(q.w)]


def validate_transform(value) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Alignment must be a finite 4x4 T_layout_lidar matrix")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
        raise ValueError("Alignment must be rigid, with no scale, reflection, or shear")
    return matrix


def timing_stats(times: list[int] | np.ndarray) -> dict:
    times = np.asarray(times, dtype=np.int64)
    delta = np.diff(times) * 1e-9
    duration = (int(times[-1]) - int(times[0])) * 1e-9 if len(times) > 1 else 0.0
    return {
        "count": len(times), "duration_s": duration,
        "rate_hz": (len(times) - 1) / duration if duration > 0 else None,
        "maximum_gap_s": float(delta.max()) if len(delta) else None,
        "nonincreasing_timestamps": int(np.count_nonzero(delta <= 0)),
    }


def voxel_cloud(xyz, intensity, frame_index, voxel_m=0.04, minimum_frames=3):
    """Keep one measured point/voxel with support from distinct sampled scans."""
    key = np.floor(np.asarray(xyz) / voxel_m).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    pairs = np.unique(np.column_stack([inverse, frame_index]), axis=0)
    support = np.bincount(pairs[:, 0], minlength=len(first))
    keep = support >= minimum_frames
    return xyz[first[keep]], intensity[first[keep]], support[keep].astype(np.uint32)


def write_cloud(path: Path, xyz, intensity, support=None) -> None:
    """Binary PLY with raw reflectivity as a scalar field for CloudCompare."""
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")]
    props = "property float x\nproperty float y\nproperty float z\nproperty float intensity\n"
    if support is not None:
        fields.append(("support_scans", "<u4"))
        props += "property uint support_scans\n"
    data = np.empty(len(xyz), dtype=fields)
    for i, name in enumerate(("x", "y", "z")):
        data[name] = xyz[:, i]
    data["intensity"] = intensity
    if support is not None:
        data["support_scans"] = support
    with path.open("wb") as stream:
        stream.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                      + props + "end_header\n").encode("ascii"))
        stream.write(data.tobytes())


def discover_recordings(root: Path) -> tuple[list[dict], list[str]]:
    entries, ignored, ids = [], [], set()
    for bag in sorted(root.iterdir()):
        if not bag.is_dir():
            continue
        match = NAME.fullmatch(bag.name)
        if not match:
            ignored.append(bag.name)
            continue
        layout, kind, number = match.group("layout", "kind", "number")
        kind = kind.replace("staionary", "stationary")
        record_id = f"{layout}_{kind}{number}"
        if record_id in ids:
            raise ValueError(f"Duplicate canonical recording ID: {record_id}")
        ids.add(record_id)
        files = sorted(bag.glob("*.mcap"))
        if not files:
            raise ValueError(f"No MCAP files in {bag}")
        role = {"clean_stationary": "reference", "clean_moving": "clean_control",
                "smoke": "smoke"}[kind]
        entries.append({"id": record_id, "layout": layout, "role": role,
                        "source_directory": str(bag.resolve()), "split": None})
    if not entries:
        raise ValueError(f"No named car recordings found in {root}")
    return entries, ignored


def _prepare_recording(entry: dict, output: Path, typestore) -> dict:
    bag = Path(entry["source_directory"])
    destination = output / "recordings" / entry["id"]
    destination.mkdir(parents=True)
    files = sorted(bag.glob("*.mcap"))
    counts, log_times = Counter(), defaultdict(list)
    frame_times, frame_counts, offset_max, header_diffs = [], [], [], []
    odom_t, odom_xyz, odom_q, velocity, angular = [], [], [], [], []
    imu_t, imu_acc, imu_gyro, imu_q = [], [], [], []
    static_transforms, lidar_frames, odom_frames, imu_frames = [], set(), set(), set()
    invalid_points, zero_points, nonmonotonic_offsets = 0, 0, 0
    candidates, stationary_samples = [], []
    for file in files:
        with file.open("rb") as stream:
            reader = make_reader(stream, validate_crcs=True)
            for schema, channel, message in reader.iter_messages():
                topic = channel.topic
                counts[topic] += 1
                log_times[topic].append(message.log_time)
                if topic == "/livox/lidar":
                    if schema.name != "livox_ros_driver2/msg/CustomMsg":
                        raise ValueError(f"Unexpected LiDAR type in {bag}: {schema.name}")
                    xyz, intensity, _, _, offsets, timebase = _custom_msg_cdr_arrays(message.data)
                    endian = "<" if int.from_bytes(message.data[:2], "big") in {1, 3} else ">"
                    sec, ns, length = struct.unpack_from(endian + "iII", message.data, 4)
                    lidar_frames.add(message.data[16:16 + length - 1].decode("utf-8"))
                    header_diffs.append(int(timebase) - (sec * 1_000_000_000 + ns))
                    if not timebase:
                        raise ValueError(f"Missing LiDAR timebase in {bag}")
                    index = len(frame_times)
                    frame_times.append(timebase)
                    frame_counts.append(len(xyz))
                    offset_max.append(float(offsets.max(initial=0)))
                    nonmonotonic_offsets += int(np.any(np.diff(offsets) < 0))
                    finite = np.isfinite(xyz).all(axis=1)
                    ranges = np.linalg.norm(xyz, axis=1)
                    invalid_points += int(np.count_nonzero(~finite))
                    zero_points += int(np.count_nonzero(ranges == 0))
                    valid = finite & (ranges >= .3) & (ranges <= 30)
                    if entry["role"] == "reference" and index % 5 == 0:
                        take = np.flatnonzero(valid)[::4]
                        stationary_samples.append((xyz[take].copy(), intensity[take].copy(),
                                                   np.full(len(take), index, dtype=np.int32)))
                    elif entry["role"] != "reference" and index % 10 == 0:
                        candidates.append((index, xyz[valid].copy(), intensity[valid].copy()))
                elif topic in ("/odom", "/livox/imu", "/tf_static"):
                    msg = typestore.deserialize_cdr(message.data, schema.name)
                    if topic == "/odom":
                        odom_frames.add((msg.header.frame_id, msg.child_frame_id))
                        odom_t.append(_stamp(msg.header))
                        odom_xyz.append(_xyz(msg.pose.pose.position))
                        odom_q.append(_quat(msg.pose.pose.orientation))
                        velocity.append(_xyz(msg.twist.twist.linear))
                        angular.append(_xyz(msg.twist.twist.angular))
                    elif topic == "/livox/imu":
                        imu_frames.add(msg.header.frame_id)
                        imu_t.append(_stamp(msg.header))
                        imu_acc.append(_xyz(msg.linear_acceleration))
                        imu_gyro.append(_xyz(msg.angular_velocity))
                        imu_q.append(_quat(msg.orientation))
                    else:
                        for tr in msg.transforms:
                            static_transforms.append({"parent": tr.header.frame_id,
                                "child": tr.child_frame_id, "matrix": pose_matrix(
                                    _xyz(tr.transform.translation), _quat(tr.transform.rotation)).tolist()})
    missing = [topic for topic in REQUIRED_TOPICS if not counts[topic]]
    if missing:
        raise ValueError(f"Missing required topics in {bag}: {missing}")
    if odom_frames != {("odom", "base_link")} or lidar_frames != {"livox_frame"}:
        raise ValueError(f"Unexpected coordinate frames in {bag}: {odom_frames}, {lidar_frames}")
    mounts = [validate_transform(t["matrix"]) for t in static_transforms
              if t["parent"] == "base_link" and t["child"] == "livox_frame"]
    if not mounts or any(not np.allclose(mounts[0], t, atol=1e-8) for t in mounts):
        raise ValueError(f"Missing/conflicting base_link -> livox_frame transforms in {bag}")
    mount = mounts[0]
    odom_t, imu_t, frame_times = (np.asarray(t, dtype=np.int64) for t in (odom_t, imu_t, frame_times))
    odom_xyz, odom_q, velocity, angular, imu_acc, imu_gyro = (
        np.asarray(a, dtype=np.float64) for a in (odom_xyz, odom_q, velocity, angular, imu_acc, imu_gyro))
    speed = np.linalg.norm(velocity, axis=1)
    angular_speed = np.linalg.norm(angular, axis=1)
    still = (speed < .02) & (angular_speed < .02)
    moving_indices = np.flatnonzero(~still)
    full_duration = (int(odom_t[-1]) - int(odom_t[0])) * 1e-9
    start_still = ((int(odom_t[moving_indices[0]]) - int(odom_t[0])) * 1e-9
                   if len(moving_indices) else full_duration)
    end_still = ((int(odom_t[-1]) - int(odom_t[moving_indices[-1]])) * 1e-9
                 if len(moving_indices) else full_duration)
    timing = {"lidar": timing_stats(frame_times), "imu": timing_stats(imu_t),
              "odom": timing_stats(odom_t)}
    warnings = []
    metadata = bag / "metadata.yaml"
    if not metadata.is_file() or metadata.stat().st_size == 0:
        warnings.append("Missing/empty ROS metadata; direct MCAP preparation works. Repair a copy for ROS replay.")
    for name, stats in timing.items():
        if stats["nonincreasing_timestamps"]:
            raise ValueError(f"Nonincreasing {name} timestamps in {bag}")
        if stats["maximum_gap_s"] is None:
            raise ValueError(f"Insufficient {name} samples in {bag}")
    if timing["lidar"]["maximum_gap_s"] > .15:
        warnings.append("LiDAR gap above 150 ms: split temporal windows at gaps.")
    if max(abs(d) for d in header_diffs) > 1_000_000:
        warnings.append("LiDAR header and timebase differ by >1 ms; resolve clock convention before deskew.")
    if nonmonotonic_offsets:
        warnings.append("Some scans have nonmonotonic point offsets; preserve original order and use per-point times.")
    norm_acc = np.linalg.norm(imu_acc, axis=1)
    accel_median = float(np.median(norm_acc))
    unit_hint = "likely_g" if .7 < accel_median < 1.3 else (
        "likely_m_s2" if 7 < accel_median < 13 else "unresolved")
    if unit_hint != "likely_m_s2":
        warnings.append(f"IMU acceleration norm median={accel_median:.3f}; {unit_hint}. Confirm units before inertial processing.")
    initial_pose = pose_matrix(odom_xyz[0], odom_q[0]) @ mount
    pose_variation = Rotation.from_quat(odom_q)
    angles = (pose_variation[0].inv() * pose_variation).magnitude()
    path_m = float(np.linalg.norm(np.diff(odom_xyz, axis=0), axis=1).sum())
    is_still = bool(still.all() and np.max(np.linalg.norm(odom_xyz - odom_xyz[0], axis=1)) < .02
                   and angles.max() < .02)
    if entry["role"] == "reference" and not is_still:
        raise ValueError(f"Stationary reference contains odometry motion: {bag}")
    frame_end = frame_times + np.rint(np.asarray(offset_max) * 1e9).astype(np.int64)
    covered = (frame_times >= odom_t[0]) & (frame_end <= odom_t[-1])
    if not covered.all():
        warnings.append(f"{int((~covered).sum())} boundary scans lack full odometry coverage; do not extrapolate.")
    if start_still < 5 or end_still < 5:
        warnings.append("Less than 5 s still at one/both ends; use available segments for motion checks.")
    np.savez_compressed(destination / "motion.npz", odom_time_ns=odom_t,
        odom_translation=odom_xyz, odom_quaternion_xyzw=odom_q, odom_linear_velocity=velocity,
        odom_angular_velocity=angular, imu_time_ns=imu_t, imu_acceleration_raw=imu_acc,
        imu_angular_velocity_raw=imu_gyro, imu_orientation_raw=np.asarray(imu_q),
        T_base_lidar=mount, frame_time_ns=frame_times, frame_point_count=np.asarray(frame_counts),
        frame_offset_max_s=np.asarray(offset_max), full_odom_coverage=covered)
    previews = []
    if stationary_samples:
        xyz, intensity, frame_index = (np.concatenate([p[i] for p in stationary_samples]) for i in range(3))
        xyz, intensity, support = voxel_cloud(xyz, intensity, frame_index)
        if not len(xyz):
            raise ValueError(f"No repeatedly observed stationary reference voxels in {bag}")
        np.savez_compressed(destination / "stationary.npz", xyz=xyz, intensity=intensity, support_scans=support)
        write_cloud(destination / "stationary.ply", xyz, intensity, support)
        previews.append("stationary.ply")
    else:
        for j in np.unique(np.linspace(0, len(candidates) - 1, min(3, len(candidates)), dtype=int)):
            index, xyz, intensity = candidates[j]
            name = f"frame_{index:06d}.ply"
            write_cloud(destination / name, xyz, intensity)
            previews.append(name)
    report = {**entry, "source_files": [{"path": str(f.resolve()), "bytes": f.stat().st_size,
        "sha256": _sha256(f)} for f in files], "topic_counts": dict(counts), "timing": timing,
        "points": int(sum(frame_counts)), "nonfinite_points": invalid_points, "zero_range_points": zero_points,
        "frame_id": "livox_frame", "imu_frame_ids": sorted(imu_frames), "T_base_lidar": mount.tolist(),
        "initial_T_odom_lidar": initial_pose.tolist(), "odom_path_m": path_m,
        "maximum_speed_m_s": float(speed.max()), "maximum_angular_speed_rad_s": float(angular_speed.max()),
        "stationary_by_wheel_odom": is_still, "start_still_s": start_still, "end_still_s": end_still,
        "imu_acceleration_norm_median_raw": accel_median, "imu_acceleration_unit_hint": unit_hint,
        "imu_orientation_is_constant": bool(np.allclose(imu_q, imu_q[0])),
        "maximum_header_timebase_difference_ns": int(max(abs(d) for d in header_diffs)),
        "warnings": warnings, "preview_files": previews, "labeling_status": "pending_geometry_review"}
    _json(destination / "audit.json", report)
    return report


def prepare_car_dataset(root: str | Path, output: str | Path) -> dict:
    root, output = Path(root).expanduser().resolve(), Path(output).expanduser().resolve()
    entries, ignored = discover_recordings(root)
    if output == root or root in output.parents:
        raise ValueError("Use a derived output directory outside the raw recording directory")
    _fresh_directory(output)
    write_run_provenance(output, {"stationary_frame_stride": 5, "point_stride": 4,
        "voxel_m": .04, "minimum_support_frames": 3, "preview_range_m": [.3, 30]},
        {"command": "car-prepare", "root": str(root), "output": str(output)})
    typestore = get_typestore(Stores.ROS2_KILTED)
    reports = []
    for entry in entries:
        print(f"Preparing {entry['id']}", flush=True)
        reports.append(_prepare_recording(entry, output, typestore))
    layouts = {}
    for layout in sorted({r["layout"] for r in reports}):
        members = [r for r in reports if r["layout"] == layout]
        refs = [r for r in members if r["role"] == "reference"]
        if not refs:
            raise ValueError(f"No stationary references for {layout}")
        anchor_inv = np.linalg.inv(np.asarray(refs[0]["initial_T_odom_lidar"]))
        review_entries = []
        for index, ref in enumerate(refs):
            npz = output / "recordings" / ref["id"] / "stationary.npz"
            review_entries.append({"id": ref["id"], "cloud": str(npz.relative_to(output)),
                "cloud_sha256": _sha256(npz), "alignment_reviewed": False,
                "T_layout_lidar": np.eye(4).tolist() if index == 0 else None,
                "unverified_odom_guess": (anchor_inv @ np.asarray(ref["initial_T_odom_lidar"])).tolist(),
                "exclude_boxes_sensor_frame": [], "notes": ""})
        review = {"schema": "car_reference_alignment_v1", "layout": layout, "anchor": refs[0]["id"],
            "convention": "p_layout = T_layout_lidar @ p_lidar; column vectors; metres",
            "layout_matches_smoke_confirmed": False, "notes": "",
            "references": review_entries,
            "clean_controls": [r["id"] for r in members if r["role"] == "clean_control"],
            "smoke_runs": [r["id"] for r in members if r["role"] == "smoke"]}
        folder = output / "layouts" / layout
        folder.mkdir(parents=True)
        _json(folder / "alignment.json", review)
        layouts[layout] = str((folder / "alignment.json").relative_to(output))
    manifest = {"schema": "car_preparation_v1", "raw_root": str(root), "layouts": layouts,
        "recordings": reports, "ignored_directories": ignored,
        "status": "reference_alignment_and_moving_pose_validation_required",
        "training_labels_created": False}
    _json(output / "manifest.json", manifest)
    lines = ["# Car recording preparation", "", "No training labels have been generated.", "",
             "| Recording | Role | Seconds | Scans | Warnings |", "| --- | --- | ---: | ---: | ---: |"]
    for r in reports:
        t = r["timing"]["lidar"]
        lines.append(f"| {r['id']} | {r['role']} | {t['duration_s']:.1f} | {t['count']} | {len(r['warnings'])} |")
    lines += ["", "## Warnings", ""]
    for r in reports:
        if r["warnings"]:
            lines += [f"### {r['id']}", ""] + [f"- {w}" for w in r["warnings"]] + [""]
    lines += ["See labeler/CAR_LABELING.md for alignment and motion-review steps.",
              "Canonical IDs are aliases; original bag names and bytes are preserved."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"output": str(output), "recordings": len(reports), "layouts": list(layouts),
            "manifest": str(output / "manifest.json"), "training_labels_created": False}


def build_car_reference(prepared: str | Path, layout: str, output: str | Path) -> dict:
    prepared, output = Path(prepared).expanduser().resolve(), Path(output).expanduser().resolve()
    manifest = json.loads((prepared / "manifest.json").read_text())
    if layout not in manifest["layouts"]:
        raise ValueError(f"Unknown layout: {layout}")
    review_path = prepared / manifest["layouts"][layout]
    review = json.loads(review_path.read_text())
    if review.get("layout_matches_smoke_confirmed") is not True:
        raise ValueError("Confirm layout consistency (or document/exclude changes) in alignment.json first")
    expected = {r["id"] for r in manifest["recordings"] if r["layout"] == layout and r["role"] == "reference"}
    if len(review["references"]) != len(expected) or {r["id"] for r in review["references"]} != expected:
        raise ValueError("Reference review entries do not match the preparation manifest")
    pieces = []
    for ref in review["references"]:
        if ref.get("alignment_reviewed") is not True:
            raise ValueError(f"Alignment has not been reviewed: {ref['id']}")
        transform = validate_transform(ref["T_layout_lidar"])
        path = prepared / ref["cloud"]
        if _sha256(path) != ref["cloud_sha256"]:
            raise ValueError(f"Reference cloud changed since preparation: {path}")
        with np.load(path, allow_pickle=False) as cloud:
            xyz, intensity, support = cloud["xyz"], cloud["intensity"], cloud["support_scans"]
        keep = np.ones(len(xyz), dtype=bool)
        for box in ref.get("exclude_boxes_sensor_frame", []):
            lo, hi = np.asarray(box["min_m"]), np.asarray(box["max_m"])
            if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(lo >= hi):
                raise ValueError("Exclusion boxes require finite min_m < max_m triples")
            keep &= ~np.all((xyz >= lo) & (xyz <= hi), axis=1)
        pieces.append((transform_points(transform, xyz[keep]), intensity[keep], support[keep]))
    xyz, intensity, support = (np.concatenate([p[i] for p in pieces]) for i in range(3))
    if not len(xyz):
        raise ValueError("Reference is empty after exclusions")
    # Retain measured points, avoiding invented voxel-centroid surfaces.
    _, first = np.unique(np.floor(xyz / .04).astype(np.int64), axis=0, return_index=True)
    xyz, intensity, support = xyz[first], intensity[first], support[first]
    _fresh_directory(output)
    np.savez_compressed(output / "reference.npz", xyz=xyz, intensity=intensity, support_scans=support)
    write_cloud(output / "reference.ply", xyz, intensity, support)
    _json(output / "alignment_used.json", review)
    result = {"layout": layout, "points": len(xyz), "reference": str(output / "reference.npz"),
              "alignment_sha256": _sha256(review_path), "reference_sha256": _sha256(output / "reference.npz"),
              "status": "reference_built_moving_pose_and_clean_control_validation_pending"}
    _json(output / "reference_manifest.json", result)
    return result
