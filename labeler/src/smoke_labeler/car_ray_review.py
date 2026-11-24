"""Directional range diagnostics for registered car scans, never training labels.

This tests nearby mapped returns along a viewing direction. A point cloud is not
an opaque mesh: agreement cannot prove a wall is solid, and unsupported directions
remain unknown. Use this pilot to review a labeling rule before adopting it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from scipy.spatial import cKDTree

from .bag import _custom_msg_cdr_arrays
from .car import _fresh_directory, _sha256, validate_transform
from .car_registration import _OdomInterpolator
from .provenance import write_run_provenance


CONFIG = {
    "angular_radius_deg": 0.75, "neighbors": 8, "minimum_support": 4,
    "maximum_depth_spread_m": 0.25, "near_m": 0.20, "mismatch_m": 0.50,
    "origin_bin_s": 0.005, "range_m": [0.3, 25.0],
}
CATEGORIES = {0: "uncertain", 1: "near_reference", 2: "before_reference", 3: "behind_reference"}
COLORS = np.array([[155, 155, 155], [60, 220, 95], [255, 175, 35], [245, 65, 220]], np.uint8)


def query_directional_ranges(reference, points, origin, config=CONFIG):
    """Require several angular neighbors with consistent depths; no surface extrapolation."""
    reference = np.asarray(reference, dtype=np.float64) - origin
    points = np.asarray(points, dtype=np.float64) - origin
    rr = np.linalg.norm(reference, axis=1)
    observed = np.linalg.norm(points, axis=1)
    keep = np.isfinite(reference).all(axis=1) & (rr > 0.3)
    reference, rr = reference[keep], rr[keep]
    expected = np.full(len(points), np.nan)
    support = np.zeros(len(points), np.int32)
    spread = np.full(len(points), np.nan)
    valid_point = np.isfinite(points).all(axis=1) & (observed > 0)
    if len(rr) and valid_point.any():
        tree = cKDTree(reference / rr[:, None])
        chord = 2 * np.sin(np.deg2rad(config["angular_radius_deg"]) / 2)
        distance, index = tree.query(
            points[valid_point] / observed[valid_point, None],
            k=list(range(1, config["neighbors"] + 1)), distance_upper_bound=chord,
        )
        found = np.isfinite(distance)
        depth = np.where(found, rr[np.minimum(index, len(rr) - 1)], np.nan)
        count = found.sum(axis=1)
        sufficient = count >= config["minimum_support"]
        target = np.flatnonzero(valid_point)[sufficient]
        support[valid_point] = count
        if len(target):
            selected = depth[sufficient]
            spread[target] = np.nanmax(selected, axis=1) - np.nanmin(selected, axis=1)
            good = spread[target] <= config["maximum_depth_spread_m"]
            expected[target[good]] = np.nanmedian(selected[good], axis=1)
    residual = observed - expected  # Positive means measured beyond mapped returns.
    category = np.zeros(len(points), np.uint8)
    category[np.abs(residual) <= config["near_m"]] = 1
    category[residual < -config["mismatch_m"]] = 2
    category[residual > config["mismatch_m"]] = 3
    return expected, residual, support, spread, category


def write_review_ply(path, reference, world, category, residual):
    xyz = np.vstack((reference, world))
    count = len(xyz)
    values = np.empty(count, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("source", "u1"), ("diagnostic_category", "u1"), ("signed_range_residual_m", "<f4")])
    for axis, name in enumerate(("x", "y", "z")):
        values[name] = xyz[:, axis]
    rgb = np.vstack((np.tile([65, 100, 130], (len(reference), 1)), COLORS[category]))
    for axis, name in enumerate(("red", "green", "blue")):
        values[name] = rgb[:, axis]
    values["source"] = np.r_[np.zeros(len(reference)), np.ones(len(world))]
    values["diagnostic_category"] = np.r_[np.full(len(reference), 255), category]
    values["signed_range_residual_m"] = np.r_[np.full(len(reference), np.nan), residual]
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property uchar source\nproperty uchar diagnostic_category\n"
        "property float signed_range_residual_m\nend_header\n")
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(values.tobytes())


def review(prepared, recording_id, reference_dir, registration, frames, output):
    manifest = json.loads((prepared / "manifest.json").read_text())
    record = next(r for r in manifest["recordings"] if r["id"] == recording_id)
    if record["role"] != "clean_control":
        raise ValueError("This diagnostic pilot is restricted to clean controls")
    invocation = json.loads((registration / "run_provenance.json").read_text())["invocation"]
    if invocation["recording"] != recording_id or Path(invocation["reference"]).resolve() != reference_dir.resolve():
        raise ValueError("Registration recording/reference mismatch")
    paths = [Path(s["path"]) for s in record["source_files"]]
    for path, source in zip(paths, record["source_files"]):
        if _sha256(path) != source["sha256"]:
            raise ValueError(f"Source changed since preparation: {path}")
    with np.load(reference_dir / "reference.npz", allow_pickle=False) as z:
        reference = z["xyz"].astype(np.float64)
    with np.load(registration / "trajectory.npz", allow_pickle=False) as z:
        poses = {int(i): (int(t), validate_transform(p)) for i, t, p in
                 zip(z["frame_index"], z["frame_time_ns"], z["T_layout_lidar"])}
    if any(i not in poses for i in frames):
        raise ValueError("Requested frame is outside the registered pilot")
    motion_path = prepared / "recordings" / recording_id / "motion.npz"
    with np.load(motion_path, allow_pickle=False) as z:
        odom = _OdomInterpolator({k: z[k] for k in z.files})
    _fresh_directory(output)
    write_run_provenance(output, CONFIG, {
        "command": "car_ray_review", "recording": recording_id, "frames": frames,
        "reference": str(reference_dir.resolve()), "registration": str(registration.resolve()),
        "input_sha256": {str(p): _sha256(p) for p in
                         [prepared / "manifest.json", reference_dir / "reference.npz",
                          registration / "trajectory.npz", motion_path]},
    })
    summaries = []
    frame = -1
    for path in paths:
        with path.open("rb") as stream:
            for _, _, message in make_reader(stream, validate_crcs=True).iter_messages(topics=["/livox/lidar"]):
                frame += 1
                if frame > max(frames):
                    break
                if frame not in frames:
                    continue
                xyz, intensity, tag, line, offsets, timebase = _custom_msg_cdr_arrays(message.data)
                timestamp, pose = poses[frame]
                if timestamp != timebase:
                    raise ValueError("Frame timestamp mismatch")
                ranges = np.linalg.norm(xyz, axis=1)
                valid = np.isfinite(xyz).all(axis=1) & (ranges >= CONFIG["range_m"][0]) & (ranges <= CONFIG["range_m"][1])
                point_index = np.flatnonzero(valid)
                xyz, intensity, tag, line, offsets = [a[valid] for a in (xyz, intensity, tag, line, offsets)]
                times = timestamp + np.rint(offsets * 1e9).astype(np.int64)
                relative = np.linalg.inv(odom.pose(timestamp))[None] @ odom.poses(times)
                point_poses = pose[None] @ relative
                origins = point_poses[:, :3, 3]
                world = np.einsum("nij,nj->ni", point_poses[:, :3, :3], xyz) + origins
                expected, residual, support, spread = [np.full(len(xyz), np.nan) for _ in range(4)]
                category = np.zeros(len(xyz), np.uint8)
                bins = np.floor(offsets / CONFIG["origin_bin_s"]).astype(np.int64)
                origin_error = 0.0
                for b in np.unique(bins):
                    take = bins == b
                    origin = origins[take].mean(axis=0)
                    origin_error = max(origin_error, float(np.linalg.norm(origins[take] - origin, axis=1).max()))
                    result = query_directional_ranges(reference, world[take], origin)
                    for dest, values in zip((expected, residual, support, spread, category), result):
                        dest[take] = values
                stem = f"frame_{frame:06d}"
                np.savez_compressed(output / f"{stem}.npz", sensor_xyz=xyz, world_xyz=world,
                    source_point_index=point_index, intensity=intensity, tag=tag, line=line,
                    point_offset_s=offsets, frame_time_ns=np.int64(timestamp),
                    expected_range_m=expected, signed_range_residual_m=residual,
                    reference_support=support.astype(np.int32), reference_depth_spread_m=spread,
                    diagnostic_category=category)
                write_review_ply(output / f"{stem}_review.ply", reference, world, category, residual)
                row = {"frame": frame, "valid_points": len(xyz),
                       "maximum_origin_approximation_m": origin_error,
                       "counts": {name: int(np.sum(category == c)) for c, name in CATEGORIES.items()}}
                summaries.append(row)
                print(json.dumps(row), flush=True)
    if len(summaries) != len(set(frames)):
        raise ValueError("Not all requested frames were found")
    report = {"status": "diagnostic_only_not_labels", "categories": CATEGORIES,
        "frames": summaries, "limitations": [
            "Nearby map directions do not prove an opaque surface or physical ray intersection.",
            "Edges, holes, sparse coverage, people, pose errors and artifacts can cause mismatches.",
            "Orange points in a clean run are false smoke candidates, not positive labels.",
            "No thresholds are calibrated or independently validated by this three-frame review."]}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "reference", "registration", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--frames", nargs="+", type=int, default=[1, 76, 150])
    args = parser.parse_args()
    review(args.prepared, args.recording, args.reference, args.registration, args.frames, args.output)


if __name__ == "__main__":
    main()
