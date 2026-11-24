"""Audit a fully registered clean car drive for frame-level geometry failures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from mcap.reader import make_reader

from .bag import _custom_msg_cdr_arrays
from .car import _fresh_directory, _sha256, validate_transform
from .car_ray_review import query_directional_ranges
from .car_registration import _OdomInterpolator
from .provenance import write_run_provenance


CONFIG = {
    "point_stride": 3,
    "range_m": [0.3, 25.0],
    "time_bin_s": 0.005,
    # Deliberately conservative candidate thresholds. They are audit flags, not labels.
    "strong_paired_fraction": 0.055,
    "strong_behind_fraction": 0.050,
    "possible_paired_fraction": 0.035,
    "possible_behind_fraction": 0.025,
    "poor_median_distance_m": 0.25,
    "poor_fraction_within_0_20_m": 0.45,
    "poor_p90_distance_m": 4.0,
    "implausible_correction_translation_m": 0.50,
    "implausible_correction_rotation_deg": 5.0,
    "footprint_reference_percentiles": [1.0, 99.0],
    "footprint_margin_m": 0.50,
    "possible_outside_footprint_fraction": 0.08,
    "strong_outside_footprint_fraction": 0.25,
}


def _runs(frames: list[int]) -> list[list[int]]:
    if not frames:
        return []
    result, start, previous = [], frames[0], frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            result.append([start, previous])
            start = frame
        previous = frame
    result.append([start, previous])
    return result


def classify_frame(row: dict[str, float]) -> tuple[str, list[str]]:
    """Return a conservative audit status and explicit machine-readable reasons."""
    reasons: list[str] = []
    pose_failure = (
        row["correction_translation_m"] >= CONFIG["implausible_correction_translation_m"]
        or row["correction_rotation_deg"] >= CONFIG["implausible_correction_rotation_deg"]
    )
    strong_pair = (
        row["paired_mismatch_fraction"] >= CONFIG["strong_paired_fraction"]
        and row["behind_reference_fraction"] >= CONFIG["strong_behind_fraction"]
    )
    possible_pair = (
        row["paired_mismatch_fraction"] >= CONFIG["possible_paired_fraction"]
        and row["behind_reference_fraction"] >= CONFIG["possible_behind_fraction"]
    )
    poor_map = (
        row["median_m"] >= CONFIG["poor_median_distance_m"]
        or row["within_0_20"] <= CONFIG["poor_fraction_within_0_20_m"]
        or row["p90_m"] >= CONFIG["poor_p90_distance_m"]
    )
    possible_outside = (
        row["outside_reference_footprint_fraction"]
        >= CONFIG["possible_outside_footprint_fraction"]
    )
    strong_outside = (
        row["outside_reference_footprint_fraction"]
        >= CONFIG["strong_outside_footprint_fraction"]
    )
    if pose_failure:
        reasons.append("implausible_pose_correction")
    if strong_pair:
        reasons.append("strong_paired_displacement")
    elif possible_pair:
        reasons.append("possible_paired_displacement")
    if poor_map:
        reasons.append("poor_map_agreement")
    if possible_outside:
        reasons.append("outside_reference_footprint")
    if pose_failure or strong_pair or (strong_outside and poor_map):
        return "reject_candidate", reasons
    if possible_pair or poor_map or possible_outside:
        return "review", reasons
    return "keep_candidate", reasons


def audit_clean_drive(prepared, recording_id, reference_dir, registration, output):
    prepared, reference_dir, registration, output = map(
        lambda value: Path(value).expanduser().resolve(),
        (prepared, reference_dir, registration, output),
    )
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    matches = [row for row in manifest["recordings"] if row["id"] == recording_id]
    if len(matches) != 1 or matches[0]["role"] != "clean_control":
        raise ValueError("Audit requires exactly one clean_control recording")
    recording = matches[0]
    source_files = [Path(item["path"]) for item in recording["source_files"]]
    for path, item in zip(source_files, recording["source_files"]):
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"Source changed since preparation: {path}")

    provenance = json.loads((registration / "run_provenance.json").read_text(encoding="utf-8"))
    invocation = provenance["invocation"]
    if invocation["recording"] != recording_id:
        raise ValueError("Registration belongs to a different recording")
    if Path(invocation["reference"]).resolve() != reference_dir:
        raise ValueError("Registration belongs to a different reference")
    with np.load(reference_dir / "reference.npz", allow_pickle=False) as cloud:
        reference = cloud["xyz"].astype(np.float64)
    footprint_center = np.median(reference[:, :2], axis=0)
    _, _, footprint_vt = np.linalg.svd(
        reference[:, :2] - footprint_center, full_matrices=False
    )
    footprint_axes = footprint_vt.T
    reference_footprint = (reference[:, :2] - footprint_center) @ footprint_axes
    footprint_low = np.percentile(
        reference_footprint, CONFIG["footprint_reference_percentiles"][0], axis=0
    ) - CONFIG["footprint_margin_m"]
    footprint_high = np.percentile(
        reference_footprint, CONFIG["footprint_reference_percentiles"][1], axis=0
    ) + CONFIG["footprint_margin_m"]
    with np.load(registration / "trajectory.npz", allow_pickle=False) as trajectory:
        poses = {
            int(frame): (int(time_ns), validate_transform(transform))
            for frame, time_ns, transform in zip(
                trajectory["frame_index"], trajectory["frame_time_ns"],
                trajectory["T_layout_lidar"],
            )
        }
    with (registration / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        registration_rows = {
            int(row["frame_index"]): row for row in csv.DictReader(stream)
        }
    if set(poses) != set(registration_rows):
        raise ValueError("Trajectory and registration metrics cover different frames")
    motion_path = prepared / "recordings" / recording_id / "motion.npz"
    with np.load(motion_path, allow_pickle=False) as motion_file:
        odom = _OdomInterpolator({name: motion_file[name] for name in motion_file.files})

    _fresh_directory(output)
    write_run_provenance(output, CONFIG, {
        "command": "car-audit-clean", "recording": recording_id,
        "reference": str(reference_dir), "registration": str(registration),
        "input_sha256": {
            str(prepared / "manifest.json"): _sha256(prepared / "manifest.json"),
            str(reference_dir / "reference.npz"): _sha256(reference_dir / "reference.npz"),
            str(registration / "trajectory.npz"): _sha256(registration / "trajectory.npz"),
            str(registration / "metrics.csv"): _sha256(registration / "metrics.csv"),
            str(motion_path): _sha256(motion_path),
        },
    })

    rows = []
    source_index = -1
    for path in source_files:
        with path.open("rb") as stream:
            for _, _, message in make_reader(stream, validate_crcs=True).iter_messages(
                topics=["/livox/lidar"]
            ):
                source_index += 1
                if source_index not in poses:
                    continue
                xyz, _, _, _, offsets, timebase = _custom_msg_cdr_arrays(
                    message.data, point_stride=CONFIG["point_stride"]
                )
                frame_time_ns, start_world = poses[source_index]
                if frame_time_ns != int(timebase):
                    raise ValueError(f"LiDAR timestamp mismatch at frame {source_index}")
                ranges = np.linalg.norm(xyz, axis=1)
                valid = (
                    np.isfinite(xyz).all(axis=1)
                    & (ranges >= CONFIG["range_m"][0])
                    & (ranges <= CONFIG["range_m"][1])
                )
                xyz, offsets = xyz[valid], offsets[valid]
                point_time_ns = frame_time_ns + np.rint(offsets * 1e9).astype(np.int64)
                relative = np.linalg.inv(odom.pose(frame_time_ns))[None] @ odom.poses(point_time_ns)
                point_world = start_world[None] @ relative
                origins = point_world[:, :3, 3]
                world = np.einsum("nij,nj->ni", point_world[:, :3, :3], xyz) + origins
                category = query_directional_ranges(
                    reference, world, origins.mean(axis=0)
                )[-1]
                footprint_xy = (world[:, :2] - footprint_center) @ footprint_axes
                outside_footprint = np.any(
                    (footprint_xy < footprint_low) | (footprint_xy > footprint_high),
                    axis=1,
                )
                bins = np.floor(offsets / CONFIG["time_bin_s"]).astype(np.int64)
                paired = 0
                peak = 0.0
                for bin_id in np.unique(bins):
                    selected = bins == bin_id
                    before = int(np.count_nonzero(selected & (category == 2)))
                    behind = int(np.count_nonzero(selected & (category == 3)))
                    pair = min(before, behind)
                    paired += pair
                    peak = max(peak, pair / max(int(np.count_nonzero(selected)), 1))
                reg = registration_rows[source_index]
                row = {
                    "frame_index": source_index,
                    "frame_time_ns": frame_time_ns,
                    "sampled_valid_points": len(category),
                    "near_reference_fraction": float(np.mean(category == 1)),
                    "before_reference_fraction": float(np.mean(category == 2)),
                    "behind_reference_fraction": float(np.mean(category == 3)),
                    "paired_mismatch_fraction": paired / len(category),
                    "peak_time_bin_paired_fraction": peak,
                    "outside_reference_footprint_fraction": float(
                        np.mean(outside_footprint)
                    ),
                    "maximum_origin_approximation_m": float(
                        np.linalg.norm(origins - origins.mean(axis=0), axis=1).max(initial=0)
                    ),
                    "median_m": float(reg["median_m"]),
                    "p90_m": float(reg["p90_m"]),
                    "within_0_20": float(reg["within_0_20"]),
                    "correction_translation_m": float(reg["correction_translation_m"]),
                    "correction_rotation_deg": float(reg["correction_rotation_deg"]),
                }
                status, reasons = classify_frame(row)
                row["status"], row["reasons"] = status, ";".join(reasons)
                rows.append(row)
                if len(rows) % 50 == 0:
                    print(f"audited {len(rows)} frames", flush=True)

    if len(rows) != len(poses):
        raise ValueError(f"Audited {len(rows)} frames but registration contains {len(poses)}")
    with (output / "audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_status = {
        status: [int(row["frame_index"]) for row in rows if row["status"] == status]
        for status in ("keep_candidate", "review", "reject_candidate")
    }
    report = {
        "schema": "car_clean_frame_audit_v1",
        "recording": recording_id,
        "status": "candidate_flags_require_representative_visual_review",
        "thresholds_are_calibrated": False,
        "frame_count": len(rows),
        "reference_footprint_corners_xy": (
            np.asarray([
                [footprint_low[0], footprint_low[1]],
                [footprint_low[0], footprint_high[1]],
                [footprint_high[0], footprint_high[1]],
                [footprint_high[0], footprint_low[1]],
            ]) @ footprint_axes.T + footprint_center
        ).tolist(),
        "counts": {status: len(frames) for status, frames in by_status.items()},
        "ranges": {status: _runs(frames) for status, frames in by_status.items()},
        "limitations": [
            "Flags are conservative diagnostics, not smoke labels or final exclusions.",
            "Directional reference support is not an opaque surface model.",
            "Map coverage, pose error and changed structure can resemble a displaced scan.",
            "Thresholds were selected from this development drive and require validation.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "flagged_frames.txt").open("w", encoding="utf-8") as stream:
        for status in ("reject_candidate", "review"):
            stream.write(f"{status}\n")
            for start, end in report["ranges"][status]:
                stream.write(f"{start}\n" if start == end else f"{start}-{end}\n")
    return report
