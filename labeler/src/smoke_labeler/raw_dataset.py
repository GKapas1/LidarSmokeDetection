from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from .bag import BagPoints, frame_layout, read_bag_points
from .core import (
    IGNORE,
    SMOKE_IMPACTED,
    UNAFFECTED,
    Calibration,
    DirectionalReference,
    build_directional_reference,
    calibrate_early_return_thresholds,
    label_points,
    query_reference,
)
from .pipeline import _write_preview
from .provenance import write_run_provenance


def _recording_name(value: str | Path) -> str:
    path = Path(value).expanduser()
    if path.suffix.lower() == ".mcap":
        return path.stem
    return path.name


def _finite_list(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def _counts(labels: np.ndarray) -> dict[str, int]:
    return {
        "unaffected": int(np.count_nonzero(labels == UNAFFECTED)),
        "smoke_impacted": int(np.count_nonzero(labels == SMOKE_IMPACTED)),
        "ignore": int(np.count_nonzero(labels == IGNORE)),
    }


def _validate_raw(points: BagPoints, role: str) -> None:
    if "CustomMsg" not in points.message_type:
        raise ValueError(
            f"{role} must contain livox_ros_driver2/msg/CustomMsg; got {points.message_type!r} "
            f"on {points.topic!r}."
        )
    ptr, _ = frame_layout(points)
    if len(ptr) <= 2:
        raise RuntimeError(f"{role} contains fewer than two complete lidar frames")


def _window_rows(
    stream: TextIO,
    recording_name: str,
    condition: str,
    session_id: str,
    labels: np.ndarray,
    frame_ptr: np.ndarray,
    frame_time_s: np.ndarray,
    sequence_length: int,
    maximum_frame_gap_s: float,
) -> tuple[int, int]:
    written = 0
    skipped_gap = 0
    for target_frame in range(sequence_length - 1, len(frame_time_s)):
        start_frame = target_frame - sequence_length + 1
        times = frame_time_s[start_frame : target_frame + 1]
        if (
            not np.isfinite(times).all()
            or (len(times) > 1 and np.any(np.diff(times) > maximum_frame_gap_s))
            or (len(times) > 1 and np.any(np.diff(times) <= 0))
        ):
            skipped_gap += 1
            continue
        point_start = int(frame_ptr[target_frame])
        point_end = int(frame_ptr[target_frame + 1])
        target_labels = labels[point_start:point_end]
        row = {
            "session_id": session_id,
            "recording": recording_name,
            "condition": condition,
            "file": f"recordings/{recording_name}.npz",
            "history_start_frame": start_frame,
            "target_frame": target_frame,
            "sequence_length": sequence_length,
            "target_point_start": point_start,
            "target_point_end": point_end,
            "target_time_s": float(frame_time_s[target_frame]),
            "unaffected_points": int(np.count_nonzero(target_labels == UNAFFECTED)),
            "smoke_impacted_points": int(np.count_nonzero(target_labels == SMOKE_IMPACTED)),
            "ignore_points": int(np.count_nonzero(target_labels == IGNORE)),
            "split": "unassigned",
        }
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        written += 1
    return written, skipped_gap


def _save_labeled_recording(
    points: BagPoints,
    recording_name: str,
    condition: str,
    session_id: str,
    reference: DirectionalReference,
    calibration: Calibration,
    config: dict[str, dict[str, Any]],
    output_dir: Path,
    manifest: TextIO,
    held_out_after_s: float | None = None,
) -> dict[str, Any]:
    ref_cfg = config["reference"]
    dataset_cfg = config["dataset"]
    output_cfg = config["output"]

    query = query_reference(
        reference,
        points.xyz,
        int(ref_cfg["min_returns_per_cell"]),
        float(ref_cfg["max_cell_depth_spread_m"]),
    )
    labels, confidence, threshold = label_points(query, calibration)
    ranges = query["observed_range"]
    input_valid = (
        np.isfinite(points.xyz).all(axis=1)
        & np.isfinite(ranges)
        & (ranges >= float(dataset_cfg["minimum_input_range_m"]))
        & (ranges <= float(dataset_cfg["maximum_input_range_m"]))
    )
    labels[~input_valid] = IGNORE
    confidence[~input_valid] = 0.0

    ptr, frame_times = frame_layout(points)
    frame_lookup = np.clip(points.frame_index, 0, max(len(frame_times) - 1, 0))
    point_offset_s = (points.time_s - frame_times[frame_lookup]).astype(np.float32)

    recordings_dir = output_dir / "recordings"
    qc_dir = output_dir / "qc"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        recordings_dir / f"{recording_name}.npz",
        schema_version=np.array("1.0"),
        session_id=np.array(session_id),
        recording=np.array(recording_name),
        condition=np.array(condition),
        topic=np.array(points.topic),
        message_type=np.array(points.message_type),
        xyz=points.xyz.astype(np.float32, copy=False),
        range_m=ranges.astype(np.float32, copy=False),
        reflectivity=points.reflectivity.astype(np.float32, copy=False),
        tag=points.tag.astype(np.uint8, copy=False),
        line=points.line.astype(np.uint8, copy=False),
        point_time_s=points.time_s.astype(np.float64, copy=False),
        point_offset_s=point_offset_s,
        frame_index=points.frame_index.astype(np.int32, copy=False),
        frame_ptr=ptr.astype(np.int64, copy=False),
        frame_time_s=frame_times.astype(np.float64, copy=False),
        input_valid=input_valid,
        label=labels,
        label_confidence=confidence.astype(np.float32, copy=False),
    )

    _write_preview(
        qc_dir / f"{recording_name}_preview.ply",
        points.xyz,
        labels,
        int(output_cfg["preview_max_points"]),
    )

    audit_count = min(len(points.xyz), int(output_cfg["preview_max_points"]))
    audit_indices = (
        np.linspace(0, len(points.xyz) - 1, audit_count, dtype=np.int64)
        if audit_count
        else np.empty(0, dtype=np.int64)
    )
    np.savez_compressed(
        qc_dir / f"{recording_name}_teacher_sample.npz",
        point_index=audit_indices,
        expected_clean_range_m=query["expected_range"][audit_indices],
        range_residual_m=query["range_residual"][audit_indices],
        expected_clean_reflectivity=query["expected_reflectivity"][audit_indices],
        early_return_threshold_m=threshold[audit_indices],
        valid_reference=query["valid_reference"][audit_indices],
        reference_count=query["reference_count"][audit_indices],
        reference_spread_m=query["reference_spread"][audit_indices],
        label=labels[audit_indices],
    )

    windows, skipped_gap = _window_rows(
        manifest,
        recording_name,
        condition,
        session_id,
        labels,
        ptr,
        frame_times,
        int(dataset_cfg["sequence_length"]),
        float(dataset_cfg["maximum_frame_gap_s"]),
    )
    counts = _counts(labels)
    labeled = counts["unaffected"] + counts["smoke_impacted"]
    summary = {
        "recording": recording_name,
        "condition": condition,
        "reader_backend": points.reader_backend,
        "source_files": points.source_files,
        "points": int(len(points.xyz)),
        "frames": int(len(frame_times)),
        "time_span_s": float(frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0.0,
        "truncated_at_configured_point_limit": bool(points.truncated),
        "labels": counts,
        "label_percent": {
            key: round(100.0 * value / max(len(labels), 1), 4) for key, value in counts.items()
        },
        "smoke_rate_among_nonignored_percent": round(
            100.0 * counts["smoke_impacted"] / max(labeled, 1), 4
        ),
        "windows": windows,
        "windows_skipped_for_timing_gap": skipped_gap,
        "data_file": f"recordings/{recording_name}.npz",
        "preview_file": f"qc/{recording_name}_preview.ply",
    }
    if held_out_after_s is not None:
        held_out = points.time_s >= held_out_after_s
        held_labels = labels[held_out]
        held_counts = _counts(held_labels)
        held_labeled = held_counts["unaffected"] + held_counts["smoke_impacted"]
        summary["held_out_clean_validation"] = {
            "starts_at_s": float(held_out_after_s),
            "points": int(len(held_labels)),
            "labels": held_counts,
            "smoke_rate_total_percent": round(
                100.0 * held_counts["smoke_impacted"] / max(len(held_labels), 1), 4
            ),
            "smoke_rate_among_nonignored_percent": round(
                100.0 * held_counts["smoke_impacted"] / max(held_labeled, 1), 4
            ),
            "ignore_rate_percent": round(
                100.0 * held_counts["ignore"] / max(len(held_labels), 1), 4
            ),
        }
    with (qc_dir / f"{recording_name}_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return summary


def _write_schema(output_dir: Path, sequence_length: int) -> None:
    schema = {
        "schema_version": "1.0",
        "description": "Frame-preserving raw Livox point-label dataset",
        "model_input_arrays": {
            "xyz": "float32 [P,3], original sensor-frame metres",
            "range_m": "float32 [P]",
            "reflectivity": "float32 [P], original Livox 0..255 values",
            "tag": "uint8 [P]",
            "line": "uint8 [P]",
            "point_offset_s": "float32 [P], point time relative to its frame",
            "frame_ptr": "int64 [F+1], point slice for each complete ROS message",
            "frame_time_s": "float64 [F], relative recording time",
            "input_valid": "bool [P], preprocessing range/finite mask",
        },
        "target_arrays": {
            "label": {"dtype": "uint8", "0": "unaffected", "1": "smoke_impacted", "255": "ignore"},
            "label_confidence": "float32 [P]",
        },
        "window_manifest": {
            "file": "windows.jsonl",
            "sequence_length": sequence_length,
            "target": "Only points in target_frame contribute to the loss",
            "split": "unassigned; assign whole sessions before training",
        },
        "teacher_only": "Clean-reference fields are stored only as sampled files under qc/ and must not be model inputs.",
    }
    with (output_dir / "dataset_schema.json").open("w", encoding="utf-8") as stream:
        json.dump(schema, stream, indent=2)
        stream.write("\n")


def run_raw_dataset(
    config: dict[str, dict[str, Any]],
    clean_reference_path: str | Path,
    clean_control_path: str | Path,
    smoke_paths: list[str | Path],
    output: str | Path,
    topic: str | None = "/livox/lidar",
    session_id: str = "",
) -> dict[str, Any]:
    if not smoke_paths:
        raise ValueError("At least one --smoke bag is required")
    output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session_id = session_id.strip() or Path(clean_reference_path).expanduser().parent.name

    sampling = config["sampling"]
    ref_cfg = config["reference"]
    cal_cfg = config["calibration"]
    selected_topic = topic.strip() if topic else None

    write_run_provenance(output_dir, config, {
        "session_id": session_id,
        "requested_topic": selected_topic,
        "clean_reference": str(Path(clean_reference_path).expanduser().resolve()),
        "clean_control": str(Path(clean_control_path).expanduser().resolve()),
        "smoke": [str(Path(path).expanduser().resolve()) for path in smoke_paths],
        "output": str(output_dir),
    })

    print(f"Reading clean reference: {clean_reference_path}")
    clean_reference = read_bag_points(
        clean_reference_path,
        topic=selected_topic,
        point_stride=int(sampling["reference_point_stride"]),
        max_points=int(sampling["max_reference_points"]),
    )
    _validate_raw(clean_reference, "Clean reference")
    selected_topic = clean_reference.topic
    print(f"  {len(clean_reference.xyz):,} sampled points in {len(frame_layout(clean_reference)[1]):,} frames")
    print("Building clean directional reference")
    reference = build_directional_reference(
        clean_reference.xyz,
        clean_reference.reflectivity,
        float(ref_cfg["angular_resolution_deg"]),
    )
    clean_reference_sources = clean_reference.source_files
    clean_message_type = clean_reference.message_type
    clean_reference_point_count = int(len(clean_reference.xyz))
    clean_reference_was_truncated = clean_reference.truncated
    del clean_reference
    gc.collect()

    print(f"Reading independent clean control for calibration: {clean_control_path}")
    calibration_points = read_bag_points(
        clean_control_path,
        topic=selected_topic,
        point_stride=int(sampling["calibration_point_stride"]),
        max_points=int(sampling["max_calibration_points"]),
    )
    _validate_raw(calibration_points, "Clean control")
    print(f"  {len(calibration_points.xyz):,} sampled calibration points")
    control_query = query_reference(
        reference,
        calibration_points.xyz,
        int(ref_cfg["min_returns_per_cell"]),
        float(ref_cfg["max_cell_depth_spread_m"]),
    )
    control_fraction = float(config["dataset"]["clean_control_calibration_fraction"])
    if not 0.2 <= control_fraction <= 0.8:
        raise ValueError("dataset.clean_control_calibration_fraction must be between 0.2 and 0.8")
    _, calibration_frame_times = frame_layout(calibration_points)
    split_frame = max(1, min(len(calibration_frame_times) - 1, int(len(calibration_frame_times) * control_fraction)))
    control_validation_start_s = float(calibration_frame_times[split_frame])
    calibration_mask = calibration_points.time_s < control_validation_start_s
    if np.count_nonzero(calibration_mask) < 1000 or np.count_nonzero(~calibration_mask) < 1000:
        raise RuntimeError("Clean control is too short to split into calibration and held-out validation portions")
    calibration = calibrate_early_return_thresholds(
        control_query["expected_range"][calibration_mask],
        control_query["range_residual"][calibration_mask],
        control_query["valid_reference"][calibration_mask],
        np.asarray(cal_cfg["range_band_edges_m"], dtype=np.float32),
        float(cal_cfg["false_positive_quantile"]),
        float(cal_cfg["minimum_early_return_m"]),
        int(cal_cfg["minimum_calibration_samples_per_band"]),
        float(cal_cfg["normal_quantile"]),
        float(cal_cfg["minimum_normal_residual_m"]),
        float(cal_cfg["maximum_usable_early_return_m"]),
    )
    np.savez_compressed(
        output_dir / "clean_reference.npz",
        comparison_mode=np.array("directional_ray"),
        topic=np.array(selected_topic),
        **reference.as_npz_dict(),
        **calibration.as_npz_dict(),
    )
    calibration_was_truncated = calibration_points.truncated
    del control_query, calibration_points
    gc.collect()

    _write_schema(output_dir, int(config["dataset"]["sequence_length"]))
    recordings: list[dict[str, Any]] = []
    warnings: list[str] = []
    if clean_reference_was_truncated:
        warnings.append("Clean reference reached max_reference_points; increase the limit if the retained time span is too short")
    if calibration_was_truncated:
        warnings.append("Clean-control calibration reached max_calibration_points")
    if not np.any(calibration.usable_band):
        warnings.append("No range band has a trustworthy raw early-return threshold")

    manifest_path = output_dir / "windows.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        targets: list[tuple[str | Path, str]] = [(clean_control_path, "clean_control")]
        targets.extend((path, "smoke") for path in smoke_paths)
        seen_names: set[str] = set()
        for path, condition in targets:
            name = _recording_name(path)
            if name in seen_names:
                raise ValueError(f"Duplicate output recording name: {name}")
            seen_names.add(name)
            print(f"Labeling {condition}: {path}")
            points = read_bag_points(
                path,
                topic=selected_topic,
                point_stride=int(sampling["target_point_stride"]),
                max_points=int(sampling["max_target_points"]),
            )
            _validate_raw(points, name)
            summary = _save_labeled_recording(
                points,
                name,
                condition,
                session_id,
                reference,
                calibration,
                config,
                output_dir,
                manifest,
                control_validation_start_s if condition == "clean_control" else None,
            )
            recordings.append(summary)
            if summary["truncated_at_configured_point_limit"]:
                warnings.append(f"{name} reached max_target_points and is incomplete")
            print(json.dumps(summary["labels"], indent=2))
            del points
            gc.collect()

    control_summary = recordings[0]
    held_out_control = control_summary["held_out_clean_validation"]
    if held_out_control["smoke_rate_among_nonignored_percent"] > 0.2:
        warnings.append("Independent clean-control smoke rate exceeds 0.2%")

    dataset_summary = {
        "schema_version": "1.0",
        "effective_config_file": "effective_config.json",
        "run_provenance_file": "run_provenance.json",
        "session_id": session_id,
        "topic": selected_topic,
        "message_type": clean_message_type,
        "clean_reference_source_files": clean_reference_sources,
        "sampled_clean_reference_points": clean_reference_point_count,
        "reference_cells": int(len(reference.cell_ids)),
        "range_band_edges_m": calibration.range_band_edges_m.tolist(),
        "normal_residual_threshold_m": _finite_list(calibration.normal_residual_threshold_m),
        "smoke_early_return_threshold_m": _finite_list(calibration.early_return_threshold_m),
        "calibration_count": calibration.calibration_count.tolist(),
        "usable_band": calibration.usable_band.tolist(),
        "sequence_length": int(config["dataset"]["sequence_length"]),
        "clean_control_calibration_fraction": control_fraction,
        "held_out_clean_validation": held_out_control,
        "label_meaning": {"0": "unaffected", "1": "smoke_impacted", "255": "ignore"},
        "recordings": recordings,
        "windows": int(sum(item["windows"] for item in recordings)),
        "warnings": warnings,
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(dataset_summary, stream, indent=2)
        stream.write("\n")
    print(f"Wrote raw frame dataset to {output_dir}")
    return dataset_summary
