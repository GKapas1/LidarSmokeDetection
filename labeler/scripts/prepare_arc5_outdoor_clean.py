"""Export the reviewed ARC-5 outdoor interval as a clean training supplement."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tomllib

import numpy as np

from smoke_labeler.grandtour import RAW_TOPIC, iter_livox_frames, resolve_grandtour_paths
from smoke_labeler.grandtour_pipeline import _isolated_return_mask
from smoke_labeler.unified_dataset import _metadata, _schema, _sha256, _write_chunk


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def export(session_dir: Path, config_path: Path, output: Path) -> dict:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    paths = resolve_grandtour_paths(session_dir)
    if config["session_id"] != paths.session_id:
        raise ValueError("Configuration and source session differ")
    time = config["time"]
    start, end = float(time["candidate_start_s"]), float(time["candidate_end_s"])
    train_start, train_end = float(time["training_start_s"]), float(time["training_end_s"])
    duration = float(time["chunk_duration_s"])
    minimum = float(config["input"]["minimum_range_m"])
    maximum = float(config["input"]["maximum_range_m"])
    isolation = float(config["input"]["maximum_isolated_neighbor_distance_m"])
    if not (0 <= start < train_start < train_end < end and duration > 0):
        raise ValueError("Invalid candidate, training or chunk intervals")
    if not (0 <= minimum < maximum and isolation >= 0):
        raise ValueError("Invalid range or isolation filter")
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Only publish a complete dataset. Partial runs disappear on failure.
    with TemporaryDirectory(prefix="arc5-clean-", dir=output.parent) as temporary:
        root = Path(temporary) / "dataset"
        (root / "chunks").mkdir(parents=True)
        pending: list[dict] = []
        chunks: list[dict] = []
        frame_rows: list[dict] = []
        origin_ns = None
        previous_ns = None
        current_key = None
        stats = {"invalid_input": 0, "outside_range": 0, "isolated_return": 0}

        def flush() -> None:
            if not pending:
                return
            index = len(chunks)
            recording_id = f"outdoor_clean_{index:03d}"
            point_counts = np.asarray([len(row["xyz"]) for row in pending], dtype=np.int64)
            ptr = np.r_[np.int64(0), np.cumsum(point_counts)]
            first_ns = pending[0]["header_ns"]
            arrays = {
                **_metadata(paths.session_id, "grandtour_ros1", recording_id, "clean_control"),
                **{name: np.concatenate([row[name] for row in pending]) for name in (
                    "xyz", "intensity", "tag", "line", "point_offset_s", "label"
                )},
                "frame_index": np.repeat(np.arange(len(pending), dtype=np.int32), point_counts),
                "frame_ptr": ptr,
                "frame_time_s": np.asarray(
                    [(row["header_ns"] - first_ns) / 1e9 for row in pending], dtype=np.float64
                ),
            }
            destination = root / "chunks" / f"grandtour_ros1__{paths.session_id}__{recording_id}.npz"
            info = _write_chunk(destination, arrays)
            role = pending[0]["role"]
            for local_index, row in enumerate(pending):
                frame_rows.append({
                    "chunk_index": index, "frame_index": local_index,
                    **{key: row[key] for key in (
                        "source_frame_index", "header_ns", "bag_ns", "source_time_s", "role"
                    )},
                    "points": len(row["label"]),
                    "unimpacted": int(np.count_nonzero(row["label"] == 0)),
                    "ignore": int(np.count_nonzero(row["label"] == 255)),
                    **row["range_counts"],
                })
            chunks.append({
                **info, "path": str(destination.relative_to(root)),
                "source_domain": "grandtour_ros1", "session_id": paths.session_id,
                "recording_id": recording_id, "condition": "clean_control", "split": role,
                "label_policy": "known_clean_range_valid_nonisolated_returns_are_unimpacted",
                "source_time_selection_s": {
                    "start": (first_ns - origin_ns) / 1e9,
                    "end": (pending[-1]["header_ns"] - origin_ns) / 1e9,
                    "end_is_last_frame_timestamp": True,
                },
                "source_frame_index_first": pending[0]["source_frame_index"],
                "source_frame_index_last": pending[-1]["source_frame_index"],
            })
            print(f"Wrote {recording_id}: {len(pending)} frames, {role}", flush=True)
            pending.clear()

        for frame in iter_livox_frames(paths.raw_livox, RAW_TOPIC):
            if origin_ns is None:
                origin_ns = frame.header_time_ns
            if previous_ns is not None and frame.header_time_ns <= previous_ns:
                raise ValueError("Raw frame timestamps are not strictly increasing")
            previous_ns = frame.header_time_ns
            seconds = (frame.header_time_ns - origin_ns) / 1e9
            if seconds < start:
                continue
            if seconds >= end:
                break
            role = "train" if train_start <= seconds < train_end else "excluded"
            segment_start = train_start if role == "train" else (start if seconds < train_start else train_end)
            key = (role, segment_start, int((seconds - segment_start) // duration))
            if key != current_key:
                flush()
                current_key = key
            offsets = ((frame.point_time_ns.astype(np.int64) - frame.header_time_ns) * 1e-9).astype(np.float32)
            ranges = np.linalg.norm(frame.xyz, axis=1)
            finite = np.isfinite(frame.xyz).all(axis=1) & np.isfinite(frame.intensity) & np.isfinite(offsets)
            # Fail explicitly instead of replacing non-finite source features.
            if not finite.all():
                raise ValueError(f"Non-finite source features in frame {frame.frame_index}")
            in_range = (ranges >= minimum) & (ranges <= maximum)
            isolated = _isolated_return_mask(frame.xyz, finite & in_range, isolation)
            valid = finite & in_range & ~isolated
            labels = np.where(valid, 0, 255).astype(np.uint8)
            stats["invalid_input"] += int(np.count_nonzero(~finite))
            stats["outside_range"] += int(np.count_nonzero(finite & ~in_range))
            stats["isolated_return"] += int(np.count_nonzero(isolated))
            counts = np.histogram(ranges[valid], bins=[0, 1, 3, 5, 8, np.inf])[0]
            pending.append({
                "xyz": frame.xyz, "intensity": frame.intensity, "tag": frame.tag,
                "line": frame.line, "point_offset_s": offsets, "label": labels,
                "header_ns": frame.header_time_ns, "bag_ns": frame.bag_time_ns,
                "source_frame_index": frame.frame_index, "source_time_s": seconds,
                "role": role,
                "range_counts": dict(zip(
                    ["clean_under_1m", "clean_1_to_3m", "clean_3_to_5m", "clean_5_to_8m", "clean_over_8m"],
                    map(int, counts), strict=True
                )),
            })
        flush()
        if not chunks or not any(row["split"] == "train" for row in chunks):
            raise ValueError("No training frames in selected interval")
        if frame_rows[-1]["source_time_s"] < end - 0.2:
            raise ValueError("Source recording ends before the requested interval")

        schema = _schema()
        schema["known_clean_policy"] = {
            **config["input"], "label_0": "finite, in-range returns with a same-frame neighbor within the isolation radius",
            "label_255": "out-of-range or isolated returns", "point_order": "all raw points retained in source order",
        }
        write_json(root / "dataset_schema.json", schema)
        manifest = {
            "schema_version": "1.0", "status": "complete", "dataset_schema": "dataset_schema.json",
            "split_policy": "Use split.json. Train-only supplement with excluded doorway buffers.",
            "chunks": chunks, "frames": sum(row["frames"] for row in chunks),
            "points": sum(row["points"] for row in chunks),
            "labels": {key: sum(row["labels"][key] for row in chunks) for key in ("unimpacted", "smoke_impacted", "ignore")},
            "provenance": {
                "source_file": str(paths.raw_livox), "source_sha256": _sha256(paths.raw_livox),
                "source_topic": RAW_TOPIC, "session_origin_header_ns": origin_ns,
                "time_origin": "first raw Livox header timestamp, identical to existing ARC-5 pipeline",
                "configuration": str(config_path.resolve()), "configuration_sha256": _sha256(config_path),
                "exporter_sha256": _sha256(Path(__file__)), "selection": time,
                "review": config["review"], "ignore_counts": stats,
            },
        }
        write_json(root / "dataset_manifest.json", manifest)
        roles = {role: [] for role in ("train", "selection", "calibration", "test")}
        excluded = []
        summaries = {role: dict(frames=0, points=0, unimpacted=0, smoke_impacted=0, ignore=0) for role in roles}
        for index, row in enumerate(chunks):
            assignment = {"chunk_index": index, "ranges": [[0, row["frames"]]]}
            if row["split"] == "train":
                roles["train"].append(assignment)
                for key, value in {"frames": row["frames"], "points": row["points"], **row["labels"]}.items():
                    summaries["train"][key] += value
            else:
                excluded.append(assignment)
        plan = {
            "version": 1, "protocol": "arc5_outdoor_clean_train_only_v1",
            "manifest": "dataset_manifest.json", "manifest_sha256": _sha256(root / "dataset_manifest.json"),
            "purge_frames_each_side": 0,
            "chunks": [{"index": index, "path": row["path"], "sha256": row["sha256"], "frames": row["frames"],
                        "stream_id": f"grandtour_ros1/{paths.session_id}"} for index, row in enumerate(chunks)],
            "roles": roles, "excluded": excluded, "summary": summaries,
        }
        canonical = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        plan["split_sha256"] = hashlib.sha256(canonical).hexdigest()
        write_json(root / "split.json", plan)
        with (root / "source_frames.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(frame_rows[0]))
            writer.writeheader()
            writer.writerows(frame_rows)
        root.rename(output)
    return {"output": str(output), "frames": manifest["frames"], "points": manifest["points"], "training": summaries["train"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=Path("data/grandtour/arc5"))
    parser.add_argument("--config", type=Path, default=Path("labeler/config/grandtour/arc5_outdoor_clean.toml"))
    parser.add_argument("--output", type=Path, default=Path("data/training/arc5_outdoor_clean_v1"))
    args = parser.parse_args()
    print(json.dumps(export(args.session, args.config, args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
