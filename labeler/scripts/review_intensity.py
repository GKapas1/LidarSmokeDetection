"""Export single-frame Livox review overlays with the original MCAP point indices."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from mcap.reader import make_reader

from smoke_labeler.bag import _custom_msg_cdr_arrays, resolve_mcap_sources
from smoke_labeler.car import _sha256


def export_frame(path, frame, xyz, intensity, cutoff):
    ranges = np.linalg.norm(xyz, axis=1)
    indices = np.flatnonzero(np.isfinite(ranges) & (ranges >= 0.3) & (ranges <= 25.0))
    weak = intensity[indices] <= cutoff
    fields = [
        ("x", "<f4", "float"), ("y", "<f4", "float"), ("z", "<f4", "float"),
        ("red", "u1", "uchar"), ("green", "u1", "uchar"), ("blue", "u1", "uchar"),
        ("intensity", "u1", "uchar"), ("weak_return", "u1", "uchar"),
        ("frame_index", "<u4", "uint"), ("source_point_index", "<u4", "uint"),
    ]
    points = np.empty(len(indices), dtype=[(name, dtype) for name, dtype, _ in fields])
    for axis, name in enumerate(("x", "y", "z")):
        points[name] = xyz[indices, axis]
    points["red"] = np.where(weak, 255, 115)
    points["green"] = np.where(weak, 151, 132)
    points["blue"] = np.where(weak, 36, 148)
    points["intensity"] = intensity[indices]
    points["weak_return"] = weak
    points["frame_index"] = frame
    points["source_point_index"] = indices
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(points)}"]
    header.extend(f"property {ply_type} {name}" for name, _, ply_type in fields)
    with path.open("wb") as stream:
        stream.write(("\n".join(header) + "\nend_header\n").encode("ascii"))
        points.tofile(stream)
    return {"raw_points": len(xyz), "displayed_points": len(points), "weak_returns": int(weak.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--frames", type=int, nargs="+", required=True, help="Zero-based LiDAR frames")
    parser.add_argument("--cutoff", type=int, default=5, help="Inclusive raw reflectivity cutoff")
    parser.add_argument("--topic", default="/livox/lidar")
    parser.add_argument("--output", type=Path, required=True, help="New directory for this review")
    args = parser.parse_args()
    if not 0 <= args.cutoff <= 255 or min(args.frames) < 0:
        parser.error("Use a cutoff from 0 to 255 and nonnegative frame indices")
    if args.output.exists():
        parser.error(f"Choose a new output directory to preserve the existing review at {args.output}")
    files = resolve_mcap_sources(args.bag)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    pending = set(args.frames)
    frame = -1
    manifest = {"topic": args.topic, "maximum_weak_reflectivity": args.cutoff,
                "display_range_m": [0.3, 25.0], "color_key": {"amber": "weak return", "gray": "other return"},
                "source_files": [], "frames": []}
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        stage = Path(tmp) / "review"
        stage.mkdir()
        for path in files:
            manifest["source_files"].append({"path": str(path), "sha256": _sha256(path)})
            with path.open("rb") as stream:
                reader = make_reader(stream, validate_crcs=True)
                for schema, _, msg in reader.iter_messages(topics=[args.topic]):
                    if schema.name != "livox_ros_driver2/msg/CustomMsg":
                        parser.error(f"Unexpected schema {schema.name} on {args.topic}")
                    frame += 1
                    if frame not in pending:
                        continue
                    xyz, intensity, _, _, _, timestamp = _custom_msg_cdr_arrays(msg.data)
                    name = f"frame_{frame:06}.ply"
                    counts = export_frame(stage / name, frame, xyz, intensity, args.cutoff)
                    manifest["frames"].append({"frame_index": frame, "frame_time_ns": timestamp,
                                               "source_file": path.name, "ply": name, **counts})
                    pending.remove(frame)
                    if not pending:
                        break
            if not pending:
                break
        if pending:
            parser.error(f"Frames missing on {args.topic}: {sorted(pending)}")
        (stage / "review.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stage.rename(args.output)
    print(args.output / "review.json")


if __name__ == "__main__":
    main()
