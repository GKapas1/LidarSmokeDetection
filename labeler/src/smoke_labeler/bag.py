from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


@dataclass
class BagPoints:
    xyz: np.ndarray
    reflectivity: np.ndarray
    tag: np.ndarray
    line: np.ndarray
    time_s: np.ndarray
    frame_index: np.ndarray
    topic: str
    message_type: str
    source_files: list[str]
    frame_ptr: np.ndarray | None = None
    frame_time_s: np.ndarray | None = None
    truncated: bool = False
    reader_backend: str = "livox_cdr_numpy"


def resolve_mcap_sources(value: str | Path) -> list[Path]:
    requested = Path(value).expanduser()
    candidates = [requested]
    if requested.suffix.lower() != ".mcap":
        candidates.append(requested.with_suffix(".mcap"))

    for candidate in candidates:
        if candidate.is_file():
            return [candidate.resolve()]
        if candidate.is_dir():
            files = sorted(candidate.glob("*.mcap"))
            if files:
                return [p.resolve() for p in files]

    parent = requested.parent if requested.parent != Path("") else Path(".")
    matches = sorted(parent.glob(f"{requested.name}*.mcap")) if parent.exists() else []
    if matches:
        return [p.resolve() for p in matches]
    raise FileNotFoundError(
        f"Could not resolve '{value}'. Expected an .mcap file or a ROS 2 bag directory containing .mcap files."
    )


def _align_cdr(offset: int, alignment: int, base: int) -> int:
    relative = offset - base
    return offset + (-relative % alignment)


def _custom_msg_cdr_arrays(
    data: bytes,
    point_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Decode Livox CustomMsg directly from CDR without creating point objects.

    ROS 2 Humble uses CDR1. CustomPoint has 4-byte alignment and is normally
    serialized at a 20-byte stride (19 data bytes plus inter-element padding).
    The packed 19-byte form is accepted as well for compatibility.
    """
    if len(data) < 32:
        raise ValueError("Livox CustomMsg CDR payload is too short")
    encapsulation = int.from_bytes(data[:2], byteorder="big", signed=False)
    endian = "<" if encapsulation in {1, 3} else ">"
    offset = 4

    # std_msgs/Header: builtin_interfaces/Time followed by frame_id string.
    offset += 8
    if offset + 4 > len(data):
        raise ValueError("Truncated Livox Header")
    string_length = struct.unpack_from(endian + "I", data, offset)[0]
    offset += 4
    if string_length > len(data) - offset:
        raise ValueError("Invalid Livox frame_id length")
    offset += int(string_length)

    candidates: list[tuple[int, int, int, int]] = []
    for alignment_base in (4, 0):
        timebase_offset = _align_cdr(offset, 8, alignment_base)
        fixed_end = timebase_offset + 8 + 4 + 1 + 3 + 4
        if fixed_end > len(data):
            continue
        timebase = int(struct.unpack_from(endian + "Q", data, timebase_offset)[0])
        point_num = int(struct.unpack_from(endian + "I", data, timebase_offset + 8)[0])
        sequence_count = int(struct.unpack_from(endian + "I", data, timebase_offset + 16)[0])
        if sequence_count != point_num or sequence_count > 2_000_000:
            continue
        remaining = len(data) - fixed_end
        if sequence_count == 0:
            candidates.append((timebase, sequence_count, fixed_end, 20))
        elif remaining >= sequence_count * 20 - 1:
            candidates.append((timebase, sequence_count, fixed_end, 20))
        elif remaining >= sequence_count * 19:
            candidates.append((timebase, sequence_count, fixed_end, 19))
    if not candidates:
        raise ValueError("Could not locate the Livox CustomPoint sequence in CDR payload")

    timebase, count, points_offset, item_stride = candidates[0]
    take = np.arange(0, count, max(1, point_stride), dtype=np.int64)
    if count == 0:
        return (
            np.empty((0, 3), np.float32),
            np.empty(0, np.float32),
            np.empty(0, np.uint8),
            np.empty(0, np.uint8),
            np.empty(0, np.float64),
            timebase,
        )

    def field_view(dtype: str, field_offset: int) -> np.ndarray:
        return np.ndarray(
            shape=(count,),
            dtype=np.dtype(endian + dtype),
            buffer=data,
            offset=points_offset + field_offset,
            strides=(item_stride,),
        )

    offsets_ns = field_view("u4", 0)[take]
    xyz = np.column_stack(
        (field_view("f4", 4)[take], field_view("f4", 8)[take], field_view("f4", 12)[take])
    ).astype(np.float32, copy=False)
    reflectivity = np.asarray(field_view("u1", 16)[take], dtype=np.float32)
    tag = np.asarray(field_view("u1", 17)[take], dtype=np.uint8)
    line = np.asarray(field_view("u1", 18)[take], dtype=np.uint8)
    offsets = np.asarray(offsets_ns, dtype=np.float64) * 1e-9
    return xyz, reflectivity, tag, line, offsets, timebase


def _read_raw_livox_points(
    files: list[Path],
    topic: str | None,
    point_stride: int,
    max_points: int,
) -> BagPoints | None:
    try:
        from mcap.reader import make_reader
    except ImportError as exc:
        raise RuntimeError("MCAP support is not installed. Run: python -m pip install -e .") from exc

    selected_topic = topic or None
    chunks_xyz: list[np.ndarray] = []
    chunks_refl: list[np.ndarray] = []
    chunks_tag: list[np.ndarray] = []
    chunks_line: list[np.ndarray] = []
    chunks_time: list[np.ndarray] = []
    chunks_frame: list[np.ndarray] = []
    frame_times: list[float] = []
    message_type = ""
    kept = 0
    frame = 0
    time_origin_ns: int | None = None
    truncated = False

    for path in files:
        with path.open("rb") as stream:
            reader = make_reader(stream)
            topics = [selected_topic] if selected_topic else None
            for schema, channel, message in reader.iter_messages(topics=topics):
                schema_name = str(getattr(schema, "name", ""))
                if schema_name != "livox_ros_driver2/msg/CustomMsg":
                    continue
                current_topic = str(channel.topic)
                if selected_topic is None:
                    selected_topic = current_topic
                if current_topic != selected_topic:
                    continue
                try:
                    xyz, reflectivity, tag, line, offsets, timebase = _custom_msg_cdr_arrays(
                        bytes(message.data), point_stride=point_stride
                    )
                except ValueError as exc:
                    raise RuntimeError(f"Failed fast Livox CDR decoding in {path}: {exc}") from exc
                if kept + len(xyz) > max_points:
                    truncated = True
                    break
                time_ns = int(timebase or message.publish_time or message.log_time)
                if time_origin_ns is None:
                    time_origin_ns = time_ns
                relative_frame_time = (time_ns - time_origin_ns) * 1e-9
                chunks_xyz.append(xyz)
                chunks_refl.append(reflectivity)
                chunks_tag.append(tag)
                chunks_line.append(line)
                chunks_time.append((relative_frame_time + offsets).astype(np.float64))
                chunks_frame.append(np.full(len(xyz), frame, dtype=np.int32))
                frame_times.append(relative_frame_time)
                message_type = schema_name
                kept += len(xyz)
                frame += 1
                if kept >= max_points:
                    truncated = True
                    break
            if truncated:
                break

    if not chunks_xyz:
        return None
    xyz_all = np.concatenate(chunks_xyz)
    frame_all = np.concatenate(chunks_frame)
    counts = np.bincount(frame_all, minlength=frame).astype(np.int64)
    frame_ptr = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
    return BagPoints(
        xyz=xyz_all,
        reflectivity=np.concatenate(chunks_refl),
        tag=np.concatenate(chunks_tag),
        line=np.concatenate(chunks_line),
        time_s=np.concatenate(chunks_time),
        frame_index=frame_all,
        topic=selected_topic or "",
        message_type=message_type,
        source_files=[str(path) for path in files],
        frame_ptr=frame_ptr,
        frame_time_s=np.asarray(frame_times, dtype=np.float64),
        truncated=truncated,
        reader_backend="livox_cdr_numpy",
    )


def read_bag_points(
    value: str | Path,
    topic: str | None = None,
    point_stride: int = 1,
    max_points: int = 8_000_000,
) -> BagPoints:
    files = resolve_mcap_sources(value)
    fast_raw = _read_raw_livox_points(files, topic, max(1, point_stride), max_points)
    if fast_raw is not None:
        return fast_raw
    raise RuntimeError(
        f"No complete livox_ros_driver2/msg/CustomMsg frames found on topic {topic!r} "
        f"within max_points={max_points} in {files}. Check the topic, bag type and point limit."
    )


def frame_layout(points: BagPoints) -> tuple[np.ndarray, np.ndarray]:
    """Return CSR-style point offsets and one timestamp per retained frame."""
    if points.frame_ptr is not None and points.frame_time_s is not None:
        return np.asarray(points.frame_ptr, dtype=np.int64), np.asarray(points.frame_time_s, dtype=np.float64)
    if len(points.frame_index) == 0:
        return np.array([0], dtype=np.int64), np.empty(0, dtype=np.float64)
    n_frames = int(np.max(points.frame_index)) + 1
    counts = np.bincount(points.frame_index, minlength=n_frames).astype(np.int64)
    ptr = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
    times = np.empty(n_frames, dtype=np.float64)
    for frame in range(n_frames):
        start, end = int(ptr[frame]), int(ptr[frame + 1])
        times[frame] = float(np.min(points.time_s[start:end])) if end > start else np.nan
    return ptr, times
