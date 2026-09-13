from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .geometry import PoseTrajectory, pose_matrix, transform_points


RAW_TOPIC = "/boxi/livox/points"
UNDISTORTED_TOPIC = "/boxi/livox/points_undistorted"
DLIO_ODOMETRY_TOPIC = "/boxi/dlio/lidar_map_odometry"


_POINT_FIELD_DTYPES = {
    1: "i1",   # INT8
    2: "u1",   # UINT8
    3: "i2",   # INT16
    4: "u2",   # UINT16
    5: "i4",   # INT32
    6: "u4",   # UINT32
    7: "f4",   # FLOAT32
    8: "f8",   # FLOAT64
}


@dataclass(frozen=True)
class GrandTourPaths:
    session_id: str
    raw_livox: Path
    undistorted_livox: Path
    dlio: Path
    tf_minimal: Path


@dataclass(frozen=True)
class LivoxFrame:
    frame_index: int
    bag_time_ns: int
    header_time_ns: int
    frame_id: str
    xyz: np.ndarray
    intensity: np.ndarray
    tag: np.ndarray
    line: np.ndarray
    point_time_ns: np.ndarray
    point_index: np.ndarray


@dataclass(frozen=True)
class PairedLivoxFrame:
    raw: LivoxFrame
    undistorted_xyz: np.ndarray


def resolve_grandtour_paths(session_dir: str | Path) -> GrandTourPaths:
    root = Path(session_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"GrandTour session directory not found: {root}")

    def exactly_one(pattern: str, role: str) -> Path:
        matches = sorted(root.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {role} file matching {pattern!r} in {root}; "
                f"found {len(matches)}"
            )
        return matches[0]

    raw = exactly_one("*_livox.bag", "raw Livox")
    prefix = raw.name.removesuffix("_livox.bag")
    return GrandTourPaths(
        session_id=prefix,
        raw_livox=raw,
        undistorted_livox=exactly_one("*_livox_undist.bag", "undistorted Livox"),
        dlio=exactly_one("*_dlio.bag", "DLIO"),
        tf_minimal=exactly_one("*_tf_minimal.bag", "minimal TF"),
    )


def _header_time_ns(header: Any) -> int:
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def pointcloud2_array(message: Any) -> np.ndarray:
    """Return a structured zero-copy view of a PointCloud2 message.

    Organized clouds with row padding are copied row-by-row into a packed array.
    Multi-count PointFields are exposed as shaped structured-array fields.
    """
    height = int(message.height)
    width = int(message.width)
    point_step = int(message.point_step)
    row_step = int(message.row_step)
    if height < 0 or width < 0 or point_step <= 0 or row_step < width * point_step:
        raise ValueError("Invalid PointCloud2 dimensions or strides")

    endian = ">" if bool(message.is_bigendian) else "<"
    names: list[str] = []
    formats: list[Any] = []
    offsets: list[int] = []
    for field in message.fields:
        code = int(field.datatype)
        if code not in _POINT_FIELD_DTYPES:
            raise ValueError(f"Unsupported PointField datatype {code} for {field.name!r}")
        count = int(field.count)
        if count <= 0:
            raise ValueError(f"Invalid PointField count {count} for {field.name!r}")
        scalar = np.dtype(endian + _POINT_FIELD_DTYPES[code])
        names.append(str(field.name))
        formats.append(scalar if count == 1 else (scalar, (count,)))
        offsets.append(int(field.offset))
        if int(field.offset) + scalar.itemsize * count > point_step:
            raise ValueError(f"PointField {field.name!r} exceeds point_step")

    dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step}
    )
    data = memoryview(message.data)
    required_bytes = row_step * height
    if len(data) < required_bytes:
        raise ValueError(
            f"PointCloud2 data is truncated: need {required_bytes} bytes, got {len(data)}"
        )
    if height == 0 or width == 0:
        return np.empty(0, dtype=dtype)
    if row_step == width * point_step:
        return np.frombuffer(data, dtype=dtype, count=height * width)

    packed = np.empty(height * width, dtype=dtype)
    for row in range(height):
        start = row * row_step
        packed[row * width : (row + 1) * width] = np.frombuffer(
            data[start : start + width * point_step], dtype=dtype, count=width
        )
    return packed


def _decode_livox_frame(message: Any, bag_time_ns: int, frame_index: int) -> LivoxFrame:
    points = pointcloud2_array(message)
    required = {"x", "y", "z", "intensity", "tag", "line", "timestamp"}
    missing = required.difference(points.dtype.names or ())
    if missing:
        raise ValueError(f"GrandTour Livox cloud is missing fields: {sorted(missing)}")

    timestamp = np.asarray(points["timestamp"])
    if timestamp.dtype.kind not in "fiu":
        raise ValueError("GrandTour Livox timestamp field must be numeric")
    if timestamp.dtype.kind == "f" and not np.isfinite(timestamp).all():
        raise ValueError("GrandTour Livox timestamp field contains non-finite values")

    # Released GrandTour Livox bags encode Unix nanoseconds in float64. Rounding
    # makes the precision loss explicit while retaining the source field in ROS.
    point_time_ns = np.rint(timestamp).astype(np.uint64)
    xyz = np.column_stack((points["x"], points["y"], points["z"])).astype(
        np.float32, copy=False
    )
    return LivoxFrame(
        frame_index=frame_index,
        bag_time_ns=int(bag_time_ns),
        header_time_ns=_header_time_ns(message.header),
        frame_id=str(message.header.frame_id),
        xyz=xyz,
        intensity=np.asarray(points["intensity"], dtype=np.float32),
        tag=np.asarray(points["tag"], dtype=np.uint8),
        line=np.asarray(points["line"], dtype=np.uint8),
        point_time_ns=point_time_ns,
        point_index=np.arange(len(points), dtype=np.uint32),
    )


def iter_livox_frames(path: str | Path, topic: str) -> Iterator[LivoxFrame]:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError("GrandTour support requires the 'rosbags' package") from exc

    source = Path(path).expanduser().resolve()
    with AnyReader([source]) as reader:
        connections = [connection for connection in reader.connections if connection.topic == topic]
        if not connections:
            available = sorted(connection.topic for connection in reader.connections)
            raise ValueError(f"Topic {topic!r} not found in {source}; available: {available}")
        for frame_index, (connection, bag_time_ns, raw) in enumerate(
            reader.messages(connections=connections)
        ):
            message = reader.deserialize(raw, connection.msgtype)
            yield _decode_livox_frame(message, bag_time_ns, frame_index)


def _require_identical_identity(raw: LivoxFrame, undistorted: LivoxFrame) -> None:
    scalar_fields = (
        ("frame_index", raw.frame_index, undistorted.frame_index),
        ("bag_time_ns", raw.bag_time_ns, undistorted.bag_time_ns),
        ("header_time_ns", raw.header_time_ns, undistorted.header_time_ns),
        ("frame_id", raw.frame_id, undistorted.frame_id),
    )
    for name, left, right in scalar_fields:
        if left != right:
            raise ValueError(f"Raw/undistorted Livox {name} mismatch: {left!r} != {right!r}")
    if len(raw.xyz) != len(undistorted.xyz):
        raise ValueError("Raw/undistorted Livox point-count mismatch")
    for name in ("intensity", "tag", "line", "point_time_ns"):
        if not np.array_equal(getattr(raw, name), getattr(undistorted, name)):
            raise ValueError(f"Raw/undistorted Livox {name} mismatch")


def iter_paired_livox_frames(paths: GrandTourPaths) -> Iterator[PairedLivoxFrame]:
    raw_frames = iter_livox_frames(paths.raw_livox, RAW_TOPIC)
    undistorted_frames = iter_livox_frames(paths.undistorted_livox, UNDISTORTED_TOPIC)
    sentinel = object()
    while True:
        raw = next(raw_frames, sentinel)
        undistorted = next(undistorted_frames, sentinel)
        if raw is sentinel and undistorted is sentinel:
            return
        if raw is sentinel or undistorted is sentinel:
            raise ValueError("Raw and undistorted Livox bags contain different frame counts")
        assert isinstance(raw, LivoxFrame) and isinstance(undistorted, LivoxFrame)
        _require_identical_identity(raw, undistorted)
        yield PairedLivoxFrame(raw=raw, undistorted_xyz=undistorted.xyz)


def load_dlio_trajectory(path: str | Path) -> PoseTrajectory:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError("GrandTour support requires the 'rosbags' package") from exc

    source = Path(path).expanduser().resolve()
    times: list[int] = []
    translations: list[tuple[float, float, float]] = []
    quaternions: list[tuple[float, float, float, float]] = []
    world_frame: str | None = None
    body_frame: str | None = None
    with AnyReader([source]) as reader:
        connections = [c for c in reader.connections if c.topic == DLIO_ODOMETRY_TOPIC]
        if not connections:
            raise ValueError(f"DLIO odometry topic not found in {source}")
        for connection, _, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            current_world = str(message.header.frame_id)
            current_body = str(message.child_frame_id)
            if world_frame is None:
                world_frame, body_frame = current_world, current_body
            elif (world_frame, body_frame) != (current_world, current_body):
                raise ValueError("DLIO odometry changes frame IDs within one bag")
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            times.append(_header_time_ns(message.header))
            translations.append((position.x, position.y, position.z))
            quaternions.append((orientation.x, orientation.y, orientation.z, orientation.w))
    return PoseTrajectory(
        np.asarray(times, dtype=np.int64),
        np.asarray(translations, dtype=np.float64),
        np.asarray(quaternions, dtype=np.float64),
        world_frame or "",
        body_frame or "",
    )


def load_static_transform(path: str | Path, parent: str, child: str) -> np.ndarray:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError("GrandTour support requires the 'rosbags' package") from exc

    source = Path(path).expanduser().resolve()
    matches: list[np.ndarray] = []
    with AnyReader([source]) as reader:
        connections = [c for c in reader.connections if c.topic == "/tf_static"]
        for connection, _, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            for transform in message.transforms:
                if (str(transform.header.frame_id), str(transform.child_frame_id)) != (parent, child):
                    continue
                p = transform.transform.translation
                q = transform.transform.rotation
                matches.append(
                    pose_matrix(
                        np.array([p.x, p.y, p.z]),
                        np.array([q.x, q.y, q.z, q.w]),
                    )
                )
    if not matches:
        raise ValueError(f"Static transform T_{parent}_{child} not found in {source}")
    first = matches[0]
    if any(not np.allclose(first, value, atol=1e-9, rtol=0.0) for value in matches[1:]):
        raise ValueError(f"Static transform T_{parent}_{child} is inconsistent in {source}")
    return first


def dlio_map_from_livox_transform(
    trajectory: PoseTrajectory,
    tf_minimal_path: str | Path,
    timestamp_ns: int,
) -> np.ndarray:
    if (trajectory.world_frame, trajectory.body_frame) != ("dlio_map", "hesai_lidar"):
        raise ValueError(
            "Expected DLIO odometry T_dlio_map_hesai_lidar; got "
            f"T_{trajectory.world_frame}_{trajectory.body_frame}"
        )
    transform_box_hesai = load_static_transform(tf_minimal_path, "box_base", "hesai_lidar")
    transform_box_livox = load_static_transform(tf_minimal_path, "box_base", "livox_lidar")
    transform_hesai_livox = np.linalg.inv(transform_box_hesai) @ transform_box_livox
    return trajectory.at(timestamp_ns) @ transform_hesai_livox


def write_map_preview(
    session_dir: str | Path,
    output_path: str | Path,
    frame_stride: int = 10,
    point_stride: int = 20,
    minimum_range_m: float = 0.1,
    maximum_range_m: float = 30.0,
) -> dict[str, Any]:
    if frame_stride < 1 or point_stride < 1:
        raise ValueError("Frame and point strides must be at least 1")
    if not 0.0 <= minimum_range_m < maximum_range_m:
        raise ValueError("Require 0 <= minimum_range_m < maximum_range_m")
    paths = resolve_grandtour_paths(session_dir)
    trajectory = load_dlio_trajectory(paths.dlio)
    transform_box_hesai = load_static_transform(paths.tf_minimal, "box_base", "hesai_lidar")
    transform_box_livox = load_static_transform(paths.tf_minimal, "box_base", "livox_lidar")
    transform_hesai_livox = np.linalg.inv(transform_box_hesai) @ transform_box_livox

    chunks: list[np.ndarray] = []
    included_frames = 0
    skipped_pose_coverage = 0
    discarded_points = 0
    for pair in iter_paired_livox_frames(paths):
        if pair.raw.frame_index % frame_stride:
            continue
        if not trajectory.time_ns[0] <= pair.raw.header_time_ns <= trajectory.time_ns[-1]:
            skipped_pose_coverage += 1
            continue
        transform_map_livox = trajectory.at(pair.raw.header_time_ns) @ transform_hesai_livox
        points = pair.undistorted_xyz[::point_stride]
        ranges = np.linalg.norm(pair.raw.xyz[::point_stride], axis=1)
        valid = (
            np.isfinite(points).all(axis=1)
            & np.isfinite(ranges)
            & (ranges >= minimum_range_m)
            & (ranges <= maximum_range_m)
        )
        discarded_points += int(np.count_nonzero(~valid))
        chunks.append(transform_points(transform_map_livox, points[valid]))
        included_frames += 1
    xyz = np.concatenate(chunks) if chunks else np.empty((0, 3), dtype=np.float32)

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(xyz)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for point in xyz:
            stream.write(f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f}\n")
    bounds = {
        "min": xyz.min(axis=0).astype(float).tolist() if len(xyz) else None,
        "max": xyz.max(axis=0).astype(float).tolist() if len(xyz) else None,
    }
    return {
        "session_id": paths.session_id,
        "output": str(output),
        "world_frame": trajectory.world_frame,
        "included_frames": included_frames,
        "skipped_frames_outside_pose_coverage": skipped_pose_coverage,
        "points": int(len(xyz)),
        "discarded_invalid_or_out_of_range_points": discarded_points,
        "frame_stride": frame_stride,
        "point_stride": point_stride,
        "minimum_range_m": minimum_range_m,
        "maximum_range_m": maximum_range_m,
        "bounds_m": bounds,
    }


def inspect_grandtour_session(session_dir: str | Path, sample_frames: int = 5) -> dict[str, Any]:
    paths = resolve_grandtour_paths(session_dir)
    if sample_frames < 1:
        raise ValueError("sample_frames must be at least 1")

    sampled = []
    for pair in iter_paired_livox_frames(paths):
        delta = np.linalg.norm(pair.raw.xyz - pair.undistorted_xyz, axis=1)
        sampled.append(
            {
                "frame_index": pair.raw.frame_index,
                "header_time_ns": pair.raw.header_time_ns,
                "frame_id": pair.raw.frame_id,
                "points": int(len(pair.raw.xyz)),
                "point_time_min_ns": int(pair.raw.point_time_ns.min()),
                "point_time_max_ns": int(pair.raw.point_time_ns.max()),
                "deskew_displacement_median_m": float(np.median(delta)),
                "deskew_displacement_max_m": float(np.max(delta)),
            }
        )
        if len(sampled) >= sample_frames:
            break

    return {
        "session_id": paths.session_id,
        "files": {
            "raw_livox": str(paths.raw_livox),
            "undistorted_livox": str(paths.undistorted_livox),
            "dlio": str(paths.dlio),
            "tf_minimal": str(paths.tf_minimal),
        },
        "raw_topic": RAW_TOPIC,
        "undistorted_topic": UNDISTORTED_TOPIC,
        "sampled_frames": sampled,
        "point_identity_check": "passed for sampled frames",
    }
