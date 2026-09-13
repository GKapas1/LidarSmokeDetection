from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def pose_matrix(translation: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    translation = np.asarray(translation, dtype=np.float64)
    quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=np.float64)
    if translation.shape != (3,) or quaternion_xyzw.shape != (4,):
        raise ValueError("A pose requires translation [3] and XYZW quaternion [4]")
    if not np.isfinite(translation).all() or not np.isfinite(quaternion_xyzw).all():
        raise ValueError("Pose contains non-finite values")
    norm = float(np.linalg.norm(quaternion_xyzw))
    if norm < 1e-12:
        raise ValueError("Pose quaternion has zero norm")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(quaternion_xyzw / norm).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def transform_points(transform_a_b: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    transform_a_b = np.asarray(transform_a_b, dtype=np.float64)
    points_b = np.asarray(points_b)
    if transform_a_b.shape != (4, 4) or points_b.ndim != 2 or points_b.shape[1] != 3:
        raise ValueError("Expected a 4x4 transform and Nx3 points")
    return (points_b @ transform_a_b[:3, :3].T + transform_a_b[:3, 3]).astype(
        np.float32, copy=False
    )


@dataclass(frozen=True)
class PoseTrajectory:
    """Poses T_world_body sampled at integer Unix nanosecond timestamps."""

    time_ns: np.ndarray
    translation: np.ndarray
    quaternion_xyzw: np.ndarray
    world_frame: str
    body_frame: str

    def __post_init__(self) -> None:
        time_ns = np.asarray(self.time_ns, dtype=np.int64)
        translation = np.asarray(self.translation, dtype=np.float64)
        quaternion = np.asarray(self.quaternion_xyzw, dtype=np.float64)
        if time_ns.ndim != 1 or len(time_ns) < 2:
            raise ValueError("Pose trajectory needs at least two timestamps")
        if translation.shape != (len(time_ns), 3) or quaternion.shape != (len(time_ns), 4):
            raise ValueError("Pose trajectory array shapes do not agree")
        if np.any(np.diff(time_ns) <= 0):
            raise ValueError("Pose timestamps must be strictly increasing")
        if not np.isfinite(translation).all() or not np.isfinite(quaternion).all():
            raise ValueError("Pose trajectory contains non-finite values")
        norms = np.linalg.norm(quaternion, axis=1)
        if np.any(norms < 1e-12):
            raise ValueError("Pose trajectory contains a zero quaternion")
        object.__setattr__(self, "time_ns", time_ns)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "quaternion_xyzw", quaternion / norms[:, None])

    def at(self, query_time_ns: int, maximum_gap_ns: int = 250_000_000) -> np.ndarray:
        query = int(query_time_ns)
        right = int(np.searchsorted(self.time_ns, query, side="left"))
        if right < len(self.time_ns) and int(self.time_ns[right]) == query:
            return pose_matrix(self.translation[right], self.quaternion_xyzw[right])
        if right == 0 or right == len(self.time_ns):
            raise ValueError(f"Pose timestamp {query} lies outside trajectory coverage")
        left = right - 1
        gap = int(self.time_ns[right]) - int(self.time_ns[left])
        if gap > int(maximum_gap_ns):
            raise ValueError(
                f"Pose interpolation gap {gap / 1e9:.6f}s exceeds "
                f"{maximum_gap_ns / 1e9:.6f}s"
            )
        fraction = (query - int(self.time_ns[left])) / gap
        translation = self.translation[left] + fraction * (
            self.translation[right] - self.translation[left]
        )
        rotations = Rotation.from_quat(self.quaternion_xyzw[[left, right]])
        rotation = Slerp([0.0, 1.0], rotations)([fraction]).as_quat()[0]
        return pose_matrix(translation, rotation)
