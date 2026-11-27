from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .data import Frame


FEATURE_NAMES = (
    "log1p_point_count",
    "intensity_mean",
    "intensity_std",
    "range_mean_m",
    "range_std_m",
    "centroid_x_m",
    "centroid_y_m",
    "centroid_z_m",
    "spread_x_m",
    "spread_y_m",
    "spread_z_m",
)


@dataclass(frozen=True)
class VoxelizedFrame:
    features: np.ndarray
    inverse: np.ndarray
    labels: np.ndarray
    voxel_coordinates: np.ndarray


@dataclass(frozen=True)
class FeatureNormalizer:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    count: int

    def transform(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError("features are incompatible with this normalizer")
        return ((features - self.mean) / self.scale).astype(np.float32, copy=False)

    def to_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "FeatureNormalizer":
        feature_names = tuple(str(name) for name in value["feature_names"])
        mean = np.asarray(value["mean"], dtype=np.float32)
        scale = np.asarray(value["scale"], dtype=np.float32)
        if not feature_names or mean.ndim != 1 or scale.shape != mean.shape:
            raise ValueError("normalizer feature schema is invalid")
        if len(feature_names) != len(mean):
            raise ValueError("normalizer feature count is invalid")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("normalizer values are invalid")
        return cls(
            feature_names=feature_names,
            mean=mean,
            scale=scale,
            count=int(value["count"]),
        )


class NormalizerAccumulator:
    def __init__(self, feature_names: tuple[str, ...]):
        if not feature_names:
            raise ValueError("feature_names must not be empty")
        self.feature_names = feature_names
        self.count = 0
        self.mean = np.zeros(len(feature_names), dtype=np.float64)
        self.m2 = np.zeros(len(feature_names), dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        if values.ndim != 2 or values.shape[1] != len(self.mean):
            raise ValueError("normalizer values have the wrong shape")
        if not len(values):
            return
        batch = values.astype(np.float64, copy=False)
        batch_count = len(batch)
        batch_mean = batch.mean(axis=0)
        batch_m2 = np.square(batch - batch_mean).sum(axis=0)
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + np.square(delta) * self.count * batch_count / total
        self.count = total

    def finalize(self) -> FeatureNormalizer:
        if self.count == 0:
            raise ValueError("cannot fit a normalizer without voxels")
        variance = self.m2 / self.count
        scale = np.sqrt(np.maximum(variance, 0.0))
        scale[scale < 1.0e-6] = 1.0
        return FeatureNormalizer(
            feature_names=self.feature_names,
            mean=self.mean.astype(np.float32),
            scale=scale.astype(np.float32),
            count=self.count,
        )


def voxelize_frame(frame: Frame, voxel_size_m: float) -> VoxelizedFrame:
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    xyz = np.asarray(frame.xyz, dtype=np.float32)
    intensity = np.asarray(frame.intensity, dtype=np.float32)
    labels = np.asarray(frame.label, dtype=np.uint8)
    if xyz.shape != (len(intensity), 3) or labels.shape != intensity.shape:
        raise ValueError("frame point arrays have incompatible shapes")
    if not len(xyz):
        return VoxelizedFrame(
            features=np.empty((0, len(FEATURE_NAMES)), dtype=np.float32),
            inverse=np.empty(0, dtype=np.int64),
            labels=labels,
            voxel_coordinates=np.empty((0, 3), dtype=np.int64),
        )

    coordinates = np.floor(xyz.astype(np.float64) / voxel_size_m).astype(np.int64)
    unique, inverse, counts = np.unique(
        coordinates, axis=0, return_inverse=True, return_counts=True
    )
    inverse = inverse.astype(np.int64, copy=False)
    counts64 = counts.astype(np.float64)
    ranges = np.linalg.norm(xyz.astype(np.float64), axis=1)

    def moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values64 = values.astype(np.float64, copy=False)
        sums = np.bincount(inverse, weights=values64, minlength=len(unique))
        square_sums = np.bincount(
            inverse, weights=np.square(values64), minlength=len(unique)
        )
        means = sums / counts64
        variance = np.maximum(square_sums / counts64 - np.square(means), 0.0)
        return means, np.sqrt(variance)

    intensity_mean, intensity_std = moments(intensity)
    range_mean, range_std = moments(ranges)
    centroids = np.empty((len(unique), 3), dtype=np.float64)
    spreads = np.empty_like(centroids)
    for axis in range(3):
        centroids[:, axis], spreads[:, axis] = moments(xyz[:, axis])

    features = np.column_stack(
        (
            np.log1p(counts64),
            intensity_mean,
            intensity_std,
            range_mean,
            range_std,
            centroids,
            spreads,
        )
    ).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("voxelization produced non-finite features")
    return VoxelizedFrame(
        features=features,
        inverse=inverse,
        labels=labels,
        voxel_coordinates=unique,
    )


def fit_feature_normalizer(
    feature_rows: Iterable[np.ndarray], feature_names: tuple[str, ...]
) -> FeatureNormalizer:
    """Fit a named feature schema from streaming 2D arrays without retaining them."""

    accumulator = NormalizerAccumulator(tuple(feature_names))
    for features in feature_rows:
        accumulator.update(features)
    return accumulator.finalize()


def fit_normalizer(
    frames: Iterable[Frame],
    voxel_size_m: float,
    *,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> FeatureNormalizer:
    if tuple(feature_names) != FEATURE_NAMES:
        raise ValueError("voxel normalization requires the local voxel feature schema")
    return fit_feature_normalizer(
        (voxelize_frame(frame, voxel_size_m).features for frame in frames),
        tuple(feature_names),
    )
