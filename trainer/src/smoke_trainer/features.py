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
    mean: np.ndarray
    scale: np.ndarray
    count: int

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.mean) / self.scale).astype(np.float32, copy=False)

    def to_dict(self) -> dict:
        return {
            "feature_names": list(FEATURE_NAMES),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "FeatureNormalizer":
        if tuple(value["feature_names"]) != FEATURE_NAMES:
            raise ValueError("normalizer feature order is incompatible")
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float32),
            scale=np.asarray(value["scale"], dtype=np.float32),
            count=int(value["count"]),
        )


class NormalizerAccumulator:
    def __init__(self, feature_count: int):
        self.count = 0
        self.mean = np.zeros(feature_count, dtype=np.float64)
        self.m2 = np.zeros(feature_count, dtype=np.float64)

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


def fit_normalizer(frames: Iterable[Frame], voxel_size_m: float) -> FeatureNormalizer:
    accumulator = NormalizerAccumulator(len(FEATURE_NAMES))
    for frame in frames:
        accumulator.update(voxelize_frame(frame, voxel_size_m).features)
    return accumulator.finalize()

