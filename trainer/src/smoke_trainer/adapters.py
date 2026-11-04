from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Protocol

import numpy as np
import torch

from .data import Frame, TemporalWindow
from .features import (
    FEATURE_NAMES,
    FeatureNormalizer,
    fit_feature_normalizer,
    fit_normalizer,
    voxelize_frame,
)


GLOBAL_POINT_FEATURE_NAMES = (
    "x_m",
    "y_m",
    "z_m",
    "range_m",
    "intensity",
)


@dataclass(frozen=True)
class PreparedFrame:
    """Model-ready inputs plus the information needed for point-level supervision."""

    features: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    # Maps each original point to a model-output row. A negative value marks an
    # invalid original point whose placeholder logit will not be supervised.
    point_index: np.ndarray | None


@dataclass(frozen=True)
class PreparedTemporalWindow:
    """Prepared past-to-current inputs; labels and mappings belong to the target only."""

    frames: tuple[PreparedFrame, ...]
    window: TemporalWindow

    @property
    def target(self) -> PreparedFrame:
        return self.frames[-1]


class FrameAdapter(Protocol):
    name: str

    @property
    def feature_names(self) -> tuple[str, ...]: ...

    def fit_normalizer(
        self, frames: Iterable[Frame], data_config: dict
    ) -> FeatureNormalizer: ...

    def prepare(self, frame: Frame, data_config: dict) -> PreparedFrame: ...

    def expand_logits(
        self, unit_logits: torch.Tensor, prepared: PreparedFrame
    ) -> torch.Tensor: ...

    def augment_features(
        self,
        features: np.ndarray,
        training_config: dict,
        rng: np.random.Generator,
    ) -> np.ndarray: ...

    def preprocessing(self, data_config: dict) -> dict: ...


def _expand_point_logits(unit_logits: torch.Tensor, prepared: PreparedFrame) -> torch.Tensor:
    if prepared.point_index is None:
        return unit_logits
    index = torch.as_tensor(prepared.point_index, device=unit_logits.device)
    if torch.all(index >= 0):
        return unit_logits[index]
    expanded = unit_logits.new_zeros(len(index))
    valid = index >= 0
    expanded[valid] = unit_logits[index[valid]]
    return expanded


def _rotation_from_training_config(
    training_config: dict, rng: np.random.Generator
) -> np.ndarray | None:
    limits = np.asarray(
        [
            float(training_config.get("roll_augmentation_deg", 0.0)),
            float(training_config.get("pitch_augmentation_deg", 0.0)),
            float(training_config.get("yaw_augmentation_deg", 0.0)),
        ],
        dtype=np.float64,
    )
    if np.any(limits < 0):
        raise ValueError("rotation augmentation limits must be nonnegative")
    if not np.any(limits):
        return None
    roll, pitch, yaw = np.deg2rad(rng.uniform(-limits, limits))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def _rotate_features(features: np.ndarray, rotation: np.ndarray | None) -> np.ndarray:
    if rotation is None or not len(features):
        return features
    augmented = features.copy()
    augmented[:, :3] = features[:, :3] @ rotation.T
    return augmented


class LocalVoxelAdapter:
    """The original L1 voxel representation, exposed through the shared interface."""

    name = "local_voxel"

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    def fit_normalizer(
        self, frames: Iterable[Frame], data_config: dict
    ) -> FeatureNormalizer:
        return fit_normalizer(
            frames,
            float(data_config["voxel_size_m"]),
            feature_names=self.feature_names,
        )

    def prepare(self, frame: Frame, data_config: dict) -> PreparedFrame:
        voxelized = voxelize_frame(frame, float(data_config["voxel_size_m"]))
        return PreparedFrame(
            features=voxelized.features,
            labels=voxelized.labels,
            valid=np.ones(len(voxelized.labels), dtype=bool),
            point_index=voxelized.inverse,
        )

    def expand_logits(
        self, unit_logits: torch.Tensor, prepared: PreparedFrame
    ) -> torch.Tensor:
        return _expand_point_logits(unit_logits, prepared)

    def augment_features(
        self,
        features: np.ndarray,
        training_config: dict,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return features

    def preprocessing(self, data_config: dict) -> dict:
        return {"voxel_size_m": float(data_config["voxel_size_m"])}


class GlobalPointAdapter:
    """Direct point features for a PointNet-style whole-scan model."""

    name = "global_points"

    @property
    def feature_names(self) -> tuple[str, ...]:
        return GLOBAL_POINT_FEATURE_NAMES

    def fit_normalizer(
        self, frames: Iterable[Frame], data_config: dict
    ) -> FeatureNormalizer:
        normalizer = fit_feature_normalizer(
            (self.prepare(frame, data_config).features for frame in frames),
            self.feature_names,
        )
        # XYZ remains centered at the sensor origin and uses one shared scale, so
        # rigid rotations commute with normalization during pose augmentation.
        mean = normalizer.mean.copy()
        scale = normalizer.scale.copy()
        mean[:3] = 0.0
        scale[:3] = float(np.max(scale[:3]))
        return FeatureNormalizer(
            feature_names=normalizer.feature_names,
            mean=mean,
            scale=scale,
            count=normalizer.count,
        )

    def prepare(self, frame: Frame, data_config: dict) -> PreparedFrame:
        xyz = np.asarray(frame.xyz, dtype=np.float32)
        intensity = np.asarray(frame.intensity, dtype=np.float32)
        labels = np.asarray(frame.label, dtype=np.uint8)
        if xyz.shape != (len(intensity), 3) or labels.shape != intensity.shape:
            raise ValueError("frame point arrays have incompatible shapes")
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(intensity)
        point_index = np.full(len(labels), -1, dtype=np.int64)
        point_index[valid] = np.arange(int(valid.sum()), dtype=np.int64)
        valid_xyz = xyz[valid]
        if len(valid_xyz):
            ranges = np.linalg.norm(valid_xyz.astype(np.float64), axis=1).astype(np.float32)
            features = np.column_stack((valid_xyz, ranges, intensity[valid])).astype(
                np.float32, copy=False
            )
        else:
            features = np.empty((0, len(self.feature_names)), dtype=np.float32)
        return PreparedFrame(
            features=features,
            labels=labels,
            valid=valid,
            point_index=point_index,
        )

    def expand_logits(
        self, unit_logits: torch.Tensor, prepared: PreparedFrame
    ) -> torch.Tensor:
        return _expand_point_logits(unit_logits, prepared)

    def augment_features(
        self,
        features: np.ndarray,
        training_config: dict,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return _rotate_features(features, _rotation_from_training_config(training_config, rng))

    def preprocessing(self, data_config: dict) -> dict:
        return {
            "point_validity": "finite_xyz_and_intensity",
            "xyz_normalization": "sensor_origin_shared_scale",
        }


class GlobalTemporalAdapter(GlobalPointAdapter):
    """G1 point preprocessing applied consistently to a bounded temporal window."""

    name = "global_temporal"
    is_temporal = True

    def prepare_window(self, window: TemporalWindow, data_config: dict) -> PreparedTemporalWindow:
        if not window.frames:
            raise ValueError("temporal window must not be empty")
        return PreparedTemporalWindow(
            frames=tuple(self.prepare(frame, data_config) for frame in window.frames),
            window=window,
        )

    def augment_window(
        self,
        prepared: PreparedTemporalWindow,
        training_config: dict,
        rng: np.random.Generator,
    ) -> PreparedTemporalWindow:
        # One physical rotation applies to the complete sequence, preserving its
        # real temporal changes instead of manufacturing scan-to-scan motion.
        rotation = _rotation_from_training_config(training_config, rng)
        return replace(
            prepared,
            frames=tuple(
                replace(frame, features=_rotate_features(frame.features, rotation))
                for frame in prepared.frames
            ),
        )

    def preprocessing(self, data_config: dict) -> dict:
        result = super().preprocessing(data_config)
        result.update(
            {
                "history_length": int(data_config["history_length"]),
                "maximum_time_gap_s": float(data_config["maximum_time_gap_s"]),
            }
        )
        return result


_ADAPTERS = {
    LocalVoxelAdapter.name: LocalVoxelAdapter,
    GlobalPointAdapter.name: GlobalPointAdapter,
    GlobalTemporalAdapter.name: GlobalTemporalAdapter,
}
_LEGACY_MODEL_ADAPTERS = {"local_mlp": LocalVoxelAdapter.name}


def create_adapter(name: str) -> FrameAdapter:
    try:
        return _ADAPTERS[name]()
    except KeyError as error:
        choices = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"unknown frame adapter: {name}; expected one of {choices}") from error


def adapter_name_for_model(model_config: dict) -> str:
    configured = model_config.get("adapter")
    if configured is not None:
        return str(configured)
    try:
        return _LEGACY_MODEL_ADAPTERS[str(model_config["name"])]
    except KeyError as error:
        raise ValueError(
            "model configuration must name a frame adapter"
        ) from error


def create_adapter_for_model(model_config: dict) -> FrameAdapter:
    return create_adapter(adapter_name_for_model(model_config))
