from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import torch

from .adapters import (
    FrameAdapter,
    GlobalTemporalAdapter,
    adapter_name_for_model,
    create_adapter,
)
from .data import Frame
from .evaluate import Calibration, adapter_frame_logits
from .features import FeatureNormalizer
from .models import GlobalTemporalPointNet, create_model


def _load_bundle_components(
    path: str | Path, device: str
) -> tuple[torch.nn.Module, FrameAdapter, dict, FeatureNormalizer, Calibration, float, torch.device]:
    resolved_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
    )
    bundle = torch.load(path, map_location=resolved_device, weights_only=False)
    normalizer = FeatureNormalizer.from_dict(bundle["normalizer"])
    if tuple(bundle.get("feature_names", ())) != normalizer.feature_names:
        raise ValueError("bundle feature schema is incompatible")
    adapter_name = bundle.get("adapter_name")
    if adapter_name is None:
        adapter_name = adapter_name_for_model(bundle["model_parameters"])
    adapter = create_adapter(str(adapter_name))
    if normalizer.feature_names != adapter.feature_names:
        raise ValueError("bundle normalizer does not match its frame adapter")
    preprocessing = bundle.get("preprocessing")
    if preprocessing is None:
        if "voxel_size_m" not in bundle:
            raise ValueError("legacy bundle does not define preprocessing")
        preprocessing = {"voxel_size_m": float(bundle["voxel_size_m"])}
    model = create_model(
        bundle["model_name"], len(normalizer.feature_names), bundle["model_parameters"]
    ).to(resolved_device)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return (
        model,
        adapter,
        dict(preprocessing),
        normalizer,
        Calibration.from_dict(bundle["calibration"]),
        float(bundle["threshold"]),
        resolved_device,
    )


@dataclass
class SmokePredictor:
    model: torch.nn.Module
    adapter: FrameAdapter
    preprocessing: dict
    normalizer: FeatureNormalizer
    calibration: Calibration
    threshold: float
    device: torch.device

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "SmokePredictor":
        components = _load_bundle_components(path, device)
        if isinstance(components[1], GlobalTemporalAdapter):
            return GlobalTemporalPredictor.from_components(*components)
        return cls(
            model=components[0],
            adapter=components[1],
            preprocessing=components[2],
            normalizer=components[3],
            calibration=components[4],
            threshold=components[5],
            device=components[6],
        )

    def predict(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        logits, _, valid = adapter_frame_logits(
            self.model,
            frame,
            self.normalizer,
            self.adapter,
            self.preprocessing,
            self.device,
        )
        probabilities = self.calibration.apply(logits)
        return probabilities, valid

    def benchmark(self, frame: Frame, repeats: int = 20) -> dict[str, float | int]:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        self.predict(frame)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        durations: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            self.predict(frame)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            durations.append((time.perf_counter() - start) * 1000.0)
        values = np.asarray(durations)
        return {
            "repeats": repeats,
            "points": len(frame.xyz),
            "mean_ms": float(values.mean()),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "max_ms": float(values.max()),
        }


class LocalPredictor(SmokePredictor):
    """Compatibility name for the original L1 predictor API."""


@dataclass
class GlobalTemporalPredictor(SmokePredictor):
    """Bounded, causal GT inference with automatic stream discontinuity resets."""

    _history: deque[torch.Tensor] = field(default_factory=deque, init=False, repr=False)
    _stream_key: tuple[str, str, str, int] | None = field(default=None, init=False, repr=False)
    _last_timestamp: float | None = field(default=None, init=False, repr=False)
    last_history_length: int = field(default=0, init=False)

    @classmethod
    def from_components(
        cls,
        model: torch.nn.Module,
        adapter: FrameAdapter,
        preprocessing: dict,
        normalizer: FeatureNormalizer,
        calibration: Calibration,
        threshold: float,
        device: torch.device,
    ) -> "GlobalTemporalPredictor":
        if not isinstance(model, GlobalTemporalPointNet) or not isinstance(adapter, GlobalTemporalAdapter):
            raise ValueError("bundle is not a global temporal point model")
        return cls(model, adapter, preprocessing, normalizer, calibration, threshold, device)

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "GlobalTemporalPredictor":
        return cls.from_components(*_load_bundle_components(path, device))

    @property
    def temporal_adapter(self) -> GlobalTemporalAdapter:
        if not isinstance(self.adapter, GlobalTemporalAdapter):
            raise RuntimeError("temporal predictor has the wrong adapter")
        return self.adapter

    @property
    def temporal_model(self) -> GlobalTemporalPointNet:
        if not isinstance(self.model, GlobalTemporalPointNet):
            raise RuntimeError("temporal predictor has the wrong model")
        return self.model

    def reset(self) -> None:
        self._history.clear()
        self._stream_key = None
        self._last_timestamp = None
        self.last_history_length = 0

    def _should_reset(self, frame: Frame) -> bool:
        key = (frame.source_domain, frame.session_id, frame.recording_id, frame.ref.chunk_index)
        if self._stream_key is None:
            return False
        if key != self._stream_key or self._last_timestamp is None:
            return True
        gap = float(frame.timestamp_s - self._last_timestamp)
        return gap <= 0.0 or gap > float(self.preprocessing["maximum_time_gap_s"])

    def predict(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        if self._should_reset(frame):
            self.reset()
        prepared = self.temporal_adapter.prepare(frame, self.preprocessing)
        features = self.normalizer.transform(prepared.features)
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            embedding, summary, _ = self.temporal_model.encode_scan(feature_tensor)
            self._history.append(summary.detach())
            while len(self._history) > self.temporal_model.history_length:
                self._history.popleft()
            unit_logits = self.temporal_model.predict_from_encoded(
                embedding, summary, tuple(self._history)
            )
        self._stream_key = (
            frame.source_domain,
            frame.session_id,
            frame.recording_id,
            frame.ref.chunk_index,
        )
        self._last_timestamp = float(frame.timestamp_s)
        self.last_history_length = len(self._history)
        expanded = self.temporal_adapter.expand_logits(unit_logits, prepared)
        return self.calibration.apply(expanded.detach().cpu().numpy()), prepared.valid
