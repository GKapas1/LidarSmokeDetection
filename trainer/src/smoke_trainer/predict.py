from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch

from .data import Frame
from .evaluate import Calibration, frame_logits
from .features import FEATURE_NAMES, FeatureNormalizer
from .models import create_model


@dataclass
class LocalPredictor:
    model: torch.nn.Module
    normalizer: FeatureNormalizer
    calibration: Calibration
    threshold: float
    voxel_size_m: float
    device: torch.device

    @classmethod
    def load(cls, path: str | Path, device: str = "auto") -> "LocalPredictor":
        resolved_device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        )
        bundle = torch.load(path, map_location=resolved_device, weights_only=False)
        if tuple(bundle["feature_names"]) != FEATURE_NAMES:
            raise ValueError("bundle feature schema is incompatible")
        model = create_model(
            bundle["model_name"], len(FEATURE_NAMES), bundle["model_parameters"]
        ).to(resolved_device)
        model.load_state_dict(bundle["model_state"])
        model.eval()
        return cls(
            model=model,
            normalizer=FeatureNormalizer.from_dict(bundle["normalizer"]),
            calibration=Calibration.from_dict(bundle["calibration"]),
            threshold=float(bundle["threshold"]),
            voxel_size_m=float(bundle["voxel_size_m"]),
            device=resolved_device,
        )

    def predict(self, frame: Frame) -> tuple[np.ndarray, np.ndarray]:
        logits, _ = frame_logits(
            self.model, frame, self.normalizer, self.voxel_size_m, self.device
        )
        probabilities = self.calibration.apply(logits)
        valid = np.ones(len(probabilities), dtype=bool)
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

