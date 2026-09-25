from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
import torch

from .data import Frame
from .features import FeatureNormalizer, voxelize_frame
from .metrics import best_f1_threshold, metric_summary


@dataclass(frozen=True)
class Calibration:
    slope: float
    bias: float

    def apply(self, logits: np.ndarray) -> np.ndarray:
        values = self.slope * np.asarray(logits, dtype=np.float64) + self.bias
        return sigmoid(values).astype(np.float32)

    def to_dict(self) -> dict[str, float]:
        return {"slope": self.slope, "bias": self.bias}

    @classmethod
    def from_dict(cls, value: dict) -> "Calibration":
        return cls(slope=float(value["slope"]), bias=float(value["bias"]))


@dataclass(frozen=True)
class PredictionCollection:
    logits: np.ndarray
    labels: np.ndarray
    domains: np.ndarray
    domain_names: tuple[str, ...]
    frames: int


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def fit_calibration(logits: np.ndarray, labels: np.ndarray) -> Calibration:
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or not len(x):
        raise ValueError("calibration requires equal-length nonempty vectors")
    if not np.isin(y, (0.0, 1.0)).all() or len(np.unique(y)) != 2:
        raise ValueError("calibration data must contain both classes")

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        raw_slope, bias = parameters
        slope = np.logaddexp(0.0, raw_slope)
        transformed = slope * x + bias
        loss = np.mean(np.logaddexp(0.0, transformed) - y * transformed)
        residual = sigmoid(transformed) - y
        slope_derivative = sigmoid(np.asarray([raw_slope]))[0]
        gradient = np.asarray(
            [np.mean(residual * x) * slope_derivative, np.mean(residual)],
            dtype=np.float64,
        )
        return float(loss), gradient

    initial = np.asarray([np.log(np.expm1(1.0)), 0.0], dtype=np.float64)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"calibration failed: {result.message}")
    return Calibration(
        slope=float(np.logaddexp(0.0, result.x[0])),
        bias=float(result.x[1]),
    )


def frame_logits(
    model: torch.nn.Module,
    frame: Frame,
    normalizer: FeatureNormalizer,
    voxel_size_m: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    voxelized = voxelize_frame(frame, voxel_size_m)
    if not len(voxelized.features):
        return np.empty(0, dtype=np.float32), voxelized.labels
    features = torch.from_numpy(normalizer.transform(voxelized.features)).to(device)
    inverse = torch.from_numpy(voxelized.inverse).to(device)
    with torch.inference_mode():
        voxel_logits = model(features)
        point_logits = voxel_logits[inverse]
    return point_logits.detach().cpu().numpy().astype(np.float32), voxelized.labels


def collect_predictions(
    model: torch.nn.Module,
    frames: Iterable[Frame],
    normalizer: FeatureNormalizer,
    voxel_size_m: float,
    device: torch.device,
) -> PredictionCollection:
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    domains: list[np.ndarray] = []
    domain_codes: dict[str, int] = {}
    frame_count = 0
    model.eval()
    for frame in frames:
        frame_output, frame_labels = frame_logits(
            model, frame, normalizer, voxel_size_m, device
        )
        mask = frame_labels != 255
        if np.any(mask):
            code = domain_codes.setdefault(frame.source_domain, len(domain_codes))
            if code > np.iinfo(np.uint8).max:
                raise ValueError("too many source domains")
            logits.append(frame_output[mask])
            labels.append(frame_labels[mask].astype(np.uint8, copy=False))
            domains.append(np.full(int(mask.sum()), code, dtype=np.uint8))
        frame_count += 1
    if not logits:
        raise ValueError("evaluation produced no supervised points")
    return PredictionCollection(
        logits=np.concatenate(logits),
        labels=np.concatenate(labels),
        domains=np.concatenate(domains),
        domain_names=tuple(domain_codes),
        frames=frame_count,
    )


def evaluation_report(
    predictions: PredictionCollection,
    calibration: Calibration,
    threshold: float,
) -> dict:
    probabilities = calibration.apply(predictions.logits)
    groups = {"overall": np.ones(len(probabilities), dtype=bool)}
    groups.update(
        (name, predictions.domains == code)
        for code, name in enumerate(predictions.domain_names)
    )
    return {
        "frames": predictions.frames,
        "calibration": calibration.to_dict(),
        "metrics": {
            name: metric_summary(
                predictions.labels[mask], probabilities[mask], threshold=threshold
            )
            for name, mask in groups.items()
        },
    }


def choose_threshold(predictions: PredictionCollection, calibration: Calibration) -> float:
    return best_f1_threshold(predictions.labels, calibration.apply(predictions.logits))


def save_report(report: dict, output: str | Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def save_calibration_plot(
    labels: np.ndarray, probabilities: np.ndarray, output: str | Path, *, bins: int = 10
) -> None:
    destination = Path(output)
    cache = destination.parent / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = np.asarray(labels, dtype=np.uint8)
    p = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    predicted: list[float] = []
    observed: list[float] = []
    for index in range(bins):
        mask = indices == index
        if np.any(mask):
            predicted.append(float(p[mask].mean()))
            observed.append(float(y[mask].mean()))

    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Ideal")
    axis.plot(predicted, observed, marker="o", label="Model")
    axis.set(xlabel="Predicted smoke probability", ylabel="Observed smoke fraction")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)
