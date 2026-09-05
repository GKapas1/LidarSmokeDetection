from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


UNAFFECTED = np.uint8(0)
SMOKE_IMPACTED = np.uint8(1)
IGNORE = np.uint8(255)


@dataclass(frozen=True)
class DirectionalReference:
    angular_resolution_deg: float
    n_azimuth_bins: int
    cell_ids: np.ndarray
    count: np.ndarray
    range_median: np.ndarray
    range_mad: np.ndarray
    range_spread: np.ndarray
    reflectivity_median: np.ndarray
    reflectivity_mad: np.ndarray

    def as_npz_dict(self) -> dict[str, Any]:
        return {
            "angular_resolution_deg": np.float64(self.angular_resolution_deg),
            "n_azimuth_bins": np.int64(self.n_azimuth_bins),
            "cell_ids": self.cell_ids,
            "count": self.count,
            "range_median": self.range_median,
            "range_mad": self.range_mad,
            "range_spread": self.range_spread,
            "reflectivity_median": self.reflectivity_median,
            "reflectivity_mad": self.reflectivity_mad,
        }


@dataclass(frozen=True)
class Calibration:
    range_band_edges_m: np.ndarray
    normal_residual_threshold_m: np.ndarray
    early_return_threshold_m: np.ndarray
    calibration_count: np.ndarray
    usable_band: np.ndarray
    quantile: float
    normal_quantile: float

    def as_npz_dict(self) -> dict[str, Any]:
        return {
            "range_band_edges_m": self.range_band_edges_m,
            "normal_residual_threshold_m": self.normal_residual_threshold_m,
            "early_return_threshold_m": self.early_return_threshold_m,
            "calibration_count": self.calibration_count,
            "usable_band": self.usable_band,
            "false_positive_quantile": np.float64(self.quantile),
            "normal_quantile": np.float64(self.normal_quantile),
        }


def ranges_and_cell_ids(xyz: np.ndarray, angular_resolution_deg: float) -> tuple[np.ndarray, np.ndarray, int]:
    xyz = np.asarray(xyz, dtype=np.float64)
    ranges = np.linalg.norm(xyz, axis=1)
    finite = np.isfinite(xyz).all(axis=1) & np.isfinite(ranges) & (ranges > 1e-6)
    safe = np.where(finite, ranges, 1.0)
    azimuth = np.where(finite, np.arctan2(xyz[:, 1], xyz[:, 0]), 0.0)
    elevation = np.where(finite, np.arcsin(np.clip(xyz[:, 2] / safe, -1.0, 1.0)), 0.0)

    resolution = np.deg2rad(angular_resolution_deg)
    n_az = int(np.ceil(2.0 * np.pi / resolution))
    n_el = int(np.ceil(np.pi / resolution))
    az_bin = np.floor((azimuth + np.pi) / resolution).astype(np.int64)
    el_bin = np.floor((elevation + np.pi / 2.0) / resolution).astype(np.int64)
    az_bin = np.clip(az_bin, 0, n_az - 1)
    el_bin = np.clip(el_bin, 0, n_el - 1)
    cells = el_bin * n_az + az_bin
    cells[~finite] = -1
    return ranges.astype(np.float32), cells, n_az


def _group_quantile(keys: np.ndarray, values: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(keys) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    order = np.lexsort((values, keys))
    sorted_keys = keys[order]
    sorted_values = values[order]
    unique, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    positions = starts + np.rint((counts - 1) * q).astype(np.int64)
    return unique, sorted_values[positions].astype(np.float32), counts.astype(np.int64)


def _align_values(target_keys: np.ndarray, source_keys: np.ndarray, values: np.ndarray, fill: float = np.nan) -> np.ndarray:
    out = np.full(len(target_keys), fill, dtype=np.float32)
    if len(source_keys) == 0:
        return out
    positions = np.searchsorted(target_keys, source_keys)
    good = (positions < len(target_keys)) & (target_keys[np.minimum(positions, len(target_keys) - 1)] == source_keys)
    out[positions[good]] = values[good]
    return out


def build_directional_reference(
    xyz: np.ndarray,
    reflectivity: np.ndarray,
    angular_resolution_deg: float,
) -> DirectionalReference:
    ranges, cells, n_az = ranges_and_cell_ids(xyz, angular_resolution_deg)
    valid = (cells >= 0) & np.isfinite(ranges)
    cells_v = cells[valid]
    ranges_v = ranges[valid]

    cell_ids, range_median, count = _group_quantile(cells_v, ranges_v, 0.5)
    positions = np.searchsorted(cell_ids, cells_v)
    deviations = np.abs(ranges_v - range_median[positions])
    mad_ids, range_mad, _ = _group_quantile(cells_v, deviations, 0.5)
    lo_ids, q05, _ = _group_quantile(cells_v, ranges_v, 0.05)
    hi_ids, q95, _ = _group_quantile(cells_v, ranges_v, 0.95)

    if not (np.array_equal(cell_ids, mad_ids) and np.array_equal(cell_ids, lo_ids) and np.array_equal(cell_ids, hi_ids)):
        raise RuntimeError("Internal directional aggregation error")

    reflectivity = np.asarray(reflectivity, dtype=np.float32)
    refl_valid = valid & np.isfinite(reflectivity)
    refl_ids, refl_median, _ = _group_quantile(cells[refl_valid], reflectivity[refl_valid], 0.5)
    refl_positions = np.searchsorted(refl_ids, cells[refl_valid])
    refl_for_points = refl_median[refl_positions]
    refl_dev = np.abs(reflectivity[refl_valid] - refl_for_points)
    refl_mad_ids, refl_mad, _ = _group_quantile(cells[refl_valid], refl_dev, 0.5)

    return DirectionalReference(
        angular_resolution_deg=float(angular_resolution_deg),
        n_azimuth_bins=n_az,
        cell_ids=cell_ids.astype(np.int64),
        count=count.astype(np.int32),
        range_median=range_median,
        range_mad=range_mad,
        range_spread=(q95 - q05).astype(np.float32),
        reflectivity_median=_align_values(cell_ids, refl_ids, refl_median),
        reflectivity_mad=_align_values(cell_ids, refl_mad_ids, refl_mad),
    )


def query_reference(
    reference: DirectionalReference,
    xyz: np.ndarray,
    min_returns_per_cell: int,
    max_cell_depth_spread_m: float,
) -> dict[str, np.ndarray]:
    observed_range, cells, _ = ranges_and_cell_ids(xyz, reference.angular_resolution_deg)
    raw_positions = np.searchsorted(reference.cell_ids, cells)
    positions = np.clip(raw_positions, 0, max(len(reference.cell_ids) - 1, 0))
    matched = np.zeros(len(cells), dtype=bool)
    if len(reference.cell_ids):
        matched = (cells >= 0) & (raw_positions < len(reference.cell_ids)) & (reference.cell_ids[positions] == cells)

    expected = np.full(len(cells), np.nan, dtype=np.float32)
    count = np.zeros(len(cells), dtype=np.int32)
    spread = np.full(len(cells), np.inf, dtype=np.float32)
    expected_refl = np.full(len(cells), np.nan, dtype=np.float32)
    expected[matched] = reference.range_median[positions[matched]]
    count[matched] = reference.count[positions[matched]]
    spread[matched] = reference.range_spread[positions[matched]]
    expected_refl[matched] = reference.reflectivity_median[positions[matched]]

    valid = (
        matched
        & (count >= min_returns_per_cell)
        & (spread <= max_cell_depth_spread_m)
        & np.isfinite(observed_range)
    )
    return {
        "observed_range": observed_range,
        "expected_range": expected,
        "range_residual": (observed_range - expected).astype(np.float32),
        "expected_reflectivity": expected_refl,
        "valid_reference": valid,
        "reference_count": count,
        "reference_spread": spread,
    }


def calibrate_early_return_thresholds(
    expected_range: np.ndarray,
    range_residual: np.ndarray,
    valid_reference: np.ndarray,
    range_band_edges_m: np.ndarray,
    false_positive_quantile: float,
    minimum_early_return_m: float,
    minimum_samples_per_band: int,
    normal_quantile: float = 0.95,
    minimum_normal_residual_m: float = 0.02,
    maximum_usable_early_return_m: float = np.inf,
) -> Calibration:
    edges = np.asarray(range_band_edges_m, dtype=np.float32)
    if len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("range_band_edges_m must be strictly increasing")
    if not 0 < normal_quantile < false_positive_quantile < 1:
        raise ValueError("Require 0 < normal_quantile < false_positive_quantile < 1")
    valid = valid_reference & np.isfinite(expected_range) & np.isfinite(range_residual)
    early_error = -range_residual[valid]
    absolute_error = np.abs(range_residual[valid])

    normal_thresholds = np.full(len(edges) - 1, np.nan, dtype=np.float32)
    thresholds = np.full(len(edges) - 1, np.nan, dtype=np.float32)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    expected_valid = expected_range[valid]
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (expected_valid >= low) & (expected_valid < high)
        counts[i] = int(np.count_nonzero(selected))
        if counts[i] >= minimum_samples_per_band:
            normal_thresholds[i] = max(
                minimum_normal_residual_m,
                float(np.quantile(absolute_error[selected], normal_quantile)),
            )
            thresholds[i] = max(
                minimum_early_return_m,
                float(np.quantile(early_error[selected], false_positive_quantile)),
                float(normal_thresholds[i]),
            )
    usable = np.isfinite(thresholds) & (thresholds <= maximum_usable_early_return_m)
    return Calibration(
        edges,
        normal_thresholds,
        thresholds,
        counts,
        usable,
        float(false_positive_quantile),
        float(normal_quantile),
    )


def thresholds_for_range(calibration: Calibration, expected_range: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bands = np.searchsorted(calibration.range_band_edges_m, expected_range, side="right") - 1
    valid = (bands >= 0) & (bands < len(calibration.early_return_threshold_m))
    clipped = np.clip(bands, 0, len(calibration.early_return_threshold_m) - 1)
    return calibration.early_return_threshold_m[clipped], valid & calibration.usable_band[clipped]


def label_points(query: dict[str, np.ndarray], calibration: Calibration) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = query["range_residual"]
    thresholds, in_band = thresholds_for_range(calibration, query["expected_range"])
    bands = np.searchsorted(calibration.range_band_edges_m, query["expected_range"], side="right") - 1
    clipped = np.clip(bands, 0, len(calibration.normal_residual_threshold_m) - 1)
    normal_thresholds = calibration.normal_residual_threshold_m[clipped]
    valid = query["valid_reference"] & in_band & np.isfinite(residual)

    labels = np.full(len(residual), IGNORE, dtype=np.uint8)
    confidence = np.zeros(len(residual), dtype=np.float32)
    early_amount = -residual
    smoke = valid & (early_amount > thresholds)
    normal = valid & (np.abs(residual) <= normal_thresholds)
    labels[normal] = UNAFFECTED
    labels[smoke] = SMOKE_IMPACTED
    confidence[normal] = np.clip(
        1.0 - np.abs(residual[normal]) / normal_thresholds[normal], 0.0, 1.0
    )
    confidence[smoke] = np.clip((early_amount[smoke] - thresholds[smoke]) / (2.0 * thresholds[smoke]), 0.0, 1.0)
    return labels, confidence, thresholds.astype(np.float32)
