"""Finite object surfaces for offline car-label review.

Surface support uses separate time blocks and never uses existing class labels.
Thresholds are empirical tolerances, not pose uncertainty estimates.
"""
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def floor_coordinates(world, floor):
    """Express observations in a rigid frame with height above the fitted floor."""
    normal = np.asarray(floor["floor_normal"], dtype=float)
    normal /= np.linalg.norm(normal)
    horizontal = Rotation.from_euler("z", 19, degrees=True).as_matrix()[:, 0]
    horizontal = horizontal - np.dot(horizontal, normal) * normal
    horizontal /= np.linalg.norm(horizontal)
    transform = np.column_stack((horizontal, np.cross(normal, horizontal), normal))
    points = world @ transform
    points[:, 2] += floor["floor_offset_m"]
    return points


@dataclass
class Panel:
    center: np.ndarray
    normal: np.ndarray
    tangent: np.ndarray
    vertical: np.ndarray
    low: np.ndarray
    high: np.ndarray
    anchors: np.ndarray
    blocks: np.ndarray

    @cached_property
    def tree(self):
        return cKDTree(self.coordinates(self.anchors))

    def coordinates(self, points):
        delta = points - self.center
        return np.column_stack((delta @ self.tangent, delta @ self.vertical))

    def support(self, points, block, radius=.10, minimum_blocks=3):
        """Require nearby observed face samples from other time blocks."""
        uv = self.coordinates(points)
        inside = np.all((uv >= self.low) & (uv <= self.high), axis=1)
        selected = np.flatnonzero(inside)
        result = np.zeros(len(points), bool)
        if not len(selected):
            return result
        distance, index = self.tree.query(uv[selected], k=32, distance_upper_bound=radius)
        valid = np.isfinite(distance)
        blocks = self.blocks[np.minimum(index, len(self.blocks) - 1)].copy()
        blocks[~valid | (blocks == block)] = -1
        blocks.sort(axis=1)
        unique = (blocks[:, 0] >= 0).astype(int)
        unique += np.sum((blocks[:, 1:] != blocks[:, :-1]) & (blocks[:, 1:] >= 0), axis=1)
        result[selected] = unique >= minimum_blocks
        return result


def fit_panels(points, blocks, seed=17):
    """Fit upright finite faces in an object ROI using measured endpoints."""
    rng = np.random.default_rng(seed)
    remaining = np.arange(len(points))
    panels = []
    for _ in range(16):
        if len(remaining) < 150:
            break
        sample_ids = rng.choice(remaining, min(6000, len(remaining)), replace=False)
        sample = points[sample_ids]
        best = np.zeros(len(sample), dtype=bool)
        for _ in range(220):
            triangle = sample[rng.choice(len(sample), 3, replace=False)]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            length = np.linalg.norm(normal)
            if length < .01:
                continue
            normal /= length
            if abs(normal[2]) > .18:
                continue
            inside = np.abs((sample - triangle[0]) @ normal) < .022
            if inside.sum() > best.sum():
                best = inside
        if best.sum() < 100:
            break
        center = sample[best].mean(axis=0)
        _, _, axes = np.linalg.svd(sample[best] - center, full_matrices=False)
        normal = axes[-1]
        candidate = remaining[np.abs((points[remaining] - center) @ normal) < .025]
        cloud = points[candidate]
        center = cloud.mean(axis=0)
        _, _, axes = np.linalg.svd(cloud - center, full_matrices=False)
        normal = axes[-1]
        tangent = np.cross([0., 0., 1.], normal)
        tangent /= np.linalg.norm(tangent)
        vertical = np.cross(normal, tangent)
        uv = np.column_stack(((cloud - center) @ tangent, (cloud - center) @ vertical))
        low, high = np.quantile(uv, [.01, .99], axis=0)
        enough = (high[0] - low[0] > .40 and high[1] - low[1] > .30
                  and len(np.unique(blocks[candidate])) >= 6 and abs(normal[2]) < .18)
        if enough:
            # Retain one anchor per face cell and block so dense scans cannot
            # crowd all independent temporal observations out of a query.
            keys = np.column_stack((np.floor(uv / .04).astype(int), blocks[candidate]))
            _, keep = np.unique(keys, axis=0, return_index=True)
            panels.append(Panel(center, normal, tangent, vertical, low, high,
                                cloud[keep], blocks[candidate][keep]))
        remaining = remaining[~np.isin(remaining, candidate)]
    return panels


def panel_evidence(points, origins, panels, block):
    """Distinguish face endpoints, surface ambiguity, and bounded early returns."""
    surface = np.zeros(len(points), bool)
    ambiguous = np.zeros(len(points), bool)
    early = np.zeros(len(points), bool)
    ray = points - origins
    length = np.linalg.norm(ray, axis=1)
    for panel in panels:
        signed = (points - panel.center) @ panel.normal
        nearby = np.flatnonzero(np.abs(signed) <= .055)
        supported = np.zeros(len(points), bool)
        supported[nearby] = panel.support(points[nearby], block)
        denominator = ray @ panel.normal
        incidence = np.abs(denominator) / np.maximum(length, 1e-9)
        surface |= supported & (np.abs(signed) <= .018) & (np.abs(signed) <= .045 * incidence)
        ambiguous |= supported & (np.abs(signed) <= .055)
        factor = np.divide(-((origins - panel.center) @ panel.normal), denominator,
                           out=np.full(len(points), np.nan), where=np.abs(denominator) > 1e-9)
        intersection = origins + factor[:, None] * ray
        gap = (factor - 1) * length
        possible = np.flatnonzero(np.isfinite(gap) & (gap >= .12) & (gap <= .80)
                                  & (np.abs(signed) > .065) & (incidence > .15))
        if len(possible):
            early[possible] |= panel.support(intersection[possible], block)
    early &= ~ambiguous
    return surface, ambiguous & ~surface, early


def fit_trunk(points, blocks, center_bounds):
    """Fit a circular lower trunk from a reviewed object region."""
    if (len(points) < 20 or len(np.unique(blocks)) < 6
            or not all(np.any(blocks % 2 == parity) for parity in [0, 1])):
        return dict(accepted=False, reason="Insufficient independent trunk observations")
    center = np.median(points[:, :2], axis=0)
    low, high = np.asarray(center_bounds)
    center = np.clip(center, low + .001, high - .001)

    def residual(parameters, xy):
        return np.linalg.norm(xy - parameters[:2], axis=1) - parameters[2]

    def fit(xy):
        result = least_squares(residual, np.r_[center, .045], args=(xy,),
                               bounds=(np.r_[low, .015], np.r_[high, .10]),
                               loss="soft_l1", f_scale=.008)
        return result.x

    fitted = fit(points[:, :2])
    halves = [fit(points[blocks % 2 == parity, :2]) for parity in [0, 1]]
    difference = float(np.linalg.norm(halves[0][:2] - halves[1][:2]))
    median_error = float(np.median(np.abs(residual(fitted, points[:, :2]))))
    accepted = (difference <= .025 and abs(halves[0][2] - halves[1][2]) <= .02
                and median_error <= .018 and len(np.unique(blocks)) >= 6)
    return dict(center=fitted[:2].tolist(), radius=float(fitted[2]), accepted=bool(accepted),
                alternate_block_center_difference_m=difference, median_residual_m=median_error,
                blocks=int(len(np.unique(blocks))))


def trunk_evidence(points, model, height_bounds):
    residual = np.abs(np.linalg.norm(points[:, :2] - model["center"], axis=1) - model["radius"])
    height = (points[:, 2] >= height_bounds[0]) & (points[:, 2] <= height_bounds[1])
    surface = height & (residual <= .014) & model["accepted"]
    ambiguous = height & (residual <= .04) & ~surface & model["accepted"]
    return surface, ambiguous


def revise_labels(labels, surface, ambiguous, smoke):
    """Preserve existing positives unless a supported surface overrides them."""
    result = labels.copy()
    result[smoke & ~surface & ~ambiguous & (labels == 255)] = 1
    result[surface] = 0
    return result
