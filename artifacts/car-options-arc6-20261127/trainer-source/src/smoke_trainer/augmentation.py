"""Geometry augmentations applied to training frames before feature extraction."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from .data import Frame


def validate_augmentation(config: dict, model_name: str) -> None:
    if not config:
        return
    if model_name not in {"global_pointnet", "global_temporal_pointnet", "local_mlp", "voxel_pointnet"}:
        raise ValueError("Frame augmentation is unsupported for this model")
    unknown = set(config) - {"smoke_relocation", "background_mask"}
    if unknown:
        raise ValueError(f"Unknown augmentation sections: {sorted(unknown)}")
    for name, options in config.items():
        if not 0 <= float(options["probability"]) <= 1:
            raise ValueError(f"{name} probability must be between zero and one")
    smoke = config.get("smoke_relocation")
    if smoke:
        if not (0 < smoke["target_min_m"] < smoke["target_max_m"]):
            raise ValueError("Invalid smoke target range")
        if min(smoke["patch_radius_m"], smoke["clearance_m"], smoke["ray_tolerance_rad"]) <= 0:
            raise ValueError("Smoke geometry tolerances must be positive")
        if smoke["target_max_m"] - smoke["target_min_m"] <= 2 * smoke["patch_radius_m"]:
            raise ValueError("Smoke target interval must fit the patch diameter")
        if int(smoke["minimum_points"]) < 2 or int(smoke["attempts"]) < 1:
            raise ValueError("Invalid smoke point or attempt count")
    mask = config.get("background_mask")
    if mask:
        if not (0 < mask["range_min_m"] <= mask["range_max_m"]):
            raise ValueError("Invalid context masking range")
        if not (0 < mask["width_min_deg"] <= mask["width_max_deg"] < 360):
            raise ValueError("Invalid context masking sector")


def _copy_distant_smoke(frame: Frame, options: dict, rng: np.random.Generator) -> tuple[Frame, int]:
    if rng.random() >= float(options["probability"]):
        return frame, 0
    xyz = frame.xyz.astype(np.float64)
    ranges = np.linalg.norm(xyz, axis=1)
    finite = np.isfinite(xyz).all(axis=1) & (ranges > 0.3)
    smoke_ids = np.flatnonzero(finite & (frame.label == 1) & (ranges < options["target_min_m"]))
    minimum_points = int(options["minimum_points"])
    if len(smoke_ids) < minimum_points:
        return frame, 0
    measured = finite & (frame.label != 1)
    background_ids = np.flatnonzero(measured & (frame.label == 0))
    if len(background_ids) < minimum_points:
        return frame, 0
    measured_xyz = xyz[measured]
    measured_ranges = ranges[measured]
    measured_tree = cKDTree(measured_xyz)
    ray_tree = cKDTree(measured_xyz / measured_ranges[:, None])
    patch_radius = float(options["patch_radius_m"])
    clearance = float(options["clearance_m"])
    # Keep the chosen distance through placement retries to avoid favoring easier near placements.
    target_range = rng.uniform(options["target_min_m"] + patch_radius, options["target_max_m"] - patch_radius)
    for _ in range(int(options["attempts"])):
        seed = xyz[rng.choice(smoke_ids)]
        ids = smoke_ids[np.linalg.norm(xyz[smoke_ids] - seed, axis=1) <= patch_radius]
        if len(ids) < minimum_points:
            continue
        patch = xyz[ids]
        center = patch.mean(axis=0)
        extent = float(np.linalg.norm(patch - center, axis=1).max())
        destinations = background_ids[ranges[background_ids] > target_range + extent + clearance]
        if not len(destinations):
            continue
        destination = int(rng.choice(destinations))
        direction = xyz[destination] / ranges[destination]
        cone = extent / target_range + float(options["ray_tolerance_rad"])
        hits = ray_tree.query_ball_point(direction, cone)
        # A supported surface behind the whole patch provides an observed free-space check.
        # Include ignored returns in the obstruction check, since they can be real objects.
        if len(hits) < minimum_points or np.min(measured_ranges[hits]) <= target_range + extent + clearance:
            continue
        moved = patch + direction * target_range - center
        moved_ranges = np.linalg.norm(moved, axis=1)
        if moved_ranges.min() < options["target_min_m"] or moved_ranges.max() > options["target_max_m"]:
            continue
        if np.any(measured_tree.query(moved, k=1)[0] < clearance):
            continue
        # Retain physical patch size and approximate its reduced angular footprint.
        retain_probability = np.minimum(1.0, np.square(ranges[ids] / moved_ranges))
        keep = rng.random(len(ids)) < retain_probability
        if np.count_nonzero(keep) < minimum_points:
            continue
        copied_ids = ids[keep]
        result = replace(
            frame,
            xyz=np.concatenate((frame.xyz, moved[keep].astype(np.float32))),
            intensity=np.concatenate((frame.intensity, frame.intensity[copied_ids])),
            label=np.concatenate((frame.label, np.ones(len(copied_ids), dtype=np.uint8))),
        )
        return result, len(copied_ids)
    return frame, 0


def _mask_far_context(frame: Frame, options: dict, rng: np.random.Generator) -> tuple[Frame, int]:
    if rng.random() >= float(options["probability"]):
        return frame, 0
    cutoff = rng.uniform(options["range_min_m"], options["range_max_m"])
    center = rng.uniform(-np.pi, np.pi)
    width = np.deg2rad(rng.uniform(options["width_min_deg"], options["width_max_deg"]))
    azimuth = np.arctan2(frame.xyz[:, 1], frame.xyz[:, 0])
    angle = np.arctan2(np.sin(azimuth - center), np.cos(azimuth - center))
    remove = (np.linalg.norm(frame.xyz, axis=1) > cutoff) & (np.abs(angle) <= width / 2)
    keep = ~remove
    return replace(frame, xyz=frame.xyz[keep], intensity=frame.intensity[keep], label=frame.label[keep]), int(remove.sum())


def augment_training_frame(frame: Frame, config: dict, rng: np.random.Generator) -> tuple[Frame, dict]:
    counts = {"copied_smoke_points": 0, "masked_context_points": 0}
    if "smoke_relocation" in config:
        frame, counts["copied_smoke_points"] = _copy_distant_smoke(frame, config["smoke_relocation"], rng)
    if "background_mask" in config:
        frame, counts["masked_context_points"] = _mask_far_context(frame, config["background_mask"], rng)
    return frame, counts
