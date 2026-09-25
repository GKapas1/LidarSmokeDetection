from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .config import public_config
from .data import UnifiedDataset, create_split_plan, iter_frames, load_split_plan
from .evaluate import (
    choose_threshold,
    collect_predictions,
    evaluation_report,
    fit_calibration,
    save_calibration_plot,
    save_report,
)
from .features import FEATURE_NAMES, FeatureNormalizer, fit_normalizer, voxelize_frame
from .metrics import average_precision
from .models import create_model


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(value: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def _checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict,
    normalizer: FeatureNormalizer,
    best_selection_ap: float,
) -> dict:
    return {
        "format_version": 1,
        "epoch": epoch,
        "model_name": config["model"]["name"],
        "model_parameters": config["model"],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "feature_names": list(FEATURE_NAMES),
        "normalizer": normalizer.to_dict(),
        "voxel_size_m": float(config["data"]["voxel_size_m"]),
        "best_selection_ap": best_selection_ap,
        "config": public_config(config),
    }


def _mean_domain_ap(predictions) -> float:
    values: list[float] = []
    for code in range(len(predictions.domain_names)):
        mask = predictions.domains == code
        value = average_precision(predictions.labels[mask], predictions.logits[mask])
        if value is not None:
            values.append(value)
    if not values:
        raise ValueError("selection data has no positive labels")
    return float(np.mean(values))


def train_local_model(
    config: dict,
    *,
    run_name: str | None = None,
    max_train_frames: int | None = None,
    max_eval_frames: int | None = None,
) -> Path:
    training = config["training"]
    seed = int(training["seed"])
    seed_everything(seed)
    device = resolve_device(str(training["device"]))
    dataset = UnifiedDataset(
        config["data"]["manifest"],
        verify_checksums=bool(config["data"].get("verify_checksums", False)),
    )
    split_path = Path(config["data"]["split"])
    if not split_path.exists():
        create_split_plan(dataset, split_path)
    split = load_split_plan(dataset, split_path)

    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("local-%Y%m%dT%H%M%SZ")
    run_dir = Path(config["output"]["root"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(public_config(config), indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "split.json").write_text(
        json.dumps(split, indent=2) + "\n", encoding="utf-8"
    )

    voxel_size = float(config["data"]["voxel_size_m"])
    normalizer = fit_normalizer(
        iter_frames(dataset, split, "train", max_frames=max_train_frames), voxel_size
    )
    (run_dir / "normalizer.json").write_text(
        json.dumps(normalizer.to_dict(), indent=2) + "\n", encoding="utf-8"
    )

    model = create_model(config["model"]["name"], len(FEATURE_NAMES), config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    positive_weight = torch.tensor(float(training["positive_weight"]), device=device)
    epochs = int(training["epochs"])
    patience = int(training["patience"])
    deadline = time.monotonic() + float(training["max_hours"]) * 3600.0
    history: list[dict] = []
    best_ap = -np.inf
    best_epoch = 0
    stop_for_budget = False

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        supervised_points = 0
        processed_frames = 0
        frames = iter_frames(
            dataset,
            split,
            "train",
            shuffle=True,
            seed=seed + epoch,
            max_frames=max_train_frames,
        )
        for frame in frames:
            voxelized = voxelize_frame(frame, voxel_size)
            valid = voxelized.labels != 255
            if not np.any(valid) or not len(voxelized.features):
                continue
            features = torch.from_numpy(normalizer.transform(voxelized.features)).to(device)
            inverse = torch.from_numpy(voxelized.inverse).to(device)
            labels = torch.from_numpy(voxelized.labels.astype(np.float32)).to(device)
            mask = torch.from_numpy(valid).to(device)
            optimizer.zero_grad(set_to_none=True)
            voxel_logits = model(features)
            point_logits = voxel_logits[inverse]
            loss = F.binary_cross_entropy_with_logits(
                point_logits[mask], labels[mask], pos_weight=positive_weight
            )
            if not torch.isfinite(loss):
                raise RuntimeError("training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            count = int(mask.sum().item())
            epoch_loss += float(loss.detach().cpu()) * count
            supervised_points += count
            processed_frames += 1
            if time.monotonic() >= deadline:
                stop_for_budget = True
                break
        if supervised_points == 0:
            raise RuntimeError("training epoch contained no supervised points")

        selection = collect_predictions(
            model,
            iter_frames(dataset, split, "selection", max_frames=max_eval_frames),
            normalizer,
            voxel_size,
            device,
        )
        selection_ap = _mean_domain_ap(selection)
        row = {
            "epoch": epoch,
            "training_loss": epoch_loss / supervised_points,
            "training_frames": processed_frames,
            "training_points": supervised_points,
            "selection_mean_domain_ap": selection_ap,
            "stopped_for_budget": stop_for_budget,
        }
        history.append(row)
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        checkpoint = _checkpoint(model, optimizer, epoch, config, normalizer, max(best_ap, selection_ap))
        _atomic_torch_save(checkpoint, run_dir / "last.pt")
        if selection_ap > best_ap:
            best_ap = selection_ap
            best_epoch = epoch
            _atomic_torch_save(checkpoint, run_dir / "best.pt")
        if stop_for_budget or epoch - best_epoch >= patience:
            break

    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    calibration_predictions = collect_predictions(
        model,
        iter_frames(dataset, split, "calibration", max_frames=max_eval_frames),
        normalizer,
        voxel_size,
        device,
    )
    calibration = fit_calibration(calibration_predictions.logits, calibration_predictions.labels)
    selection_predictions = collect_predictions(
        model,
        iter_frames(dataset, split, "selection", max_frames=max_eval_frames),
        normalizer,
        voxel_size,
        device,
    )
    threshold = choose_threshold(selection_predictions, calibration)
    report = evaluation_report(selection_predictions, calibration, threshold)
    report.update(
        {
            "run_name": run_name,
            "device": str(device),
            "best_epoch": best_epoch,
            "best_selection_mean_domain_ap": best_ap,
            "split_sha256": split["split_sha256"],
        }
    )
    report["partition"] = "selection"
    save_report(report, run_dir / "selection_metrics.json")
    save_calibration_plot(
        calibration_predictions.labels,
        calibration.apply(calibration_predictions.logits),
        run_dir / "calibration_fit.png",
    )
    bundle = {
        "format_version": 1,
        "model_name": config["model"]["name"],
        "model_parameters": config["model"],
        "model_state": model.state_dict(),
        "feature_names": list(FEATURE_NAMES),
        "normalizer": normalizer.to_dict(),
        "voxel_size_m": voxel_size,
        "calibration": calibration.to_dict(),
        "threshold": threshold,
        "split_sha256": split["split_sha256"],
        "manifest_sha256": dataset.manifest_sha256,
    }
    _atomic_torch_save(bundle, run_dir / "bundle.pt")
    return run_dir
