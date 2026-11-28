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

from .adapters import (
    FrameAdapter,
    GlobalTemporalAdapter,
    adapter_name_for_model,
    create_adapter_for_model,
)
from .config import public_config
from .augmentation import augment_training_frame, validate_augmentation
from .data import (
    TemporalWindow,
    UnifiedDataset,
    create_split_plan,
    iter_frames,
    iter_interleaved_frames,
    iter_temporal_windows,
    load_split_plan,
)
from .evaluate import (
    choose_threshold,
    collect_predictions,
    evaluation_report,
    fit_calibration,
    collect_temporal_predictions,
    save_calibration_plot,
    save_report,
)
from .features import FeatureNormalizer
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
    adapter: FrameAdapter,
    best_selection_ap: float,
) -> dict:
    return {
        "format_version": 2,
        "epoch": epoch,
        "model_name": config["model"]["name"],
        "model_parameters": config["model"],
        "adapter_name": adapter.name,
        "input_schema": list(normalizer.feature_names),
        "preprocessing": adapter.preprocessing(config["data"]),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "feature_names": list(normalizer.feature_names),
        "normalizer": normalizer.to_dict(),
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


def _regularization_loss(model: torch.nn.Module, reference: torch.Tensor) -> torch.Tensor:
    method = getattr(model, "regularization_loss", None)
    if method is None:
        return reference.new_zeros(())
    value = method()
    if not isinstance(value, torch.Tensor) or value.ndim != 0:
        raise ValueError("model regularization_loss must return a scalar tensor")
    if not torch.isfinite(value):
        raise RuntimeError("model regularization loss became non-finite")
    return value


def train_model(
    config: dict,
    *,
    run_name: str | None = None,
    resume_run: str | Path | None = None,
    max_train_frames: int | None = None,
    max_eval_frames: int | None = None,
) -> Path:
    training = config["training"]
    frame_augmentation = config.get("augmentation", {})
    validate_augmentation(frame_augmentation, config["model"]["name"])
    step_limit = training.get("max_optimizer_steps_per_epoch")
    sampling = training.get("frame_sampling", "chunk_shuffle")
    if sampling not in {"chunk_shuffle", "interleaved"}:
        raise ValueError("unknown frame_sampling")
    if sampling == "interleaved" and (step_limit is None or config["model"]["name"] == "global_temporal_pointnet"):
        raise ValueError("interleaved sampling requires a step limit and a single-scan model")
    if step_limit is not None and int(step_limit) <= 0:
        raise ValueError("max_optimizer_steps_per_epoch must be positive")
    seed = int(training["seed"])
    seed_everything(seed)
    device = resolve_device(str(training["device"]))
    dataset = UnifiedDataset(
        config["data"]["manifest"],
        verify_checksums=bool(config["data"].get("verify_checksums", False)),
    )
    adapter = create_adapter_for_model(config["model"])
    if isinstance(adapter, GlobalTemporalAdapter):
        return train_global_temporal_model(
            config,
            run_name=run_name,
            resume_run=resume_run,
            max_train_frames=max_train_frames,
            max_eval_frames=max_eval_frames,
        )
    split_path = Path(config["data"]["split"])
    requested_domains = config["data"].get("source_domains")
    if not split_path.exists():
        create_split_plan(dataset, split_path, source_domains=requested_domains)
    split = load_split_plan(dataset, split_path)
    if requested_domains is not None:
        expected_domains = {str(value) for value in requested_domains}
        actual_domains = {str(value) for value in split.get("source_domains", ())}
        if actual_domains != expected_domains:
            raise ValueError(
                "split source domains do not match the training configuration; "
                "rebuild the split before training"
            )

    if resume_run is None:
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
        normalizer = adapter.fit_normalizer(
            iter_frames(dataset, split, "train", max_frames=max_train_frames),
            config["data"],
        )
        (run_dir / "normalizer.json").write_text(
            json.dumps(normalizer.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        checkpoint = None
        history: list[dict] = []
        start_epoch = 1
        best_ap = -np.inf
        best_epoch = 0
    else:
        run_dir = Path(resume_run).expanduser().resolve()
        checkpoint_path = run_dir / "last.pt"
        history_path = run_dir / "history.json"
        if not checkpoint_path.is_file() or not history_path.is_file():
            raise FileNotFoundError("resume run requires last.pt and history.json")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["config"]["training"].get("frame_sampling", "chunk_shuffle") != sampling:
            raise ValueError("resume checkpoint frame sampling differs from the configuration")
        if checkpoint["config"].get("augmentation", {}) != frame_augmentation:
            raise ValueError("resume checkpoint augmentation differs from the configuration")
        if checkpoint["config"]["training"].get("max_optimizer_steps_per_epoch") != step_limit:
            raise ValueError("resume checkpoint optimizer step limit differs from the configuration")
        if checkpoint["model_parameters"] != config["model"]:
            raise ValueError("resume checkpoint model parameters do not match the configuration")
        if checkpoint.get("adapter_name", adapter_name_for_model(config["model"])) != adapter.name:
            raise ValueError("resume checkpoint adapter does not match the configuration")
        normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not history or int(history[-1]["epoch"]) != int(checkpoint["epoch"]):
            raise ValueError("resume history does not match the last checkpoint")
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(max(history, key=lambda row: row["selection_mean_domain_ap"])["epoch"])
        best_ap = float(checkpoint["best_selection_ap"])

    if normalizer.feature_names != adapter.feature_names:
        raise ValueError("normalizer feature schema does not match the model adapter")
    model = create_model(
        config["model"]["name"], len(normalizer.feature_names), config["model"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    positive_weight = torch.tensor(float(training["positive_weight"]), device=device)
    alignment_weight = float(config["model"].get("alignment_regularization", 0.0))
    if alignment_weight < 0:
        raise ValueError("alignment_regularization must be nonnegative")
    epochs = int(training["epochs"])
    patience = int(training["patience"])
    deadline = time.monotonic() + float(training["max_hours"]) * 3600.0
    stop_for_budget = False

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_alignment_loss = 0.0
        supervised_points = 0
        processed_frames = 0
        augmentation_counts = {"copied_smoke_points": 0, "masked_context_points": 0}
        domain_frames: dict[str, int] = {}
        frames = iter_frames(
            dataset,
            split,
            "train",
            shuffle=True,
            seed=seed + epoch,
            max_frames=max_train_frames,
        )
        augmentation_rng = np.random.default_rng(seed + epoch)
        if sampling == "interleaved":
            frames = iter_interleaved_frames(dataset, split, seed=seed + epoch,
                steps=int(step_limit), max_frames=max_train_frames)
        frame_rng = np.random.default_rng(seed + epoch + 100_000)
        for frame in frames:
            frame, counts = augment_training_frame(frame, frame_augmentation, frame_rng)
            prepared = adapter.prepare(frame, config["data"])
            valid = prepared.valid & (prepared.labels != 255)
            if not np.any(valid) or not len(prepared.features):
                continue
            augmented_features = adapter.augment_features(
                prepared.features, training, augmentation_rng
            )
            features = torch.from_numpy(normalizer.transform(augmented_features)).to(device)
            labels = torch.from_numpy(prepared.labels.astype(np.float32)).to(device)
            mask = torch.from_numpy(valid).to(device)
            optimizer.zero_grad(set_to_none=True)
            point_logits = adapter.expand_logits(model(features), prepared)
            loss = F.binary_cross_entropy_with_logits(
                point_logits[mask], labels[mask], pos_weight=positive_weight
            )
            alignment_loss = _regularization_loss(model, loss)
            loss = loss + alignment_weight * alignment_loss
            if not torch.isfinite(loss):
                raise RuntimeError("training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            count = int(mask.sum().item())
            epoch_loss += float(loss.detach().cpu()) * count
            epoch_alignment_loss += float(alignment_loss.detach().cpu()) * count
            supervised_points += count
            processed_frames += 1
            domain_frames[frame.source_domain] = domain_frames.get(frame.source_domain, 0) + 1
            for key, count in counts.items():
                augmentation_counts[key] += count
            if time.monotonic() >= deadline:
                stop_for_budget = True
                break
            if step_limit is not None and processed_frames >= int(step_limit):
                break
        if supervised_points == 0:
            raise RuntimeError("training epoch contained no supervised points")

        selection = collect_predictions(
            model,
            iter_frames(dataset, split, "selection", max_frames=max_eval_frames),
            normalizer,
            adapter,
            config["data"],
            device,
        )
        selection_ap = _mean_domain_ap(selection)
        row = {
            "epoch": epoch,
            "training_loss": epoch_loss / supervised_points,
            "training_alignment_loss": epoch_alignment_loss / supervised_points,
            "training_frames": processed_frames,
            "training_points": supervised_points,
            "selection_mean_domain_ap": selection_ap,
            "stopped_for_budget": stop_for_budget,
            "augmentation": augmentation_counts,
            "training_frames_by_domain": domain_frames,
        }
        history.append(row)
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        checkpoint = _checkpoint(
            model, optimizer, epoch, config, normalizer, adapter, max(best_ap, selection_ap)
        )
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
        adapter,
        config["data"],
        device,
    )
    calibration = fit_calibration(calibration_predictions.logits, calibration_predictions.labels)
    selection_predictions = collect_predictions(
        model,
        iter_frames(dataset, split, "selection", max_frames=max_eval_frames),
        normalizer,
        adapter,
        config["data"],
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
        "format_version": 2,
        "model_name": config["model"]["name"],
        "model_parameters": config["model"],
        "adapter_name": adapter.name,
        "input_schema": list(normalizer.feature_names),
        "preprocessing": adapter.preprocessing(config["data"]),
        "model_state": model.state_dict(),
        "feature_names": list(normalizer.feature_names),
        "normalizer": normalizer.to_dict(),
        "calibration": calibration.to_dict(),
        "threshold": threshold,
        "split_sha256": split["split_sha256"],
        "manifest_sha256": dataset.manifest_sha256,
    }
    _atomic_torch_save(bundle, run_dir / "bundle.pt")
    return run_dir


def train_global_temporal_model(
    config: dict,
    *,
    run_name: str | None = None,
    resume_run: str | Path | None = None,
    max_train_frames: int | None = None,
    max_eval_frames: int | None = None,
) -> Path:
    """Train GT on bounded windows; only each window's final scan is supervised."""

    training = config["training"]
    frame_augmentation = config.get("augmentation", {})
    validate_augmentation(frame_augmentation, config["model"]["name"])
    step_limit = training.get("max_optimizer_steps_per_epoch")
    if step_limit is not None and int(step_limit) <= 0:
        raise ValueError("max_optimizer_steps_per_epoch must be positive")
    seed = int(training["seed"])
    seed_everything(seed)
    device = resolve_device(str(training["device"]))
    adapter = create_adapter_for_model(config["model"])
    if not isinstance(adapter, GlobalTemporalAdapter):
        raise ValueError("GT training requires the global_temporal adapter")
    history_length = int(config["data"]["history_length"])
    maximum_time_gap_s = float(config["data"]["maximum_time_gap_s"])
    configured_history = int(config["model"].get("history_length", history_length))
    if configured_history != history_length:
        raise ValueError("model and data history_length must match")

    dataset = UnifiedDataset(
        config["data"]["manifest"],
        verify_checksums=bool(config["data"].get("verify_checksums", False)),
    )
    split_path = Path(config["data"]["split"])
    requested_domains = config["data"].get("source_domains")
    if not split_path.exists():
        create_split_plan(dataset, split_path, source_domains=requested_domains)
    split = load_split_plan(dataset, split_path)
    if requested_domains is not None:
        expected_domains = {str(value) for value in requested_domains}
        actual_domains = {str(value) for value in split.get("source_domains", ())}
        if actual_domains != expected_domains:
            raise ValueError(
                "split source domains do not match the training configuration; "
                "rebuild the split before training"
            )

    if resume_run is None:
        if run_name is None:
            run_name = datetime.now(timezone.utc).strftime("gt-%Y%m%dT%H%M%SZ")
        run_dir = Path(config["output"]["root"]) / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "config.json").write_text(
            json.dumps(public_config(config), indent=2) + "\n", encoding="utf-8"
        )
        (run_dir / "split.json").write_text(
            json.dumps(split, indent=2) + "\n", encoding="utf-8"
        )
        # Fit once from the unique training frames, rather than repeatedly from
        # their overlapping appearances in temporal windows.
        normalizer = adapter.fit_normalizer(
            iter_frames(dataset, split, "train", max_frames=max_train_frames),
            config["data"],
        )
        (run_dir / "normalizer.json").write_text(
            json.dumps(normalizer.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        checkpoint = None
        history: list[dict] = []
        start_epoch = 1
        best_ap = -np.inf
        best_epoch = 0
    else:
        run_dir = Path(resume_run).expanduser().resolve()
        checkpoint_path = run_dir / "last.pt"
        history_path = run_dir / "history.json"
        if not checkpoint_path.is_file() or not history_path.is_file():
            raise FileNotFoundError("resume run requires last.pt and history.json")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["model_parameters"] != config["model"]:
            raise ValueError("resume checkpoint model parameters do not match the configuration")
        if checkpoint.get("adapter_name") != adapter.name:
            raise ValueError("resume checkpoint adapter does not match the configuration")
        if checkpoint["config"].get("augmentation", {}) != frame_augmentation:
            raise ValueError("resume checkpoint augmentation differs from the configuration")
        if checkpoint["config"]["training"].get("max_optimizer_steps_per_epoch") != step_limit:
            raise ValueError("resume checkpoint optimizer step limit differs from the configuration")
        normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not history or int(history[-1]["epoch"]) != int(checkpoint["epoch"]):
            raise ValueError("resume history does not match the last checkpoint")
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(max(history, key=lambda row: row["selection_mean_domain_ap"])["epoch"])
        best_ap = float(checkpoint["best_selection_ap"])

    if normalizer.feature_names != adapter.feature_names:
        raise ValueError("normalizer feature schema does not match the model adapter")
    model = create_model(
        config["model"]["name"], len(normalizer.feature_names), config["model"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    positive_weight = torch.tensor(float(training["positive_weight"]), device=device)
    alignment_weight = float(config["model"].get("alignment_regularization", 0.0))
    if alignment_weight < 0:
        raise ValueError("alignment_regularization must be nonnegative")
    epochs = int(training["epochs"])
    patience = int(training["patience"])
    deadline = time.monotonic() + float(training["max_hours"]) * 3600.0
    stop_for_budget = False

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_alignment_loss = 0.0
        supervised_points = 0
        processed_windows = 0
        augmentation_counts = {"copied_smoke_points": 0, "masked_context_points": 0}
        windows = iter_temporal_windows(
            dataset,
            split,
            "train",
            history_length=history_length,
            maximum_time_gap_s=maximum_time_gap_s,
            shuffle=True,
            seed=seed + epoch,
            max_windows=max_train_frames,
        )
        augmentation_rng = np.random.default_rng(seed + epoch)
        frame_rng = np.random.default_rng(seed + epoch + 100_000)
        for window in windows:
            if frame_augmentation:
                target_frame, counts = augment_training_frame(window.target, frame_augmentation, frame_rng)
                window = TemporalWindow(window.frames[:-1] + (target_frame,))
                for key, count in counts.items():
                    augmentation_counts[key] += count
            prepared = adapter.prepare_window(window, config["data"])
            target = prepared.target
            valid = target.valid & (target.labels != 255)
            if not np.any(valid) or not len(target.features):
                continue
            augmented = adapter.augment_window(prepared, training, augmentation_rng)
            feature_tensors = tuple(
                torch.from_numpy(normalizer.transform(frame.features)).to(device)
                for frame in augmented.frames
            )
            labels = torch.from_numpy(target.labels.astype(np.float32)).to(device)
            mask = torch.from_numpy(valid).to(device)
            optimizer.zero_grad(set_to_none=True)
            point_logits = adapter.expand_logits(model(feature_tensors), target)
            loss = F.binary_cross_entropy_with_logits(
                point_logits[mask], labels[mask], pos_weight=positive_weight
            )
            alignment_loss = _regularization_loss(model, loss)
            loss = loss + alignment_weight * alignment_loss
            if not torch.isfinite(loss):
                raise RuntimeError("training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            count = int(mask.sum().item())
            epoch_loss += float(loss.detach().cpu()) * count
            epoch_alignment_loss += float(alignment_loss.detach().cpu()) * count
            supervised_points += count
            processed_windows += 1
            if time.monotonic() >= deadline:
                stop_for_budget = True
                break
            if step_limit is not None and processed_windows >= int(step_limit):
                break
        if supervised_points == 0:
            raise RuntimeError("training epoch contained no supervised points")

        selection = collect_temporal_predictions(
            model,
            iter_temporal_windows(
                dataset,
                split,
                "selection",
                history_length=history_length,
                maximum_time_gap_s=maximum_time_gap_s,
                max_windows=max_eval_frames,
            ),
            normalizer,
            adapter,
            config["data"],
            device,
        )
        selection_ap = _mean_domain_ap(selection)
        history.append(
            {
                "epoch": epoch,
                "training_loss": epoch_loss / supervised_points,
                "training_alignment_loss": epoch_alignment_loss / supervised_points,
                "training_windows": processed_windows,
                "training_points": supervised_points,
                "augmentation": augmentation_counts,
                "selection_mean_domain_ap": selection_ap,
                "stopped_for_budget": stop_for_budget,
            }
        )
        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        checkpoint = _checkpoint(
            model, optimizer, epoch, config, normalizer, adapter, max(best_ap, selection_ap)
        )
        _atomic_torch_save(checkpoint, run_dir / "last.pt")
        if selection_ap > best_ap:
            best_ap = selection_ap
            best_epoch = epoch
            _atomic_torch_save(checkpoint, run_dir / "best.pt")
        if stop_for_budget or epoch - best_epoch >= patience:
            break

    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    calibration_predictions = collect_temporal_predictions(
        model,
        iter_temporal_windows(
            dataset,
            split,
            "calibration",
            history_length=history_length,
            maximum_time_gap_s=maximum_time_gap_s,
            max_windows=max_eval_frames,
        ),
        normalizer,
        adapter,
        config["data"],
        device,
    )
    calibration = fit_calibration(calibration_predictions.logits, calibration_predictions.labels)
    selection_predictions = collect_temporal_predictions(
        model,
        iter_temporal_windows(
            dataset,
            split,
            "selection",
            history_length=history_length,
            maximum_time_gap_s=maximum_time_gap_s,
            max_windows=max_eval_frames,
        ),
        normalizer,
        adapter,
        config["data"],
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
            "history_length": history_length,
            "maximum_time_gap_s": maximum_time_gap_s,
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
        "format_version": 2,
        "model_name": config["model"]["name"],
        "model_parameters": config["model"],
        "adapter_name": adapter.name,
        "input_schema": list(normalizer.feature_names),
        "preprocessing": adapter.preprocessing(config["data"]),
        "model_state": model.state_dict(),
        "feature_names": list(normalizer.feature_names),
        "normalizer": normalizer.to_dict(),
        "calibration": calibration.to_dict(),
        "threshold": threshold,
        "split_sha256": split["split_sha256"],
        "manifest_sha256": dataset.manifest_sha256,
    }
    _atomic_torch_save(bundle, run_dir / "bundle.pt")
    return run_dir


def train_local_model(*args, **kwargs) -> Path:
    """Compatibility entry point for callers of the original L1-only trainer."""

    return train_model(*args, **kwargs)
