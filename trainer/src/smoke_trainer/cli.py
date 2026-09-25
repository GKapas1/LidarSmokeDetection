from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smoke-train")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="validate and summarize a dataset manifest")
    inspect.add_argument("--manifest", required=True)
    inspect.add_argument("--verify-checksums", action="store_true")

    split = commands.add_parser("split", help="create the chronological development split")
    split.add_argument("--manifest", required=True)
    split.add_argument("--output", required=True)
    split.add_argument("--purge-frames", type=int, default=20)
    split.add_argument("--verify-checksums", action="store_true")

    train = commands.add_parser("train", help="train and package the local voxel MLP")
    train.add_argument("--config", required=True)
    train.add_argument("--run-name")
    train.add_argument("--max-train-frames", type=int)
    train.add_argument("--max-eval-frames", type=int)

    evaluate = commands.add_parser("evaluate", help="evaluate a bundle on a saved split")
    evaluate.add_argument("--bundle", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--split", required=True)
    evaluate.add_argument("--partition", choices=("selection", "calibration", "test"), default="test")
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--max-frames", type=int)

    predict = commands.add_parser("predict", help="predict and benchmark one dataset frame")
    predict.add_argument("--bundle", required=True)
    predict.add_argument("--manifest", required=True)
    predict.add_argument("--chunk", type=int, required=True)
    predict.add_argument("--frame", type=int, required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--device", default="auto")
    predict.add_argument("--repeats", type=int, default=20)
    return parser


def _inspect(args: argparse.Namespace) -> None:
    from .data import UnifiedDataset

    dataset = UnifiedDataset(args.manifest, verify_checksums=args.verify_checksums)
    result = {
        "manifest": str(dataset.manifest_path),
        "manifest_sha256": dataset.manifest_sha256,
        "chunks": len(dataset.chunks),
        "frames": sum(chunk.frames for chunk in dataset.chunks),
        "points": sum(chunk.points for chunk in dataset.chunks),
        "streams": sorted({chunk.stream_id for chunk in dataset.chunks}),
    }
    print(json.dumps(result, indent=2))


def _split(args: argparse.Namespace) -> None:
    from .data import UnifiedDataset, create_split_plan

    dataset = UnifiedDataset(args.manifest, verify_checksums=args.verify_checksums)
    plan = create_split_plan(dataset, args.output, purge_frames=args.purge_frames)
    print(json.dumps({"output": str(Path(args.output).resolve()), "summary": plan["summary"]}, indent=2))


def _train(args: argparse.Namespace) -> None:
    from .config import load_config
    from .train import train_local_model

    run_dir = train_local_model(
        load_config(args.config),
        run_name=args.run_name,
        max_train_frames=args.max_train_frames,
        max_eval_frames=args.max_eval_frames,
    )
    print(json.dumps({"run_directory": str(run_dir), "bundle": str(run_dir / "bundle.pt")}, indent=2))


def _evaluate(args: argparse.Namespace) -> None:
    from .data import UnifiedDataset, iter_frames, load_split_plan
    from .evaluate import collect_predictions, evaluation_report, save_report
    from .predict import LocalPredictor

    predictor = LocalPredictor.load(args.bundle, device=args.device)
    dataset = UnifiedDataset(args.manifest)
    split = load_split_plan(dataset, args.split)
    predictions = collect_predictions(
        predictor.model,
        iter_frames(dataset, split, args.partition, max_frames=args.max_frames),
        predictor.normalizer,
        predictor.voxel_size_m,
        predictor.device,
    )
    report = evaluation_report(predictions, predictor.calibration, predictor.threshold)
    report["partition"] = args.partition
    save_report(report, args.output)
    print(json.dumps(report, indent=2))


def _predict(args: argparse.Namespace) -> None:
    from .data import UnifiedDataset
    from .predict import LocalPredictor

    predictor = LocalPredictor.load(args.bundle, device=args.device)
    dataset = UnifiedDataset(args.manifest)
    chunk = dataset.chunks[args.chunk]
    arrays = dataset.load_chunk(args.chunk)
    frame = dataset.frame(chunk, arrays, args.frame)
    probabilities, valid = predictor.predict(frame)
    benchmark = predictor.benchmark(frame, repeats=args.repeats)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        xyz=frame.xyz,
        smoke_probability=probabilities,
        reliability_probability=1.0 - probabilities,
        predicted_label=(probabilities >= predictor.threshold).astype(np.uint8),
        valid=valid,
        source_label=frame.label,
    )
    print(json.dumps({"output": str(output), "benchmark": benchmark}, indent=2))


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    handlers = {
        "inspect": _inspect,
        "split": _split,
        "train": _train,
        "evaluate": _evaluate,
        "predict": _predict,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()

