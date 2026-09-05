from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import inspect_bag, load_config
from .raw_dataset import run_raw_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smoke-label",
        description="Create stationary raw Livox smoke-impact datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Check that a bag can be decoded and show its selected lidar topic")
    inspect.add_argument("bag")
    inspect.add_argument("--topic", default=None)

    dataset = sub.add_parser(
        "dataset",
        help="Build a frame-preserving raw Livox training dataset using independent clean reference/control bags",
    )
    dataset.add_argument("--config", default="config/raw_dataset.toml")
    dataset.add_argument("--clean-reference", required=True)
    dataset.add_argument("--clean-control", required=True)
    dataset.add_argument(
        "--smoke",
        action="append",
        required=True,
        help="Smoky bag path; repeat this option for low/medium/high recordings",
    )
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--topic", default="/livox/lidar")
    dataset.add_argument("--session-id", default="")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            inspect_bag(args.bag, args.topic)
            return
        if args.command == "dataset":
            config = load_config(Path(args.config))
            run_raw_dataset(
                config,
                args.clean_reference,
                args.clean_control,
                args.smoke,
                args.output,
                args.topic,
                args.session_id,
            )
            return
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
