from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import inspect_bag, load_config
from .raw_dataset import run_raw_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smoke-label",
        description="Create frame-preserving Livox MID-360 smoke-impact datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Check that a bag can be decoded and show its selected lidar topic")
    inspect.add_argument("bag")
    inspect.add_argument("--topic", default=None)

    grandtour_inspect = sub.add_parser(
        "grandtour-inspect",
        help="Inspect a GrandTour session and verify raw/undistorted Livox point identity",
    )
    grandtour_inspect.add_argument("session_dir")
    grandtour_inspect.add_argument("--sample-frames", type=int, default=5)

    grandtour_preview = sub.add_parser(
        "grandtour-preview",
        help="Build a downsampled MID-360 map preview in the session's DLIO frame",
    )
    grandtour_preview.add_argument("session_dir")
    grandtour_preview.add_argument("--output", required=True)
    grandtour_preview.add_argument("--frame-stride", type=int, default=10)
    grandtour_preview.add_argument("--point-stride", type=int, default=20)
    grandtour_preview.add_argument("--minimum-range-m", type=float, default=0.1)
    grandtour_preview.add_argument("--maximum-range-m", type=float, default=30.0)

    grandtour_reference = sub.add_parser(
        "grandtour-reference",
        help="Build a stable voxelized ARC-5 MID-360 clean reference",
    )
    grandtour_reference.add_argument("session_dir")
    grandtour_reference.add_argument("--config", required=True)
    grandtour_reference.add_argument("--output", required=True)

    grandtour_pilot = sub.add_parser(
        "grandtour-pilot",
        help="Generate preliminary ARC-6 distance labels for manual review",
    )
    grandtour_pilot.add_argument("session_dir")
    grandtour_pilot.add_argument("--config", required=True)
    grandtour_pilot.add_argument("--reference", required=True)
    grandtour_pilot.add_argument("--output", required=True)

    grandtour_session = sub.add_parser(
        "grandtour-session",
        help="Label a complete GrandTour session in bounded time chunks",
    )
    grandtour_session.add_argument("session_dir")
    grandtour_session.add_argument("--config", required=True)
    grandtour_session.add_argument("--reference", required=True)
    grandtour_session.add_argument("--output", required=True)

    grandtour_validate = sub.add_parser(
        "grandtour-validate-clean",
        help="Validate GrandTour thresholds on the reserved ARC-5 clean interval",
    )
    grandtour_validate.add_argument("session_dir")
    grandtour_validate.add_argument("--reference-config", required=True)
    grandtour_validate.add_argument("--label-config", required=True)
    grandtour_validate.add_argument("--reference", required=True)
    grandtour_validate.add_argument("--output", required=True)

    grandtour_export = sub.add_parser(
        "grandtour-export-training",
        help="Export reviewed GrandTour labels without teacher-only features",
    )
    grandtour_export.add_argument("source_manifest")
    grandtour_export.add_argument("--output", required=True)

    unified_export = sub.add_parser(
        "unified-export",
        help="Convert stationary and GrandTour labels to one trainer-facing schema",
    )
    unified_export.add_argument(
        "--stationary",
        action="append",
        default=[],
        help="Stationary labeled-set directory; repeatable",
    )
    unified_export.add_argument(
        "--grandtour",
        action="append",
        default=[],
        help="GrandTour training_manifest.json; repeatable",
    )
    unified_export.add_argument("--output", required=True)

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
        if args.command == "grandtour-inspect":
            from .grandtour import inspect_grandtour_session

            print(json.dumps(inspect_grandtour_session(args.session_dir, args.sample_frames), indent=2))
            return
        if args.command == "grandtour-preview":
            from .grandtour import write_map_preview

            result = write_map_preview(
                args.session_dir,
                args.output,
                args.frame_stride,
                args.point_stride,
                args.minimum_range_m,
                args.maximum_range_m,
            )
            print(json.dumps(result, indent=2))
            return
        if args.command == "grandtour-reference":
            from .grandtour_pipeline import build_grandtour_reference

            result = build_grandtour_reference(args.session_dir, args.config, args.output)
            print(json.dumps(result, indent=2))
            return
        if args.command == "grandtour-pilot":
            from .grandtour_pipeline import label_grandtour_pilot

            result = label_grandtour_pilot(
                args.session_dir, args.config, args.reference, args.output
            )
            print(json.dumps(result, indent=2))
            return
        if args.command == "grandtour-session":
            from .grandtour_pipeline import label_grandtour_session

            result = label_grandtour_session(
                args.session_dir, args.config, args.reference, args.output
            )
            print(json.dumps(result, indent=2))
            return
        if args.command == "grandtour-validate-clean":
            from .grandtour_pipeline import validate_grandtour_clean

            result = validate_grandtour_clean(
                args.session_dir,
                args.reference_config,
                args.label_config,
                args.reference,
                args.output,
            )
            print(json.dumps(result, indent=2))
            return
        if args.command == "grandtour-export-training":
            from .grandtour_pipeline import export_grandtour_training_dataset

            result = export_grandtour_training_dataset(args.source_manifest, args.output)
            print(json.dumps(result, indent=2))
            return
        if args.command == "unified-export":
            from .unified_dataset import export_unified_dataset

            result = export_unified_dataset(args.stationary, args.grandtour, args.output)
            print(json.dumps(result, indent=2))
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
