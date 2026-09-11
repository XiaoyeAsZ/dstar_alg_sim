#!/usr/bin/env python3
"""Command-line entry point for DSTAR profiling, inference, evaluation, and simulation."""

from __future__ import annotations

import argparse
import json

try:
    from .api import evaluate_generated_quality, profile_mask, run_inference, simulate
except ImportError:  # Allow ``python dstar.py`` from the artifact directory.
    from api import evaluate_generated_quality, profile_mask, run_inference, simulate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    profile = subparsers.add_parser("profile-mask", help="Profile and save sparse attention masks")
    profile.add_argument("--config", required=True)

    infer = subparsers.add_parser("infer", help="Run quantized and optionally masked inference")
    infer.add_argument("--config", required=True)
    infer.add_argument("--mask-path")
    infer.add_argument("--quant-mode", default="adapt", choices=["adapt", "group", "ditto"])
    infer.add_argument("--no-mask", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate generated images/videos")
    evaluate.add_argument("output_dir")
    evaluate.add_argument("--reference-dir")
    evaluate.add_argument("--json")
    evaluate.add_argument("--metrics", nargs="+", default=["auto"], choices=["auto", "clip", "fid", "fvd", "is"])
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--splits", type=int, default=10)

    simulation = subparsers.add_parser("simulate", help="Simulate DSTAR traces")
    simulation.add_argument("trace_path")
    simulation.add_argument("--log-path")

    args = parser.parse_args()
    if args.command == "profile-mask":
        profile_mask(args.config)
    elif args.command == "infer":
        run_inference(args.config, mask_path=args.mask_path, quant_mode=args.quant_mode, use_mask=not args.no_mask)
    elif args.command == "evaluate":
        print(
            json.dumps(
                evaluate_generated_quality(
                    args.output_dir,
                    args.reference_dir,
                    args.json,
                    args.metrics,
                    args.batch_size,
                    args.splits,
                ),
                indent=2,
            )
        )
    else:
        print(json.dumps(simulate(args.trace_path, args.log_path), indent=2))


if __name__ == "__main__":
    main()
