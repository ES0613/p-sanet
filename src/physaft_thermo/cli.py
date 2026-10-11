"""Command-line interface for reviewer-facing inference."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

from .constants import DEFAULT_MODELS, FRAMEWORK_NAME
from .pipeline import PROPERTIES, predict_range
from .smi_ted import default_smi_ted_dir, download_smi_ted


def _models(value: str) -> tuple[str, ...]:
    models = tuple(item.strip() for item in value.split(",") if item.strip())
    if not models:
        raise argparse.ArgumentTypeError("Models must be a comma-separated list.")
    return models


def _properties(value: str) -> tuple[str, ...]:
    selected = tuple(item.strip() for item in value.split(","))
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(PROPERTIES):
        raise argparse.ArgumentTypeError(f"Choose distinct comma-separated properties from {PROPERTIES}.")
    return selected


def _slug(index: int, smiles: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", smiles).strip("_")[:40] or "molecule"
    return f"{index:04d}_{safe}"


def _configure_smi_ted(path: Path | None) -> None:
    if path is not None:
        os.environ["SMI_TED_HOME"] = str(path.expanduser().resolve())


def _run_predict(args: argparse.Namespace) -> int:
    _configure_smi_ted(args.smi_ted_dir)
    result = predict_range(
        args.smiles,
        args.t_min,
        args.t_max,
        n_points=args.n_points,
        models=args.models,
        device=args.device,
        properties=args.properties,
    )
    output = result.write(args.output)
    print(f"Wrote prediction artifacts to {output}")
    if args.strict and result.has_failures:
        print("Strict mode: at least one model/temperature failed.", file=sys.stderr)
        return 2
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    _configure_smi_ted(args.smi_ted_dir)
    frame = pd.read_csv(args.input)
    required = {"smiles", "t_min_k", "t_max_k"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Batch CSV is missing columns: {missing}")
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object]] = []
    any_failure = False
    for index, row in frame.iterrows():
        smiles = str(row["smiles"])
        destination = output_root / _slug(index + 1, smiles)
        n_points = int(row["n_points"]) if "n_points" in frame.columns and pd.notna(row["n_points"]) else args.n_points
        try:
            result = predict_range(
                smiles,
                float(row["t_min_k"]),
                float(row["t_max_k"]),
                n_points=n_points,
                models=args.models,
                device=args.device,
                properties=args.properties,
            )
            result.write(destination)
            failed = result.has_failures
            any_failure = any_failure or failed
            summaries.append(
                {
                    "row": index + 1,
                    "smiles": smiles,
                    "output_directory": destination.name,
                    "status": "partial_failure" if failed else "ok",
                    "message": "",
                }
            )
        except Exception as exc:
            any_failure = True
            summaries.append(
                {
                    "row": index + 1,
                    "smiles": smiles,
                    "output_directory": destination.name,
                    "status": "failed",
                    "message": str(exc),
                }
            )
            if args.strict:
                pd.DataFrame(summaries).to_csv(output_root / "batch_summary.csv", index=False)
                print(f"Batch stopped at row {index + 1}: {exc}", file=sys.stderr)
                return 2
    pd.DataFrame(summaries).to_csv(output_root / "batch_summary.csv", index=False)
    print(f"Wrote batch artifacts to {output_root}")
    return 2 if args.strict and any_failure else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physaft-thermo",
        description=f"{FRAMEWORK_NAME} five-model PC-SAFT property inference.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-smi-ted", help="Install pinned SMI-TED-Light assets.")
    download.add_argument("--destination", type=Path, default=None)
    download.add_argument("--source-dir", type=Path, default=None, help="Verified offline source directory.")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--models", type=_models, default=DEFAULT_MODELS)
    common.add_argument("--device", default="cpu")
    common.add_argument("--smi-ted-dir", type=Path, default=None)
    common.add_argument("--strict", action="store_true")
    common.add_argument("--properties", type=_properties, default=None,
                        help="Comma-separated properties: " + ", ".join(PROPERTIES))

    single = subparsers.add_parser("predict", parents=[common], help="Predict one molecule.")
    single.add_argument("--smiles", required=True)
    single.add_argument("--t-min", type=float, required=True)
    single.add_argument("--t-max", type=float, required=True)
    single.add_argument("--n-points", type=int, default=125)
    single.add_argument("--output", type=Path, required=True)

    batch = subparsers.add_parser("batch", parents=[common], help="Predict rows in a CSV file.")
    batch.add_argument("--input", type=Path, required=True)
    batch.add_argument("--output", type=Path, required=True)
    batch.add_argument("--n-points", type=int, default=125, help="Default when the CSV omits n_points.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "download-smi-ted":
            destination = download_smi_ted(args.destination or default_smi_ted_dir(), args.source_dir)
            print(f"Verified SMI-TED-Light assets in {destination}")
            return 0
        if args.command == "predict":
            return _run_predict(args)
        if args.command == "batch":
            return _run_batch(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
