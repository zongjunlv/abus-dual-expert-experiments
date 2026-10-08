#!/usr/bin/env python3
"""Collect per-run and aggregate training times from checkpoint summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_input(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise argparse.ArgumentTypeError("--input must have the form label=checkpoint_root")
    label, raw_path = specification.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--input label and path must be non-empty")
    return label.strip(), Path(raw_path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=parse_input, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    source_files: list[str] = []
    for label, root in args.input:
        paths = sorted(root.glob("**/best_model_multiseed_summary.csv"))
        if not paths:
            raise FileNotFoundError(f"No training summaries found below {root}")
        for path in paths:
            frame = pd.read_csv(path)
            if "train_time_sec" not in frame or frame.empty:
                raise ValueError(f"{path} lacks a non-empty train_time_sec column")
            source_files.append(str(path))
            for seconds in pd.to_numeric(frame["train_time_sec"], errors="coerce"):
                if not np.isfinite(seconds):
                    raise ValueError(f"{path} contains a non-finite train_time_sec")
                rows.append(
                    {
                        "experiment": label,
                        "summary_path": str(path),
                        "train_time_sec": float(seconds),
                        "train_time_hours": float(seconds / 3600.0),
                    }
                )

    per_run = pd.DataFrame(rows)
    summary = (
        per_run.groupby("experiment", sort=True)["train_time_hours"]
        .agg(
            run_n="size",
            mean_hours="mean",
            sd_hours="std",
            median_hours="median",
            total_hours="sum",
        )
        .reset_index()
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(args.output_dir / "training_time_per_run.csv", index=False)
    summary.to_csv(args.output_dir / "training_time_summary.csv", index=False)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "time_source": "train_time_sec recorded by trainer.runner.train_one_seed",
                "source_file_count": len(source_files),
                "source_files": source_files,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
