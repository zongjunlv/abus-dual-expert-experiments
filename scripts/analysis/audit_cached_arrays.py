#!/usr/bin/env python3
"""Audit cached-array provenance without exporting patient-level predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_path(value: object) -> str:
    return str(PurePosixPath(str(value).replace("\\", "/")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    csv_records = []
    for csv_path in args.csv:
        frame = pd.read_csv(csv_path)
        if "npy_path" not in frame or "label" not in frame:
            raise ValueError(f"{csv_path} must contain npy_path and label")
        frames.append(frame)
        csv_records.append(
            {
                "path": normalized_path(csv_path),
                "sha256": sha256(csv_path),
                "rows": int(len(frame)),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["npy_path"] = combined["npy_path"].map(normalized_path)
    if combined.groupby("npy_path")["label"].nunique().max() > 1:
        raise ValueError("A cached path is associated with multiple labels")
    unique = combined.drop_duplicates("npy_path").reset_index(drop=True)

    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    missing = []
    global_min = float("inf")
    global_max = float("-inf")
    for raw_path in unique["npy_path"]:
        path = Path(raw_path)
        if not path.is_file():
            missing.append(raw_path)
            continue
        array = np.load(path, mmap_mode="r")
        shape_counts["x".join(str(value) for value in array.shape)] += 1
        dtype_counts[str(array.dtype)] += 1
        global_min = min(global_min, float(np.min(array)))
        global_max = max(global_max, float(np.max(array)))

    payload = {
        "csv_files": csv_records,
        "input_rows_before_deduplication": int(len(combined)),
        "unique_cached_cases": int(len(unique)),
        "label_counts": {
            str(key): int(value)
            for key, value in unique["label"].value_counts().sort_index().items()
        },
        "shape_counts": dict(sorted(shape_counts.items())),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "global_min": None if not shape_counts else global_min,
        "global_max": None if not shape_counts else global_max,
        "missing_array_count": len(missing),
        "missing_arrays": missing,
        "note": (
            "This audit describes the arrays consumed by Medical_Dataset. "
            "It does not by itself prove the upstream raw-image orientation or crop provenance."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
