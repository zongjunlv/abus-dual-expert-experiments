#!/usr/bin/env python3
"""Materialize the exact tracked TDSC repeated-CV memberships on a new cache.

The tracked membership file contains public case identifiers, labels, and roles
only.  This script joins those identifiers to a user-built TDSC image cache and
writes the 45 train/selection-validation/outer-test CSVs consumed by the final
public-dataset training queue.  No historical local cache namespace is used to
resolve paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


EXPECTED_SPLIT_SEEDS = (42, 1234, 3407)
EXPECTED_FOLDS = (1, 2, 3, 4, 5)
EXPECTED_ROLE_COUNTS = {"train": 136, "selection_val": 24, "outer_test": 40}
EXPECTED_CASES = 200
MEMBERSHIP_COLUMNS = {
    "split_seed",
    "fold",
    "role",
    "public_case_id",
    "source_filename",
    "label",
}
KNOWN_IMAGE_SUFFIXES = (".nii.gz", ".nrrd", ".nii", ".mha", ".mhd", ".npy")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_case_id(value: object) -> str:
    """Return the public case identifier encoded by a column or file path."""

    text = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    lowered = text.lower()
    for suffix in KNOWN_IMAGE_SUFFIXES:
        if lowered.endswith(suffix):
            return text[: -len(suffix)]
    return text


def parse_binary_label(value: object) -> int:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized == "B":
            return 0
        if normalized == "M":
            return 1
    label = int(value)
    if label not in {0, 1}:
        raise ValueError(f"Labels must be binary B/M or 0/1, got {value!r}.")
    return label


def discover_cache_csvs(cache_root: Path) -> list[Path]:
    candidates = [
        cache_root / "labels_train_cache.csv",
        cache_root / "labels_val_cache.csv",
        cache_root / "labels_test_cache.csv",
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != len(candidates):
        missing = [str(path) for path in candidates if not path.is_file()]
        raise FileNotFoundError(
            "TDSC cache root must contain all three cache CSVs; missing: "
            + ", ".join(missing)
        )
    return existing


def _resolve_npy_path(value: object, source_csv: Path) -> Path:
    raw = Path(str(value).strip()).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    candidates = ((source_csv.parent / raw).resolve(), (Path.cwd() / raw).resolve())
    existing = list(dict.fromkeys(path for path in candidates if path.is_file()))
    if len(existing) > 1:
        raise ValueError(
            f"Ambiguous relative cached-array path {raw!s} in {source_csv}: "
            + ", ".join(str(path) for path in existing)
        )
    return existing[0] if existing else candidates[0]


def _row_case_id(row: pd.Series) -> str:
    for column in ("public_case_id", "case_id"):
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            return normalize_case_id(row[column])
    for column in ("source_filename", "data_path"):
        if column in row and pd.notna(row[column]) and str(row[column]).strip():
            return normalize_case_id(row[column])
    raise ValueError(
        "Each cache row must identify its public case through public_case_id, "
        "case_id, source_filename, or data_path."
    )


def load_cache_index(
    cache_csvs: list[Path], *, check_files: bool = True
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Load cache CSVs and return one validated row per public case."""

    records: list[dict[str, object]] = []
    provenance: list[dict[str, str]] = []
    for csv_path in cache_csvs:
        csv_path = csv_path.resolve()
        frame = pd.read_csv(csv_path)
        required = {"label", "npy_path"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{csv_path} lacks required columns: {sorted(missing)}")
        provenance.append({"path": str(csv_path), "sha256": _sha256(csv_path)})
        for _, row in frame.iterrows():
            case = _row_case_id(row)
            if pd.isna(row["npy_path"]) or not str(row["npy_path"]).strip():
                raise ValueError(f"Cache row for {case} has an empty npy_path in {csv_path}.")
            npy_path = _resolve_npy_path(row["npy_path"], csv_path)
            if check_files and not npy_path.is_file():
                raise FileNotFoundError(f"Cached array not found for {case}: {npy_path}")
            records.append(
                {
                    "case_id": case,
                    "npy_path": str(npy_path),
                    "label": parse_binary_label(row["label"]),
                }
            )

    cache = pd.DataFrame(records)
    if cache.empty:
        raise ValueError("No cache rows were loaded.")
    conflicts = cache.groupby("case_id", sort=False).agg(
        labels=("label", "nunique"), paths=("npy_path", "nunique")
    )
    conflicting = conflicts[(conflicts.labels != 1) | (conflicts.paths != 1)]
    if not conflicting.empty:
        raise ValueError(
            "Conflicting duplicate cache rows for public cases: "
            + ", ".join(conflicting.index.astype(str).tolist()[:10])
        )
    return cache.drop_duplicates("case_id").set_index("case_id"), provenance


def validate_membership(membership: pd.DataFrame) -> pd.DataFrame:
    missing = MEMBERSHIP_COLUMNS.difference(membership.columns)
    if missing:
        raise ValueError(f"Membership CSV lacks required columns: {sorted(missing)}")

    result = membership.copy()
    if (
        result["public_case_id"].isna().any()
        or result["source_filename"].isna().any()
        or result["public_case_id"].astype(str).str.strip().eq("").any()
        or result["source_filename"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Membership case identifiers and source filenames must be nonempty.")
    result["public_case_id"] = result["public_case_id"].map(normalize_case_id)
    result["label"] = result["label"].map(parse_binary_label)
    result["split_seed"] = pd.to_numeric(result["split_seed"], errors="raise").astype(int)
    result["fold"] = pd.to_numeric(result["fold"], errors="raise").astype(int)
    if not set(result["label"].unique()).issubset({0, 1}):
        raise ValueError("Membership labels must be binary (0/1).")
    if set(result["split_seed"].unique()) != set(EXPECTED_SPLIT_SEEDS):
        raise ValueError(f"Expected split seeds {EXPECTED_SPLIT_SEEDS}.")
    if set(result["fold"].unique()) != set(EXPECTED_FOLDS):
        raise ValueError(f"Expected folds {EXPECTED_FOLDS}.")
    if set(result["role"].unique()) != set(EXPECTED_ROLE_COUNTS):
        raise ValueError(f"Expected roles {sorted(EXPECTED_ROLE_COUNTS)}.")

    canonical = result[["public_case_id", "label", "source_filename"]].drop_duplicates()
    conflicts = canonical.groupby("public_case_id").agg(
        labels=("label", "nunique"), filenames=("source_filename", "nunique")
    )
    if ((conflicts.labels != 1) | (conflicts.filenames != 1)).any():
        raise ValueError("Membership labels or source filenames are inconsistent across folds.")
    if len(conflicts) != EXPECTED_CASES:
        raise ValueError(f"Expected {EXPECTED_CASES} unique public cases, found {len(conflicts)}.")

    for (split_seed, fold), group in result.groupby(["split_seed", "fold"], sort=True):
        counts = group["role"].value_counts().to_dict()
        if counts != EXPECTED_ROLE_COUNTS:
            raise ValueError(
                f"split_seed={split_seed}, fold={fold} has role counts {counts}; "
                f"expected {EXPECTED_ROLE_COUNTS}."
            )
        if len(group) != EXPECTED_CASES or group["public_case_id"].nunique() != EXPECTED_CASES:
            raise ValueError(
                f"split_seed={split_seed}, fold={fold} does not contain each case exactly once."
            )

    for split_seed, group in result.groupby("split_seed", sort=True):
        outer = group[group.role.eq("outer_test")]
        counts = outer["public_case_id"].value_counts()
        if len(counts) != EXPECTED_CASES or not counts.eq(1).all():
            raise ValueError(
                f"Each public case must appear in outer_test exactly once for split_seed={split_seed}."
            )
    return result


def materialize_exact_splits(
    *,
    membership_csv: Path,
    cache_csvs: list[Path],
    output_dir: Path,
    check_files: bool = True,
) -> dict[str, object]:
    membership_csv = membership_csv.resolve()
    membership = validate_membership(pd.read_csv(membership_csv))
    cache, cache_provenance = load_cache_index(cache_csvs, check_files=check_files)

    member_cases = set(membership["public_case_id"])
    missing_cache = sorted(member_cases.difference(cache.index))
    extra_cache = sorted(set(cache.index).difference(member_cases))
    if missing_cache or extra_cache:
        raise ValueError(
            "Cache and tracked membership must contain the same 200 public cases; "
            f"missing={missing_cache[:10]}, extra={extra_cache[:10]}."
        )

    labels = membership[["public_case_id", "label"]].drop_duplicates().set_index("public_case_id")
    joined = labels.join(cache, how="left", rsuffix="_cache")
    label_mismatch = joined.index[joined.label.ne(joined.label_cache)].tolist()
    if label_mismatch:
        raise ValueError(
            "Cache labels disagree with tracked public labels for: "
            + ", ".join(label_mismatch[:10])
        )

    output_dir = output_dir.resolve()
    split_root = output_dir / "splits"
    written: list[dict[str, object]] = []
    for (split_seed, fold, role), group in membership.groupby(
        ["split_seed", "fold", "role"], sort=True
    ):
        destination = split_root / f"split_seed_{split_seed}" / f"fold_{fold}" / f"{role}.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = group[["public_case_id", "source_filename", "label"]].copy()
        rows = rows.rename(columns={"public_case_id": "case_id"})
        rows["npy_path"] = rows["case_id"].map(cache["npy_path"])
        rows = rows[["case_id", "source_filename", "npy_path", "label"]]
        rows.to_csv(destination, index=False)
        written.append(
            {
                "split_seed": int(split_seed),
                "fold": int(fold),
                "role": str(role),
                "rows": int(len(rows)),
                "path": str(destination),
                "sha256": _sha256(destination),
            }
        )

    manifest: dict[str, object] = {
        "protocol": "exact tracked TDSC 3x5 repeated stratified cross-validation membership",
        "membership_csv": str(membership_csv),
        "membership_sha256": _sha256(membership_csv),
        "cache_csvs": cache_provenance,
        "unique_public_cases": EXPECTED_CASES,
        "split_seeds": list(EXPECTED_SPLIT_SEEDS),
        "folds": list(EXPECTED_FOLDS),
        "role_counts_per_fold": EXPECTED_ROLE_COUNTS,
        "array_shape_contract": [1, 128, 128, 64],
        "path_resolution": "cache CSV npy_path resolved relative to its CSV; historical namespaces ignored",
        "written_csvs": written,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "materialization_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--membership-csv",
        type=Path,
        default=Path(
            "outputs/tables/aim_xyzsafe_revalidation_20260914/"
            "reproducibility/tdsc_public_membership.csv"
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--cache-csv",
        type=Path,
        action="append",
        help="Cache CSV to join (repeat for train/validation/test cache CSVs).",
    )
    source.add_argument(
        "--cache-root",
        type=Path,
        help="Root containing labels_{train,val,test}_cache.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/protocol/aim_tdsc_public_exact_repeated_cv"),
    )
    parser.add_argument(
        "--skip-array-existence-check",
        action="store_true",
        help="Materialize paths without checking files (intended only for protocol inspection).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_csvs = args.cache_csv or discover_cache_csvs(args.cache_root)
    manifest = materialize_exact_splits(
        membership_csv=args.membership_csv,
        cache_csvs=cache_csvs,
        output_dir=args.output_dir,
        check_files=not args.skip_array_existence_check,
    )
    print(
        f"materialized {len(manifest['written_csvs'])} split CSVs for "
        f"{manifest['unique_public_cases']} public cases under {args.output_dir}"
    )


if __name__ == "__main__":
    main()
