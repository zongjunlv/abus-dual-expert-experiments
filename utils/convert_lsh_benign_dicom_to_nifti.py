#!/usr/bin/env python3
"""Convert LSH benign DICOM volumes to compressed NIfTI files.

The original DICOM files are preserved. Converted files are written to a
separate directory so downstream preprocessing can opt into the harmonized
NIfTI source without mixing .dcm and .nii.gz files in the same case lookup.
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import SimpleITK as sitk


DEFAULT_INPUT_DIR = Path("dataset/LSH/benign/img")
DEFAULT_OUTPUT_DIR = Path("dataset/LSH/benign/img_nii_gz")
MANIFEST_FIELDS = [
    "case_id",
    "source_dicom",
    "target_nii_gz",
    "status",
    "dimension",
    "size_xyz",
    "spacing_xyz",
    "message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LSH benign DICOM volumes to .nii.gz files."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing LSH benign .dcm files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for converted .nii.gz files.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="CSV manifest path. Defaults to <output-dir>/manifest.csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N cases, useful for smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .nii.gz files.",
    )
    parser.add_argument(
        "--allow-non-3d",
        action="store_true",
        help="Allow non-3D DICOM objects instead of marking them as errors.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on the first conversion error.",
    )
    return parser.parse_args()


def iter_dicom_files(input_dir: Path, limit: Optional[int] = None) -> List[Path]:
    """Return sorted DICOM files from the input directory."""
    files = sorted(path for path in input_dir.glob("*.dcm") if path.is_file())
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive when provided")
        files = files[:limit]
    return files


def format_tuple(values: Iterable[object]) -> str:
    """Format tuple-like metadata without introducing locale-dependent output."""
    return ";".join(str(value) for value in values)


def target_for_source(source_path: Path, output_dir: Path) -> Path:
    """Build a .nii.gz target path for a source DICOM file."""
    return output_dir / f"{source_path.stem}.nii.gz"


def convert_one(
    source_path: Path,
    target_path: Path,
    overwrite: bool = False,
    allow_non_3d: bool = False,
) -> Dict[str, str]:
    """Convert one DICOM file and return a manifest row."""
    row = {
        "case_id": source_path.stem,
        "source_dicom": str(source_path),
        "target_nii_gz": str(target_path),
        "status": "pending",
        "dimension": "",
        "size_xyz": "",
        "spacing_xyz": "",
        "message": "",
    }

    if target_path.exists() and not overwrite:
        row["status"] = "skipped_existing"
        row["message"] = "target exists; pass --overwrite to regenerate"
        return row

    image = sitk.ReadImage(str(source_path))
    dimension = image.GetDimension()
    row["dimension"] = str(dimension)
    row["size_xyz"] = format_tuple(image.GetSize())
    row["spacing_xyz"] = format_tuple(image.GetSpacing())

    if dimension != 3 and not allow_non_3d:
        row["status"] = "error"
        row["message"] = f"expected 3D volume, got dimension={dimension}"
        return row

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.parent / f".{target_path.name}.tmp.nii.gz"
    if tmp_path.exists():
        tmp_path.unlink()

    sitk.WriteImage(image, str(tmp_path), True)
    tmp_path.replace(target_path)

    row["status"] = "converted"
    return row


def write_manifest(manifest_path: Path, rows: List[Dict[str, str]]) -> None:
    """Write conversion status rows to CSV."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def validate_paths(input_dir: Path, output_dir: Path) -> None:
    """Validate conversion paths before touching data."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Output directory must differ from input directory")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.csv"

    validate_paths(input_dir, output_dir)
    source_files = iter_dicom_files(input_dir, limit=args.limit)
    if not source_files:
        raise ValueError(f"No .dcm files found under {input_dir}")

    print(f"Input: {input_dir}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Cases: {len(source_files)}", flush=True)

    rows: List[Dict[str, str]] = []
    for index, source_path in enumerate(source_files, start=1):
        target_path = target_for_source(source_path, output_dir)
        try:
            row = convert_one(
                source_path=source_path,
                target_path=target_path,
                overwrite=args.overwrite,
                allow_non_3d=args.allow_non_3d,
            )
        except Exception as exc:  # noqa: BLE001 - keep batch conversion moving.
            row = {
                "case_id": source_path.stem,
                "source_dicom": str(source_path),
                "target_nii_gz": str(target_path),
                "status": "error",
                "dimension": "",
                "size_xyz": "",
                "spacing_xyz": "",
                "message": str(exc),
            }
            if args.fail_fast:
                rows.append(row)
                write_manifest(manifest_path, rows)
                raise

        rows.append(row)
        print(
            f"[{index}/{len(source_files)}] {source_path.name}: {row['status']}",
            flush=True,
        )

    write_manifest(manifest_path, rows)
    status_counts: Dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1

    print(f"Manifest: {manifest_path}", flush=True)
    print(f"Status: {status_counts}", flush=True)

    if status_counts.get("error", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
