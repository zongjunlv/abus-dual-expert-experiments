#!/usr/bin/env python3
"""Prepare LSH data in the same TDSC-like layout used by the cache builder."""
import argparse
import csv
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process LSH dataset to TDSC-like structure")
    parser.add_argument("--benign-dir", default="dataset/LSH/benign", help="Benign data directory")
    parser.add_argument("--malignant-dir", default="dataset/LSH/malignant", help="Malignant data directory")
    parser.add_argument("--output-dir", default="dataset/LSH/processed", help="Output directory")
    parser.add_argument("--ratios", default="7,1,2", help="Split ratios (train,val,test)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "symlink", "move"),
        default="hardlink",
        help="How to place source files into the TDSC-like output tree.",
    )
    parser.add_argument("--apply", action="store_true", help="Execute file operations")
    parser.add_argument(
        "--benign-flat-layout",
        choices=("auto", "normal", "swapped"),
        default="auto",
        help="How to interpret benign flat img/roi directories. Use swapped when img contains ROI masks.",
    )
    parser.add_argument(
        "--benign-flat-image-subdir",
        default="auto",
        help=(
            "Benign flat image subdirectory. Use auto to prefer img_nii_gz "
            "when present, otherwise img."
        ),
    )
    parser.add_argument(
        "--benign-flat-mask-subdir",
        default="roi",
        help="Benign flat ROI/mask subdirectory.",
    )
    parser.add_argument(
        "--malignant-flat-layout",
        choices=("auto", "normal", "swapped"),
        default="auto",
        help="How to interpret malignant flat img/roi directories. Use swapped when img contains ROI masks.",
    )
    return parser.parse_args()


def parse_ratios(ratio_string: str) -> Tuple[int, int, int]:
    parts = [item.strip() for item in ratio_string.split(",")]
    if len(parts) != 3:
        raise ValueError("Ratios must be train,val,test (e.g., 7,1,2)")
    values = tuple(int(item) for item in parts)
    if any(item <= 0 for item in values):
        raise ValueError("All ratios must be positive")
    return values


def allocate_counts(total: int, ratios: Sequence[int]) -> List[int]:
    """Allocate total counts based on ratios."""
    ratio_sum = sum(ratios)
    raw_values = [total * item / ratio_sum for item in ratios]
    counts = [math.floor(v) for v in raw_values]
    remainder = total - sum(counts)
    ranked = sorted(
        range(len(ratios)),
        key=lambda i: (raw_values[i] - counts[i], ratios[i]),
        reverse=True,
    )
    for i in ranked[:remainder]:
        counts[i] += 1
    return counts


def stratified_split(
    samples: List[Dict[str, str]],
    ratios: Tuple[int, int, int],
    seed: int,
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, Dict[str, int]]]:
    """Stratified split by label."""
    split_names = ("train", "val", "test")
    target_counts = allocate_counts(len(samples), ratios)

    # Group by label
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["label"]].append(sample)

    # Calculate allocations per label
    label_allocations = {}
    for label, label_samples in grouped.items():
        count = len(label_samples)
        ratio_sum = sum(ratios)
        raw_values = [count * item / ratio_sum for item in ratios]
        base_values = [math.floor(v) for v in raw_values]
        remainder = count - sum(base_values)
        ranked = sorted(
            range(len(ratios)),
            key=lambda i: (raw_values[i] - base_values[i], ratios[i]),
            reverse=True,
        )
        for i in ranked[:remainder]:
            base_values[i] += 1
        label_allocations[label] = base_values

    # Rebalance to match target totals
    split_totals = [0] * 3
    for alloc in label_allocations.values():
        for i, count in enumerate(alloc):
            split_totals[i] += count

    while split_totals != list(target_counts):
        surplus = [i for i in range(3) if split_totals[i] > target_counts[i]]
        deficit = [i for i in range(3) if split_totals[i] < target_counts[i]]
        if not surplus or not deficit:
            raise RuntimeError("Cannot rebalance allocations")

        best_move = None
        best_cost = None

        for label, alloc in label_allocations.items():
            for src in surplus:
                if alloc[src] <= 0:
                    continue
                for dst in deficit:
                    new_alloc = list(alloc)
                    new_alloc[src] -= 1
                    new_alloc[dst] += 1

                    # Calculate cost as deviation from ideal
                    count = len(grouped[label])
                    ratio_sum = sum(ratios)
                    ideal = [count * item / ratio_sum for item in ratios]
                    cost = sum(abs(new_alloc[i] - ideal[i]) for i in range(3))

                    move_signature = (cost, src, dst, label)
                    if best_move is None or move_signature < best_cost:
                        best_move = (label, src, dst)
                        best_cost = move_signature

        if best_move is None:
            raise RuntimeError("No valid move found")

        label, src, dst = best_move
        label_allocations[label][src] -= 1
        label_allocations[label][dst] += 1
        split_totals[src] -= 1
        split_totals[dst] += 1

    # Shuffle and allocate
    rng = random.Random(seed)
    split_rows = {name: [] for name in split_names}
    split_labels = {name: defaultdict(int) for name in split_names}

    for label in sorted(grouped.keys()):
        samples_label = list(grouped[label])
        rng.shuffle(samples_label)
        alloc = label_allocations[label]
        start = 0
        for split_idx, split_name in enumerate(split_names):
            count = alloc[split_idx]
            chunk = samples_label[start:start + count]
            split_rows[split_name].extend(chunk)
            split_labels[split_name][label] += len(chunk)
            start += count

        if start != len(samples_label):
            raise RuntimeError(f"Not all samples for label {label} were allocated")

    # Sort by case_id
    for split_name in split_names:
        split_rows[split_name].sort(
            key=lambda r: int(r["case_id"]) if r["case_id"].isdigit() else r["case_id"]
        )

    return split_rows, split_labels


def strip_medical_suffix(path: Path) -> str:
    """Return a case id after removing .nii, .nii.gz, or common medical suffixes."""
    name = path.name
    for suffix in (".nii.gz", ".nii", ".nrrd", ".dcm"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def build_file_lookup(files: Iterable[Path]) -> Dict[str, Path]:
    """Build a unique case-id lookup from medical image files."""
    lookup: Dict[str, Path] = {}
    duplicates = []
    for file_path in sorted(files):
        if not file_path.is_file():
            continue
        case_id = strip_medical_suffix(file_path)
        if case_id in lookup:
            duplicates.append(case_id)
            continue
        lookup[case_id] = file_path

    if duplicates:
        duplicate_list = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"Duplicate LSH case ids detected: {duplicate_list}")
    return lookup


def list_medical_files(directory: Path) -> List[Path]:
    """List NIfTI files regardless of compression."""
    return sorted(path for path in directory.glob("*.nii*") if path.is_file())


def list_image_files(directory: Path) -> List[Path]:
    """List supported image volumes for LSH flat img directories."""
    patterns = ("*.nii", "*.nii.gz", "*.nrrd", "*.dcm")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(path for path in directory.glob(pattern) if path.is_file())
    return sorted(files)


def median_file_size(paths: Sequence[Path]) -> float:
    """Return the median file size in bytes for existing files."""
    sizes = sorted(path.stat().st_size for path in paths if path.exists())
    if not sizes:
        return 0.0
    mid = len(sizes) // 2
    if len(sizes) % 2:
        return float(sizes[mid])
    return float((sizes[mid - 1] + sizes[mid]) / 2)


def infer_flat_pair_swap(img_dir: Path, roi_dir: Path, layout: str) -> bool:
    """Infer whether flat img/roi directories are reversed.

    Some LSH benign folders store tiny binary ROI masks under img/ and full
    image volumes under roi/. Auto mode flags that pattern without reading the
    full medical volumes.
    """
    if layout == "normal":
        return False
    if layout == "swapped":
        return True

    img_files = list_medical_files(img_dir)
    roi_files = list_medical_files(roi_dir)
    if not img_files or not roi_files:
        return False

    img_median = median_file_size(img_files)
    roi_median = median_file_size(roi_files)
    tiny_image_threshold = 5 * 1024 * 1024
    return img_median < tiny_image_threshold and roi_median > img_median * 20


def scan_flat_pair_dirs(
    cohort_dir: Path,
    label: str,
    source_dir: str,
    flat_layout: str = "auto",
    image_subdir: str = "img",
    mask_subdir: str = "roi",
) -> List[Dict[str, str]]:
    """Scan a flat LSH directory with paired img/ and roi/ files."""
    img_dir = cohort_dir / image_subdir
    roi_dir = cohort_dir / mask_subdir
    if not img_dir.is_dir() or not roi_dir.is_dir():
        return []

    swap_dirs = infer_flat_pair_swap(img_dir, roi_dir, flat_layout)
    image_dir = roi_dir if swap_dirs else img_dir
    mask_dir = img_dir if swap_dirs else roi_dir
    if swap_dirs:
        print(f"  Warning: {source_dir} flat img/roi layout appears swapped; using roi/ as image and img/ as mask")

    image_files = build_file_lookup(list_image_files(image_dir))
    mask_files = build_file_lookup(list_medical_files(mask_dir))
    matched_ids = sorted(set(image_files) & set(mask_files))

    missing_masks = sorted(set(image_files) - set(mask_files))
    missing_images = sorted(set(mask_files) - set(image_files))
    if missing_masks or missing_images:
        print(
            f"  Warning: {source_dir} unmatched files: "
            f"{len(missing_masks)} images without mask, {len(missing_images)} masks without image"
        )

    return [
        {
            "case_id": case_id,
            "label": label,
            "data_path": str(image_files[case_id]),
            "mask_path": str(mask_files[case_id]),
            "source_dir": source_dir,
        }
        for case_id in matched_ids
    ]


def resolve_benign_image_subdir(benign_dir: Path, image_subdir: str = "auto") -> str:
    """Resolve the benign image subdirectory for flat LSH preprocessing."""
    if image_subdir != "auto":
        return image_subdir

    converted_dir = benign_dir / "img_nii_gz"
    if converted_dir.is_dir() and list_image_files(converted_dir):
        return "img_nii_gz"
    return "img"


def scan_benign(
    benign_dir: Path,
    flat_layout: str = "auto",
    image_subdir: str = "auto",
    mask_subdir: str = "roi",
) -> List[Dict[str, str]]:
    """Scan benign directory for patient data."""
    resolved_image_subdir = resolve_benign_image_subdir(benign_dir, image_subdir)
    flat_samples = scan_flat_pair_dirs(
        benign_dir,
        label="B",
        source_dir=f"benign/{resolved_image_subdir}",
        flat_layout=flat_layout,
        image_subdir=resolved_image_subdir,
        mask_subdir=mask_subdir,
    )
    if flat_samples:
        return flat_samples

    samples = []
    for patient_dir in benign_dir.iterdir():
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name
        nii_file = None
        nii_gz_file = None

        for file in patient_dir.iterdir():
            if file.suffix == ".nii" and not file.name.endswith(".gz"):
                nii_file = file
            elif file.suffixes == [".nii", ".gz"]:
                nii_gz_file = file

        if nii_file and nii_gz_file:
            samples.append({
                "case_id": patient_id,
                "label": "B",
                "data_path": str(nii_gz_file),  # ABUS data (nii.gz)
                "mask_path": str(nii_file),     # ROI (nii)
                "source_dir": "benign",
            })

    return samples


def scan_malignant(malignant_dir: Path, flat_layout: str = "auto") -> List[Dict[str, str]]:
    """Scan malignant directory for patient data."""
    original_roi_dir = malignant_dir / "original_roi_nii_gz"
    if (malignant_dir / "img").is_dir() and original_roi_dir.is_dir():
        image_files = build_file_lookup(list_image_files(malignant_dir / "img"))
        mask_files = build_file_lookup(list_medical_files(original_roi_dir))
        matched_ids = sorted(set(image_files) & set(mask_files))
        missing_masks = sorted(set(image_files) - set(mask_files))
        missing_images = sorted(set(mask_files) - set(image_files))
        if missing_masks or missing_images:
            print(
                "  Warning: malignant flat img/original_roi_nii_gz unmatched files: "
                f"{len(missing_masks)} images without mask, {len(missing_images)} masks without image"
            )
        if matched_ids:
            return [
                {
                    "case_id": case_id,
                    "label": "M",
                    "data_path": str(image_files[case_id]),
                    "mask_path": str(mask_files[case_id]),
                    "source_dir": "malignant",
                }
                for case_id in matched_ids
            ]

    flat_samples = scan_flat_pair_dirs(malignant_dir, label="M", source_dir="malignant", flat_layout=flat_layout)
    if flat_samples:
        return flat_samples

    samples = []

    # Year directories
    year_dirs = ["ABUS2020", "ABUS2021", "ABUS2022"]

    for year_dir_name in year_dirs:
        year_dir = malignant_dir / year_dir_name
        if not year_dir.exists():
            continue

        img_dir = year_dir / "img"
        roi_dir = year_dir / "roi"

        if not img_dir.exists() or not roi_dir.exists():
            continue

        # Build ROI lookup (patient_id -> roi_file)
        roi_files = {}
        for roi_file in roi_dir.glob("*.nii*"):  # Match both .nii and .nii.gz
            # Remove all .nii and .gz extensions to get patient_id
            patient_id = roi_file.name
            patient_id = patient_id.replace('.nii', '').replace('.gz', '')
            roi_files[patient_id] = roi_file

        # Build img lookup (patient_id -> img_file)
        img_files = {}
        for img_path in img_dir.iterdir():
            if img_path.is_dir():
                patient_id = img_path.name
                subdir_files = [item for item in img_path.iterdir() if item.is_file()]
                if subdir_files:
                    img_files[patient_id] = sorted(subdir_files)[0]
            elif img_path.is_file():
                patient_id = strip_medical_suffix(img_path)
                img_files[patient_id] = img_path

        # Find matches
        matched_ids = set(img_files.keys()) & set(roi_files.keys())

        for patient_id in matched_ids:
            samples.append({
                "case_id": patient_id,
                "label": "M",
                "data_path": str(img_files[patient_id]),
                "mask_path": str(roi_files[patient_id]),
                "source_dir": f"malignant/{year_dir_name}",
            })

    return samples


def plan_moves(
    split_rows: Dict[str, List[Dict[str, str]]],
    output_dir: Path,
) -> Tuple[List[Tuple[Path, Path]], Dict[str, List[Dict[str, str]]]]:
    """Plan file moves to TDSC-like structure."""
    move_pairs = []
    updated_rows = {split: [] for split in split_rows}

    structure = {
        "train": {
            "data_dir": "train/imagesTr_origin",
            "mask_dir": "train/labelsTr_origin",
        },
        "val": {
            "data_dir": "val/imagesVal_origin",
            "mask_dir": "val/labelsVal_origin",
        },
        "test": {
            "data_dir": "test/imagesTs_origin",
            "mask_dir": "test/labelsTs_origin",
        },
    }

    # Track used names to avoid collisions
    used_names = set()

    for split_name, rows in split_rows.items():
        data_subdir = structure[split_name]["data_dir"]
        mask_subdir = structure[split_name]["mask_dir"]

        for row in rows:
            # Create new paths with collision avoidance
            src_data = Path(row["data_path"])
            src_mask = Path(row["mask_path"])
            case_id = row["case_id"]

            # Generate unique names
            data_name = src_data.name
            mask_name = src_mask.name

            # Handle name collisions - add suffix before extensions
            suffix = ""
            while data_name in used_names or mask_name in used_names:
                suffix = f"_{case_id}"

                # Properly handle file extensions
                def add_suffix(name, suffix):
                    # Find the first dot for extension
                    if '.' in name:
                        parts = name.split('.', 1)
                        return f"{parts[0]}{suffix}.{parts[1]}"
                    return f"{name}{suffix}"

                data_name = add_suffix(src_data.name, suffix)
                mask_name = add_suffix(src_mask.name, suffix)

            dst_data = output_dir / data_subdir / data_name
            dst_mask = output_dir / mask_subdir / mask_name

            updated_row = dict(row)
            # Use absolute path format like TDSC
            updated_row["data_path"] = str(dst_data)
            updated_row["mask_path"] = str(dst_mask)

            move_pairs.append((src_data, dst_data))
            move_pairs.append((src_mask, dst_mask))

            used_names.add(data_name)
            used_names.add(mask_name)

            updated_rows[split_name].append(updated_row)

    return move_pairs, updated_rows


def validate_moves(move_pairs: List[Tuple[Path, Path]], link_mode: str) -> None:
    """Validate move plan."""
    seen_sources = set()
    seen_destinations = set()

    for src, dst in move_pairs:
        if not src.exists():
            raise FileNotFoundError(f"Source not found: {src}")
        if src == dst:
            continue
        if dst.exists():
            if link_mode in {"hardlink", "symlink"} and dst.samefile(src):
                continue
            raise FileExistsError(f"Destination exists: {dst}")
        if src in seen_sources:
            raise ValueError(f"Duplicate source: {src}")
        if dst in seen_destinations:
            raise ValueError(f"Duplicate destination: {dst}")
        seen_sources.add(src)
        seen_destinations.add(dst)


def execute_moves(move_pairs: List[Tuple[Path, Path]], link_mode: str) -> None:
    """Execute file placement into the TDSC-like tree."""
    for src, dst in move_pairs:
        if src == dst:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and link_mode in {"hardlink", "symlink"} and dst.samefile(src):
            continue
        if link_mode == "hardlink":
            os.link(src, dst)
        elif link_mode == "copy":
            shutil.copy2(src, dst)
        elif link_mode == "symlink":
            os.symlink(src.resolve(), dst)
        elif link_mode == "move":
            shutil.move(str(src), str(dst))
        else:
            raise ValueError(f"Unsupported link mode: {link_mode}")


def write_csv(output_path: Path, rows: List[Dict[str, str]]) -> None:
    """Write CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "label", "data_path", "mask_path"]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            # Convert numeric labels to M/B format
            label = row["label"]
            if label == "0":
                label = "B"  # Benign
            elif label == "1":
                label = "M"  # Malignant

            writer.writerow({
                "case_id": row.get("case_id", ""),
                "label": label,
                "data_path": row.get("data_path", ""),
                "mask_path": row.get("mask_path", ""),
            })


def main():
    args = parse_args()
    ratios = parse_ratios(args.ratios)

    # Scan data
    print("Scanning benign data...")
    benign_samples = scan_benign(
        Path(args.benign_dir),
        flat_layout=args.benign_flat_layout,
        image_subdir=args.benign_flat_image_subdir,
        mask_subdir=args.benign_flat_mask_subdir,
    )
    print(f"  Found {len(benign_samples)} benign samples")

    print("Scanning malignant data...")
    malignant_samples = scan_malignant(Path(args.malignant_dir), flat_layout=args.malignant_flat_layout)
    print(f"  Found {len(malignant_samples)} malignant samples")

    all_samples = benign_samples + malignant_samples
    print(f"Total: {len(all_samples)} samples")

    if not all_samples:
        raise ValueError("No samples found")

    # Split data
    print("\nSplitting data...")
    split_rows, split_labels = stratified_split(all_samples, ratios, args.seed)

    for split_name in ("train", "val", "test"):
        label_dist = dict(sorted(split_labels[split_name].items()))
        print(f"  {split_name}: {len(split_rows[split_name])} samples, labels={label_dist}")

    # Plan moves
    output_dir = Path(args.output_dir)
    print("\nPlanning file moves...")
    move_pairs, updated_rows = plan_moves(split_rows, output_dir)
    print(f"  Planned {len(move_pairs)} file placements via {args.link_mode}")

    # Validate
    print("\nValidating moves...")
    validate_moves(move_pairs, link_mode=args.link_mode)

    if not args.apply:
        print("\n[DRY-RUN] Run with --apply to execute")
        return

    # Execute
    print("\nExecuting file placements...")
    execute_moves(move_pairs, link_mode=args.link_mode)

    # Write CSVs
    print("\nWriting CSV files...")
    for split_name in ("train", "val", "test"):
        csv_path = output_dir / f"labels_{split_name}.csv"
        write_csv(csv_path, updated_rows[split_name])
        print(f"  Wrote {csv_path}")

    trainval_rows = updated_rows["train"] + updated_rows["val"]
    trainval_path = output_dir / "labels_trainval.csv"
    write_csv(trainval_path, trainval_rows)
    print(f"  Wrote {trainval_path}")

    print("\n✓ Processing complete!")


if __name__ == "__main__":
    main()
