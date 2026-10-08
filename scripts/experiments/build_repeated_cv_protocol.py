#!/usr/bin/env python3
"""Build a leakage-aware repeated cross-validation command plan.

This script only creates split CSVs, a command plan, and a manifest. It never
starts training. Split seeds control data partitions; train seeds control model
initialization and are deliberately represented as separate protocol fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)


DEFAULT_MODELS = [
    "dual_backbone_view_consistency_reliability",
    "r3d",
    "r2plus1d",
]
DEFAULT_SPLIT_SEEDS = [3407, 42, 1234]
DEFAULT_TRAIN_SEEDS = [3407, 42, 1234]
GROUP_COLUMN_CANDIDATES = (
    "patient_id",
    "patientid",
    "patient",
    "subject_id",
    "subjectid",
    "subject",
)
PATH_COLUMNS = ("npy_path", "data_path", "mask_path")


def parse_label(value: object) -> int:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized == "B":
            return 0
        if normalized == "M":
            return 1
    return int(value)


def detect_group_column(frame: pd.DataFrame, requested: str | None = None) -> str | None:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Requested group column is missing: {requested}")
        return requested
    normalized = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in GROUP_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolutize_paths(frame: pd.DataFrame, source_csv: Path) -> pd.DataFrame:
    result = frame.copy()
    for column in PATH_COLUMNS:
        if column not in result.columns:
            continue

        def resolve(value: object) -> object:
            if not isinstance(value, str) or not value.strip():
                return value
            candidate = Path(os.path.normpath(value.strip())).expanduser()
            if candidate.is_absolute():
                return str(candidate)
            candidates = (Path.cwd() / candidate, source_csv.parent / candidate)
            for item in candidates:
                if item.exists():
                    return str(item.resolve())
            return str(candidates[0].resolve())

        result[column] = result[column].map(resolve)
    return result


def _outer_splits(
    labels: np.ndarray,
    groups: np.ndarray | None,
    n_splits: int,
    split_seed: int,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    dummy = np.zeros(len(labels), dtype=np.uint8)
    if groups is None:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=split_seed,
        )
        return splitter.split(dummy, labels)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=split_seed,
    )
    return splitter.split(dummy, labels, groups)


def _selection_split(
    outer_train_indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < fraction < 0.5:
        raise ValueError("inner_val_fraction must be in (0, 0.5).")
    outer_labels = labels[outer_train_indices]
    if groups is None:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=fraction,
            random_state=seed,
        )
        train_local, val_local = next(
            splitter.split(np.zeros(len(outer_train_indices)), outer_labels)
        )
    else:
        outer_groups = groups[outer_train_indices]
        splitter = GroupShuffleSplit(
            n_splits=64,
            test_size=fraction,
            random_state=seed,
        )
        target_prevalence = float(np.mean(outer_labels))
        best: tuple[float, np.ndarray, np.ndarray] | None = None
        for train_candidate, val_candidate in splitter.split(
            np.zeros(len(outer_train_indices)), outer_labels, outer_groups
        ):
            if np.unique(outer_labels[train_candidate]).size < 2:
                continue
            if np.unique(outer_labels[val_candidate]).size < 2:
                continue
            size_error = abs(len(val_candidate) / len(outer_train_indices) - fraction)
            prevalence_error = abs(
                float(np.mean(outer_labels[val_candidate])) - target_prevalence
            )
            score = size_error + prevalence_error
            if best is None or score < best[0]:
                best = (score, train_candidate, val_candidate)
        if best is None:
            raise ValueError(
                "Could not create a group-disjoint inner selection split containing both classes."
            )
        _, train_local, val_local = best
    return outer_train_indices[train_local], outer_train_indices[val_local]


def _assert_disjoint(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    groups: np.ndarray | None,
) -> None:
    index_sets = [set(train_indices), set(val_indices), set(test_indices)]
    if any(index_sets[i] & index_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Train, selection-validation, and outer-test rows overlap.")
    if groups is None:
        return
    group_sets = [set(groups[index]) for index in (train_indices, val_indices, test_indices)]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Patient/group leakage detected across protocol splits.")


def _write_split(frame: pd.DataFrame, indices: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.iloc[indices].reset_index(drop=True).to_csv(path, index=False)


def _commands(
    *,
    python_executable: str,
    model: str,
    train_seed: int,
    train_csv: Path,
    selection_val_csv: Path,
    outer_test_csv: Path,
    checkpoint: Path,
    log_dir: Path,
    prediction_csv: Path,
    epochs: int,
    batch_size: int,
    accum_steps: int,
    num_workers: int,
    target_shape: str,
    spatial_aug_policy: str,
) -> tuple[str, str]:
    train = [
        python_executable,
        "scripts/train.py",
        "--train-csv",
        str(train_csv),
        "--val-csv",
        str(selection_val_csv),
        "--model",
        model,
        "--seed",
        str(train_seed),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--accum-steps",
        str(accum_steps),
        "--num-workers",
        str(num_workers),
        "--target-shape",
        target_shape,
        "--spatial-aug-policy",
        spatial_aug_policy,
        "--save-path",
        str(checkpoint),
        "--log-dir",
        str(log_dir),
        "--swanlab-mode",
        "disabled",
    ]
    evaluate = [
        python_executable,
        "scripts/test.py",
        "--test-csv",
        str(outer_test_csv),
        "--val-csv",
        str(selection_val_csv),
        "--model",
        model,
        "--checkpoint",
        str(checkpoint),
        "--batch-size",
        "1",
        "--num-workers",
        str(num_workers),
        "--target-shape",
        target_shape,
        "--export-pred-csv",
        str(prediction_csv),
        "--swanlab-mode",
        "disabled",
    ]
    return shlex.join(train), shlex.join(evaluate)


def _source_export_command(
    *,
    python_executable: str,
    data_csv: Path,
    checkpoint: Path,
    output_csv: Path,
    num_workers: int,
    target_shape: str,
) -> str:
    spatial_shape = [part.strip() for part in target_shape.replace("x", ",").split(",")]
    if len(spatial_shape) != 3 or any(not part for part in spatial_shape):
        raise ValueError("target_shape must contain three dimensions for source export.")
    command = [
        python_executable,
        "scripts/analysis/export_db_vcrf_sources.py",
        "--data-csv",
        str(data_csv),
        "--checkpoint",
        str(checkpoint),
        "--output-csv",
        str(output_csv),
        "--target-shape",
        *spatial_shape,
        "--batch-size",
        "1",
        "--num-workers",
        str(num_workers),
    ]
    return shlex.join(command)


def build_repeated_cv_protocol(
    *,
    input_csv: str | Path,
    output_dir: str | Path,
    models: list[str],
    split_seeds: list[int],
    train_seeds: list[int],
    n_splits: int = 5,
    group_column: str | None = None,
    inner_val_fraction: float = 0.15,
    checkpoint_root: str | Path = "outputs/checkpoints/aim_revision_repeated_cv",
    result_root: str | Path = "outputs/results/aim_revision_repeated_cv",
    log_root: str | Path = "outputs/logs/aim_revision_repeated_cv",
    python_executable: str = "python",
    epochs: int = 80,
    batch_size: int = 1,
    accum_steps: int = 4,
    num_workers: int = 8,
    target_shape: str = "128,128,64",
    spatial_aug_policy: str = "full",
) -> tuple[pd.DataFrame, dict[str, object]]:
    source_path = Path(input_csv).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source_path}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if not models or not split_seeds or not train_seeds:
        raise ValueError("models, split_seeds, and train_seeds must be non-empty.")

    raw_frame = pd.read_csv(source_path)
    if "label" not in raw_frame.columns:
        raise ValueError("Input CSV must contain a label column.")
    frame = _absolutize_paths(raw_frame, source_path)
    labels = raw_frame["label"].map(parse_label).to_numpy(dtype=np.int64)
    resolved_group_column = detect_group_column(raw_frame, group_column)
    groups = (
        raw_frame[resolved_group_column].astype(str).to_numpy()
        if resolved_group_column is not None
        else None
    )

    output_path = Path(output_dir).resolve()
    checkpoint_root = Path(checkpoint_root)
    result_root = Path(result_root)
    log_root = Path(log_root)
    rows: list[dict[str, object]] = []
    fold_manifest: list[dict[str, object]] = []

    for split_seed in split_seeds:
        split_iter = _outer_splits(labels, groups, n_splits, int(split_seed))
        for fold, (outer_train, outer_test) in enumerate(split_iter, start=1):
            train_indices, selection_val_indices = _selection_split(
                np.asarray(outer_train),
                labels,
                groups,
                inner_val_fraction,
                seed=int(split_seed) * 1000 + fold,
            )
            outer_test_indices = np.asarray(outer_test)
            _assert_disjoint(
                train_indices,
                selection_val_indices,
                outer_test_indices,
                groups,
            )

            split_dir = output_path / "splits" / f"split_seed_{split_seed}" / f"fold_{fold}"
            train_csv = split_dir / "train.csv"
            selection_val_csv = split_dir / "selection_val.csv"
            outer_test_csv = split_dir / "outer_test.csv"
            _write_split(frame, train_indices, train_csv)
            _write_split(frame, selection_val_indices, selection_val_csv)
            _write_split(frame, outer_test_indices, outer_test_csv)

            fold_record: dict[str, object] = {
                "split_seed": int(split_seed),
                "fold": fold,
                "train_n": int(len(train_indices)),
                "selection_val_n": int(len(selection_val_indices)),
                "outer_test_n": int(len(outer_test_indices)),
                "outer_test_positive_n": int(labels[outer_test_indices].sum()),
                "train_csv": str(train_csv),
                "selection_val_csv": str(selection_val_csv),
                "outer_test_csv": str(outer_test_csv),
            }
            if resolved_group_column is not None:
                fold_record.update(
                    {
                        "train_group_n": int(np.unique(groups[train_indices]).size),
                        "selection_val_group_n": int(
                            np.unique(groups[selection_val_indices]).size
                        ),
                        "outer_test_group_n": int(
                            np.unique(groups[outer_test_indices]).size
                        ),
                    }
                )
            fold_manifest.append(fold_record)

            for model in models:
                for train_seed in train_seeds:
                    run_key = (
                        f"{model}/split_seed_{split_seed}/fold_{fold}/"
                        f"train_seed_{train_seed}"
                    )
                    checkpoint = Path(checkpoint_root) / run_key / "best_model.pth"
                    log_dir = Path(log_root) / run_key
                    prediction_csv = Path(result_root) / run_key / "outer_test_predictions.csv"
                    train_command, eval_command = _commands(
                        python_executable=python_executable,
                        model=model,
                        train_seed=int(train_seed),
                        train_csv=train_csv,
                        selection_val_csv=selection_val_csv,
                        outer_test_csv=outer_test_csv,
                        checkpoint=checkpoint,
                        log_dir=log_dir,
                        prediction_csv=prediction_csv,
                        epochs=epochs,
                        batch_size=batch_size,
                        accum_steps=accum_steps,
                        num_workers=num_workers,
                        target_shape=target_shape,
                        spatial_aug_policy=spatial_aug_policy,
                    )
                    source_export_command = ""
                    source_export_selection_command = ""
                    if model == "dual_backbone_view_consistency_reliability":
                        source_export_command = _source_export_command(
                            python_executable=python_executable,
                            data_csv=outer_test_csv,
                            checkpoint=checkpoint,
                            output_csv=Path(result_root)
                            / run_key
                            / "outer_test_source_predictions.csv",
                            num_workers=num_workers,
                            target_shape=target_shape,
                        )
                        source_export_selection_command = _source_export_command(
                            python_executable=python_executable,
                            data_csv=selection_val_csv,
                            checkpoint=checkpoint,
                            output_csv=Path(result_root)
                            / run_key
                            / "selection_val_source_predictions.csv",
                            num_workers=num_workers,
                            target_shape=target_shape,
                        )
                    rows.append(
                        {
                            "run_key": run_key,
                            "model": model,
                            "split_seed": int(split_seed),
                            "train_seed": int(train_seed),
                            "fold": fold,
                            "group_column": resolved_group_column or "",
                            "train_csv": str(train_csv),
                            "selection_val_csv": str(selection_val_csv),
                            "outer_test_csv": str(outer_test_csv),
                            "checkpoint": str(checkpoint),
                            "prediction_csv": str(prediction_csv),
                            "train_command": train_command,
                            "eval_command": eval_command,
                            "source_export_command": source_export_command,
                            "source_export_selection_command": source_export_selection_command,
                        }
                    )

    protocol = pd.DataFrame(rows)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "input_csv": str(source_path),
        "input_csv_sha256": _sha256(source_path),
        "n_rows": int(len(frame)),
        "class_counts": {
            str(key): int(value)
            for key, value in raw_frame["label"].map(parse_label).value_counts().sort_index().items()
        },
        "group_column": resolved_group_column,
        "group_aware": resolved_group_column is not None,
        "n_groups": int(np.unique(groups).size) if groups is not None else None,
        "n_splits": int(n_splits),
        "split_seeds": [int(seed) for seed in split_seeds],
        "train_seeds": [int(seed) for seed in train_seeds],
        "models": list(models),
        "inner_val_fraction": float(inner_val_fraction),
        "planned_runs": int(len(protocol)),
        "planned_training_runs": int(len(protocol)),
        "planned_source_exports": int(
            2
            * len(
                protocol[
                    protocol["model"]
                    == "dual_backbone_view_consistency_reliability"
                ]
            )
        ),
        "folds": fold_manifest,
        "note": (
            "Outer folds are evaluation-only. A disjoint inner selection-validation "
            "split is used for checkpoint and threshold selection. Source-export "
            "commands reuse DB-VCRF checkpoints and do not add training runs; fixed "
            "fusion controls are post-hoc analyses and require no separate training."
        ),
    }
    return protocol, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/protocol/aim_revision_repeated_cv")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--group-column", default=None)
    parser.add_argument("--inner-val-fraction", type=float, default=0.15)
    parser.add_argument("--checkpoint-root", default="outputs/checkpoints/aim_revision_repeated_cv")
    parser.add_argument("--result-root", default="outputs/results/aim_revision_repeated_cv")
    parser.add_argument("--log-root", default="outputs/logs/aim_revision_repeated_cv")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--target-shape", default="128,128,64")
    parser.add_argument(
        "--spatial-aug-policy",
        choices=("full", "hw_only", "slice_xy", "xy_flip_only", "none"),
        default="full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol, manifest = build_repeated_cv_protocol(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        models=args.models,
        split_seeds=args.split_seeds,
        train_seeds=args.train_seeds,
        n_splits=args.num_folds,
        group_column=args.group_column,
        inner_val_fraction=args.inner_val_fraction,
        checkpoint_root=args.checkpoint_root,
        result_root=args.result_root,
        log_root=args.log_root,
        python_executable=args.python_executable,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        num_workers=args.num_workers,
        target_shape=args.target_shape,
        spatial_aug_policy=args.spatial_aug_policy,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "command_plan.csv"
    manifest_path = output_dir / "manifest.json"
    protocol.to_csv(protocol_path, index=False)
    manifest["command_plan_csv"] = str(protocol_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Command plan: {protocol_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Planned runs: {len(protocol)}")


if __name__ == "__main__":
    main()
