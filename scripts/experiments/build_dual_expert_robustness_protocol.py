#!/usr/bin/env python3
"""Build locked robustness plans for the R3D + ConvNeXt-MIL pair.

The builder supports repeated 70/10/20 holdouts and repeated outer K-fold CV.
Both experts always receive identical case partitions, while their established
model-specific training recipes remain fixed.  This script only writes CSV and
JSON artifacts; it never starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


SPLIT_SEEDS = (3407, 42, 1234)
PATH_COLUMNS = ("npy_path", "data_path", "mask_path", "mask_npy_path")
MODEL_RECIPES = {
    "r3d": {
        "train_seed": 1234,
        "batch_size": 28,
        "backbone_lr_scale": 0.05,
        "weight_decay": None,
        "spatial_aug_policy": "hw_only",
    },
    "convnext_slice_mil": {
        "train_seed": 1234,
        "batch_size": 36,
        "backbone_lr_scale": 0.1,
        "weight_decay": 0.05,
        "spatial_aug_policy": "slice_xy",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_label(value: object) -> int:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized == "B":
            return 0
        if normalized == "M":
            return 1
    label = int(value)
    if label not in (0, 1):
        raise ValueError(f"Labels must be binary, got {value!r}")
    return label


def _load_inputs(paths: list[str | Path]) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    frames = []
    sources = []
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        if "label" not in frame:
            raise ValueError(f"{path} lacks a label column")
        frame = frame.copy()
        for column in PATH_COLUMNS:
            if column not in frame:
                continue

            def resolve(item: object) -> object:
                if not isinstance(item, str) or not item.strip():
                    return item
                candidate = Path(os.path.normpath(item.strip())).expanduser()
                if candidate.is_absolute():
                    return str(candidate)
                cwd_candidate = Path.cwd() / candidate
                source_candidate = path.parent / candidate
                selected = cwd_candidate if cwd_candidate.exists() else source_candidate
                return str(selected.resolve())

            frame[column] = frame[column].map(resolve)
        frames.append(frame)
        sources.append({"path": str(path), "sha256": _sha256(path), "rows": str(len(frame))})
    combined = pd.concat(frames, ignore_index=True)
    identity_column = "npy_path" if "npy_path" in combined else "data_path"
    if combined[identity_column].astype(str).duplicated().any():
        raise ValueError(f"Duplicate cases found by {identity_column}")
    return combined, sources


def _write_split(frame: pd.DataFrame, indices: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.iloc[indices].reset_index(drop=True).to_csv(path, index=False)


def _train_command(
    model: str,
    train_csv: Path,
    val_csv: Path,
    checkpoint: Path,
    log_dir: Path,
    python_executable: str,
) -> str:
    recipe = MODEL_RECIPES[model]
    command = [
        python_executable,
        "scripts/train.py",
        "--train-csv", str(train_csv),
        "--val-csv", str(val_csv),
        "--model", model,
        "--seed", str(recipe["train_seed"]),
        "--epochs", "80",
        "--batch-size", str(recipe["batch_size"]),
        "--accum-steps", "1",
        "--num-workers", "0",
        "--lr", "1e-4",
        "--backbone-lr-scale", str(recipe["backbone_lr_scale"]),
        "--warmup-epochs", "5",
        "--patience", "10",
        "--freeze-backbone-epochs", "5",
        "--monitor-start-epoch", "1",
        "--target-shape", "128,128,64",
        "--spatial-aug-policy", str(recipe["spatial_aug_policy"]),
        "--disable-selection-threshold-search",
        "--score-weight-auc", "1.0",
        "--score-weight-mcc", "0.0",
        "--score-weight-bias", "0.0",
        "--score-min-delta", "0.0",
        "--min-sensitivity", "0.0",
        "--min-specificity", "0.0",
        "--max-bias-gap", "1.0",
        "--save-path", str(checkpoint),
        "--log-dir", str(log_dir),
        "--swanlab-mode", "disabled",
    ]
    if recipe["weight_decay"] is not None:
        command.extend(["--weight-decay", str(recipe["weight_decay"])])
    return shlex.join(command)


def _eval_command(
    model: str,
    test_csv: Path,
    checkpoint: Path,
    prediction_csv: Path,
    python_executable: str,
) -> str:
    return shlex.join([
        python_executable,
        "scripts/test.py",
        "--test-csv", str(test_csv),
        "--model", model,
        "--checkpoint", str(checkpoint),
        "--batch-size", "1",
        "--num-workers", "0",
        "--target-shape", "128,128,64",
        "--export-pred-csv", str(prediction_csv),
        "--swanlab-mode", "disabled",
    ])


def build_protocol(
    *,
    input_csvs: list[str | Path],
    output_dir: str | Path,
    mode: str,
    split_seeds: tuple[int, ...] = SPLIT_SEEDS,
    n_splits: int = 5,
    test_fraction: float = 0.2,
    val_fraction_total: float = 0.1,
    checkpoint_root: str | Path,
    result_root: str | Path,
    log_root: str | Path,
    python_executable: str = "python",
) -> tuple[pd.DataFrame, dict[str, object]]:
    if mode not in {"repeated_holdout", "repeated_cv"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if not 0 < val_fraction_total < 1 - test_fraction:
        raise ValueError("val_fraction_total must fit inside the non-test fraction")
    frame, sources = _load_inputs(input_csvs)
    labels = frame["label"].map(_parse_label).to_numpy(np.int64)
    output_dir = Path(output_dir).resolve()
    checkpoint_root = Path(checkpoint_root)
    result_root = Path(result_root)
    log_root = Path(log_root)
    rows: list[dict[str, object]] = []
    splits_manifest: list[dict[str, object]] = []

    for split_seed in split_seeds:
        if mode == "repeated_holdout":
            outer = StratifiedShuffleSplit(
                n_splits=1, test_size=test_fraction, random_state=split_seed
            )
            outer_pairs = list(outer.split(np.zeros(len(labels)), labels))
        else:
            outer = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=split_seed
            )
            outer_pairs = list(outer.split(np.zeros(len(labels)), labels))

        for fold, (development, outer_test) in enumerate(outer_pairs, start=1):
            relative_val_fraction = val_fraction_total / (1.0 - test_fraction)
            inner = StratifiedShuffleSplit(
                n_splits=1,
                test_size=relative_val_fraction,
                random_state=split_seed * 1000 + fold,
            )
            train_local, val_local = next(
                inner.split(np.zeros(len(development)), labels[development])
            )
            train_indices = np.asarray(development)[train_local]
            val_indices = np.asarray(development)[val_local]
            test_indices = np.asarray(outer_test)
            sets = [set(train_indices), set(val_indices), set(test_indices)]
            if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
                raise AssertionError("Train, validation, and test indices overlap")

            split_dir = output_dir / "splits" / f"split_seed_{split_seed}" / f"fold_{fold}"
            train_csv = split_dir / "train.csv"
            val_csv = split_dir / "selection_val.csv"
            test_csv = split_dir / "outer_test.csv"
            _write_split(frame, train_indices, train_csv)
            _write_split(frame, val_indices, val_csv)
            _write_split(frame, test_indices, test_csv)
            splits_manifest.append({
                "split_seed": split_seed,
                "fold": fold,
                "train_n": len(train_indices),
                "selection_val_n": len(val_indices),
                "outer_test_n": len(test_indices),
                "outer_test_positive_n": int(labels[test_indices].sum()),
                "train_csv": str(train_csv),
                "selection_val_csv": str(val_csv),
                "outer_test_csv": str(test_csv),
            })

            for model, recipe in MODEL_RECIPES.items():
                run_key = (
                    f"{model}/split_seed_{split_seed}/fold_{fold}/"
                    f"train_seed_{recipe['train_seed']}"
                )
                checkpoint = checkpoint_root / run_key / "best_model.pth"
                prediction_csv = result_root / run_key / "outer_test_predictions.csv"
                log_dir = log_root / run_key
                rows.append({
                    "run_key": run_key,
                    "model": model,
                    "split_seed": split_seed,
                    "train_seed": recipe["train_seed"],
                    "fold": fold,
                    "train_csv": str(train_csv),
                    "selection_val_csv": str(val_csv),
                    "outer_test_csv": str(test_csv),
                    "checkpoint": str(checkpoint),
                    "prediction_csv": str(prediction_csv),
                    "train_command": _train_command(
                        model, train_csv, val_csv, checkpoint, log_dir, python_executable
                    ),
                    "eval_command": _eval_command(
                        model, test_csv, checkpoint, prediction_csv, python_executable
                    ),
                    "source_export_command": "",
                    "source_export_selection_command": "",
                })

    plan = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "command_plan.csv"
    plan.to_csv(plan_path, index=False)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "protocol": mode,
        "input_sources": sources,
        "n_cases": len(frame),
        "class_counts": {
            str(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()
        },
        "split_seeds": list(split_seeds),
        "n_outer_folds_per_repeat": 1 if mode == "repeated_holdout" else n_splits,
        "test_fraction": test_fraction,
        "validation_fraction_total": val_fraction_total,
        "models": MODEL_RECIPES,
        "fusion": "fixed 0.5/0.5 arithmetic mean of positive-class logits",
        "selection": "checkpoint selected by validation AUC; test labels evaluation-only",
        "planned_training_runs": len(plan),
        "splits": splits_manifest,
        "command_plan_csv": str(plan_path),
        "command_plan_sha256": _sha256(plan_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return plan, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("repeated_holdout", "repeated_cv"), required=True)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(SPLIT_SEEDS))
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--val-fraction-total", type=float, default=0.1)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--python-executable", default="python")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan, manifest = build_protocol(
        input_csvs=args.input_csv,
        output_dir=args.output_dir,
        mode=args.mode,
        split_seeds=tuple(args.split_seeds),
        n_splits=args.num_folds,
        test_fraction=args.test_fraction,
        val_fraction_total=args.val_fraction_total,
        checkpoint_root=args.checkpoint_root,
        result_root=args.result_root,
        log_root=args.log_root,
        python_executable=args.python_executable,
    )
    print(f"Command plan: {manifest['command_plan_csv']}")
    print(f"Planned training runs: {len(plan)}")


if __name__ == "__main__":
    main()
