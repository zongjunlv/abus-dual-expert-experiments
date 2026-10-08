#!/usr/bin/env python3
"""Summarize repeated outer-CV predictions for the AIIM major revision.

Each split seed is treated as one complete repeated 5-fold CV estimate.  The
five disjoint outer folds are pooled before metrics are calculated.  Paired
confidence intervals resample unique cases and retain all repeat predictions,
so the three appearances of a case are never treated as independent samples.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


METRICS = ("auc", "accuracy", "sensitivity", "specificity", "f1", "precision", "mcc")
BOOTSTRAP_METRICS = ("auc", "accuracy", "f1", "mcc")


def canonical_case_id(value: object) -> str:
    """Return a case key robust to migration between repository prefixes."""
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    if not parts:
        raise ValueError("Empty case path")
    return "/".join(parts[-4:])


def _metrics_from_arrays(y_true: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    return {
        "n": int(y_true.size),
        "auc": float(roc_auc_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }


def binary_metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Calculate metrics using fold-specific validation-locked predictions."""
    return _metrics_from_arrays(
        frame["label_true"].to_numpy(dtype=int),
        frame["label_pred"].to_numpy(dtype=int),
        frame["prob_1"].to_numpy(dtype=float),
    )


def exact_mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, float | int]:
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {
        "a_correct_b_wrong": a_only,
        "a_wrong_b_correct": b_only,
        "discordant_n": discordant,
        "p_value": float(p_value),
    }


def _repeat_metric(frame: pd.DataFrame, prob_column: str, pred_column: str, metric: str) -> float:
    values = []
    for _, repeat_frame in frame.groupby("split_seed", sort=True):
        result = _metrics_from_arrays(
            repeat_frame["label_true"].to_numpy(dtype=int),
            repeat_frame[pred_column].to_numpy(dtype=int),
            repeat_frame[prob_column].to_numpy(dtype=float),
        )
        values.append(result[metric])
    return float(np.mean(values))


def _bootstrap_summary(samples: np.ndarray, observed: float) -> dict[str, float]:
    lower, upper = np.quantile(samples, [0.025, 0.975])
    p_lower = (np.sum(samples <= 0.0) + 1.0) / (samples.size + 1.0)
    p_upper = (np.sum(samples >= 0.0) + 1.0) / (samples.size + 1.0)
    return {
        "observed_delta": observed,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "two_sided_p": float(min(1.0, 2.0 * min(p_lower, p_upper))),
    }


def cluster_bootstrap_deltas(paired: pd.DataFrame, repeats: int = 10000, seed: int = 3407) -> pd.DataFrame:
    """Paired bootstrap of A-minus-B metrics, clustered by unique case."""
    case_ids = np.array(sorted(paired["case_id"].unique()))
    rng = np.random.default_rng(seed)
    observed = {
        metric: _repeat_metric(paired, "a_prob", "a_pred", metric)
        - _repeat_metric(paired, "b_prob", "b_pred", metric)
        for metric in BOOTSTRAP_METRICS
    }
    samples = {metric: np.empty(repeats, dtype=float) for metric in BOOTSTRAP_METRICS}
    indexed = {split_seed: frame.set_index("case_id") for split_seed, frame in paired.groupby("split_seed")}
    case_labels = paired.drop_duplicates("case_id").set_index("case_id")["label_true"]
    for iteration in range(repeats):
        for _ in range(1000):
            sampled_ids = rng.choice(case_ids, size=case_ids.size, replace=True)
            if case_labels.loc[sampled_ids].nunique() == 2:
                break
        else:
            raise RuntimeError("Could not draw a two-class case bootstrap sample")
        repeat_deltas = {metric: [] for metric in BOOTSTRAP_METRICS}
        for frame in indexed.values():
            sampled = frame.loc[sampled_ids]
            a = _metrics_from_arrays(
                sampled["label_true"].to_numpy(int),
                sampled["a_pred"].to_numpy(int),
                sampled["a_prob"].to_numpy(float),
            )
            b = _metrics_from_arrays(
                sampled["label_true"].to_numpy(int),
                sampled["b_pred"].to_numpy(int),
                sampled["b_prob"].to_numpy(float),
            )
            for metric in BOOTSTRAP_METRICS:
                repeat_deltas[metric].append(a[metric] - b[metric])
        for metric in BOOTSTRAP_METRICS:
            samples[metric][iteration] = float(np.mean(repeat_deltas[metric]))

    rows = []
    for metric in BOOTSTRAP_METRICS:
        row = {
            "metric": metric,
            "bootstrap_unit": "case_id",
            "n_unique_cases": int(case_ids.size),
            "n_repeats": int(paired["split_seed"].nunique()),
            "bootstrap_replicates": repeats,
        }
        row.update(_bootstrap_summary(samples[metric], observed[metric]))
        rows.append(row)
    return pd.DataFrame(rows)


def _load_model_oof(result_root: Path, model: str) -> pd.DataFrame:
    files = sorted((result_root / model).glob("split_seed_*/fold_*/train_seed_*/outer_test_predictions.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction files found for {model} below {result_root}")
    frames = []
    for path in files:
        split_seed = int(next(part for part in path.parts if part.startswith("split_seed_")).split("_")[-1])
        fold = int(next(part for part in path.parts if part.startswith("fold_")).split("_")[-1])
        frame = pd.read_csv(path)
        path_column = "npy_path" if "npy_path" in frame.columns else "data_path"
        required = {"label_true", "label_pred", "prob_1", path_column}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        frame = frame.copy()
        frame["case_id"] = frame[path_column].map(canonical_case_id)
        frame["split_seed"] = split_seed
        frame["fold"] = fold
        frames.append(frame[["case_id", "split_seed", "fold", "label_true", "label_pred", "prob_1"]])
    combined = pd.concat(frames, ignore_index=True)
    for split_seed, repeat_frame in combined.groupby("split_seed"):
        if repeat_frame["case_id"].duplicated().any():
            raise ValueError(f"Duplicate outer-test cases for {model}, split seed {split_seed}")
    return combined


def _paired_frame(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    paired = a.merge(
        b,
        on=["case_id", "split_seed"],
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )
    if len(paired) != len(a) or len(paired) != len(b):
        raise ValueError("Models do not contain identical repeated-CV case sets")
    if not np.array_equal(paired["label_true_a"], paired["label_true_b"]):
        raise ValueError("Paired model labels differ")
    return paired.rename(
        columns={
            "label_true_a": "label_true",
            "label_pred_a": "a_pred",
            "prob_1_a": "a_prob",
            "label_pred_b": "b_pred",
            "prob_1_b": "b_prob",
        }
    )[["case_id", "split_seed", "label_true", "a_pred", "a_prob", "b_pred", "b_prob"]]


def _repeat_metrics(models: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, frame in models.items():
        for split_seed, repeat_frame in frame.groupby("split_seed", sort=True):
            row = {"model": model, "split_seed": int(split_seed)}
            row.update(binary_metrics(repeat_frame))
            row["folds"] = int(repeat_frame["fold"].nunique())
            rows.append(row)
    return pd.DataFrame(rows)


def _model_summary(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, frame in repeat_metrics.groupby("model", sort=True):
        row: dict[str, float | int | str] = {
            "model": model,
            "repeat_n": int(len(frame)),
            "cases_per_repeat": int(frame["n"].iloc[0]),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_sd"] = float(frame[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _training_time_summary(checkpoint_root: Path, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        paths = sorted((checkpoint_root / model).glob("split_seed_*/fold_*/train_seed_*/best_model_multiseed_summary.csv"))
        values = []
        for path in paths:
            frame = pd.read_csv(path)
            if "train_time_sec" in frame and not frame.empty:
                values.extend(pd.to_numeric(frame["train_time_sec"], errors="coerce").dropna().tolist())
        if values:
            array = np.asarray(values, dtype=float)
            rows.append(
                {
                    "model": model,
                    "run_n": int(array.size),
                    "mean_hours_per_fold": float(array.mean() / 3600.0),
                    "sd_hours_per_fold": float(array.std(ddof=1) / 3600.0),
                    "median_hours_per_fold": float(np.median(array) / 3600.0),
                    "total_gpu_hours": float(array.sum() / 3600.0),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-a", default="dual_backbone_view_consistency_reliability")
    parser.add_argument("--model-b", default="r2plus1d")
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = {
        args.model_a: _load_model_oof(args.result_root, args.model_a),
        args.model_b: _load_model_oof(args.result_root, args.model_b),
    }
    repeat_metrics = _repeat_metrics(models)
    model_summary = _model_summary(repeat_metrics)
    paired = _paired_frame(models[args.model_a], models[args.model_b])

    paired_repeat_rows = []
    mcnemar_rows = []
    for split_seed, frame in paired.groupby("split_seed", sort=True):
        a = _metrics_from_arrays(frame["label_true"].to_numpy(int), frame["a_pred"].to_numpy(int), frame["a_prob"].to_numpy(float))
        b = _metrics_from_arrays(frame["label_true"].to_numpy(int), frame["b_pred"].to_numpy(int), frame["b_prob"].to_numpy(float))
        paired_repeat_rows.append(
            {"split_seed": int(split_seed), **{f"delta_{metric}": a[metric] - b[metric] for metric in METRICS}}
        )
        mcnemar_rows.append(
            {
                "split_seed": int(split_seed),
                **exact_mcnemar(frame["label_true"].to_numpy(int), frame["a_pred"].to_numpy(int), frame["b_pred"].to_numpy(int)),
            }
        )

    bootstrap = cluster_bootstrap_deltas(
        paired,
        repeats=args.bootstrap_repeats,
        seed=args.bootstrap_seed,
    )
    bootstrap.insert(0, "contrast", f"{args.model_a} minus {args.model_b}")
    training_time = _training_time_summary(args.checkpoint_root, list(models))

    repeat_metrics.to_csv(args.output_dir / "repeat_oof_metrics.csv", index=False)
    model_summary.to_csv(args.output_dir / "model_repeat_summary.csv", index=False)
    pd.DataFrame(paired_repeat_rows).to_csv(args.output_dir / "paired_repeat_deltas.csv", index=False)
    pd.DataFrame(mcnemar_rows).to_csv(args.output_dir / "mcnemar_exact_per_repeat.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_case_cluster_bootstrap.csv", index=False)
    training_time.to_csv(args.output_dir / "training_time_summary.csv", index=False)

    manifest = {
        "protocol": "3x repeated stratified 5-fold case-level CV; inner validation threshold locked per outer fold",
        "threshold_metric": "MCC (test.py default)",
        "model_a": args.model_a,
        "model_b": args.model_b,
        "bootstrap_unit": "unique case_id with all repeat predictions retained",
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.bootstrap_seed,
        "notes": [
            "The public CSV has no patient identifier, so this is case-level rather than patient-level CV.",
            "Fold-exported label_pred values preserve each fold's inner-validation-selected threshold.",
            "The five disjoint outer folds are pooled before each repeat metric is computed.",
        ],
    }
    (args.output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
