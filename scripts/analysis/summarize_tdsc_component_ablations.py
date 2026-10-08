#!/usr/bin/env python3
"""Locked, paired TDSC component ablations; exports aggregate evidence only.

The five variants reuse the published 3 x 5-fold memberships, optimization
seed, and final depth-safe expert checkpoints. Confidence intervals are
exploratory percentile intervals, unadjusted for the ten planned contrasts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

try:
    from scripts.analysis.summarize_triplanar_stability import (
        EPS, case_id, expit, load_prediction, logit, youden_threshold,
    )
    from scripts.analysis.summarize_triplanar_tdsc_cv import metrics_with_predictions
except ModuleNotFoundError:
    from summarize_triplanar_stability import EPS, case_id, expit, load_prediction, logit, youden_threshold
    from summarize_triplanar_tdsc_cv import metrics_with_predictions


SPLIT_SEEDS = (3407, 42, 1234)
FOLDS = (1, 2, 3, 4, 5)
TRAIN_SEED = 3407
VARIANTS = ("center_only", "mean_pool", "no_position", "no_plane", "z_only")
FULL_MODEL = "convnext_triplanar_25d_mil"
METRICS = ("auc", "nll", "accuracy", "sensitivity", "specificity", "f1_positive", "mcc")
ROLE_COUNTS = {"train": 136, "selection_val": 24, "outer_test": 40}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction_path(root: Path, model: str, seed: int, fold: int, role: str) -> Path:
    return root / model / f"split_seed_{seed}" / f"fold_{fold}" / f"train_seed_{TRAIN_SEED}" / f"{role}_predictions.csv"


def _binary_label(value: object) -> int:
    if isinstance(value, str) and value.strip().upper() in {"B", "M"}:
        return int(value.strip().upper() == "M")
    number = float(value)
    if not np.isfinite(number) or number not in (0.0, 1.0):
        raise ValueError("Protocol labels must be binary")
    return int(number)


def load_split(path: Path, expected_n: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    label_column = next((name for name in ("label", "label_true") if name in frame), None)
    if "npy_path" not in frame or label_column is None:
        raise ValueError(f"Protocol CSV lacks paths/labels: {path}")
    if frame["npy_path"].isnull().any() or frame[label_column].isnull().any():
        raise ValueError(f"Protocol contains missing ID/label: {path}")
    result = pd.DataFrame({
        "case_id": frame.npy_path.map(case_id),
        "label_true": frame[label_column].map(_binary_label),
    })
    if len(result) != expected_n or result.case_id.duplicated().any():
        raise ValueError(f"Protocol {path} has wrong count or duplicate case IDs")
    if result.label_true.nunique() != 2:
        raise ValueError(f"Protocol role {path} must contain both classes")
    return result.sort_values("case_id").reset_index(drop=True)


def load_protocol(root: Path) -> tuple[dict[tuple[int, int, str], pd.DataFrame], list[dict]]:
    roles = {}
    hashes = []
    canonical: pd.DataFrame | None = None
    for seed in SPLIT_SEEDS:
        outer = []
        for fold in FOLDS:
            per_fold = []
            for role, count in ROLE_COUNTS.items():
                path = root / "splits" / f"split_seed_{seed}" / f"fold_{fold}" / f"{role}.csv"
                value = load_split(path, count)
                roles[(seed, fold, role)] = value
                hashes.append({"split_seed": seed, "fold": fold, "role": role, "sha256": sha256(path)})
                per_fold.append(value)
            joined = pd.concat(per_fold, ignore_index=True).sort_values("case_id").reset_index(drop=True)
            if len(joined) != 200 or joined.case_id.duplicated().any():
                raise ValueError(f"Split {seed}/{fold} roles overlap or do not span 200 unique cases")
            if canonical is None:
                canonical = joined
            elif not joined.equals(canonical):
                raise ValueError("Protocol case sets or labels vary across folds/repeats")
            outer.append(roles[(seed, fold, "outer_test")])
        repeated = pd.concat(outer, ignore_index=True).sort_values("case_id").reset_index(drop=True)
        if repeated.case_id.duplicated().any() or not repeated.equals(canonical):
            raise ValueError(f"Repeat {seed} does not test every case exactly once")
    return roles, hashes


def aligned_prediction(path: Path, expected: pd.DataFrame, name: str) -> tuple[pd.DataFrame, dict]:
    observed = load_prediction(path, name).set_index("case_id")
    if set(observed.index) != set(expected.case_id):
        raise ValueError(f"Prediction IDs do not match the locked role: {path}")
    observed = observed.loc[expected.case_id].reset_index()
    if not np.array_equal(observed[f"label_{name}"].to_numpy(int), expected.label_true.to_numpy(int)):
        raise ValueError(f"Prediction labels differ from the locked protocol: {path}")
    probability = observed[f"prob_{name}"].to_numpy(float)
    record = {
        "path": str(path), "sha256": sha256(path), "n": len(observed),
        "probability_endpoint_clipped_n": int(np.sum((probability < EPS) | (probability > 1 - EPS))),
    }
    return observed, record


def fuse_probabilities(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Equal raw-logit binary fusion reconstructed from softmax log-odds.

    log(p/(1-p)) = logit_1 - logit_0. Averaging these margins is equivalent
    to softmax of averaged class logits, up to the inherited EPS endpoint
    clipping. No temperature or weight is estimated from either test role.
    """
    return expit(0.5 * logit(first) + 0.5 * logit(second))


def analyze_all(
    roles: dict[tuple[int, int, str], pd.DataFrame],
    r3d_root: Path,
    full_root: Path,
    variant_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    fold_rows = []
    outer_rows = []
    hashes = []
    variants = VARIANTS if variant_root is not None else ()
    for seed in SPLIT_SEEDS:
        for fold in FOLDS:
            partitions = {}
            for role in ("selection_val", "outer_test"):
                expected = roles[(seed, fold, role)]
                paired = expected.copy()
                sources = [("r3d", r3d_root, "r3d"), ("trimil_full", full_root, FULL_MODEL)]
                sources += [(f"trimil_{name}", variant_root, f"{FULL_MODEL}_{name}") for name in variants]
                for name, root, model in sources:
                    path = prediction_path(root, model, seed, fold, role)
                    observed, record = aligned_prediction(path, expected, name)
                    paired[f"prob_{name}"] = observed[f"prob_{name}"].to_numpy(float)
                    hashes.append({"split_seed": seed, "fold": fold, "role": role, "method": name, **record})
                for name in ("full", *variants):
                    paired[f"prob_fusion_{name}"] = fuse_probabilities(
                        paired.prob_r3d.to_numpy(float), paired[f"prob_trimil_{name}"].to_numpy(float),
                    )
                partitions[role] = paired
            validation, outer = partitions["selection_val"], partitions["outer_test"]
            for column in (value for value in outer.columns if value.startswith("prob_")):
                method = column.removeprefix("prob_")
                threshold = youden_threshold(validation.label_true.to_numpy(int), validation[column].to_numpy(float))
                outer[f"pred_{method}"] = outer[column].to_numpy(float) >= threshold
                outer[f"threshold_{method}"] = threshold
                fold_rows.append({
                    "split_seed": seed, "fold": fold, "method": method, "n": len(outer),
                    "validation_locked_threshold": threshold,
                    **metrics_with_predictions(outer.label_true.to_numpy(int), outer[column].to_numpy(float), outer[f"pred_{method}"].to_numpy(bool)),
                })
            outer["split_seed"], outer["fold"] = seed, fold
            outer_rows.append(outer)
    return pd.DataFrame(fold_rows), pd.concat(outer_rows, ignore_index=True), hashes


def validate_oof(frame: pd.DataFrame, expected_cases: int = 200) -> list[pd.DataFrame]:
    required = {"case_id", "label_true", "split_seed", "fold"}
    if not required.issubset(frame.columns) or frame[list(required)].isnull().any().any():
        raise ValueError("OOF frame lacks complete required columns")
    if set(frame.split_seed.unique()) != set(SPLIT_SEEDS):
        raise ValueError("OOF frame must contain all three locked repeats")
    if not set(frame.label_true.unique()).issubset({0, 1}) or frame.label_true.nunique() != 2:
        raise ValueError("OOF labels must contain both binary classes")
    if frame.groupby("case_id").label_true.nunique().max() != 1:
        raise ValueError("Labels differ across repeats for the same case ID")
    canonical = frame[frame.split_seed.eq(SPLIT_SEEDS[0])].sort_values("case_id").reset_index(drop=True)
    if len(canonical) != expected_cases or canonical.case_id.duplicated().any():
        raise ValueError("OOF repeat has duplicate/missing case IDs")
    output = []
    for seed in SPLIT_SEEDS:
        repeat = frame[frame.split_seed.eq(seed)].sort_values("case_id").reset_index(drop=True)
        if len(repeat) != expected_cases or repeat.case_id.duplicated().any():
            raise ValueError("OOF repeat has duplicate/missing case IDs")
        if not repeat[["case_id", "label_true"]].equals(canonical[["case_id", "label_true"]]):
            raise ValueError("OOF case IDs/labels differ across repeats")
        for column in (value for value in repeat.columns if value.startswith("prob_")):
            values = repeat[column].to_numpy(float)
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError(f"Invalid OOF probabilities: {column}")
        output.append(repeat)
    return output


def repeat_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    output = []
    for seed, repeat in frame.groupby("split_seed", sort=True):
        for column in (value for value in repeat.columns if value.startswith("prob_")):
            method = column.removeprefix("prob_")
            output.append({
                "split_seed": int(seed), "method": method, "n": len(repeat), "folds": repeat.fold.nunique(),
                **metrics_with_predictions(repeat.label_true.to_numpy(int), repeat[column].to_numpy(float), repeat[f"pred_{method}"].to_numpy(bool)),
            })
    return pd.DataFrame(output)


def model_summary(repeats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in repeats.groupby("method", sort=True):
        rows.append({
            "method": method, "repeat_n": len(group),
            **{f"{metric}_{stat}": float(getattr(group[metric], stat)(**({"ddof": 1} if stat == "std" else {})))
               for metric in METRICS for stat in ("mean", "std")},
        })
    return pd.DataFrame(rows).rename(columns={f"{metric}_std": f"{metric}_sd" for metric in METRICS})


def assert_full_baseline(summary: pd.DataFrame, expected_path: Path, tolerance: float = 1e-10) -> None:
    expected = pd.read_csv(expected_path).set_index("method")
    current = summary.set_index("method")
    for old, new in (("r3d", "r3d"), ("triplanar", "trimil_full"), ("equal_logit", "fusion_full")):
        if old not in expected.index or new not in current.index:
            raise ValueError("Expected full model is missing from the baseline summary")
        for metric in METRICS:
            for stat in ("mean", "sd"):
                column = f"{metric}_{stat}"
                if not np.isclose(expected.loc[old, column], current.loc[new, column], atol=tolerance, rtol=0):
                    raise ValueError(f"Baseline verification failed: {old}/{column}")


def weighted_auc_many(labels: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Vectorized, tie-corrected weighted ROC AUC for many bootstrap draws."""
    labels = np.asarray(labels, int)
    probability = np.asarray(probability, float)
    weights = np.asarray(weights, float)
    if weights.ndim != 2 or weights.shape[1] != len(labels) or probability.shape != labels.shape:
        raise ValueError("Labels, probability and bootstrap weights are not aligned")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Weights must be finite and nonnegative")
    order = np.argsort(probability, kind="stable")
    ordered_probability = probability[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(ordered_probability) != 0)]
    ends = np.r_[starts[1:] - 1, len(labels) - 1]
    group = np.repeat(np.arange(len(starts)), ends - starts + 1)
    negative = weights[:, order] * (labels[order] == 0)
    positive = weights[:, order] * (labels[order] == 1)
    cumulative = np.cumsum(negative, axis=1)
    padded = np.pad(cumulative, ((0, 0), (1, 0)))
    before = padded[:, starts[group]]
    through = cumulative[:, ends[group]]
    denominator = negative.sum(axis=1) * positive.sum(axis=1)
    if np.any(denominator <= 0):
        raise ValueError("Each AUC resample must contain both classes")
    return np.sum(positive * (before + through) * 0.5, axis=1) / denominator


def planned_contrasts() -> list[tuple[str, str, str]]:
    return [
        (f"{scope}_full_minus_{name}", f"{scope}_full", f"{scope}_{name}")
        for name in VARIANTS for scope in ("trimil", "fusion")
    ]


def hierarchical_bootstrap(
    frame: pd.DataFrame, replicates: int = 10_000, seed: int = 20260929,
    *, expected_cases: int = 200,
) -> pd.DataFrame:
    if replicates <= 0:
        raise ValueError("Bootstrap replicate count must be positive")
    repeats = validate_oof(frame, expected_cases)
    labels = repeats[0].label_true.to_numpy(int)
    methods = sorted({method for _, first, second in planned_contrasts() for method in (first, second)})
    rng = np.random.default_rng(seed)
    # One class-stratified unique-ID draw is shared by every repeat, every
    # method and every contrast. Repeats are then sampled (not outer folds).
    weights = np.zeros((replicates, len(labels)), dtype=np.int64)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=replicates)
    sampled_repeats = rng.integers(0, len(repeats), size=(replicates, len(repeats)))
    rows_index = np.arange(replicates)[:, None]
    estimates = {}
    observed = {}
    for method in methods:
        auc_samples, nll_samples, auc_point, nll_point = [], [], [], []
        for repeat in repeats:
            probability = repeat[f"prob_{method}"].to_numpy(float)
            clipped = np.clip(probability, EPS, 1 - EPS)
            case_loss = -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
            auc_samples.append(weighted_auc_many(labels, probability, weights))
            nll_samples.append((weights @ case_loss) / len(labels))
            auc_point.append(roc_auc_score(labels, probability))
            nll_point.append(log_loss(labels, clipped, labels=[0, 1]))
        for metric, samples, points in (("auc", auc_samples, auc_point), ("nll", nll_samples, nll_point)):
            matrix = np.asarray(samples).T
            estimates[(method, metric)] = matrix[rows_index, sampled_repeats].mean(axis=1)
            observed[(method, metric)] = float(np.mean(points))
    output = []
    for contrast, full, ablated in planned_contrasts():
        for metric in ("auc", "nll"):
            values = estimates[(full, metric)] - estimates[(ablated, metric)]
            lower, upper = np.quantile(values, [0.025, 0.975])
            output.append({
                "contrast": contrast, "metric": f"{metric}_delta",
                "observed_delta": observed[(full, metric)] - observed[(ablated, metric)],
                "ci95_lower": float(lower), "ci95_upper": float(upper),
                "probability_favorable_to_full": float(np.mean(values > 0 if metric == "auc" else values < 0)),
                "bootstrap_unit": "class-stratified unique case ID plus repeat; shared ID weights across all methods/repeats",
                "replicates": replicates, "multiplicity_adjustment": "none; exploratory planned contrasts",
            })
    return pd.DataFrame(output)


def validate_variant_completion(root: Path) -> list[dict]:
    cells = []
    locked_settings = {
        "train_seed": TRAIN_SEED, "epochs": 80, "batch_size": 46,
        "eval_batch_size": 64, "accum_steps": 1, "num_workers": 0,
        "lr": 1e-4, "backbone_lr_scale": 0.05, "weight_decay": 0.05,
        "warmup_epochs": 5, "patience": 10, "freeze_backbone_epochs": 5,
        "monitor_start_epoch": 1, "target_shape": [128, 128, 64],
        "spatial_aug_policy": "slice_xy",
        "checkpoint_selection": "selection-validation AUC only",
    }
    common_source_hashes = None
    for name in VARIANTS:
        model = f"{FULL_MODEL}_{name}"
        for seed in SPLIT_SEEDS:
            for fold in FOLDS:
                run = prediction_path(root, model, seed, fold, "outer_test").parent
                for marker in ("TRAIN_DONE", "DONE", "configuration.json", "prediction_sha256.json"):
                    if not (run / marker).is_file() or (run / marker).stat().st_size == 0:
                        raise ValueError(f"Incomplete ablation cell: {model}/{seed}/{fold}; missing {marker}")
                signed = json.loads((run / "configuration.json").read_text())
                config = signed["configuration"]
                encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
                if hashlib.sha256(encoded).hexdigest() != signed["signature_sha256"]:
                    raise ValueError(f"Invalid cell signature: {run}")
                if config["model"] != model or config["split_seed"] != seed or config["fold"] != fold or any(config.get(key) != value for key, value in locked_settings.items()):
                    raise ValueError(f"Cell configuration differs from locked protocol: {run}")
                source_hashes = {path: digest for path, digest in config["input_sha256"].items() if not path.endswith(".csv")}
                if common_source_hashes is None:
                    common_source_hashes = source_hashes
                elif common_source_hashes != source_hashes:
                    raise ValueError("Ablation cells were produced by different source versions")
                for input_path, digest in config["input_sha256"].items():
                    path = Path(input_path)
                    if not path.is_file() or sha256(path) != digest:
                        raise ValueError(f"Signed input changed after training: {input_path}")
                hashes = json.loads((run / "prediction_sha256.json").read_text())
                for role in ("selection_val", "outer_test"):
                    if sha256(run / f"{role}_predictions.csv") != hashes[role]:
                        raise ValueError(f"Completed prediction was modified: {run}/{role}")
                summary_path = root.parent / "checkpoints" / model / f"split_seed_{seed}" / f"fold_{fold}" / f"train_seed_{TRAIN_SEED}" / "best_model_multiseed_summary.csv"
                summary = pd.read_csv(summary_path)
                if len(summary) != 1 or int(summary.seed.iloc[0]) != TRAIN_SEED or not np.isfinite(summary.train_time_sec.iloc[0]) or summary.train_time_sec.iloc[0] <= 0 or not 1 <= int(summary.best_epoch.iloc[0]) <= 80:
                    raise ValueError(f"Unexpected training summary: {summary_path}")
                cells.append({
                    "model": model, "split_seed": seed, "fold": fold, "train_seed": TRAIN_SEED,
                    "signature_sha256": signed["signature_sha256"], "train_time_sec": float(summary.train_time_sec.iloc[0]),
                    "best_epoch": int(summary.best_epoch.iloc[0]),
                })
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--baseline-result-root", type=Path, required=True)
    parser.add_argument("--full-trimil-result-root", type=Path, required=True)
    parser.add_argument("--expected-full-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260929)
    args = parser.parse_args()
    if not args.preflight and args.result_root is None:
        parser.error("Full analysis requires --result-root")
    roles, protocol_hashes = load_protocol(args.protocol_root)
    baseline_fold, baseline_oof, baseline_hashes = analyze_all(roles, args.baseline_result_root, args.full_trimil_result_root)
    validate_oof(baseline_oof)
    baseline_summary = model_summary(repeat_metrics(baseline_oof))
    assert_full_baseline(baseline_summary, args.expected_full_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight = {
        "passed": True, "unique_cases": 200, "repeats": 3, "outer_folds_per_repeat": 5,
        "training_seed": TRAIN_SEED, "expected_full_summary_sha256": sha256(args.expected_full_summary),
        "baseline_recomputed_exactly": True, "comparison_tolerance": 1e-10,
        "protocol_role_sha256": protocol_hashes, "baseline_prediction_sha256": baseline_hashes,
        "full_fusion_auc_mean": float(baseline_summary.set_index("method").loc["fusion_full", "auc_mean"]),
        "full_fusion_nll_mean": float(baseline_summary.set_index("method").loc["fusion_full", "nll_mean"]),
    }
    (args.output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if args.preflight:
        print(json.dumps({key: value for key, value in preflight.items() if key not in {"protocol_role_sha256", "baseline_prediction_sha256"}}, indent=2))
        return

    cells = validate_variant_completion(args.result_root)
    fold_metrics, oof, prediction_hashes = analyze_all(roles, args.baseline_result_root, args.full_trimil_result_root, args.result_root)
    validate_oof(oof)
    repeats = repeat_metrics(oof)
    summary = model_summary(repeats)
    assert_full_baseline(summary, args.expected_full_summary)
    bootstrap = hierarchical_bootstrap(oof, args.bootstrap_replicates, args.bootstrap_seed)
    deltas = []
    for seed, group in repeats.groupby("split_seed", sort=True):
        indexed = group.set_index("method")
        for contrast, full, ablated in planned_contrasts():
            deltas.append({"split_seed": int(seed), "contrast": contrast, **{
                f"{metric}_delta": float(indexed.loc[full, metric] - indexed.loc[ablated, metric]) for metric in METRICS
            }})
    fold_metrics.to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    repeats.to_csv(args.output_dir / "repeat_oof_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "model_repeat_summary.csv", index=False)
    pd.DataFrame(deltas).to_csv(args.output_dir / "paired_repeat_deltas.csv", index=False)
    bootstrap.to_csv(args.output_dir / "hierarchical_case_repeat_bootstrap.csv", index=False)
    pd.DataFrame(cells).to_csv(args.output_dir / "training_completion_summary.csv", index=False)
    metadata = {
        "status": "complete", "trained_cells": len(cells), "variants": list(VARIANTS),
        "protocol": "TDSC 3 repeated stratified 5-fold CV; 200 pooled OOF cases per repeat; fixed train seed 3407",
        "thresholds": "Youden threshold from each fold's selection validation only; no outer-test fitting",
        "fusion": "0.5/0.5 raw-logit-equivalent binary log-odds reconstruction from exported prob_1; no temperature/gate fitting",
        "endpoint_clip_epsilon": EPS,
        "endpoint_clipped_prediction_total": sum(record["probability_endpoint_clipped_n"] for record in prediction_hashes),
        "prediction_sha256": prediction_hashes,
        "bootstrap": "class-stratified unique case ID plus repeat; shared ID weights across methods and repeats; folds are not resampled",
        "bootstrap_replicates": args.bootstrap_replicates, "bootstrap_seed": args.bootstrap_seed,
        "uncertainty": "exploratory percentile 95% intervals; ten planned contrasts; no multiplicity adjustment",
        "contrast_direction": "full minus ablated; positive AUC and negative NLL favor full",
        "z_only_boundary": "Z-only is a joint plane-coverage/instance-count control (24 versus 72 instances), not isolated evidence for plane embedding",
        "evaluation_boundary": "same public developmental cohort; not an independent untouched external cohort",
        "negative_result_policy": "report all planned variants regardless of improvement or deterioration",
        "baseline_verified": True,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "complete", "trained_cells": len(cells), "summary": str(args.output_dir / "model_repeat_summary.csv"), "bootstrap": str(args.output_dir / "hierarchical_case_repeat_bootstrap.csv")}, indent=2))


if __name__ == "__main__":
    main()
