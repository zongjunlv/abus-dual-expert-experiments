#!/usr/bin/env python3
"""Summarize the locked 3x3 R3D + tri-planar MIL stability experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)


EPS = 1e-7
SPLIT_SEEDS = (3407, 42, 1234)
TRAIN_SEEDS = (3407, 42, 1234)
METHODS = ("r3d", "triplanar", "equal_logit")


def case_id(value: object) -> str:
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    return "/".join(parts[-4:])


def logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, float), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def expit(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -50.0, 50.0)))


def load_prediction(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"npy_path", "label_true", "prob_1"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks {sorted(missing)}")
    result = frame[["npy_path", "label_true", "prob_1"]].copy()
    result["case_id"] = result["npy_path"].map(case_id)
    if result["case_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate cases")
    labels = pd.to_numeric(result["label_true"], errors="coerce").to_numpy(float)
    probabilities = pd.to_numeric(result["prob_1"], errors="coerce").to_numpy(float)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{path} has invalid labels")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(f"{path} has invalid probabilities")
    return pd.DataFrame({
        "case_id": result["case_id"], f"label_{name}": labels.astype(int),
        f"prob_{name}": probabilities,
    })


def pair_predictions(r3d_path: Path, tri_path: Path) -> pd.DataFrame:
    r3d = load_prediction(r3d_path, "r3d")
    tri = load_prediction(tri_path, "triplanar")
    paired = r3d.merge(tri, on="case_id", how="outer", validate="one_to_one")
    if paired.isna().any().any() or len(paired) != len(r3d) or len(paired) != len(tri):
        raise ValueError("R3D and tri-planar case sets differ")
    if not np.array_equal(paired["label_r3d"], paired["label_triplanar"]):
        raise ValueError("R3D and tri-planar labels differ")
    paired = paired.rename(columns={"label_r3d": "label_true"}).drop(columns="label_triplanar")
    paired["prob_equal_logit"] = expit(
        0.5 * logit(paired["prob_r3d"]) + 0.5 * logit(paired["prob_triplanar"])
    )
    return paired.sort_values("case_id").reset_index(drop=True)


def youden_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, probability)
    finite = np.isfinite(thresholds)
    scores = np.where(finite, tpr - fpr, -np.inf)
    best = np.flatnonzero(np.isclose(scores, np.max(scores), rtol=0.0, atol=1e-12))
    return float(thresholds[best[np.argmin(np.abs(thresholds[best] - 0.5))]])


def metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probability >= threshold
    positive = labels == 1
    negative = ~positive
    return {
        "auc": float(roc_auc_score(labels, probability)),
        "nll": float(log_loss(labels, np.clip(probability, EPS, 1 - EPS), labels=[0, 1])),
        "accuracy": float(accuracy_score(labels, predicted)),
        "sensitivity": float(np.mean(predicted[positive])),
        "specificity": float(np.mean(~predicted[negative])),
        "f1_positive": float(f1_score(labels, predicted)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "threshold": float(threshold),
    }


def run_analysis(result_root: Path, split_seed: int, train_seed: int):
    def path(model: str, partition: str) -> Path:
        return result_root / model / f"split_seed_{split_seed}" / "fold_1" / f"train_seed_{train_seed}" / f"{partition}_predictions.csv"

    validation = pair_predictions(path("r3d", "selection_val"), path("convnext_triplanar_25d_mil", "selection_val"))
    test = pair_predictions(path("r3d", "outer_test"), path("convnext_triplanar_25d_mil", "outer_test"))
    thresholds = {
        method: youden_threshold(validation.label_true.to_numpy(int), validation[f"prob_{method}"].to_numpy(float))
        for method in METHODS
    }
    labels = test.label_true.to_numpy(int)
    rows = []
    for method in METHODS:
        row = {"split_seed": split_seed, "train_seed": train_seed, "method": method}
        row.update(metrics(labels, test[f"prob_{method}"].to_numpy(float), thresholds[method]))
        rows.append(row)
    by_method = {row["method"]: row for row in rows}
    r3d_correct = test.prob_r3d.to_numpy() >= thresholds["r3d"]
    fusion_correct = test.prob_equal_logit.to_numpy() >= thresholds["equal_logit"]
    r3d_correct = r3d_correct == labels
    fusion_correct = fusion_correct == labels
    delta = {
        "split_seed": split_seed, "train_seed": train_seed,
        "auc_delta": by_method["equal_logit"]["auc"] - by_method["r3d"]["auc"],
        "nll_delta": by_method["equal_logit"]["nll"] - by_method["r3d"]["nll"],
        "sensitivity_delta_locked": by_method["equal_logit"]["sensitivity"] - by_method["r3d"]["sensitivity"],
        "specificity_delta_locked": by_method["equal_logit"]["specificity"] - by_method["r3d"]["specificity"],
        "pearson_r3d_triplanar": float(test.prob_r3d.corr(test.prob_triplanar)),
        "corrected_r3d_errors": int(np.sum(~r3d_correct & fusion_correct)),
        "harmed_r3d_correct": int(np.sum(r3d_correct & ~fusion_correct)),
    }
    test = test.assign(
        split_seed=split_seed,
        train_seed=train_seed,
        threshold_r3d=thresholds["r3d"],
        threshold_triplanar=thresholds["triplanar"],
        threshold_equal_logit=thresholds["equal_logit"],
    )
    return rows, delta, test


def weighted_rate(mask: np.ndarray, condition: np.ndarray, weights: np.ndarray) -> float:
    selected = weights * condition
    return float(np.sum(weights * mask * condition) / np.sum(selected))


def hierarchical_bootstrap(
    runs: dict[tuple[int, int], pd.DataFrame],
    repeats: int,
    seed: int,
    baseline: str = "r3d",
) -> pd.DataFrame:
    if baseline not in {"r3d", "triplanar"}:
        raise ValueError(f"Unsupported baseline: {baseline}")
    labels_by_case: dict[str, int] = {}
    for frame in runs.values():
        for key, label in zip(frame.case_id, frame.label_true):
            if key in labels_by_case and labels_by_case[key] != int(label):
                raise ValueError(f"Label mismatch across splits for {key}")
            labels_by_case[key] = int(label)
    negative = np.array([key for key, value in labels_by_case.items() if value == 0], object)
    positive = np.array([key for key, value in labels_by_case.items() if value == 1], object)
    rng = np.random.default_rng(seed)
    draws = {name: np.empty(repeats) for name in ("auc_delta", "nll_delta", "sensitivity_delta_locked", "specificity_delta_locked")}
    split_values = np.array(SPLIT_SEEDS)
    train_values = np.array(TRAIN_SEEDS)
    for index in range(repeats):
        sampled = np.concatenate((rng.choice(negative, len(negative), replace=True), rng.choice(positive, len(positive), replace=True)))
        unique, counts = np.unique(sampled, return_counts=True)
        cluster_weights = dict(zip(unique.tolist(), counts.tolist()))
        current = {name: [] for name in draws}
        for split_seed in rng.choice(split_values, len(split_values), replace=True):
            for train_seed in rng.choice(train_values, len(train_values), replace=True):
                frame = runs[(int(split_seed), int(train_seed))]
                weights = frame.case_id.map(cluster_weights).fillna(0).to_numpy(float)
                keep = weights > 0
                labels = frame.label_true.to_numpy(int)[keep]
                weights = weights[keep]
                baseline_probability = frame[f"prob_{baseline}"].to_numpy(float)[keep]
                fusion = frame.prob_equal_logit.to_numpy(float)[keep]
                if np.unique(labels[weights > 0]).size != 2:
                    continue
                current["auc_delta"].append(roc_auc_score(labels, fusion, sample_weight=weights) - roc_auc_score(labels, baseline_probability, sample_weight=weights))
                current["nll_delta"].append(log_loss(labels, np.clip(fusion, EPS, 1-EPS), labels=[0,1], sample_weight=weights) - log_loss(labels, np.clip(baseline_probability, EPS, 1-EPS), labels=[0,1], sample_weight=weights))
                positive_mask = labels == 1
                negative_mask = labels == 0
                baseline_pred = baseline_probability >= float(frame[f"threshold_{baseline}"].iloc[0])
                fusion_pred = fusion >= float(frame.threshold_equal_logit.iloc[0])
                current["sensitivity_delta_locked"].append(weighted_rate(fusion_pred, positive_mask, weights) - weighted_rate(baseline_pred, positive_mask, weights))
                current["specificity_delta_locked"].append(weighted_rate(~fusion_pred, negative_mask, weights) - weighted_rate(~baseline_pred, negative_mask, weights))
        for name in draws:
            draws[name][index] = float(np.mean(current[name]))
    rows = []
    for name, values in draws.items():
        favorable = values > 0 if name != "nll_delta" else values < 0
        lower, upper = np.quantile(values, [0.025, 0.975])
        observed = []
        for frame in runs.values():
            labels = frame.label_true.to_numpy(int)
            baseline_probability = frame[f"prob_{baseline}"].to_numpy(float)
            fusion = frame.prob_equal_logit.to_numpy(float)
            if name == "auc_delta":
                value = roc_auc_score(labels, fusion) - roc_auc_score(labels, baseline_probability)
            elif name == "nll_delta":
                value = log_loss(labels, np.clip(fusion, EPS, 1-EPS), labels=[0,1]) - log_loss(labels, np.clip(baseline_probability, EPS, 1-EPS), labels=[0,1])
            else:
                positive_mask = labels == 1
                negative_mask = labels == 0
                baseline_pred = baseline_probability >= float(frame[f"threshold_{baseline}"].iloc[0])
                fusion_pred = fusion >= float(frame.threshold_equal_logit.iloc[0])
                if name == "sensitivity_delta_locked":
                    value = float(np.mean(fusion_pred[positive_mask]) - np.mean(baseline_pred[positive_mask]))
                else:
                    value = float(np.mean(~fusion_pred[negative_mask]) - np.mean(~baseline_pred[negative_mask]))
            observed.append(float(value))
        rows.append({
            "contrast": f"equal_logit minus {baseline}",
            "metric": name,
            "observed_delta": float(np.mean(observed)),
            "bootstrap_mean": float(values.mean()),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "probability_favorable": float(np.mean(favorable)),
            "bootstrap_unit": "split seed, training seed, and stratified unique case",
            "replicates": repeats,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    args = parser.parse_args()
    metric_rows, delta_rows, paired_rows = [], [], []
    runs = {}
    for split_seed in SPLIT_SEEDS:
        for train_seed in TRAIN_SEEDS:
            metrics_rows, delta, paired = run_analysis(args.result_root, split_seed, train_seed)
            metric_rows.extend(metrics_rows); delta_rows.append(delta); paired_rows.append(paired)
            runs[(split_seed, train_seed)] = paired
    metric_frame = pd.DataFrame(metric_rows)
    delta_frame = pd.DataFrame(delta_rows)
    paired_frame = pd.concat(paired_rows, ignore_index=True)
    bootstrap = pd.concat(
        [
            hierarchical_bootstrap(
                runs,
                args.bootstrap_repeats,
                args.bootstrap_seed + offset,
                baseline=baseline,
            )
            for offset, baseline in enumerate(("r3d", "triplanar"))
        ],
        ignore_index=True,
    )
    r3d_bootstrap = bootstrap[bootstrap.contrast.eq("equal_logit minus r3d")]
    gates = {
        "mean_auc_delta_at_least_0p008": float(delta_frame.auc_delta.mean()) >= 0.008,
        "auc_positive_at_least_7_of_9": int((delta_frame.auc_delta > 0).sum()) >= 7,
        "bootstrap_probability_auc_gain_at_least_0p95": float(r3d_bootstrap.loc[r3d_bootstrap.metric.eq("auc_delta"), "probability_favorable"].iloc[0]) >= 0.95,
        "mean_nll_delta_nonpositive": float(delta_frame.nll_delta.mean()) <= 0,
        "nll_nonworse_at_least_6_of_9": int((delta_frame.nll_delta <= 0).sum()) >= 6,
        "mean_locked_specificity_delta_at_least_minus_0p02": float(delta_frame.specificity_delta_locked.mean()) >= -0.02,
        "mean_locked_sensitivity_delta_nonnegative": float(delta_frame.sensitivity_delta_locked.mean()) >= 0,
        "total_corrected_at_least_total_harmed": int(delta_frame.corrected_r3d_errors.sum()) >= int(delta_frame.harmed_r3d_correct.sum()),
        "mean_r3d_triplanar_pearson_at_most_0p86": float(delta_frame.pearson_r3d_triplanar.mean()) <= 0.86,
    }
    summary = {
        "protocol": "LSH 3 split seeds x 3 paired training seeds; validation-locked Youden thresholds",
        "fusion": "fixed 0.5/0.5 equal-logit; no learned reliability weights",
        "run_count": 9,
        "observed": {
            "mean_auc_delta": float(delta_frame.auc_delta.mean()), "sd_auc_delta": float(delta_frame.auc_delta.std(ddof=1)),
            "auc_positive_runs": int((delta_frame.auc_delta > 0).sum()),
            "mean_nll_delta": float(delta_frame.nll_delta.mean()), "nll_nonworse_runs": int((delta_frame.nll_delta <= 0).sum()),
            "mean_locked_sensitivity_delta": float(delta_frame.sensitivity_delta_locked.mean()),
            "mean_locked_specificity_delta": float(delta_frame.specificity_delta_locked.mean()),
            "mean_pearson": float(delta_frame.pearson_r3d_triplanar.mean()),
            "total_corrected": int(delta_frame.corrected_r3d_errors.sum()), "total_harmed": int(delta_frame.harmed_r3d_correct.sum()),
        },
        "gates": gates, "may_proceed_to_tdsc": bool(all(gates.values())),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_frame.to_csv(args.output_dir / "per_run_metrics.csv", index=False)
    delta_frame.to_csv(args.output_dir / "per_run_deltas.csv", index=False)
    paired_frame.to_csv(args.output_dir / "paired_predictions.csv", index=False)
    bootstrap.to_csv(args.output_dir / "hierarchical_bootstrap_ci.csv", index=False)
    (args.output_dir / "decision.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
