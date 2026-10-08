#!/usr/bin/env python3
"""Summarize paired R3D + tri-planar repeated five-fold TDSC CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, matthews_corrcoef, roc_auc_score

try:
    from scripts.analysis.summarize_triplanar_stability import (
        EPS,
        METHODS,
        pair_predictions,
        youden_threshold,
    )
except ModuleNotFoundError:  # direct execution via python scripts/analysis/...
    from summarize_triplanar_stability import EPS, METHODS, pair_predictions, youden_threshold


SPLIT_SEEDS = (3407, 42, 1234)
FOLDS = (1, 2, 3, 4, 5)
TRAIN_SEED = 3407
BOOTSTRAP_METRICS = ("auc_delta", "nll_delta", "sensitivity_delta", "specificity_delta")


def metrics_with_predictions(labels: np.ndarray, probability: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    positive = labels == 1
    negative = labels == 0
    return {
        "auc": float(roc_auc_score(labels, probability)),
        "nll": float(log_loss(labels, np.clip(probability, EPS, 1 - EPS), labels=[0, 1])),
        "accuracy": float(accuracy_score(labels, predicted)),
        "sensitivity": float(np.mean(predicted[positive])),
        "specificity": float(np.mean(~predicted[negative])),
        "f1_positive": float(f1_score(labels, predicted)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
    }


def _prediction_path(result_root: Path, model: str, split_seed: int, fold: int, partition: str) -> Path:
    return (
        result_root
        / model
        / f"split_seed_{split_seed}"
        / f"fold_{fold}"
        / f"train_seed_{TRAIN_SEED}"
        / f"{partition}_predictions.csv"
    )


def analyze_fold(result_root: Path, split_seed: int, fold: int) -> tuple[list[dict[str, float]], pd.DataFrame]:
    validation = pair_predictions(
        _prediction_path(result_root, "r3d", split_seed, fold, "selection_val"),
        _prediction_path(result_root, "convnext_triplanar_25d_mil", split_seed, fold, "selection_val"),
    )
    outer = pair_predictions(
        _prediction_path(result_root, "r3d", split_seed, fold, "outer_test"),
        _prediction_path(result_root, "convnext_triplanar_25d_mil", split_seed, fold, "outer_test"),
    )
    labels_val = validation.label_true.to_numpy(int)
    thresholds = {
        method: youden_threshold(labels_val, validation[f"prob_{method}"].to_numpy(float))
        for method in METHODS
    }
    labels = outer.label_true.to_numpy(int)
    rows = []
    for method in METHODS:
        probability = outer[f"prob_{method}"].to_numpy(float)
        predicted = probability >= thresholds[method]
        row: dict[str, float | int | str] = {
            "split_seed": split_seed,
            "fold": fold,
            "method": method,
            "threshold": thresholds[method],
            "n": len(outer),
        }
        row.update(metrics_with_predictions(labels, probability, predicted))
        rows.append(row)
        outer[f"pred_{method}"] = predicted
        outer[f"threshold_{method}"] = thresholds[method]
    outer["split_seed"] = split_seed
    outer["fold"] = fold
    return rows, outer


def repeat_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_seed, repeat in frame.groupby("split_seed", sort=True):
        labels = repeat.label_true.to_numpy(int)
        for method in METHODS:
            row: dict[str, float | int | str] = {
                "split_seed": int(split_seed),
                "method": method,
                "n": len(repeat),
                "folds": int(repeat.fold.nunique()),
            }
            row.update(
                metrics_with_predictions(
                    labels,
                    repeat[f"prob_{method}"].to_numpy(float),
                    repeat[f"pred_{method}"].to_numpy(bool),
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def repeat_deltas(repeats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ("auc", "nll", "accuracy", "sensitivity", "specificity", "f1_positive", "mcc")
    for split_seed, frame in repeats.groupby("split_seed", sort=True):
        indexed = frame.set_index("method")
        row = {"split_seed": int(split_seed)}
        for metric in metrics:
            row[f"{metric}_delta"] = float(indexed.loc["equal_logit", metric] - indexed.loc["r3d", metric])
        rows.append(row)
    return pd.DataFrame(rows)


def _repeat_delta(frame: pd.DataFrame, baseline: str = "r3d") -> dict[str, float]:
    labels = frame.label_true.to_numpy(int)
    baseline_result = metrics_with_predictions(
        labels,
        frame[f"prob_{baseline}"].to_numpy(float),
        frame[f"pred_{baseline}"].to_numpy(bool),
    )
    fusion = metrics_with_predictions(
        labels,
        frame.prob_equal_logit.to_numpy(float),
        frame.pred_equal_logit.to_numpy(bool),
    )
    return {
        "auc_delta": fusion["auc"] - baseline_result["auc"],
        "nll_delta": fusion["nll"] - baseline_result["nll"],
        "sensitivity_delta": fusion["sensitivity"] - baseline_result["sensitivity"],
        "specificity_delta": fusion["specificity"] - baseline_result["specificity"],
    }


def _validate_repeated_case_frame(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[int, pd.DataFrame]]:
    labels_by_case = frame.drop_duplicates("case_id").set_index("case_id")["label_true"]
    negative = labels_by_case.index[labels_by_case.eq(0)].to_numpy(object)
    positive = labels_by_case.index[labels_by_case.eq(1)].to_numpy(object)
    indexed = {
        int(split_seed): repeat.set_index("case_id")
        for split_seed, repeat in frame.groupby("split_seed", sort=True)
    }
    expected_cases = set(labels_by_case.index)
    for split_seed, repeat in indexed.items():
        if repeat.index.has_duplicates or set(repeat.index) != expected_cases:
            raise ValueError(f"Repeat {split_seed} does not contain each case exactly once")
    return negative, positive, indexed


def _bootstrap_rows(
    samples: dict[str, np.ndarray],
    observed: dict[str, float],
    *,
    contrast: str,
    bootstrap_unit: str,
    replicates: int,
) -> pd.DataFrame:
    rows = []
    for metric, values in samples.items():
        favorable = values < 0 if metric == "nll_delta" else values > 0
        lower, upper = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "contrast": contrast,
                "metric": metric,
                "observed_delta": observed[metric],
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "probability_favorable": float(np.mean(favorable)),
                "bootstrap_unit": bootstrap_unit,
                "replicates": replicates,
            }
        )
    return pd.DataFrame(rows)


def paired_case_bootstrap(
    frame: pd.DataFrame,
    replicates: int,
    seed: int,
    baseline: str = "r3d",
) -> pd.DataFrame:
    negative, positive, indexed = _validate_repeated_case_frame(frame)
    rng = np.random.default_rng(seed)
    samples = {metric: np.empty(replicates, float) for metric in BOOTSTRAP_METRICS}
    for iteration in range(replicates):
        sampled = np.concatenate(
            (
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            )
        )
        deltas = [
            _repeat_delta(repeat.loc[sampled].reset_index(), baseline=baseline)
            for repeat in indexed.values()
        ]
        for metric in BOOTSTRAP_METRICS:
            samples[metric][iteration] = float(np.mean([delta[metric] for delta in deltas]))
    observed = {
        metric: float(
            np.mean([_repeat_delta(repeat, baseline=baseline)[metric] for repeat in indexed.values()])
        )
        for metric in BOOTSTRAP_METRICS
    }
    return _bootstrap_rows(
        samples,
        observed,
        contrast=f"equal_logit minus {baseline}",
        bootstrap_unit="stratified unique case; all repeat predictions retained",
        replicates=replicates,
    )


def hierarchical_split_case_bootstrap(
    frame: pd.DataFrame,
    replicates: int,
    seed: int,
    baseline: str = "r3d",
) -> pd.DataFrame:
    """Resample split seeds and stratified cases as a conservative sensitivity analysis."""
    negative, positive, indexed = _validate_repeated_case_frame(frame)
    split_seeds = np.asarray(sorted(indexed), dtype=int)
    rng = np.random.default_rng(seed)
    samples = {metric: np.empty(replicates, float) for metric in BOOTSTRAP_METRICS}
    for iteration in range(replicates):
        sampled_cases = np.concatenate(
            (
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            )
        )
        sampled_splits = rng.choice(split_seeds, len(split_seeds), replace=True)
        deltas = [
            _repeat_delta(
                indexed[int(split)].loc[sampled_cases].reset_index(),
                baseline=baseline,
            )
            for split in sampled_splits
        ]
        for metric in BOOTSTRAP_METRICS:
            samples[metric][iteration] = float(np.mean([delta[metric] for delta in deltas]))
    observed = {
        metric: float(
            np.mean([_repeat_delta(repeat, baseline=baseline)[metric] for repeat in indexed.values()])
        )
        for metric in BOOTSTRAP_METRICS
    }
    return _bootstrap_rows(
        samples,
        observed,
        contrast=f"equal_logit minus {baseline}",
        bootstrap_unit="split seed plus stratified unique case",
        replicates=replicates,
    )


def fixed_threshold_repeat_metrics(frame: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    fixed = frame.copy()
    for method in METHODS:
        fixed[f"pred_{method}"] = fixed[f"prob_{method}"].to_numpy(float) >= threshold
    result = repeat_metrics(fixed)
    result.insert(2, "threshold", float(threshold))
    return result


def model_summary(repeats: pd.DataFrame) -> pd.DataFrame:
    metric_names = ("auc", "nll", "accuracy", "sensitivity", "specificity", "f1_positive", "mcc")
    rows = []
    for method, frame in repeats.groupby("method", sort=True):
        row: dict[str, float | int | str] = {"method": method, "repeat_n": len(frame)}
        for metric in metric_names:
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_sd"] = float(frame[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    args = parser.parse_args()

    fold_rows: list[dict[str, float]] = []
    paired = []
    for split_seed in SPLIT_SEEDS:
        for fold in FOLDS:
            metrics_rows, predictions = analyze_fold(args.result_root, split_seed, fold)
            fold_rows.extend(metrics_rows)
            paired.append(predictions)
    paired_frame = pd.concat(paired, ignore_index=True)
    repeat_frame = repeat_metrics(paired_frame)
    delta_frame = repeat_deltas(repeat_frame)
    case_bootstraps = pd.concat(
        [
            paired_case_bootstrap(
                paired_frame,
                args.bootstrap_repeats,
                args.bootstrap_seed + offset,
                baseline=baseline,
            )
            for offset, baseline in enumerate(("r3d", "triplanar"))
        ],
        ignore_index=True,
    )
    hierarchical_bootstraps = pd.concat(
        [
            hierarchical_split_case_bootstrap(
                paired_frame,
                args.bootstrap_repeats,
                args.bootstrap_seed + 100 + offset,
                baseline=baseline,
            )
            for offset, baseline in enumerate(("r3d", "triplanar"))
        ],
        ignore_index=True,
    )
    fixed_repeat_frame = fixed_threshold_repeat_metrics(paired_frame)
    fixed_summary_frame = model_summary(fixed_repeat_frame)
    fixed_paired_frame = paired_frame.copy()
    for method in METHODS:
        fixed_paired_frame[f"pred_{method}"] = (
            fixed_paired_frame[f"prob_{method}"].to_numpy(float) >= 0.5
        )
    fixed_hierarchical_bootstraps = pd.concat(
        [
            hierarchical_split_case_bootstrap(
                fixed_paired_frame,
                args.bootstrap_repeats,
                args.bootstrap_seed + 200 + offset,
                baseline=baseline,
            )
            for offset, baseline in enumerate(("r3d", "triplanar"))
        ],
        ignore_index=True,
    )
    summary_frame = model_summary(repeat_frame)

    r3d_case = case_bootstraps[
        case_bootstraps.contrast.eq("equal_logit minus r3d")
    ].set_index("metric")
    auc_ci = r3d_case.loc["auc_delta"]
    nll_ci = r3d_case.loc["nll_delta"]
    hierarchical_indexed = hierarchical_bootstraps.set_index(["contrast", "metric"])
    gates = {
        "auc_positive_at_least_2_of_3_repeats": int((delta_frame.auc_delta > 0).sum()) >= 2,
        "case_conditioned_auc_noninferiority_ci_lower_above_minus_0p01": float(auc_ci.ci95_lower) > -0.01,
        "case_conditioned_auc_superiority_ci_lower_above_zero": float(auc_ci.ci95_lower) > 0,
        "nll_nonworse_at_least_2_of_3_repeats": int((delta_frame.nll_delta <= 0).sum()) >= 2,
        "case_conditioned_nll_improvement_ci_upper_below_zero": float(nll_ci.ci95_upper) < 0,
        "mean_specificity_delta_at_least_minus_0p02": float(delta_frame.specificity_delta.mean()) >= -0.02,
    }
    conservative_r3d_auc = hierarchical_indexed.loc[("equal_logit minus r3d", "auc_delta")]
    conservative_r3d_nll = hierarchical_indexed.loc[("equal_logit minus r3d", "nll_delta")]
    conservative_tri_auc = hierarchical_indexed.loc[("equal_logit minus triplanar", "auc_delta")]
    conservative_tri_nll = hierarchical_indexed.loc[("equal_logit minus triplanar", "nll_delta")]
    decision = {
        "protocol": "TDSC 3 repeated stratified 5-fold case-level CV; fold-specific selection-validation Youden thresholds",
        "fusion": "fixed 0.5/0.5 equal-logit; no learned gate, temperature, or test-set tuning",
        "train_seed": TRAIN_SEED,
        "outer_test_predictions": int(len(paired_frame)),
        "unique_cases": int(paired_frame.case_id.nunique()),
        "observed": {
            "mean_auc_delta": float(delta_frame.auc_delta.mean()),
            "mean_nll_delta": float(delta_frame.nll_delta.mean()),
            "mean_sensitivity_delta": float(delta_frame.sensitivity_delta.mean()),
            "mean_specificity_delta": float(delta_frame.specificity_delta.mean()),
            "auc_positive_repeats": int((delta_frame.auc_delta > 0).sum()),
            "nll_nonworse_repeats": int((delta_frame.nll_delta <= 0).sum()),
        },
        "gates": gates,
        "interpretation": {
            "case_conditioned_auc_superior_to_r3d": gates[
                "case_conditioned_auc_superiority_ci_lower_above_zero"
            ],
            "case_conditioned_auc_noninferior_to_r3d": gates[
                "case_conditioned_auc_noninferiority_ci_lower_above_minus_0p01"
            ],
            "case_conditioned_nll_improved_over_r3d": gates[
                "case_conditioned_nll_improvement_ci_upper_below_zero"
            ],
            "conservative_hierarchical_auc_superior_to_r3d": bool(
                conservative_r3d_auc.ci95_lower > 0
            ),
            "conservative_hierarchical_nll_improved_over_r3d": bool(
                conservative_r3d_nll.ci95_upper < 0
            ),
            "conservative_hierarchical_auc_superior_to_triplanar": bool(
                conservative_tri_auc.ci95_lower > 0
            ),
            "conservative_hierarchical_nll_improved_over_triplanar": bool(
                conservative_tri_nll.ci95_upper < 0
            ),
            "ready_for_manuscript_revision": bool(
                gates["auc_positive_at_least_2_of_3_repeats"]
                and gates["case_conditioned_auc_noninferiority_ci_lower_above_minus_0p01"]
                and gates["nll_nonworse_at_least_2_of_3_repeats"]
                and gates["mean_specificity_delta_at_least_minus_0p02"]
            ),
            "claim_boundary": (
                "Use split-seed-plus-case hierarchical intervals for confirmatory "
                "wording; case-conditioned intervals are sensitivity analyses."
            ),
        },
        "sensitivity_analysis": {
            "hierarchical_auc_ci_vs_r3d": [
                float(hierarchical_indexed.loc[("equal_logit minus r3d", "auc_delta"), "ci95_lower"]),
                float(hierarchical_indexed.loc[("equal_logit minus r3d", "auc_delta"), "ci95_upper"]),
            ],
            "hierarchical_auc_ci_vs_triplanar": [
                float(
                    hierarchical_indexed.loc[
                        ("equal_logit minus triplanar", "auc_delta"), "ci95_lower"
                    ]
                ),
                float(
                    hierarchical_indexed.loc[
                        ("equal_logit minus triplanar", "auc_delta"), "ci95_upper"
                    ]
                ),
            ],
            "hierarchical_nll_ci_vs_r3d": [
                float(hierarchical_indexed.loc[("equal_logit minus r3d", "nll_delta"), "ci95_lower"]),
                float(hierarchical_indexed.loc[("equal_logit minus r3d", "nll_delta"), "ci95_upper"]),
            ],
            "hierarchical_nll_ci_vs_triplanar": [
                float(
                    hierarchical_indexed.loc[
                        ("equal_logit minus triplanar", "nll_delta"), "ci95_lower"
                    ]
                ),
                float(
                    hierarchical_indexed.loc[
                        ("equal_logit minus triplanar", "nll_delta"), "ci95_upper"
                    ]
                ),
            ],
            "note": "Hierarchical split+case intervals are the conservative claim boundary.",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "per_fold_metrics.csv", index=False)
    paired_frame.to_csv(args.output_dir / "paired_oof_predictions.csv", index=False)
    repeat_frame.to_csv(args.output_dir / "repeat_oof_metrics.csv", index=False)
    delta_frame.to_csv(args.output_dir / "paired_repeat_deltas.csv", index=False)
    summary_frame.to_csv(args.output_dir / "model_repeat_summary.csv", index=False)
    case_bootstraps.to_csv(args.output_dir / "paired_case_cluster_bootstrap.csv", index=False)
    hierarchical_bootstraps.to_csv(
        args.output_dir / "hierarchical_split_case_bootstrap.csv", index=False
    )
    fixed_repeat_frame.to_csv(args.output_dir / "fixed_0p5_repeat_metrics.csv", index=False)
    fixed_summary_frame.to_csv(args.output_dir / "fixed_0p5_model_summary.csv", index=False)
    fixed_hierarchical_bootstraps.to_csv(
        args.output_dir / "fixed_0p5_hierarchical_split_case_bootstrap.csv",
        index=False,
    )
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
