#!/usr/bin/env python3
"""Compare matched-protocol R(2+1)D with the locked dual-expert system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.summarize_triplanar_stability import youden_threshold  # noqa: E402
from scripts.analysis.summarize_triplanar_tdsc_cv import metrics_with_predictions  # noqa: E402


SPLIT_SEEDS = (3407, 42, 1234)
FOLDS = (1, 2, 3, 4, 5)
METHODS = ("r2plus1d", "r3d", "triplanar", "equal_logit")
CONTRAST_METHODS = ("r3d", "triplanar", "equal_logit")
METRICS = ("auc", "nll", "accuracy", "sensitivity", "specificity", "f1_positive", "mcc")


def case_id(value: object) -> str:
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    return "/".join(parts[-4:])


def load_r2_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"npy_path", "label_true", "prob_1"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks {sorted(missing)}")
    result = pd.DataFrame(
        {
            "case_id": frame["npy_path"].map(case_id),
            "label_true": pd.to_numeric(frame["label_true"], errors="coerce"),
            "prob_r2plus1d": pd.to_numeric(frame["prob_1"], errors="coerce"),
        }
    )
    if result.isna().any().any() or result.case_id.duplicated().any():
        raise ValueError(f"{path} has missing, nonnumeric, or duplicate rows")
    return result


def r2_path(root: Path, dataset: str, split_seed: int, fold: int, partition: str) -> Path:
    return (
        root / dataset / "r2plus1d" / f"split_seed_{split_seed}" / f"fold_{fold}"
        / "train_seed_3407" / f"{partition}_predictions.csv"
    )


def merge_checked(reference: pd.DataFrame, r2: pd.DataFrame, context: str) -> pd.DataFrame:
    merged = reference.merge(r2, on="case_id", how="outer", suffixes=("", "_r2"), validate="one_to_one")
    if merged.isna().any().any() or len(merged) != len(reference) or len(merged) != len(r2):
        raise ValueError(f"Case sets differ for {context}")
    if not np.array_equal(merged.label_true.to_numpy(int), merged.label_true_r2.to_numpy(int)):
        raise ValueError(f"Labels differ for {context}")
    return merged.drop(columns="label_true_r2")


def prepare_tdsc(paired_path: Path, root: Path) -> pd.DataFrame:
    paired = pd.read_csv(paired_path)
    runs = []
    for split_seed in SPLIT_SEEDS:
        for fold in FOLDS:
            validation = load_r2_prediction(r2_path(root, "tdsc", split_seed, fold, "selection_val"))
            threshold = youden_threshold(
                validation.label_true.to_numpy(int), validation.prob_r2plus1d.to_numpy(float)
            )
            outer = load_r2_prediction(r2_path(root, "tdsc", split_seed, fold, "outer_test"))
            reference = paired.query("split_seed == @split_seed and fold == @fold").copy()
            merged = merge_checked(reference, outer, f"TDSC split={split_seed} fold={fold}")
            merged["threshold_r2plus1d"] = threshold
            merged["pred_r2plus1d"] = merged.prob_r2plus1d >= threshold
            runs.append(merged)
    return pd.concat(runs, ignore_index=True)


def prepare_lsh(paired_path: Path, root: Path) -> pd.DataFrame:
    paired = pd.read_csv(paired_path).query("train_seed == 3407").copy()
    if "threshold_triplanar" not in paired.columns:
        raise ValueError(f"{paired_path} lacks validation-locked tri-planar thresholds")
    runs = []
    for split_seed in SPLIT_SEEDS:
        validation = load_r2_prediction(r2_path(root, "lsh", split_seed, 1, "selection_val"))
        threshold = youden_threshold(
            validation.label_true.to_numpy(int), validation.prob_r2plus1d.to_numpy(float)
        )
        outer = load_r2_prediction(r2_path(root, "lsh", split_seed, 1, "outer_test"))
        reference = paired.query("split_seed == @split_seed").copy()
        merged = merge_checked(reference, outer, f"LSH split={split_seed}")
        merged["threshold_r2plus1d"] = threshold
        merged["pred_r2plus1d"] = merged.prob_r2plus1d >= threshold
        for method in ("r3d", "equal_logit"):
            merged[f"pred_{method}"] = merged[f"prob_{method}"] >= merged[f"threshold_{method}"]
        # The paired LSH stability file stores the Youden threshold selected
        # on each run's selection-validation cases. Preserve that locked value.
        tri_thresholds = pd.to_numeric(merged["threshold_triplanar"], errors="coerce")
        if (
            not np.isfinite(tri_thresholds).all()
            or not tri_thresholds.between(0.0, 1.0).all()
            or tri_thresholds.nunique() != 1
        ):
            raise ValueError(f"Invalid tri-planar threshold for LSH split={split_seed}")
        merged["pred_triplanar"] = merged.prob_triplanar >= float(tri_thresholds.iloc[0])
        runs.append(merged)
    return pd.concat(runs, ignore_index=True)


def per_repeat_metrics(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for split_seed, repeat in frame.groupby("split_seed", sort=True):
        labels = repeat.label_true.to_numpy(int)
        for method in METHODS:
            row = {
                "dataset": dataset,
                "split_seed": int(split_seed),
                "method": method,
                "n": len(repeat),
                "threshold_policy": "selection_validation_locked_youden",
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


def aggregate_metrics(per_repeat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, method), group in per_repeat.groupby(["dataset", "method"], sort=True):
        for metric_name in METRICS:
            values = group[metric_name].to_numpy(float)
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "metric": metric_name,
                    "repeat_count": len(values),
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def _auc_nll(
    labels: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float]:
    from sklearn.metrics import log_loss, roc_auc_score

    return (
        float(roc_auc_score(labels, probability, sample_weight=sample_weight)),
        float(
            log_loss(
                labels,
                np.clip(probability, 1e-7, 1 - 1e-7),
                labels=[0, 1],
                sample_weight=sample_weight,
            )
        ),
    )


def cluster_labels(frame: pd.DataFrame) -> pd.Series:
    """Return one label per cluster and reject cross-split label mismatches."""
    label_counts = frame.groupby("case_id", sort=False).label_true.nunique(dropna=False)
    mismatched = label_counts.index[label_counts.ne(1)].tolist()
    if mismatched:
        raise ValueError(f"Cluster labels differ across repeats: {mismatched[:5]}")
    return frame.drop_duplicates("case_id").set_index("case_id").label_true.astype(int)


def stratified_cluster_weights(
    labels_by_cluster: pd.Series,
    rng: np.random.Generator,
) -> pd.Series:
    """Draw one stratified bootstrap weight for every cluster in the union."""
    negative = labels_by_cluster.index[labels_by_cluster.eq(0)].to_numpy(object)
    positive = labels_by_cluster.index[labels_by_cluster.eq(1)].to_numpy(object)
    if not len(negative) or not len(positive):
        raise ValueError("Both classes are required for stratified cluster bootstrap")
    sampled = np.concatenate(
        (
            rng.choice(negative, len(negative), replace=True),
            rng.choice(positive, len(positive), replace=True),
        )
    )
    clusters, counts = np.unique(sampled, return_counts=True)
    return pd.Series(counts.astype(float), index=clusters, name="cluster_weight")


def map_cluster_weights(run: pd.DataFrame, cluster_weights: pd.Series) -> np.ndarray:
    """Map a union-level draw while preserving a cluster's cross-split weight."""
    return run.index.to_series().map(cluster_weights).fillna(0.0).to_numpy(float)


def bootstrap_contrasts(
    frame: pd.DataFrame,
    dataset: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    labels_by_case = cluster_labels(frame)
    runs = {int(key): value.set_index("case_id") for key, value in frame.groupby("split_seed")}
    if any(run.index.has_duplicates for run in runs.values()):
        raise ValueError(f"{dataset} repeats contain duplicate cases")
    cluster_kind = "patient" if dataset == "lsh" else "case"
    units = (
        f"stratified_{cluster_kind}_cluster",
        f"split_seed_and_stratified_{cluster_kind}_cluster",
    )
    rng = np.random.default_rng(seed)
    samples = {
        (method, metric_name, unit): np.empty(replicates)
        for method in CONTRAST_METHODS
        for metric_name in ("auc_delta", "nll_delta")
        for unit in units
    }
    split_seeds = np.asarray(sorted(runs))

    observed = {method: {"auc_delta": [], "nll_delta": []} for method in CONTRAST_METHODS}
    for run in runs.values():
        labels = run.label_true.to_numpy(int)
        r2_auc, r2_nll = _auc_nll(labels, run.prob_r2plus1d.to_numpy(float))
        for method in CONTRAST_METHODS:
            auc, nll = _auc_nll(labels, run[f"prob_{method}"].to_numpy(float))
            observed[method]["auc_delta"].append(auc - r2_auc)
            observed[method]["nll_delta"].append(nll - r2_nll)

    for iteration in range(replicates):
        cluster_weights = stratified_cluster_weights(labels_by_case, rng)
        for unit in units:
            resample_splits = unit.startswith("split_seed_and_")
            selected_splits = (
                rng.choice(split_seeds, len(split_seeds), replace=True)
                if resample_splits
                else split_seeds
            )
            deltas = {method: {"auc_delta": [], "nll_delta": []} for method in CONTRAST_METHODS}
            for split_seed in selected_splits:
                run = runs[int(split_seed)]
                weights = map_cluster_weights(run, cluster_weights)
                keep = weights > 0
                labels = run.label_true.to_numpy(int)[keep]
                weights = weights[keep]
                if len(labels) == 0 or np.unique(labels).size != 2:
                    continue
                r2_auc, r2_nll = _auc_nll(
                    labels,
                    run.prob_r2plus1d.to_numpy(float)[keep],
                    sample_weight=weights,
                )
                for method in CONTRAST_METHODS:
                    auc, nll = _auc_nll(
                        labels,
                        run[f"prob_{method}"].to_numpy(float)[keep],
                        sample_weight=weights,
                    )
                    deltas[method]["auc_delta"].append(auc - r2_auc)
                    deltas[method]["nll_delta"].append(nll - r2_nll)
            for method in CONTRAST_METHODS:
                for metric_name in ("auc_delta", "nll_delta"):
                    values = deltas[method][metric_name]
                    if not values:
                        raise ValueError(
                            f"Bootstrap replicate {iteration} has no valid "
                            f"two-class split for {unit}"
                        )
                    samples[(method, metric_name, unit)][iteration] = np.mean(values)

    rows = []
    for (method, metric_name, unit), values in samples.items():
        lower, upper = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "dataset": dataset,
                "contrast": f"{method} minus r2plus1d",
                "metric": metric_name,
                "bootstrap_unit": unit,
                "cluster_scope": f"union_of_outer_test_{cluster_kind}s",
                "split_seed_resampled": unit.startswith("split_seed_and_"),
                "replicates": replicates,
                "observed_delta": float(np.mean(observed[method][metric_name])),
                "mean_delta": float(np.mean(values)),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "probability_favorable": float(
                    np.mean(values > 0 if metric_name == "auc_delta" else values < 0)
                ),
            }
        )
    result = pd.DataFrame(rows)
    expected = {
        (f"{method} minus r2plus1d", metric, unit)
        for method in CONTRAST_METHODS
        for metric in ("auc_delta", "nll_delta")
        for unit in units
    }
    actual = set(zip(result.contrast, result.metric, result.bootstrap_unit))
    if actual != expected or result.duplicated(["contrast", "metric", "bootstrap_unit"]).any():
        raise RuntimeError(f"Incomplete or duplicate bootstrap contrasts for {dataset}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tdsc-paired", type=Path, required=True)
    parser.add_argument(
        "--lsh-paired",
        type=Path,
        help="Optional LSH paired predictions; omit for public-TDSC-only analysis.",
    )
    parser.add_argument("--r2-result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260912)
    return parser.parse_args()


def summarize_prepared_datasets(
    datasets: dict[str, pd.DataFrame],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the three output tables for full or public-TDSC-only analysis."""
    if not datasets:
        raise ValueError("At least one prepared dataset is required")
    unknown = set(datasets).difference({"tdsc", "lsh"})
    if unknown:
        raise ValueError(f"Unsupported datasets: {sorted(unknown)}")
    ordered = [(name, datasets[name]) for name in ("tdsc", "lsh") if name in datasets]
    per_repeat = pd.concat(
        [per_repeat_metrics(frame, name) for name, frame in ordered],
        ignore_index=True,
    )
    bootstrap = pd.concat(
        [
            bootstrap_contrasts(
                frame,
                name,
                bootstrap_replicates,
                bootstrap_seed + (1 if name == "lsh" else 0),
            )
            for name, frame in ordered
        ],
        ignore_index=True,
    )
    return per_repeat, aggregate_metrics(per_repeat), bootstrap


def main() -> None:
    args = parse_args()
    datasets = {"tdsc": prepare_tdsc(args.tdsc_paired, args.r2_result_root)}
    if args.lsh_paired is not None:
        datasets["lsh"] = prepare_lsh(args.lsh_paired, args.r2_result_root)
    per_repeat, aggregate, bootstrap = summarize_prepared_datasets(
        datasets,
        args.bootstrap_replicates,
        args.bootstrap_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_repeat.to_csv(args.output_dir / "metrics_by_repeat.csv", index=False)
    aggregate.to_csv(args.output_dir / "metrics_summary.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap_contrasts.csv", index=False)
    metadata = {
        "datasets": list(datasets),
        "protocol": (
            "Matched to final R3D: identical splits, train seed 3407, batch 28, "
            "X-Y flip-only augmentation with Z order preserved, AUC-only "
            "checkpoint selection."
        ),
        "tdsc": (
            "Three repeated five-fold CV; fold-specific selection-validation "
            "Youden thresholds."
        ),
        "tdsc_bootstrap_case_scope": (
            "The union of cases appearing in at least one outer-test partition."
        ),
        "bootstrap_cluster_policy": (
            "Each replicate draws one label-stratified cluster-weight vector "
            "from the relevant outer-test union (cases for TDSC and patients "
            "for LSH) and maps the same cluster weight to every split in which "
            "that cluster appears."
        ),
        "bootstrap_split_policy": (
            "The cluster analysis keeps all split seeds once; the "
            "hierarchical sensitivity analysis additionally resamples split "
            "seeds with replacement while retaining the shared cluster weights."
        ),
        "multiplicity": (
            f"No multiplicity adjustment; all {3 * len(datasets)} "
            "dataset-by-reference contrasts "
            "and both AUC/NLL intervals are exploratory."
        ),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
    }
    if "lsh" in datasets:
        metadata["lsh"] = (
            "Three repeated holdouts; all thresholded metrics use the "
            "selection-validation-locked Youden thresholds stored in the "
            "paired stability predictions, including tri-planar MIL."
        )
        metadata["lsh_bootstrap_patient_scope"] = (
            "The union of patients appearing in at least one outer-test partition."
        )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
