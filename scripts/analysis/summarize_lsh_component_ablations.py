#!/usr/bin/env python3
"""Locked LSH component ablations with patient-clustered paired uncertainty.

Nine runs reuse three private holdouts and three optimization seeds. The
source cohort supplies one case per patient; patient identities are not
independently reconstructed from image metadata. Only aggregate artifacts
are exported. Intervals condition on the historically selected architecture.
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
    from scripts.analysis.summarize_tdsc_component_ablations import (
        EPS, FULL_MODEL, METRICS, VARIANTS, aligned_prediction,
        fuse_probabilities, load_split, planned_contrasts, sha256,
        weighted_auc_many,
    )
    from scripts.analysis.summarize_triplanar_stability import metrics, youden_threshold
except ModuleNotFoundError:
    from summarize_tdsc_component_ablations import (
        EPS, FULL_MODEL, METRICS, VARIANTS, aligned_prediction,
        fuse_probabilities, load_split, planned_contrasts, sha256,
        weighted_auc_many,
    )
    from summarize_triplanar_stability import metrics, youden_threshold


SPLIT_SEEDS = (3407, 42, 1234)
TRAIN_SEEDS = (3407, 42, 1234)
ROLE_COUNTS = {"train": 415, "selection_val": 60, "outer_test": 119}


def prediction_path(root: Path, model: str, split_seed: int, train_seed: int, role: str) -> Path:
    return root / model / f"split_seed_{split_seed}" / "fold_1" / f"train_seed_{train_seed}" / f"{role}_predictions.csv"


def load_protocol(root: Path) -> tuple[dict[tuple[int, str], pd.DataFrame], list[dict]]:
    roles, hashes = {}, []
    canonical = None
    for split_seed in SPLIT_SEEDS:
        per_split = []
        for role, count in ROLE_COUNTS.items():
            path = root / "splits" / f"split_seed_{split_seed}" / "fold_1" / f"{role}.csv"
            value = load_split(path, count)
            roles[(split_seed, role)] = value
            hashes.append({"split_seed": split_seed, "role": role, "sha256": sha256(path)})
            per_split.append(value)
        joined = pd.concat(per_split, ignore_index=True).sort_values("case_id").reset_index(drop=True)
        if len(joined) != 594 or joined.case_id.duplicated().any():
            raise ValueError("LSH roles overlap or do not span 594 unique source patient-case records")
        if canonical is None:
            canonical = joined
        elif not joined.equals(canonical):
            raise ValueError("LSH source patient-case sets or labels vary across splits")
    return roles, hashes


def analyze_all(
    roles: dict[tuple[int, str], pd.DataFrame], r3d_root: Path,
    full_root: Path, variant_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[tuple[int, int], pd.DataFrame], list[dict]]:
    metric_rows, runs, hashes = [], {}, []
    variants = VARIANTS if variant_root is not None else ()
    for split_seed in SPLIT_SEEDS:
        for train_seed in TRAIN_SEEDS:
            partitions = {}
            for role in ("selection_val", "outer_test"):
                expected = roles[(split_seed, role)]
                paired = expected.copy()
                sources = [("r3d", r3d_root, "r3d"), ("trimil_full", full_root, FULL_MODEL)]
                sources += [(f"trimil_{name}", variant_root, f"{FULL_MODEL}_{name}") for name in variants]
                for name, root, model in sources:
                    observed, record = aligned_prediction(
                        prediction_path(root, model, split_seed, train_seed, role), expected, name,
                    )
                    paired[f"prob_{name}"] = observed[f"prob_{name}"].to_numpy(float)
                    hashes.append({"split_seed": split_seed, "train_seed": train_seed, "role": role, "method": name, **record})
                for name in ("full", *variants):
                    paired[f"prob_fusion_{name}"] = fuse_probabilities(
                        paired.prob_r3d.to_numpy(float), paired[f"prob_trimil_{name}"].to_numpy(float),
                    )
                partitions[role] = paired
            validation, outer = partitions["selection_val"], partitions["outer_test"]
            for column in (value for value in outer.columns if value.startswith("prob_")):
                method = column.removeprefix("prob_")
                threshold = youden_threshold(validation.label_true.to_numpy(int), validation[column].to_numpy(float))
                outer[f"threshold_{method}"] = threshold
                metric_rows.append({
                    "split_seed": split_seed, "train_seed": train_seed, "method": method, "n": len(outer),
                    **metrics(outer.label_true.to_numpy(int), outer[column].to_numpy(float), threshold),
                })
            runs[(split_seed, train_seed)] = outer
    return pd.DataFrame(metric_rows), runs, hashes


def model_summary(per_run: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in per_run.groupby("method", sort=True):
        rows.append({"method": method, "run_n": len(group), **{
            f"{metric}_{stat}": float(group[metric].mean() if stat == "mean" else group[metric].std(ddof=1))
            for metric in METRICS for stat in ("mean", "sd")
        }})
    return pd.DataFrame(rows)


def assert_full_baseline(current: pd.DataFrame, expected_path: Path, tolerance: float = 1e-10) -> None:
    """Verify the actual archived per-run file, not a nonexistent summary."""
    expected = pd.read_csv(expected_path)
    current = current.copy()
    current.method = current.method.replace({"trimil_full": "triplanar", "fusion_full": "equal_logit"})
    methods = {"r3d", "triplanar", "equal_logit"}
    keys = ["split_seed", "train_seed", "method"]
    required = set(keys) | set(METRICS) | {"threshold"}
    if not required.issubset(expected) or not required.issubset(current):
        raise ValueError("Baseline per-run metrics lacks required locked metrics/thresholds")
    expected = expected[expected.method.isin(methods)].sort_values(keys).reset_index(drop=True)
    current = current[current.method.isin(methods)].sort_values(keys).reset_index(drop=True)
    expected_keys = {(split, train, method) for split in SPLIT_SEEDS for train in TRAIN_SEEDS for method in methods}
    for frame in (expected, current):
        if len(frame) != 27 or frame.duplicated(keys).any() or set(frame[keys].itertuples(index=False, name=None)) != expected_keys:
            raise ValueError("Baseline must contain all three methods and exactly nine locked paired runs")
    if not expected[keys].equals(current[keys]):
        raise ValueError("Baseline keys differ")
    for column in (*METRICS, "threshold"):
        old, new = expected[column].to_numpy(float), current[column].to_numpy(float)
        if not np.isfinite(old).all() or not np.isfinite(new).all() or not np.allclose(old, new, atol=tolerance, rtol=0):
            raise ValueError(f"Baseline verification failed: {column}")
    # Per-run equality implies the reported means and sample SDs also agree;
    # check explicitly to guard accidental summary-estimator changes.
    old_summary, new_summary = model_summary(expected).set_index("method"), model_summary(current).set_index("method")
    for metric in METRICS:
        for stat in ("mean", "sd"):
            column = f"{metric}_{stat}"
            if not np.allclose(old_summary.loc[sorted(methods), column], new_summary.loc[sorted(methods), column], atol=tolerance, rtol=0):
                raise ValueError(f"Baseline summary verification failed: {column}")


def validate_runs(runs: dict[tuple[int, int], pd.DataFrame], *, expected_n: int = 119) -> pd.DataFrame:
    expected_keys = {(split, train) for split in SPLIT_SEEDS for train in TRAIN_SEEDS}
    if set(runs) != expected_keys:
        raise ValueError("LSH analysis needs every locked split/train-seed pair")
    labels_by_id = {}
    for split_seed in SPLIT_SEEDS:
        canonical = None
        for train_seed in TRAIN_SEEDS:
            frame = runs[(split_seed, train_seed)]
            required = ["case_id", "label_true"]
            if not set(required).issubset(frame) or frame[required].isnull().any().any():
                raise ValueError("LSH run lacks complete IDs/labels")
            if len(frame) != expected_n or frame.case_id.duplicated().any():
                raise ValueError("LSH outer-test run has missing or duplicate patient-case records")
            current = frame[required].sort_values("case_id").reset_index(drop=True)
            if current.label_true.nunique() != 2 or not set(current.label_true.unique()).issubset({0, 1}):
                raise ValueError("LSH outer-test labels must contain both binary classes")
            if canonical is None:
                canonical = current
            elif not current.equals(canonical):
                raise ValueError("Training seeds changed the locked outer-test membership/labels")
            for identifier, label in current.itertuples(index=False, name=None):
                if identifier in labels_by_id and labels_by_id[identifier] != int(label):
                    raise ValueError("Labels differ across splits for a source patient-case ID")
                labels_by_id[identifier] = int(label)
            for column in (value for value in frame if value.startswith("prob_")):
                values = frame[column].to_numpy(float)
                if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                    raise ValueError("LSH outer-test probabilities are invalid")
    return pd.DataFrame(sorted(labels_by_id.items()), columns=["case_id", "label_true"])


def hierarchical_bootstrap(
    runs: dict[tuple[int, int], pd.DataFrame], replicates: int = 10_000,
    seed: int = 20260929, *, expected_n: int = 119,
) -> pd.DataFrame:
    """Formal hierarchy: shared union-ID weights, split, then train within split.

    Unlike pooled predictions, weighted metrics are first computed per run,
    then averaged over the selected split/training occurrences. Vectorization
    changes computation order, not the formal hierarchy or estimand. Run
    resamples lacking one class are skipped, as in the archived LSH analysis.
    """
    if replicates <= 0:
        raise ValueError("Bootstrap replicate count must be positive")
    patients = validate_runs(runs, expected_n=expected_n)
    labels = patients.label_true.to_numpy(int)
    id_index = {identifier: index for index, identifier in enumerate(patients.case_id)}
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(patients)), dtype=np.int64)
    for label in (0, 1):
        members = np.flatnonzero(labels == label)
        weights[:, members] = rng.multinomial(len(members), np.full(len(members), 1 / len(members)), size=replicates)
    selected_splits = rng.integers(len(SPLIT_SEEDS), size=(replicates, len(SPLIT_SEEDS)))
    selected_trains = rng.integers(len(TRAIN_SEEDS), size=(replicates, len(SPLIT_SEEDS), len(TRAIN_SEEDS)))
    selected_runs = (selected_splits[:, :, None] * len(TRAIN_SEEDS) + selected_trains).reshape(replicates, -1)
    keys = [(split, train) for split in SPLIT_SEEDS for train in TRAIN_SEEDS]
    methods = sorted({method for _, first, second in planned_contrasts() for method in (first, second)})
    estimates, observed, invalid_draws = {}, {}, np.zeros((replicates, len(keys)), dtype=bool)
    for method in methods:
        samples = {"auc": np.full((replicates, len(keys)), np.nan), "nll": np.full((replicates, len(keys)), np.nan)}
        points = {"auc": [], "nll": []}
        for run_index, key in enumerate(keys):
            frame = runs[key]
            y, probability = frame.label_true.to_numpy(int), frame[f"prob_{method}"].to_numpy(float)
            run_weights = weights[:, [id_index[identifier] for identifier in frame.case_id]]
            valid = (run_weights[:, y == 0].sum(axis=1) > 0) & (run_weights[:, y == 1].sum(axis=1) > 0)
            invalid_draws[:, run_index] = ~valid
            clipped = np.clip(probability, EPS, 1 - EPS)
            case_loss = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
            samples["auc"][valid, run_index] = weighted_auc_many(y, probability, run_weights[valid])
            samples["nll"][valid, run_index] = (run_weights[valid] @ case_loss) / run_weights[valid].sum(axis=1)
            points["auc"].append(roc_auc_score(y, probability))
            points["nll"].append(log_loss(y, clipped, labels=[0, 1]))
        for metric in ("auc", "nll"):
            chosen = np.take_along_axis(samples[metric], selected_runs, axis=1)
            count = np.sum(np.isfinite(chosen), axis=1)
            if np.any(count == 0):
                raise ValueError("A complete hierarchical draw lacks both classes in every selected run")
            estimates[(method, metric)] = np.nansum(chosen, axis=1) / count
            observed[(method, metric)] = float(np.mean(points[metric]))
    rows = []
    for contrast, full, ablated in planned_contrasts():
        for metric in ("auc", "nll"):
            difference = estimates[(full, metric)] - estimates[(ablated, metric)]
            low, high = np.quantile(difference, [0.025, 0.975])
            rows.append({
                "contrast": contrast, "metric": f"{metric}_delta",
                "observed_delta": observed[(full, metric)] - observed[(ablated, metric)],
                "bootstrap_mean": float(np.mean(difference)), "ci95_lower": float(low), "ci95_upper": float(high),
                "probability_favorable_to_full": float(np.mean(difference > 0 if metric == "auc" else difference < 0)),
                "bootstrap_unit": "class-stratified unique outer-test patient-case ID union; split seed; within-split optimization seed",
                "shared_id_weights": True, "unique_outer_test_patients": len(patients),
                "run_class_degenerate_draws": int(invalid_draws.sum()),
                "replicates": replicates, "multiplicity_adjustment": "none; ten exploratory planned contrasts",
                "selection_conditioning": "conditional on historical outer-test-informed architecture selection and retained pilot checkpoint",
            })
    return pd.DataFrame(rows)


def validate_variant_completion(root: Path) -> list[dict]:
    locked = {
        "epochs": 80, "batch_size": 46, "eval_batch_size": 64, "accum_steps": 1,
        "num_workers": 0, "lr": 1e-4, "backbone_lr_scale": 0.05, "weight_decay": 0.05,
        "warmup_epochs": 5, "patience": 10, "freeze_backbone_epochs": 5,
        "monitor_start_epoch": 1, "target_shape": [128, 128, 64],
        "spatial_aug_policy": "slice_xy", "checkpoint_selection": "selection-validation AUC only",
        "loss_function": "cross_entropy", "label_smoothing": 0.05,
        "class_weight_mode": "sqrt_balanced",
    }
    cells, common_source_hashes = [], None
    for variant in VARIANTS:
        model = f"{FULL_MODEL}_{variant}"
        for split_seed in SPLIT_SEEDS:
            for train_seed in TRAIN_SEEDS:
                run = prediction_path(root, model, split_seed, train_seed, "outer_test").parent
                for marker in ("TRAIN_DONE", "DONE", "configuration.json", "prediction_sha256.json"):
                    if not (run / marker).is_file() or (run / marker).stat().st_size == 0:
                        raise ValueError(f"Incomplete LSH ablation cell: {model}/{split_seed}/{train_seed}; missing {marker}")
                signed = json.loads((run / "configuration.json").read_text())
                config = signed["configuration"]
                encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
                if hashlib.sha256(encoded).hexdigest() != signed["signature_sha256"]:
                    raise ValueError("Invalid LSH cell signature")
                if config.get("model") != model or config.get("split_seed") != split_seed or config.get("train_seed") != train_seed or config.get("fold") != 1 or any(config.get(key) != value for key, value in locked.items()):
                    raise ValueError("LSH cell configuration differs from the locked protocol")
                source_hashes = {path: digest for path, digest in config["input_sha256"].items() if not path.endswith(".csv")}
                if common_source_hashes is None:
                    common_source_hashes = source_hashes
                elif common_source_hashes != source_hashes:
                    raise ValueError("LSH ablation cells were produced by different source versions")
                for input_path, digest in config["input_sha256"].items():
                    path = Path(input_path)
                    if not path.is_file() or sha256(path) != digest:
                        raise ValueError(f"Signed LSH input changed after training: {input_path}")
                prediction_hashes = json.loads((run / "prediction_sha256.json").read_text())
                for role in ("selection_val", "outer_test"):
                    if sha256(run / f"{role}_predictions.csv") != prediction_hashes[role]:
                        raise ValueError("Completed LSH prediction was modified")
                summary_path = root.parent / "checkpoints" / model / f"split_seed_{split_seed}" / "fold_1" / f"train_seed_{train_seed}" / "best_model_multiseed_summary.csv"
                summary = pd.read_csv(summary_path)
                if len(summary) != 1 or int(summary.seed.iloc[0]) != train_seed or not np.isfinite(summary.train_time_sec.iloc[0]) or summary.train_time_sec.iloc[0] <= 0 or not 1 <= int(summary.best_epoch.iloc[0]) <= 80:
                    raise ValueError("Unexpected LSH training summary")
                cells.append({"model": model, "split_seed": split_seed, "train_seed": train_seed,
                              "signature_sha256": signed["signature_sha256"], "train_time_sec": float(summary.train_time_sec.iloc[0]),
                              "best_epoch": int(summary.best_epoch.iloc[0])})
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--baseline-result-root", type=Path, required=True)
    parser.add_argument("--full-trimil-result-root", type=Path, required=True)
    parser.add_argument("--expected-full-summary", type=Path, required=True, help="Archived LSH per_run_metrics.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260929)
    args = parser.parse_args()
    if not args.preflight and args.result_root is None:
        parser.error("Full analysis requires --result-root")
    roles, protocol_hashes = load_protocol(args.protocol_root)
    baseline_metrics, baseline_runs, baseline_hashes = analyze_all(roles, args.baseline_result_root, args.full_trimil_result_root)
    patients = validate_runs(baseline_runs)
    assert_full_baseline(baseline_metrics, args.expected_full_summary)
    baseline_summary = model_summary(baseline_metrics).set_index("method")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preflight = {
        "passed": True, "source_patients": 594, "outer_test_n_per_run": 119,
        "paired_runs": 9, "unique_outer_test_patients": len(patients),
        "split_seeds": list(SPLIT_SEEDS), "train_seeds": list(TRAIN_SEEDS),
        "comparison_tolerance": 1e-10, "baseline_per_run_metrics_recomputed_exactly": True,
        "baseline_summary_recomputed_exactly": True,
        "expected_full_per_run_metrics_sha256": sha256(args.expected_full_summary),
        "protocol_role_sha256": protocol_hashes, "baseline_prediction_sha256": baseline_hashes,
        "full_fusion_auc_mean": float(baseline_summary.loc["fusion_full", "auc_mean"]),
        "full_fusion_nll_mean": float(baseline_summary.loc["fusion_full", "nll_mean"]),
        "full_fusion_auc_sd": float(baseline_summary.loc["fusion_full", "auc_sd"]),
        "full_fusion_nll_sd": float(baseline_summary.loc["fusion_full", "nll_sd"]),
        "patient_identity_boundary": "one case per patient per source record; independent patient identifier verification unavailable",
    }
    (args.output_dir / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if args.preflight:
        print(json.dumps({key: value for key, value in preflight.items() if key not in {"protocol_role_sha256", "baseline_prediction_sha256"}}, indent=2))
        return
    cells = validate_variant_completion(args.result_root)
    per_run, runs, prediction_hashes = analyze_all(roles, args.baseline_result_root, args.full_trimil_result_root, args.result_root)
    validate_runs(runs)
    assert_full_baseline(per_run, args.expected_full_summary)
    summary = model_summary(per_run)
    bootstrap = hierarchical_bootstrap(runs, args.bootstrap_replicates, args.bootstrap_seed)
    delta_rows = []
    for (split_seed, train_seed), frame in per_run.groupby(["split_seed", "train_seed"], sort=True):
        indexed = frame.set_index("method")
        for contrast, full, ablated in planned_contrasts():
            delta_rows.append({"split_seed": int(split_seed), "train_seed": int(train_seed), "contrast": contrast,
                               **{f"{metric}_delta": float(indexed.loc[full, metric] - indexed.loc[ablated, metric]) for metric in METRICS}})
    per_run.to_csv(args.output_dir / "per_run_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "model_run_summary.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(args.output_dir / "paired_run_deltas.csv", index=False)
    bootstrap.to_csv(args.output_dir / "hierarchical_patient_split_train_bootstrap.csv", index=False)
    pd.DataFrame(cells).to_csv(args.output_dir / "training_completion_summary.csv", index=False)
    metadata = {
        "status": "complete", "trained_cells": len(cells), "variants": list(VARIANTS),
        "protocol": "LSH three locked repeated holdouts x three paired optimization seeds; 415/60/119 train/selection-validation/outer-test",
        "source_patients": 594, "paired_runs": 9, "unique_outer_test_patients": len(patients),
        "patient_identity_boundary": preflight["patient_identity_boundary"],
        "thresholds": "ordinary Youden index from selection validation only; no outer-test fitting",
        "fusion": "fixed 0.5/0.5 raw-logit-equivalent binary log-odds reconstruction from exported prob_1; no gate/temperature fitting",
        "endpoint_clip_epsilon": EPS,
        "endpoint_clipped_prediction_total": sum(record["probability_endpoint_clipped_n"] for record in prediction_hashes),
        "prediction_sha256": prediction_hashes, "protocol_role_sha256": protocol_hashes,
        "bootstrap": "class-stratified unique outer-test patient ID union with globally shared ID weights; resample splits then optimization seeds within each selected split",
        "bootstrap_replicates": args.bootstrap_replicates, "bootstrap_seed": args.bootstrap_seed,
        "uncertainty": "exploratory percentile 95% intervals; ten planned contrasts; no multiplicity adjustment",
        "contrast_direction": "full minus ablated; positive AUC and negative NLL favor full",
        "z_only_boundary": "joint plane-coverage/instance-count control: 24 versus 72 instances; not an isolated plane-embedding control",
        "mean_pool_boundary": "embeddings are retained before token mean, becoming constant learned offsets rather than instance identities; CLS/Transformer removed",
        "evaluation_boundary": "historically outer-test-informed architecture selection; baseline pilot checkpoint retained; intervals condition on selected architecture and omit architecture-selection uncertainty",
        "negative_result_policy": "all planned variants reported regardless of direction; new ablations do not remove historical selection bias",
        "baseline_verified": True,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "complete", "trained_cells": len(cells), "paired_runs": 9,
                      "summary": str(args.output_dir / "model_run_summary.csv"), "bootstrap": str(args.output_dir / "hierarchical_patient_split_train_bootstrap.csv")}, indent=2))


if __name__ == "__main__":
    main()
