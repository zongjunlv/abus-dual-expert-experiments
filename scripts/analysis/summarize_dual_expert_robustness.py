#!/usr/bin/env python3
"""Summarize repeated-split R3D + ConvNeXt-MIL robustness experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


EPSILON = 1e-7
METHODS = ("r3d", "convnext_slice_mil", "equal_logit")


def _case_id(value: object) -> str:
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    return "/".join(parts[-4:])


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability.astype(float), EPSILON, 1.0 - EPSILON)
    return np.log(probability / (1.0 - probability))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "n": int(labels.size),
        "auc": float(roc_auc_score(labels, probabilities)),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
        "accuracy_0.5": float(np.mean((probabilities >= 0.5) == labels)),
    }


def _load_model(result_root: Path, model: str) -> pd.DataFrame:
    paths = sorted(
        (result_root / model).glob(
            "split_seed_*/fold_*/train_seed_*/outer_test_predictions.csv"
        )
    )
    if not paths:
        raise FileNotFoundError(f"No predictions for {model} below {result_root}")
    frames = []
    for path in paths:
        split_seed = int(
            next(part for part in path.parts if part.startswith("split_seed_")).split("_")[-1]
        )
        fold = int(next(part for part in path.parts if part.startswith("fold_")).split("_")[-1])
        frame = pd.read_csv(path)
        path_column = "npy_path" if "npy_path" in frame else "data_path"
        required = {path_column, "label_true", "prob_1"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        frames.append(pd.DataFrame({
            "case_id": frame[path_column].map(_case_id),
            "split_seed": split_seed,
            "fold": fold,
            "label_true": frame["label_true"].astype(int),
            f"prob_{model}": frame["prob_1"].astype(float),
        }))
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["case_id", "split_seed"]).any():
        raise ValueError(f"Duplicate case within a repeat for {model}")
    return combined


def load_paired(result_root: Path) -> pd.DataFrame:
    r3d = _load_model(result_root, "r3d")
    mil = _load_model(result_root, "convnext_slice_mil")
    paired = r3d.merge(
        mil,
        on=["case_id", "split_seed", "fold"],
        suffixes=("_r3d", "_mil"),
        validate="one_to_one",
    )
    if len(paired) != len(r3d) or len(paired) != len(mil):
        raise ValueError("R3D and MIL prediction case sets differ")
    if not np.array_equal(paired["label_true_r3d"], paired["label_true_mil"]):
        raise ValueError("R3D and MIL labels differ")
    paired = paired.rename(columns={"label_true_r3d": "label_true"}).drop(
        columns="label_true_mil"
    )
    paired["prob_equal_logit"] = _sigmoid(
        0.5 * _logit(paired["prob_r3d"].to_numpy())
        + 0.5 * _logit(paired["prob_convnext_slice_mil"].to_numpy())
    )
    return paired


def repeat_metrics(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_seed, frame in paired.groupby("split_seed", sort=True):
        labels = frame["label_true"].to_numpy(int)
        for method in METHODS:
            row = {"split_seed": int(split_seed), "method": method}
            row.update(_metrics(labels, frame[f"prob_{method}"].to_numpy(float)))
            row["folds"] = int(frame["fold"].nunique())
            rows.append(row)
    return pd.DataFrame(rows)


def _draw_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    return np.concatenate([
        rng.choice(negative, negative.size, replace=True),
        rng.choice(positive, positive.size, replace=True),
    ])


def _delta(frame: pd.DataFrame, indices: np.ndarray, metric: str) -> float:
    sampled = frame.iloc[indices]
    labels = sampled["label_true"].to_numpy(int)
    candidate = sampled["prob_equal_logit"].to_numpy(float)
    reference = sampled["prob_r3d"].to_numpy(float)
    if metric == "auc":
        return float(roc_auc_score(labels, candidate) - roc_auc_score(labels, reference))
    return float(
        log_loss(labels, candidate, labels=[0, 1])
        - log_loss(labels, reference, labels=[0, 1])
    )


def paired_bootstrap(
    paired: pd.DataFrame, repeats: int = 10_000, seed: int = 20260910
) -> pd.DataFrame:
    grouped = {key: value.reset_index(drop=True) for key, value in paired.groupby("split_seed")}
    rng = np.random.default_rng(seed)
    rows = []
    repeat_keys = sorted(grouped)
    repeat_points = {
        metric: np.array([
            _delta(frame, np.arange(len(frame)), metric)
            for frame in grouped.values()
        ])
        for metric in ("auc", "nll")
    }
    for metric in ("auc", "nll"):
        per_repeat_draws = {key: np.empty(repeats) for key in repeat_keys}
        hierarchical = np.empty(repeats)
        for iteration in range(repeats):
            current = []
            for key in repeat_keys:
                frame = grouped[key]
                indices = _draw_indices(frame["label_true"].to_numpy(int), rng)
                value = _delta(frame, indices, metric)
                per_repeat_draws[key][iteration] = value
                current.append(value)
            selected = rng.integers(0, len(current), size=len(current))
            hierarchical[iteration] = float(np.mean(np.asarray(current)[selected]))

        for key, draws in per_repeat_draws.items():
            lower, upper = np.quantile(draws, [0.025, 0.975])
            rows.append({
                "analysis_mode": "per_repeat_case_stratified",
                "split_seed": key,
                "metric": metric,
                "point_delta": float(repeat_points[metric][repeat_keys.index(key)]),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "probability_favorable": float(np.mean(draws > 0 if metric == "auc" else draws < 0)),
                "bootstrap_replicates": repeats,
            })
        lower, upper = np.quantile(hierarchical, [0.025, 0.975])
        rows.append({
            "analysis_mode": "hierarchical_repeat_and_case_stratified",
            "split_seed": "all",
            "metric": metric,
            "point_delta": float(repeat_points[metric].mean()),
            "ci95_lower": float(lower),
            "ci95_upper": float(upper),
            "probability_favorable": float(np.mean(hierarchical > 0 if metric == "auc" else hierarchical < 0)),
            "bootstrap_replicates": repeats,
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260910)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired = load_paired(args.result_root)
    metrics = repeat_metrics(paired)
    bootstrap = paired_bootstrap(paired, args.bootstrap_repeats, args.bootstrap_seed)
    summary_rows = []
    for method, frame in metrics.groupby("method", sort=True):
        row = {"method": method, "repeat_n": len(frame)}
        for metric in ("auc", "nll", "accuracy_0.5"):
            row[f"{metric}_mean"] = float(frame[metric].mean())
            row[f"{metric}_sd"] = float(frame[metric].std(ddof=1))
        summary_rows.append(row)
    metrics.to_csv(args.output_dir / "repeat_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.output_dir / "model_summary.csv", index=False)
    bootstrap.to_csv(args.output_dir / "paired_bootstrap_ci.csv", index=False)
    paired.to_csv(args.output_dir / "paired_predictions.csv", index=False)
    manifest = {
        "protocol": args.protocol,
        "candidate": "R3D + ConvNeXt slice-MIL fixed 0.5/0.5 equal-logit",
        "reference": "R3D",
        "delta": "candidate minus reference",
        "bootstrap": "label-stratified paired case resampling with hierarchical repeat resampling",
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.bootstrap_seed,
        "repeat_count": int(paired["split_seed"].nunique()),
        "case_rows": int(len(paired)),
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
