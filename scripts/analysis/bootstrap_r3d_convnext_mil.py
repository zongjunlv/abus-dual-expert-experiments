#!/usr/bin/env python3
"""Paired stratified bootstrap for the R3D + ConvNeXt slice-MIL fusion.

The script treats cases as the sampling unit.  Within every bootstrap replicate,
benign and malignant cases are sampled separately with replacement.  The same
case indices are used for the candidate and reference, for both metrics, and
across all training seeds.  This preserves the paired design and class counts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


DEFAULT_INPUT_ROOT = Path("outputs/tables/aim_convnext_mil_dual_20260910")
DEFAULT_GLOB = "convnext_slice_mil_s64_seed*_bs36/paired_predictions_lsh_test.csv"
EPSILON = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired case-stratified bootstrap for fixed equal-logit fusion."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--prediction-glob", default=DEFAULT_GLOB)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_INPUT_ROOT / "paired_bootstrap",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=20260910)
    return parser.parse_args()


def _extract_seed(path: Path) -> int:
    match = re.search(r"seed(\d+)", str(path))
    if match is None:
        raise ValueError(f"Cannot extract a training seed from {path}")
    return int(match.group(1))


def _logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(probability / (1.0 - probability))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def load_seed_predictions(paths: Iterable[Path]) -> dict[int, pd.DataFrame]:
    """Load and align all seed-level prediction files by case path."""
    frames: dict[int, pd.DataFrame] = {}
    required = {"npy_path", "label_true", "prob_r3d", "prob_mil"}
    for path in sorted(paths):
        seed = _extract_seed(path)
        if seed in frames:
            raise ValueError(f"Duplicate prediction file for seed {seed}")
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame["npy_path"].duplicated().any():
            raise ValueError(f"{path} contains duplicate npy_path values")
        frame = frame.sort_values("npy_path").reset_index(drop=True)
        labels = frame["label_true"].to_numpy(dtype=int)
        if not set(np.unique(labels)).issubset({0, 1}) or np.unique(labels).size != 2:
            raise ValueError(f"{path} must contain both binary classes")

        recomputed_equal = _sigmoid(
            0.5 * _logit(frame["prob_r3d"].to_numpy())
            + 0.5 * _logit(frame["prob_mil"].to_numpy())
        )
        if "prob_equal_logit" in frame and not np.allclose(
            frame["prob_equal_logit"].to_numpy(dtype=float),
            recomputed_equal,
            rtol=1e-7,
            atol=1e-9,
        ):
            raise ValueError(f"{path} contains inconsistent prob_equal_logit values")
        frame["prob_equal_logit"] = recomputed_equal
        frames[seed] = frame

    if not frames:
        raise ValueError("No prediction files matched the requested path/glob")

    first_seed = next(iter(frames))
    reference = frames[first_seed]
    for seed, frame in frames.items():
        if not np.array_equal(
            frame["npy_path"].to_numpy(), reference["npy_path"].to_numpy()
        ):
            raise ValueError(
                f"Case-path mismatch between seeds {first_seed} and {seed}"
            )
        if not np.array_equal(
            frame["label_true"].to_numpy(), reference["label_true"].to_numpy()
        ):
            raise ValueError(f"Label mismatch between seeds {first_seed} and {seed}")
        if not np.allclose(
            frame["prob_r3d"], reference["prob_r3d"], rtol=0.0, atol=1e-10
        ):
            raise ValueError(f"R3D reference predictions differ for seed {seed}")

    legacy_column = "prob_legacy_r3d_r2plus1d_equal_logit"
    legacy_presence = [legacy_column in frame for frame in frames.values()]
    if any(legacy_presence) and not all(legacy_presence):
        raise ValueError(f"{legacy_column} must be present for every seed or no seed")
    if all(legacy_presence):
        for seed, frame in frames.items():
            if not np.allclose(
                frame[legacy_column],
                reference[legacy_column],
                rtol=0.0,
                atol=1e-10,
            ):
                raise ValueError(f"Legacy reference predictions differ for seed {seed}")
    return frames


def stratified_bootstrap_indices(
    labels: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return separate class-wise draws, preserving the observed class counts."""
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    labels = np.asarray(labels, dtype=int)
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    if negative.size == 0 or positive.size == 0:
        raise ValueError("Both classes are required for stratified AUC bootstrap")
    negative_draws = rng.choice(
        negative, size=(n_bootstrap, negative.size), replace=True
    )
    positive_draws = rng.choice(
        positive, size=(n_bootstrap, positive.size), replace=True
    )
    return negative_draws, positive_draws


def _binary_nll_per_case(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float), EPSILON, 1.0 - EPSILON
    )
    labels = np.asarray(labels, dtype=int)
    return -(
        labels * np.log(probabilities)
        + (1 - labels) * np.log(1.0 - probabilities)
    )


def _auc_from_class_probabilities(negative: np.ndarray, positive: np.ndarray) -> float:
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0))


def bootstrap_metric_deltas(
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    negative_draws: np.ndarray,
    positive_draws: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute candidate-minus-reference deltas using shared paired draws."""
    labels = np.asarray(labels, dtype=int)
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if not (labels.shape == candidate.shape == reference.shape):
        raise ValueError("labels, candidate, and reference must have identical shapes")
    n_bootstrap = negative_draws.shape[0]
    if positive_draws.shape[0] != n_bootstrap:
        raise ValueError(
            "Class-wise bootstrap arrays must have the same number of rows"
        )

    auc_delta = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        negative_index = negative_draws[index]
        positive_index = positive_draws[index]
        candidate_auc = _auc_from_class_probabilities(
            candidate[negative_index], candidate[positive_index]
        )
        reference_auc = _auc_from_class_probabilities(
            reference[negative_index], reference[positive_index]
        )
        auc_delta[index] = candidate_auc - reference_auc

    candidate_loss = _binary_nll_per_case(labels, candidate)
    reference_loss = _binary_nll_per_case(labels, reference)
    sampled_candidate_nll = (
        candidate_loss[negative_draws].sum(axis=1)
        + candidate_loss[positive_draws].sum(axis=1)
    ) / labels.size
    sampled_reference_nll = (
        reference_loss[negative_draws].sum(axis=1)
        + reference_loss[positive_draws].sum(axis=1)
    ) / labels.size
    return {
        "auc": auc_delta,
        "nll": sampled_candidate_nll - sampled_reference_nll,
    }


def _point_metric_delta(
    labels: np.ndarray, candidate: np.ndarray, reference: np.ndarray, metric: str
) -> float:
    if metric == "auc":
        return float(
            roc_auc_score(labels, candidate) - roc_auc_score(labels, reference)
        )
    if metric == "nll":
        return float(
            log_loss(labels, candidate, labels=[0, 1])
            - log_loss(labels, reference, labels=[0, 1])
        )
    raise ValueError(f"Unsupported metric: {metric}")


def _ci_row(
    *,
    mode: str,
    seed: int | str,
    comparison: str,
    metric: str,
    point_delta: float,
    draws: np.ndarray,
    confidence_level: float,
) -> dict[str, object]:
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    favorable = draws > 0.0 if metric == "auc" else draws < 0.0
    return {
        "analysis_mode": mode,
        "seed": seed,
        "comparison": comparison,
        "metric": metric,
        "delta_definition": "candidate_minus_reference",
        "favorable_direction": "positive" if metric == "auc" else "negative",
        "point_delta": float(point_delta),
        "bootstrap_mean_delta": float(np.mean(draws)),
        "bootstrap_std_delta": float(np.std(draws, ddof=1)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": confidence_level,
        "n_bootstrap": int(draws.size),
        "probability_favorable": float(np.mean(favorable)),
        "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
    }


def analyze(
    frames: dict[int, pd.DataFrame],
    *,
    n_bootstrap: int,
    confidence_level: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    seeds = sorted(frames)
    first = frames[seeds[0]]
    labels = first["label_true"].to_numpy(dtype=int)
    rng = np.random.default_rng(random_seed)
    negative_draws, positive_draws = stratified_bootstrap_indices(
        labels, n_bootstrap, rng
    )
    seed_draw_indices = rng.integers(0, len(seeds), size=(n_bootstrap, len(seeds)))

    references = {"vs_r3d": "prob_r3d"}
    legacy_column = "prob_legacy_r3d_r2plus1d_equal_logit"
    if all(legacy_column in frame for frame in frames.values()):
        references["vs_legacy_r3d_r2plus1d_equal_logit"] = legacy_column

    point_rows: list[dict[str, object]] = []
    ci_rows: list[dict[str, object]] = []
    for comparison, reference_column in references.items():
        per_metric_draws: dict[str, list[np.ndarray]] = {"auc": [], "nll": []}
        per_metric_points: dict[str, list[float]] = {"auc": [], "nll": []}
        for seed in seeds:
            frame = frames[seed]
            candidate = frame["prob_equal_logit"].to_numpy(dtype=float)
            reference = frame[reference_column].to_numpy(dtype=float)
            draw_deltas = bootstrap_metric_deltas(
                labels, candidate, reference, negative_draws, positive_draws
            )
            for metric in ("auc", "nll"):
                point = _point_metric_delta(labels, candidate, reference, metric)
                point_rows.append(
                    {
                        "seed": seed,
                        "comparison": comparison,
                        "metric": metric,
                        "delta_definition": "candidate_minus_reference",
                        "value": point,
                    }
                )
                per_metric_draws[metric].append(draw_deltas[metric])
                per_metric_points[metric].append(point)
                ci_rows.append(
                    _ci_row(
                        mode="per_seed_case_stratified",
                        seed=seed,
                        comparison=comparison,
                        metric=metric,
                        point_delta=point,
                        draws=draw_deltas[metric],
                        confidence_level=confidence_level,
                    )
                )

        for metric in ("auc", "nll"):
            draw_matrix = np.column_stack(per_metric_draws[metric])
            point_mean = float(np.mean(per_metric_points[metric]))
            shared_case_seed_mean = draw_matrix.mean(axis=1)
            ci_rows.append(
                _ci_row(
                    mode="seed_mean_shared_case_stratified",
                    seed="all",
                    comparison=comparison,
                    metric=metric,
                    point_delta=point_mean,
                    draws=shared_case_seed_mean,
                    confidence_level=confidence_level,
                )
            )
            hierarchical = np.take_along_axis(
                draw_matrix, seed_draw_indices, axis=1
            ).mean(axis=1)
            ci_rows.append(
                _ci_row(
                    mode="hierarchical_seed_and_case_stratified",
                    seed="all",
                    comparison=comparison,
                    metric=metric,
                    point_delta=point_mean,
                    draws=hierarchical,
                    confidence_level=confidence_level,
                )
            )

    ci_frame = pd.DataFrame(ci_rows)
    point_frame = pd.DataFrame(point_rows)
    summary: dict[str, object] = {
        "protocol": {
            "sampling_unit": "case",
            "case_sampling": "stratified by binary label with replacement",
            "pairing": (
                "shared case indices for candidate/reference, metrics, and "
                "training seeds"
            ),
            "candidate": "R3D + ConvNeXt slice-MIL fixed 0.5/0.5 equal-logit fusion",
            "delta_definition": "candidate minus reference",
            "n_bootstrap": n_bootstrap,
            "confidence_level": confidence_level,
            "interval": "percentile",
            "random_seed": random_seed,
            "training_seeds": seeds,
            "case_count": int(labels.size),
            "class_counts": {
                "negative": int(np.sum(labels == 0)),
                "positive": int(np.sum(labels == 1)),
            },
            "modes": {
                "per_seed_case_stratified": "case uncertainty for each training seed",
                "seed_mean_shared_case_stratified": (
                    "mean seed effect under the same resampled cases; training "
                    "seeds are fixed"
                ),
                "hierarchical_seed_and_case_stratified": (
                    "resamples both training seeds and stratified cases"
                ),
            },
        },
        "confidence_intervals": ci_frame.to_dict(orient="records"),
    }
    return ci_frame, point_frame, summary


def main() -> None:
    args = parse_args()
    paths = list(args.input_root.glob(args.prediction_glob))
    frames = load_seed_predictions(paths)
    ci_frame, point_frame, summary = analyze(
        frames,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        random_seed=args.random_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ci_frame.to_csv(args.output_dir / "paired_bootstrap_ci.csv", index=False)
    point_frame.to_csv(args.output_dir / "per_seed_point_deltas.csv", index=False)
    with (args.output_dir / "paired_bootstrap_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
