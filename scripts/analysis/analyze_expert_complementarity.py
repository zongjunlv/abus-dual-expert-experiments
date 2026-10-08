#!/usr/bin/env python3
"""Quantify complementarity between two case-aligned binary predictors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


PROBABILITY_EPSILON = 1e-7
CASE_KEY_COLUMNS = ("data_path", "npy_path")


def _case_key_column(frame: pd.DataFrame, source_name: str) -> str:
    candidates = [column for column in CASE_KEY_COLUMNS if column in frame.columns]
    if not candidates:
        raise ValueError(
            f"{source_name} must contain one case-key column: "
            + ", ".join(CASE_KEY_COLUMNS)
        )
    return candidates[0]


def _prepare(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    required = {"label_true", "prob_1"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")
    key_column = _case_key_column(frame, source_name)
    prepared = frame[[key_column, "label_true", "prob_1"]].copy()
    prepared.columns = ["case_key", "label_true", "prob_1"]
    prepared["case_key"] = prepared["case_key"].astype(str)
    if prepared["case_key"].duplicated().any():
        duplicates = prepared.loc[prepared["case_key"].duplicated(), "case_key"].tolist()
        raise ValueError(f"{source_name} has duplicate case keys: {duplicates[:5]}")
    labels = pd.to_numeric(prepared["label_true"], errors="coerce").to_numpy(float)
    probabilities = pd.to_numeric(prepared["prob_1"], errors="coerce").to_numpy(float)
    if not np.isfinite(labels).all() or not set(np.unique(labels)).issubset({0.0, 1.0}):
        raise ValueError(f"{source_name} labels must be finite binary values")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError(f"{source_name} prob_1 values must be finite and within [0, 1]")
    prepared["label_true"] = labels.astype(int)
    prepared["prob_1"] = probabilities
    return prepared.set_index("case_key").sort_index()


def _align_predictions(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> pd.DataFrame:
    left = _prepare(frame_a, "model_a")
    right = _prepare(frame_b, "model_b")
    if not left.index.equals(right.index):
        only_a = left.index.difference(right.index).tolist()
        only_b = right.index.difference(left.index).tolist()
        raise ValueError(
            "Prediction CSV case keys do not match; "
            f"only model_a={only_a[:5]}, only model_b={only_b[:5]}"
        )
    mismatched = left["label_true"].ne(right["label_true"])
    if mismatched.any():
        cases = left.index[mismatched].tolist()
        raise ValueError(f"Prediction CSV labels do not match for case keys: {cases[:5]}")
    labels = left["label_true"].to_numpy(int)
    if np.unique(labels).size != 2:
        raise ValueError("Both binary classes are required to compute AUC")
    return pd.DataFrame(
        {
            "case_key": left.index,
            "label_true": labels,
            "prob_a": left["prob_1"].to_numpy(float),
            "prob_b": right["prob_1"].to_numpy(float),
        }
    )


def _stable_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return np.log(clipped) - np.log1p(-clipped)


def _stable_expit(logits: np.ndarray) -> np.ndarray:
    output = np.empty_like(logits, dtype=float)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponent = np.exp(logits[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "nll": float(log_loss(labels, clipped, labels=[0, 1])),
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _quartile_statistics(cases: pd.DataFrame) -> pd.DataFrame:
    ranked = cases["disagreement"].rank(method="first")
    cases = cases.copy()
    cases["disagreement_quartile"] = pd.qcut(
        ranked, q=4, labels=["Q1", "Q2", "Q3", "Q4"]
    ).astype(str)
    rows: list[dict[str, Any]] = []
    for quartile, group in cases.groupby("disagreement_quartile", sort=True, observed=True):
        rows.append(
            {
                "disagreement_quartile": quartile,
                "case_count": int(len(group)),
                "disagreement_min": float(group["disagreement"].min()),
                "disagreement_max": float(group["disagreement"].max()),
                "disagreement_mean": float(group["disagreement"].mean()),
                "model_a_accuracy": float(group["correct_a"].mean()),
                "model_b_accuracy": float(group["correct_b"].mean()),
                "equal_probability_accuracy": float(group["correct_equal_probability"].mean()),
                "equal_logit_accuracy": float(group["correct_equal_logit"].mean()),
                "oracle_per_case_accuracy": float((group["correct_a"] | group["correct_b"]).mean()),
                "both_wrong_count": int((~group["correct_a"] & ~group["correct_b"]).sum()),
                "one_model_only_correct_count": int((group["correct_a"] ^ group["correct_b"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def analyze_predictions(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Analyze two prediction tables and return a JSON-ready summary and quartiles."""
    if not model_a_name or not model_b_name or model_a_name == model_b_name:
        raise ValueError("Model names must be non-empty and distinct")
    cases = _align_predictions(frame_a, frame_b)
    labels = cases["label_true"].to_numpy(int)
    probability_a = cases["prob_a"].to_numpy(float)
    probability_b = cases["prob_b"].to_numpy(float)
    equal_probability = (probability_a + probability_b) / 2.0
    equal_logit = _stable_expit((_stable_logit(probability_a) + _stable_logit(probability_b)) / 2.0)
    oracle = np.where(labels == 1, np.maximum(probability_a, probability_b), np.minimum(probability_a, probability_b))

    prediction_a = probability_a >= 0.5
    prediction_b = probability_b >= 0.5
    correct_a = prediction_a == labels
    correct_b = prediction_b == labels
    cases["disagreement"] = np.abs(probability_a - probability_b)
    cases["correct_a"] = correct_a
    cases["correct_b"] = correct_b
    cases["correct_equal_probability"] = (equal_probability >= 0.5) == labels
    cases["correct_equal_logit"] = (equal_logit >= 0.5) == labels

    summary: dict[str, Any] = {
        "case_count": int(len(cases)),
        "threshold": 0.5,
        "oracle_definition": (
            "Label-aware per-case upper bound: choose the larger source probability for positive "
            "cases and the smaller source probability for negative cases."
        ),
        "metrics": {
            model_a_name: _metrics(labels, probability_a),
            model_b_name: _metrics(labels, probability_b),
            "equal_probability": _metrics(labels, equal_probability),
            "equal_logit": _metrics(labels, equal_logit),
            "oracle_per_case": _metrics(labels, oracle),
        },
        "error_overlap": {
            "both_correct_count": int((correct_a & correct_b).sum()),
            f"{model_a_name}_only_correct_count": int((correct_a & ~correct_b).sum()),
            f"{model_b_name}_only_correct_count": int((~correct_a & correct_b).sum()),
            "both_wrong_count": int((~correct_a & ~correct_b).sum()),
            "either_model_correct_count": int((correct_a | correct_b).sum()),
            "complementary_correction_count": int((correct_a ^ correct_b).sum()),
        },
        "prediction_correlation": {
            "pearson": _finite_or_none(pd.Series(probability_a).corr(pd.Series(probability_b), method="pearson")),
            "spearman": _finite_or_none(pd.Series(probability_a).corr(pd.Series(probability_b), method="spearman")),
        },
        "disagreement": {
            "mean_absolute_probability_difference": float(cases["disagreement"].mean()),
            "median_absolute_probability_difference": float(cases["disagreement"].median()),
        },
    }
    return summary, _quartile_statistics(cases)


def _summary_tables(summary: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.DataFrame.from_dict(summary["metrics"], orient="index").rename_axis("method").reset_index()
    complementarity = {
        "case_count": summary["case_count"],
        **summary["error_overlap"],
        **{f"correlation_{key}": value for key, value in summary["prediction_correlation"].items()},
        **summary["disagreement"],
    }
    return metrics, pd.DataFrame([complementarity])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-a", type=Path, required=True)
    parser.add_argument("--predictions-b", type=Path, required=True)
    parser.add_argument("--model-a-name", default="model_a")
    parser.add_argument("--model-b-name", default="model_b")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, quartiles = analyze_predictions(
        pd.read_csv(args.predictions_a),
        pd.read_csv(args.predictions_b),
        args.model_a_name,
        args.model_b_name,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, complementarity = _summary_tables(summary)
    (args.output_dir / "expert_complementarity.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    metrics.to_csv(args.output_dir / "expert_metrics.csv", index=False)
    complementarity.to_csv(args.output_dir / "error_complementarity.csv", index=False)
    quartiles.to_csv(args.output_dir / "disagreement_quartiles.csv", index=False)


if __name__ == "__main__":
    main()
