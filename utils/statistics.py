from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auc_ci(
    labels: np.ndarray,
    positive_probs: np.ndarray,
    n_bootstraps: int = 1000,
    confidence: float = 0.95,
    seed: int = 3407,
) -> tuple[float, float]:
    """Estimate a percentile bootstrap CI for binary AUC."""
    labels = np.asarray(labels).astype(int)
    positive_probs = np.asarray(positive_probs).astype(float)
    if labels.ndim != 1 or positive_probs.ndim != 1:
        raise ValueError("labels and positive_probs must be one-dimensional arrays.")
    if labels.shape[0] != positive_probs.shape[0]:
        raise ValueError("labels and positive_probs must have the same length.")
    if labels.size == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(int(n_bootstraps)):
        indices = rng.integers(0, labels.size, labels.size)
        sample_labels = labels[indices]
        if np.unique(sample_labels).size < 2:
            continue
        scores.append(float(roc_auc_score(sample_labels, positive_probs[indices])))

    if not scores:
        return float("nan"), float("nan")

    alpha = 1.0 - float(confidence)
    return (
        float(np.quantile(scores, alpha / 2.0)),
        float(np.quantile(scores, 1.0 - alpha / 2.0)),
    )
