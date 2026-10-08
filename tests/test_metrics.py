import numpy as np

from utils.metrics import find_best_binary_threshold


def test_find_best_binary_threshold_can_apply_sensitivity_floor():
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    probs = np.array([0.551, 0.311, 0.108, 0.577, 0.420, 0.248, 0.589, 0.912, 0.947, 0.486])

    _, unconstrained = find_best_binary_threshold(labels, probs, metric_name="mcc")
    threshold, constrained = find_best_binary_threshold(
        labels,
        probs,
        metric_name="mcc",
        min_sensitivity=1.0,
    )

    assert unconstrained["sensitivity"] == 0.75
    assert constrained["sensitivity"] == 1.0
    assert constrained["mcc"] < unconstrained["mcc"]
    assert 0.420 < threshold < 0.486


def test_find_best_binary_threshold_falls_back_when_constraints_are_impossible():
    labels = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
    probs = np.array([0.551, 0.311, 0.108, 0.577, 0.420, 0.248, 0.589, 0.912, 0.947, 0.486])

    _, metrics = find_best_binary_threshold(
        labels,
        probs,
        metric_name="mcc",
        min_sensitivity=1.0,
        min_specificity=1.0,
    )

    assert metrics["sensitivity"] == 0.75
    assert metrics["specificity"] == 1.0
