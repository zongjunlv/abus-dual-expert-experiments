from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from scripts.analysis.summarize_tdsc_component_ablations import (
    FULL_MODEL, SPLIT_SEEDS, VARIANTS, aligned_prediction, analyze_all,
    assert_full_baseline, fuse_probabilities, hierarchical_bootstrap,
    load_protocol, model_summary, planned_contrasts, repeat_metrics,
    validate_oof, weighted_auc_many,
)


@pytest.mark.parametrize("probability", [
    np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]),
    np.array([0.2, 0.2, 0.6, 0.6, 0.6, 0.8]),
    np.full(6, 0.5),
])
def test_vectorized_weighted_auc_matches_sklearn_including_ties(probability):
    labels = np.array([0, 1, 0, 1, 0, 1])
    weights = np.array([[1, 2, 3, 1, 2, 1], [0, 1, 1, 2, 2, 0], [1, 1, 1, 1, 1, 1]])
    expected = np.array([roc_auc_score(labels, probability, sample_weight=row) for row in weights])
    np.testing.assert_allclose(weighted_auc_many(labels, probability, weights), expected, atol=1e-14)


def test_binary_log_odds_fusion_matches_softmax_of_raw_logit_average():
    first = np.array([[-1.0, 2.0], [4.0, 1.0], [1.0, 1.0]])
    second = np.array([[3.0, 2.0], [-2.0, 2.0], [6.0, 7.0]])
    softmax = lambda values: np.exp(values[:, 1]) / np.exp(values).sum(axis=1)
    np.testing.assert_allclose(fuse_probabilities(softmax(first), softmax(second)), softmax((first + second) / 2), atol=1e-14)
    assert np.isfinite(fuse_probabilities(np.array([0.0, 1.0]), np.array([0.0, 1.0]))).all()


def _frame() -> pd.DataFrame:
    rows = []
    labels = (0, 0, 1, 1, 0, 0, 1, 1)
    full = (0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.8, 0.9)
    weak = (0.1, 0.7, 0.3, 0.9, 0.1, 0.7, 0.3, 0.9)
    for seed in SPLIT_SEEDS:
        for index, label in enumerate(labels):
            row = {"case_id": f"case-{index}", "label_true": label, "split_seed": seed, "fold": 1 + index // 4}
            for scope in ("trimil", "fusion"):
                row[f"prob_{scope}_full"] = full[index]
                for variant in VARIANTS:
                    row[f"prob_{scope}_{variant}"] = full[index] if variant == "no_position" else weak[index]
            for column in list(row):
                if column.startswith("prob_"):
                    row[column.replace("prob_", "pred_", 1)] = row[column] >= 0.5
            rows.append(row)
    return pd.DataFrame(rows)


def test_bootstrap_has_all_planned_contrasts_and_shared_weights_for_noop():
    result = hierarchical_bootstrap(_frame(), replicates=100, seed=8, expected_cases=8)
    assert len(result) == 20
    assert set(result.contrast) == {contrast for contrast, _, _ in planned_contrasts()}
    noop = result[result.contrast.str.endswith("minus_no_position")]
    np.testing.assert_array_equal(noop[["observed_delta", "ci95_lower", "ci95_upper"]].to_numpy(), np.zeros((4, 3)))
    meaningful = result[result.contrast.str.endswith("minus_center_only")].set_index("metric")
    assert (meaningful.loc["auc_delta", "observed_delta"] > 0).all()
    assert (meaningful.loc["nll_delta", "observed_delta"] < 0).all()
    assert set(result.multiplicity_adjustment) == {"none; exploratory planned contrasts"}


@pytest.mark.parametrize("alteration", ["missing", "duplicate", "label", "nan"])
def test_oof_integrity_rejects_invalid_repeated_cases(alteration):
    frame = _frame()
    if alteration == "missing":
        frame = frame.drop(index=0)
    elif alteration == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    elif alteration == "label":
        frame.loc[0, "label_true"] = 1
    else:
        frame.loc[0, "prob_fusion_full"] = np.nan
    with pytest.raises(ValueError):
        validate_oof(frame, expected_cases=8)


def test_validation_locked_prediction_alignment_does_not_depend_on_row_order(tmp_path):
    expected = pd.DataFrame({"case_id": ["cases/a.npy", "cases/b.npy"], "label_true": [0, 1]})
    path = tmp_path / "predictions.csv"
    pd.DataFrame({"npy_path": ["cases/b.npy", "cases/a.npy"], "label_true": [1, 0], "prob_1": [0.8, 0.2]}).to_csv(path, index=False)
    observed, metadata = aligned_prediction(path, expected, "variant")
    assert observed.case_id.tolist() == expected.case_id.tolist()
    assert observed.prob_variant.tolist() == [0.2, 0.8]
    assert metadata["n"] == 2 and metadata["probability_endpoint_clipped_n"] == 0
    pd.DataFrame({"npy_path": ["cases/b.npy", "cases/a.npy"], "label_true": [0, 0], "prob_1": [0.8, 0.2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="labels differ"):
        aligned_prediction(path, expected, "variant")


def test_exact_baseline_guard_rejects_result_drift(tmp_path):
    current = model_summary(repeat_metrics(_frame()))
    r3d = current[current.method.eq("trimil_full")].assign(method="r3d")
    current = pd.concat([current, r3d], ignore_index=True)
    expected = current[current.method.isin(("r3d", "trimil_full", "fusion_full"))].copy()
    expected.method = expected.method.replace({"trimil_full": "triplanar", "fusion_full": "equal_logit"})
    path = tmp_path / "expected.csv"
    expected.to_csv(path, index=False)
    assert_full_baseline(current, path)
    current.loc[current.method.eq("fusion_full"), "auc_mean"] += 0.0001
    with pytest.raises(ValueError, match="Baseline verification failed"):
        assert_full_baseline(current, path)


def test_protocol_roles_are_disjoint_and_baseline_repeats_pool_exactly_200(tmp_path):
    protocol = tmp_path / "protocol"
    baseline, full = tmp_path / "r3d_results", tmp_path / "full_results"
    indices = np.arange(200)
    for seed in SPLIT_SEEDS:
        for fold in (1, 2, 3, 4, 5):
            outer = indices[(fold - 1) * 40:fold * 40]
            development = np.setdiff1d(indices, outer)
            for role, members in (("train", development[24:]), ("selection_val", development[:24]), ("outer_test", outer)):
                source = pd.DataFrame({"npy_path": [f"cases/case-{index:03}.npy" for index in members], "label": members % 2})
                directory = protocol / "splits" / f"split_seed_{seed}" / f"fold_{fold}"
                directory.mkdir(parents=True, exist_ok=True)
                source.to_csv(directory / f"{role}.csv", index=False)
                if role != "train":
                    for root, model in ((baseline, "r3d"), (full, FULL_MODEL)):
                        target = root / model / f"split_seed_{seed}" / f"fold_{fold}" / "train_seed_3407"
                        target.mkdir(parents=True, exist_ok=True)
                        source.rename(columns={"label": "label_true"}).assign(prob_1=0.1 + 0.8 * source.label).to_csv(target / f"{role}_predictions.csv", index=False)
    roles, hashes = load_protocol(protocol)
    assert len(hashes) == 45
    fold_metrics, oof, prediction_hashes = analyze_all(roles, baseline, full)
    assert len(fold_metrics) == 45 and len(oof) == 600 and len(prediction_hashes) == 60
    assert all(len(repeat) == 200 for repeat in validate_oof(oof))
    repeat = repeat_metrics(oof)
    assert set(repeat["n"]) == {200} and set(repeat["folds"]) == {5}
    # A role-overlap error must not be treated as a new split or reduced sample.
    path = protocol / "splits/split_seed_3407/fold_1/train.csv"
    train = pd.read_csv(path)
    train.loc[0, "npy_path"] = "cases/case-000.npy"
    train.to_csv(path, index=False)
    with pytest.raises(ValueError, match="roles overlap"):
        load_protocol(protocol)
