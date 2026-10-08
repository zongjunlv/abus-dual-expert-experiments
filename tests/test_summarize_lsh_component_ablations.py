from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss, roc_auc_score

from scripts.analysis.summarize_lsh_component_ablations import (
    EPS, FULL_MODEL, METRICS, SPLIT_SEEDS, TRAIN_SEEDS, VARIANTS,
    analyze_all, assert_full_baseline, hierarchical_bootstrap, load_protocol,
    model_summary, planned_contrasts, prediction_path, validate_runs,
    validate_variant_completion,
)


def _runs() -> dict:
    """Three overlapping holdouts, not nine independent patient cohorts."""
    runs = {}
    for split_index, split_seed in enumerate(SPLIT_SEEDS):
        identifiers = (np.arange(40) + 16 * split_index) % 80
        labels = identifiers % 2
        for train_index, train_seed in enumerate(TRAIN_SEEDS):
            full = np.where(labels == 1, 0.75, 0.25) + train_index * 0.015
            weaker = full.copy()
            weaker[identifiers % 5 == 0] = 1 - weaker[identifiers % 5 == 0]
            frame = pd.DataFrame({"case_id": [f"patient-{index:03}" for index in identifiers], "label_true": labels})
            for scope in ("trimil", "fusion"):
                frame[f"prob_{scope}_full"] = full
                for variant in VARIANTS:
                    frame[f"prob_{scope}_{variant}"] = full if variant == "no_position" else weaker
            runs[(split_seed, train_seed)] = frame
    return runs


def test_patient_union_is_shared_across_optimization_seeds_and_overlapping_splits():
    patients = validate_runs(_runs(), expected_n=40)
    assert len(patients) == 72
    assert len(patients) < 40 * 9
    assert not patients.case_id.duplicated().any()


def test_hierarchical_bootstrap_noop_is_exact_and_all_contrasts_are_reported():
    result = hierarchical_bootstrap(_runs(), replicates=100, seed=42, expected_n=40)
    assert len(result) == 20
    assert set(result.contrast) == {name for name, _, _ in planned_contrasts()}
    assert result.shared_id_weights.all()
    assert set(result.unique_outer_test_patients) == {72}
    assert set(result.run_class_degenerate_draws) == {0}
    assert set(result.multiplicity_adjustment) == {"none; ten exploratory planned contrasts"}
    identical = result[result.contrast.str.endswith("minus_no_position")]
    np.testing.assert_array_equal(identical[["observed_delta", "ci95_lower", "ci95_upper"]], np.zeros((4, 3)))
    meaningful = result[result.contrast.str.endswith("minus_center_only")]
    assert (meaningful[meaningful.metric.eq("auc_delta")].observed_delta > 0).all()
    assert (meaningful[meaningful.metric.eq("nll_delta")].observed_delta < 0).all()


def test_observed_effect_is_mean_of_run_metrics_not_pooled_patient_predictions():
    runs = _runs()
    result = hierarchical_bootstrap(runs, replicates=80, seed=11, expected_n=40)
    auc = np.mean([roc_auc_score(frame.label_true, frame.prob_fusion_full) - roc_auc_score(frame.label_true, frame.prob_fusion_center_only) for frame in runs.values()])
    nll = np.mean([log_loss(frame.label_true, np.clip(frame.prob_fusion_full, EPS, 1-EPS), labels=[0, 1]) - log_loss(frame.label_true, np.clip(frame.prob_fusion_center_only, EPS, 1-EPS), labels=[0, 1]) for frame in runs.values()])
    values = result[result.contrast.eq("fusion_full_minus_center_only")].set_index("metric")
    assert values.loc["auc_delta", "observed_delta"] == pytest.approx(auc)
    assert values.loc["nll_delta", "observed_delta"] == pytest.approx(nll)


def test_vectorized_hierarchy_matches_scalar_patient_cluster_oracle():
    runs, replicates, seed = _runs(), 50, 812
    patients = validate_runs(runs, expected_n=40)
    labels = patients.label_true.to_numpy(int)
    rng = np.random.default_rng(seed)
    weights = np.zeros((replicates, len(patients)), dtype=int)
    for label in (0, 1):
        members = np.flatnonzero(labels == label)
        weights[:, members] = rng.multinomial(len(members), np.full(len(members), 1 / len(members)), size=replicates)
    selected_splits = rng.integers(len(SPLIT_SEEDS), size=(replicates, len(SPLIT_SEEDS)))
    selected_trains = rng.integers(len(TRAIN_SEEDS), size=(replicates, len(SPLIT_SEEDS), len(TRAIN_SEEDS)))
    id_index = {identifier: index for index, identifier in enumerate(patients.case_id)}
    oracle = {"auc_delta": [], "nll_delta": []}
    for draw in range(replicates):
        per_draw = {metric: [] for metric in oracle}
        # The train-seed draw belongs to each selected split occurrence.
        # It is not an unconditional sample of nine exchangeable run rows.
        for occurrence, split_index in enumerate(selected_splits[draw]):
            for train_index in selected_trains[draw, occurrence]:
                frame = runs[(SPLIT_SEEDS[split_index], TRAIN_SEEDS[train_index])]
                sample_weights = weights[draw, [id_index[identifier] for identifier in frame.case_id]]
                y = frame.label_true.to_numpy(int)
                if not (sample_weights[y == 0].sum() and sample_weights[y == 1].sum()):
                    continue
                full, ablated = frame.prob_fusion_full.to_numpy(float), frame.prob_fusion_center_only.to_numpy(float)
                per_draw["auc_delta"].append(roc_auc_score(y, full, sample_weight=sample_weights) - roc_auc_score(y, ablated, sample_weight=sample_weights))
                per_draw["nll_delta"].append(log_loss(y, np.clip(full, EPS, 1-EPS), sample_weight=sample_weights, labels=[0, 1]) - log_loss(y, np.clip(ablated, EPS, 1-EPS), sample_weight=sample_weights, labels=[0, 1]))
        for metric in oracle:
            oracle[metric].append(np.mean(per_draw[metric]))
    actual = hierarchical_bootstrap(runs, replicates=replicates, seed=seed, expected_n=40)
    actual = actual[actual.contrast.eq("fusion_full_minus_center_only")].set_index("metric")
    for metric, draws in oracle.items():
        np.testing.assert_allclose(actual.loc[metric, ["bootstrap_mean", "ci95_lower", "ci95_upper"]].to_numpy(float),
                                   [np.mean(draws), *np.quantile(draws, [0.025, 0.975])], atol=1e-14, rtol=0)


@pytest.mark.parametrize("alteration", ["missing_run", "missing_patient", "duplicate", "changed_seed_membership", "label", "nan"])
def test_repeated_holdout_integrity_rejects_invalid_runs(alteration):
    runs = _runs()
    key = (SPLIT_SEEDS[0], TRAIN_SEEDS[0])
    if alteration == "missing_run":
        del runs[key]
    elif alteration == "missing_patient":
        runs[key] = runs[key].iloc[1:]
    elif alteration == "duplicate":
        runs[key].loc[1, "case_id"] = runs[key].case_id.iloc[0]
    elif alteration == "changed_seed_membership":
        runs[key].loc[0, "case_id"] = "changed-patient"
    elif alteration == "label":
        for train_seed in TRAIN_SEEDS:
            # This ID also appears in another split: training-pair equality
            # alone must not let an inconsistent source label through.
            frame = runs[(SPLIT_SEEDS[1], train_seed)]
            frame.loc[frame.case_id.eq("patient-016"), "label_true"] = 1
    else:
        runs[key].loc[0, "prob_fusion_full"] = np.nan
    with pytest.raises(ValueError):
        validate_runs(runs, expected_n=40)


def _baseline_metrics():
    rows = []
    for split_seed in SPLIT_SEEDS:
        for train_seed in TRAIN_SEEDS:
            for method in ("r3d", "trimil_full", "fusion_full"):
                rows.append({"split_seed": split_seed, "train_seed": train_seed, "method": method,
                             "threshold": 0.48, **{metric: 0.9 if metric != "nll" else 0.3 for metric in METRICS}})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("alteration", ["auc", "threshold", "missing_run"])
def test_baseline_guard_checks_all_nine_paired_runs_and_thresholds(tmp_path, alteration):
    current = _baseline_metrics()
    expected = current.copy()
    expected.method = expected.method.replace({"trimil_full": "triplanar", "fusion_full": "equal_logit"})
    path = tmp_path / "per_run_metrics.csv"
    expected.to_csv(path, index=False)
    assert_full_baseline(current, path)
    if alteration == "missing_run":
        current = current.iloc[1:]
    else:
        current.loc[0, alteration] += 0.0001
    with pytest.raises(ValueError):
        assert_full_baseline(current, path)


def test_model_summary_uses_sample_sd_and_preserves_nine_run_estimand():
    frame = _baseline_metrics()
    frame.loc[frame.method.eq("fusion_full"), "auc"] = np.arange(9) / 100 + 0.9
    summary = model_summary(frame).set_index("method")
    assert set(summary.run_n) == {9}
    assert summary.loc["fusion_full", "auc_mean"] == pytest.approx(0.94)
    assert summary.loc["fusion_full", "auc_sd"] == pytest.approx(np.std(np.arange(9) / 100, ddof=1))


def test_locked_protocol_and_prediction_roles_are_exact_and_validation_driven(tmp_path):
    protocol, baseline = tmp_path / "protocol", tmp_path / "baseline"
    for split_index, split_seed in enumerate(SPLIT_SEEDS):
        indices = np.roll(np.arange(594), split_index * 130)
        memberships = {"train": indices[:415], "selection_val": indices[415:475], "outer_test": indices[475:]}
        for role, members in memberships.items():
            source = pd.DataFrame({"npy_path": [f"cases/patient-{index:03}.npy" for index in members], "label": members % 2})
            directory = protocol / "splits" / f"split_seed_{split_seed}" / "fold_1"
            directory.mkdir(parents=True, exist_ok=True)
            source.to_csv(directory / f"{role}.csv", index=False)
            if role != "train":
                for model in ("r3d", FULL_MODEL):
                    for train_seed in TRAIN_SEEDS:
                        destination = prediction_path(baseline, model, split_seed, train_seed, role)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        prediction = source.rename(columns={"label": "label_true"})
                        prediction = prediction.assign(prob_1=np.where(source.label == 1, 0.8, 0.2))
                        prediction.iloc[::-1].to_csv(destination, index=False)
    roles, hashes = load_protocol(protocol)
    assert len(roles) == 9 and len(hashes) == 9
    per_run, runs, prediction_hashes = analyze_all(roles, baseline, baseline)
    assert len(per_run) == 27 and len(runs) == 9 and len(prediction_hashes) == 36
    assert set(per_run["n"]) == {119}
    # Threshold is determined by selection-validation predictions, not by
    # the order of exports or a default 0.5 cut-off.
    np.testing.assert_allclose(per_run.threshold, np.full(27, 0.8), atol=1e-14)
    assert (per_run.accuracy == 1).all()
    validate_runs(runs)
    path = protocol / "splits" / f"split_seed_{SPLIT_SEEDS[0]}" / "fold_1" / "train.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "npy_path"] = roles[(SPLIT_SEEDS[0], "outer_test")].case_id.iloc[0]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="roles overlap"):
        load_protocol(protocol)


def test_incomplete_cell_is_not_accepted_as_finished_training(tmp_path):
    with pytest.raises(ValueError, match="Incomplete LSH ablation cell"):
        validate_variant_completion(tmp_path)


def test_tampered_cell_signature_is_rejected_before_any_statistics(tmp_path):
    run = prediction_path(tmp_path, f"{FULL_MODEL}_{VARIANTS[0]}", SPLIT_SEEDS[0], TRAIN_SEEDS[0], "outer_test").parent
    run.mkdir(parents=True)
    for marker in ("TRAIN_DONE", "DONE", "prediction_sha256.json"):
        (run / marker).write_text("{}\n")
    config = {"model": "wrong-model"}
    signed = {"configuration": config, "signature_sha256": hashlib.sha256(b"unrelated configuration").hexdigest()}
    (run / "configuration.json").write_text(json.dumps(signed))
    with pytest.raises(ValueError, match="Invalid LSH cell signature"):
        validate_variant_completion(tmp_path)
