from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from scripts.analysis.summarize_r2plus1d_evidence import (
    bootstrap_contrasts,
    case_id,
    cluster_labels,
    map_cluster_weights,
    parse_args,
    per_repeat_metrics,
    prepare_lsh,
    r2_path,
    stratified_cluster_weights,
    summarize_prepared_datasets,
)


def _paired_frame() -> pd.DataFrame:
    rows = []
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    r2 = np.array([0.1, 0.2, 0.6, 0.7, 0.3, 0.4, 0.8, 0.9])
    r3d = np.array([0.1, 0.2, 0.3, 0.7, 0.3, 0.7, 0.8, 0.9])
    tri = np.array([0.2, 0.1, 0.4, 0.6, 0.2, 0.6, 0.9, 0.8])
    fusion = (r3d + tri) / 2
    for split_seed in (42, 3407, 1234):
        for index, label in enumerate(labels):
            row = {
                "case_id": f"case-{index}",
                "label_true": label,
                "split_seed": split_seed,
                "prob_r2plus1d": r2[index],
                "prob_r3d": r3d[index],
                "prob_triplanar": tri[index],
                "prob_equal_logit": fusion[index],
            }
            for method in ("r2plus1d", "r3d", "triplanar", "equal_logit"):
                row[f"threshold_{method}"] = 0.5
                row[f"pred_{method}"] = row[f"prob_{method}"] >= 0.5
            rows.append(row)
    return pd.DataFrame(rows)


def test_case_id_normalizes_machine_specific_prefix() -> None:
    assert case_id("/root/data/TDSC/test/imagesTs_cache/0003.npy") == "TDSC/test/imagesTs_cache/0003.npy"


def test_metrics_and_bootstrap_compare_all_three_methods_with_r2() -> None:
    frame = _paired_frame()
    by_repeat = per_repeat_metrics(frame, "tdsc")
    bootstrap = bootstrap_contrasts(frame, "tdsc", replicates=20, seed=7)

    assert len(by_repeat) == 12
    assert set(by_repeat.method) == {"r2plus1d", "r3d", "triplanar", "equal_logit"}
    assert len(bootstrap) == 3 * 2 * 2
    assert set(bootstrap.bootstrap_unit) == {
        "stratified_case_cluster",
        "split_seed_and_stratified_case_cluster",
    }
    assert set(bootstrap.cluster_scope) == {"union_of_outer_test_cases"}
    assert set(bootstrap.split_seed_resampled) == {False, True}
    fusion_auc = bootstrap.query(
        "contrast == 'equal_logit minus r2plus1d' and metric == 'auc_delta'"
    )
    assert (fusion_auc.mean_delta > 0).all()


def test_bootstrap_supports_repeated_holdouts_with_different_cases() -> None:
    frame = _paired_frame()
    frame = frame.loc[
        ~(
            ((frame.split_seed == 42) & frame.case_id.isin(["case-0", "case-4"]))
            | ((frame.split_seed == 3407) & frame.case_id.isin(["case-1", "case-5"]))
            | ((frame.split_seed == 1234) & frame.case_id.isin(["case-2", "case-6"]))
        )
    ]
    bootstrap = bootstrap_contrasts(frame, "lsh", replicates=10, seed=9)

    assert set(bootstrap.bootstrap_unit) == {
        "stratified_patient_cluster",
        "split_seed_and_stratified_patient_cluster",
    }
    assert set(bootstrap.cluster_scope) == {"union_of_outer_test_patients"}
    assert np.isfinite(bootstrap[["observed_delta", "ci95_lower", "ci95_upper"]]).all().all()


def test_union_cluster_weight_is_shared_for_patient_across_splits() -> None:
    frame = _paired_frame()
    frame = frame.loc[
        ~(
            ((frame.split_seed == 42) & frame.case_id.isin(["case-0", "case-4"]))
            | ((frame.split_seed == 3407) & frame.case_id.isin(["case-1", "case-5"]))
            | ((frame.split_seed == 1234) & frame.case_id.isin(["case-2", "case-6"]))
        )
    ]
    labels = cluster_labels(frame)
    weights = stratified_cluster_weights(labels, np.random.default_rng(19))
    mapped = {}
    for split_seed, run in frame.groupby("split_seed"):
        indexed = run.set_index("case_id")
        mapped[int(split_seed)] = pd.Series(
            map_cluster_weights(indexed, weights), index=indexed.index
        )

    shared = set.intersection(*(set(values.index) for values in mapped.values()))
    assert shared
    for patient in shared:
        observed = {values.loc[patient] for values in mapped.values()}
        assert observed == {float(weights.get(patient, 0.0))}


def test_two_datasets_emit_all_six_contrasts_for_both_metrics() -> None:
    frame = _paired_frame()
    combined = pd.concat(
        [
            bootstrap_contrasts(frame, "tdsc", replicates=10, seed=7),
            bootstrap_contrasts(frame, "lsh", replicates=10, seed=8),
        ],
        ignore_index=True,
    )
    hierarchical = combined.loc[
        combined.bootstrap_unit.str.startswith("split_seed_and_stratified_")
    ]
    expected_pairs = {
        (dataset, f"{method} minus r2plus1d")
        for dataset in ("tdsc", "lsh")
        for method in ("r3d", "triplanar", "equal_logit")
    }
    assert set(zip(hierarchical.dataset, hierarchical.contrast)) == expected_pairs
    metric_sets = hierarchical.groupby(["dataset", "contrast"]).metric.apply(set)
    assert all(values == {"auc_delta", "nll_delta"} for values in metric_sets)


def test_cli_allows_public_tdsc_only_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_r2plus1d_evidence.py",
            "--tdsc-paired",
            "tdsc.csv",
            "--r2-result-root",
            "results",
            "--output-dir",
            "tables",
        ],
    )
    args = parse_args()

    assert args.tdsc_paired.name == "tdsc.csv"
    assert args.lsh_paired is None
    assert args.r2_result_root.name == "results"


def test_public_tdsc_only_output_tables_are_complete() -> None:
    per_repeat, aggregate, bootstrap = summarize_prepared_datasets(
        {"tdsc": _paired_frame()},
        bootstrap_replicates=10,
        bootstrap_seed=11,
    )

    assert set(per_repeat.dataset) == {"tdsc"}
    assert len(per_repeat) == 3 * 4
    assert len(aggregate) == 4 * 7
    assert set(aggregate.dataset) == {"tdsc"}
    assert len(bootstrap) == 3 * 2 * 2
    assert set(bootstrap.dataset) == {"tdsc"}
    assert set(bootstrap.contrast) == {
        "r3d minus r2plus1d",
        "triplanar minus r2plus1d",
        "equal_logit minus r2plus1d",
    }
    assert set(bootstrap.metric) == {"auc_delta", "nll_delta"}
    assert set(bootstrap.bootstrap_unit) == {
        "stratified_case_cluster",
        "split_seed_and_stratified_case_cluster",
    }
    assert set(bootstrap.cluster_scope) == {"union_of_outer_test_cases"}


def test_lsh_triplanar_uses_saved_selection_validation_threshold(tmp_path) -> None:
    paired_rows = []
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    tri_probs = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    for split_seed in (3407, 42, 1234):
        for index, (label, tri_prob) in enumerate(zip(labels, tri_probs)):
            paired_rows.append(
                {
                    "case_id": f"LSH/test/imagesTs_cache/{index:04d}.npy",
                    "label_true": label,
                    "split_seed": split_seed,
                    "train_seed": 3407,
                    "prob_r3d": tri_prob,
                    "prob_triplanar": tri_prob,
                    "prob_equal_logit": tri_prob,
                    "threshold_r3d": 0.5,
                    "threshold_triplanar": 0.8,
                    "threshold_equal_logit": 0.5,
                }
            )
        for partition in ("selection_val", "outer_test"):
            prediction_path = r2_path(tmp_path, "lsh", split_seed, 1, partition)
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "npy_path": [
                        f"/root/data/LSH/test/imagesTs_cache/{index:04d}.npy"
                        for index in range(len(labels))
                    ],
                    "label_true": labels,
                    "prob_1": tri_probs,
                }
            ).to_csv(prediction_path, index=False)
    paired_path = tmp_path / "paired.csv"
    pd.DataFrame(paired_rows).to_csv(paired_path, index=False)

    prepared = prepare_lsh(paired_path, tmp_path)
    summary = per_repeat_metrics(prepared, "lsh")
    tri = summary.loc[summary.method.eq("triplanar")]

    assert len(tri) == 3
    assert set(prepared.threshold_triplanar) == {0.8}
    assert set(tri.threshold_policy) == {"selection_validation_locked_youden"}
    assert np.allclose(tri.accuracy, 0.75)


def test_lsh_triplanar_rejects_missing_locked_threshold(tmp_path) -> None:
    paired_path = tmp_path / "paired.csv"
    pd.DataFrame(
        {"train_seed": [3407], "split_seed": [3407], "case_id": ["case-0"]}
    ).to_csv(paired_path, index=False)

    with np.testing.assert_raises_regex(ValueError, "lacks validation-locked"):
        prepare_lsh(paired_path, tmp_path)
