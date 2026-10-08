from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.experiments.materialize_tdsc_public_exact_splits import (
    EXPECTED_FOLDS,
    EXPECTED_ROLE_COUNTS,
    EXPECTED_SPLIT_SEEDS,
    load_cache_index,
    materialize_exact_splits,
)
from utils.pre_cache_binary_tdsc import TARGET_SHAPE


def _write_exact_membership(path: Path) -> pd.DataFrame:
    cases = [f"DATA_{index:03d}" for index in range(1, 201)]
    labels = {case: int(index >= 85) for index, case in enumerate(cases)}
    rows = []
    for split_seed in EXPECTED_SPLIT_SEEDS:
        # The exact membership validator does not regenerate folds; this
        # deterministic layout only supplies a complete valid fixture.
        rotated = cases[split_seed % len(cases) :] + cases[: split_seed % len(cases)]
        for fold in EXPECTED_FOLDS:
            start = (fold - 1) * 40
            outer = set(rotated[start : start + 40])
            remaining = [case for case in rotated if case not in outer]
            selection = set(remaining[:24])
            for case in cases:
                role = (
                    "outer_test"
                    if case in outer
                    else "selection_val"
                    if case in selection
                    else "train"
                )
                rows.append(
                    {
                        "split_seed": split_seed,
                        "fold": fold,
                        "role": role,
                        "public_case_id": case,
                        "tdsc_official_split": "train",
                        "historical_local_cache_namespace": "unused",
                        "source_filename": f"{case}.nrrd",
                        "label": labels[case],
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def _write_cache(path: Path) -> pd.DataFrame:
    rows = []
    for index in range(1, 201):
        case = f"DATA_{index:03d}"
        rows.append(
            {
                "case_id": case,
                "source_filename": f"{case}.nrrd",
                "data_path": f"raw/{case}.nrrd",
                "npy_path": f"cache/{case}.npy",
                "label": int(index > 85),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def test_exact_public_membership_materializes_all_45_split_csvs(tmp_path):
    membership_csv = tmp_path / "tdsc_public_membership.csv"
    _write_exact_membership(membership_csv)
    cache_csv = tmp_path / "labels_all_cache.csv"
    _write_cache(cache_csv)

    output = tmp_path / "protocol"
    manifest = materialize_exact_splits(
        membership_csv=membership_csv,
        cache_csvs=[cache_csv],
        output_dir=output,
        check_files=False,
    )

    assert manifest["unique_public_cases"] == 200
    assert manifest["array_shape_contract"] == [1, 128, 128, 64]
    assert len(manifest["written_csvs"]) == 45
    assert TARGET_SHAPE == (128, 128, 64)
    assert (output / "materialization_manifest.json").is_file()

    for split_seed in EXPECTED_SPLIT_SEEDS:
        outer_seen = []
        for fold in EXPECTED_FOLDS:
            role_frames = {}
            for role, count in EXPECTED_ROLE_COUNTS.items():
                split_csv = (
                    output
                    / "splits"
                    / f"split_seed_{split_seed}"
                    / f"fold_{fold}"
                    / f"{role}.csv"
                )
                frame = pd.read_csv(split_csv)
                assert len(frame) == count
                assert list(frame.columns) == ["case_id", "source_filename", "npy_path", "label"]
                assert frame["npy_path"].str.startswith(str(tmp_path)).all()
                role_frames[role] = set(frame["case_id"])
            assert role_frames["train"].isdisjoint(role_frames["selection_val"])
            assert role_frames["train"].isdisjoint(role_frames["outer_test"])
            assert role_frames["selection_val"].isdisjoint(role_frames["outer_test"])
            outer_seen.extend(role_frames["outer_test"])
        assert len(outer_seen) == 200
        assert len(set(outer_seen)) == 200


def test_exact_public_membership_rejects_label_disagreement(tmp_path):
    membership_csv = tmp_path / "tdsc_public_membership.csv"
    _write_exact_membership(membership_csv)
    cache_csv = tmp_path / "labels_all_cache.csv"
    cache = _write_cache(cache_csv)
    cache.loc[0, "label"] = 1 - int(cache.loc[0, "label"])
    cache.to_csv(cache_csv, index=False)

    with pytest.raises(ValueError, match="Cache labels disagree"):
        materialize_exact_splits(
            membership_csv=membership_csv,
            cache_csvs=[cache_csv],
            output_dir=tmp_path / "protocol",
            check_files=False,
        )


def test_cache_index_resolves_legacy_project_relative_array_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache_root = tmp_path / "dataset" / "TDSC"
    array = cache_root / "train" / "imagesTr_cache" / "0001.npy"
    array.parent.mkdir(parents=True)
    array.touch()
    cache_csv = cache_root / "labels_train_cache.csv"
    pd.DataFrame(
        [
            {
                "data_path": "dataset/TDSC/train/imagesTr_origin/DATA_001.nrrd",
                "npy_path": "dataset/TDSC/train/imagesTr_cache/0001.npy",
                "label": 0,
            }
        ]
    ).to_csv(cache_csv, index=False)

    index, _ = load_cache_index([cache_csv])
    assert index.loc["DATA_001", "npy_path"] == str(array)


def test_public_cleanroom_queue_does_not_reference_historical_artifacts():
    script = Path("scripts/remote/run_aim_tdsc_public_cleanroom_queue.sh").read_text()

    assert "OLD_" not in script
    assert "cp -" not in script
    assert "run_aim_xyzsafe_revalidation_queue" not in script
    assert "for model in r3d convnext_triplanar_25d_mil r2plus1d" in script
    assert "--target-shape 128,128,64" in script
    assert "summarize_r2plus1d_evidence.py" in script
    assert "--lsh-paired" not in script
