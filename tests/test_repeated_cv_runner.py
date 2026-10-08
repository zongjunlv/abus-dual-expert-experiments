from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.experiments.run_repeated_cv_protocol import (
    StateStore,
    build_stages,
    filter_rows,
    load_plan,
    parse_gpu_ids,
    parse_gpu_inventory,
    row_is_complete,
    run_row,
    validate_command,
)


def _row(tmp_path: Path, model: str = "dual_backbone_view_consistency_reliability"):
    checkpoint = tmp_path / "checkpoint.pth"
    prediction = tmp_path / "prediction.csv"
    selection_source = tmp_path / "selection_sources.csv"
    outer_source = tmp_path / "outer_sources.csv"
    base = {
        "run_key": f"{model}/split_seed_11/fold_1/train_seed_101",
        "model": model,
        "split_seed": "11",
        "train_seed": "101",
        "fold": "1",
        "checkpoint": str(checkpoint),
        "prediction_csv": str(prediction),
        "train_command": f"python scripts/train.py --save-path {checkpoint}",
        "eval_command": f"python scripts/test.py --export-pred-csv {prediction}",
        "source_export_selection_command": "",
        "source_export_command": "",
    }
    if model == "dual_backbone_view_consistency_reliability":
        base["source_export_selection_command"] = (
            "python scripts/analysis/export_db_vcrf_sources.py "
            f"--output-csv {selection_source}"
        )
        base["source_export_command"] = (
            "python scripts/analysis/export_db_vcrf_sources.py "
            f"--output-csv {outer_source}"
        )
    return base


def test_gpu_inventory_marks_any_active_compute_process_as_busy():
    inventory = parse_gpu_inventory(
        "0, 00000000:01:00.0\n1, 00000000:02:00.0\n2, 00000000:03:00.0\n",
        "00000000:02:00.0\n",
        ["0", "1"],
    )
    assert inventory.all_gpu_ids == ("0", "1")
    assert inventory.free_gpu_ids == ("0",)
    assert inventory.busy_gpu_ids == ("1",)


def test_command_validation_rejects_non_whitelisted_or_shell_commands():
    assert validate_command("train", "python scripts/train.py --epochs 1")[:2] == [
        "python",
        "scripts/train.py",
    ]
    with pytest.raises(ValueError, match="Unexpected script"):
        validate_command("train", "python scripts/test.py")
    with pytest.raises(ValueError, match="Shell"):
        validate_command("train", "python scripts/train.py ; touch unsafe")


def test_stage_artifacts_must_match_command_outputs(tmp_path):
    row = _row(tmp_path, model="r3d")
    row["checkpoint"] = str(tmp_path / "different.pth")
    with pytest.raises(ValueError, match="save-path"):
        build_stages(row)


def test_build_stages_preserves_train_eval_and_source_export_order(tmp_path):
    stages = build_stages(_row(tmp_path))
    assert [stage.name for stage in stages] == [
        "train",
        "eval",
        "source_export_selection",
        "source_export_outer",
    ]
    baseline = build_stages(_row(tmp_path, model="r3d"))
    assert [stage.name for stage in baseline] == ["train", "eval"]


def test_run_row_resumes_from_existing_artifacts(tmp_path):
    row = _row(tmp_path)
    stages = build_stages(row)
    stages[0].artifact.write_text("checkpoint", encoding="utf-8")
    invoked: list[list[str]] = []

    def fake_run(arguments, **kwargs):
        invoked.append(list(arguments))
        if "scripts/test.py" in arguments:
            stages[1].artifact.write_text("predictions", encoding="utf-8")
        else:
            output_index = arguments.index("--output-csv") + 1
            Path(arguments[output_index]).write_text("sources", encoding="utf-8")
        return subprocess.CompletedProcess(arguments, 0)

    state = StateStore(tmp_path / "state")
    assert run_row(
        row,
        "3",
        state,
        project_dir=tmp_path,
        run_command=fake_run,
    )
    assert len(invoked) == 3
    assert all(command[1] != "scripts/train.py" for command in invoked)
    assert row_is_complete(row)
    status = state.read(row["run_key"])
    assert status["status"] == "success"
    assert status["stages"]["train"]["status"] == "skipped_existing"


def test_load_and_filter_plan_uses_only_standard_library_csv(tmp_path):
    rows = [_row(tmp_path, "r3d"), _row(tmp_path, "r2plus1d")]
    plan = tmp_path / "command_plan.csv"
    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    loaded = load_plan(plan)
    selected = filter_rows(loaded, models={"r3d"}, split_seeds={"11"})
    assert len(selected) == 1
    assert selected[0]["model"] == "r3d"

    script = Path("scripts/experiments/run_repeated_cv_protocol.py").resolve()
    dry_run = subprocess.run(
        [sys.executable, str(script), "--command-plan", str(plan), "--dry-run", "--max-runs", "1"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "RUN r3d/split_seed_11/fold_1/train_seed_101" in dry_run.stdout
    assert "scripts/train.py" in dry_run.stdout


def test_gpu_id_parser_rejects_duplicates():
    assert parse_gpu_ids("0,2,7") == ["0", "2", "7"]
    with pytest.raises(Exception, match="duplicate"):
        parse_gpu_ids("0,0")
