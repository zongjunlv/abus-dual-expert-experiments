#!/usr/bin/env python3
"""Safely execute a repeated-CV command plan on compute-free GPUs.

The runner reserves at most one plan row per physical GPU. Within a row it
executes training, ordinary evaluation, selection-view source export, and
outer-test source export in that order. It never uses a shell, and accepts only
the repository scripts emitted by ``build_repeated_cv_protocol.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence


ALLOWED_STAGE_SCRIPTS = {
    "train": "scripts/train.py",
    "eval": "scripts/test.py",
    "source_export_selection": "scripts/analysis/export_db_vcrf_sources.py",
    "source_export_outer": "scripts/analysis/export_db_vcrf_sources.py",
}
SHELL_TOKENS = {";", "&&", "||", "|", ">", ">>", "<", "<<"}


@dataclass(frozen=True)
class Stage:
    name: str
    command: str
    artifact: Path


@dataclass(frozen=True)
class GpuInventory:
    all_gpu_ids: tuple[str, ...]
    free_gpu_ids: tuple[str, ...]
    busy_gpu_ids: tuple[str, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def command_option(command: str, option: str) -> str | None:
    parts = shlex.split(command)
    try:
        index = parts.index(option)
    except ValueError:
        return None
    if index + 1 >= len(parts):
        raise ValueError(f"Missing value after {option}: {command}")
    return parts[index + 1]


def validate_command(stage_name: str, command: str) -> list[str]:
    if stage_name not in ALLOWED_STAGE_SCRIPTS:
        raise ValueError(f"Unknown stage: {stage_name}")
    parts = shlex.split(command)
    if len(parts) < 2:
        raise ValueError(f"Incomplete {stage_name} command.")
    executable = Path(parts[0]).name.lower()
    if not executable.startswith("python"):
        raise ValueError(f"Only Python commands are allowed, got: {parts[0]}")
    if parts[1] != ALLOWED_STAGE_SCRIPTS[stage_name]:
        raise ValueError(
            f"Unexpected script for {stage_name}: {parts[1]} "
            f"(expected {ALLOWED_STAGE_SCRIPTS[stage_name]})"
        )
    if any(token in SHELL_TOKENS for token in parts):
        raise ValueError("Shell control/redirection tokens are not allowed.")
    return parts


def build_stages(row: dict[str, str]) -> list[Stage]:
    required = ("run_key", "checkpoint", "prediction_csv", "train_command", "eval_command")
    missing = [column for column in required if not row.get(column, "").strip()]
    if missing:
        raise ValueError(f"Plan row is missing required values: {missing}")
    checkpoint = Path(row["checkpoint"])
    prediction_csv = Path(row["prediction_csv"])
    train_output = command_option(row["train_command"], "--save-path")
    eval_output = command_option(row["eval_command"], "--export-pred-csv")
    if train_output is None or Path(train_output) != checkpoint:
        raise ValueError("train_command --save-path must match the checkpoint column.")
    if eval_output is None or Path(eval_output) != prediction_csv:
        raise ValueError("eval_command --export-pred-csv must match prediction_csv.")
    stages = [
        Stage("train", row["train_command"], checkpoint),
        Stage("eval", row["eval_command"], prediction_csv),
    ]
    optional = (
        ("source_export_selection", "source_export_selection_command"),
        ("source_export_outer", "source_export_command"),
    )
    for stage_name, column in optional:
        command = row.get(column, "").strip()
        if not command:
            continue
        output = command_option(command, "--output-csv")
        if not output:
            raise ValueError(f"{column} must include --output-csv.")
        stages.append(Stage(stage_name, command, Path(output)))
    return stages


def parse_gpu_inventory(
    gpu_query_output: str,
    compute_query_output: str,
    allowed_gpu_ids: Sequence[str] | None = None,
) -> GpuInventory:
    index_to_bus: dict[str, str] = {}
    for line in gpu_query_output.splitlines():
        if not line.strip() or "," not in line:
            continue
        gpu_id, bus_id = (part.strip() for part in line.split(",", 1))
        index_to_bus[gpu_id] = bus_id.lower()
    if not index_to_bus:
        raise RuntimeError("nvidia-smi returned no physical GPUs.")

    requested = list(allowed_gpu_ids) if allowed_gpu_ids else list(index_to_bus)
    unknown = [gpu_id for gpu_id in requested if gpu_id not in index_to_bus]
    if unknown:
        raise ValueError(f"Requested GPU IDs do not exist: {unknown}")
    busy_buses = {
        line.strip().lower()
        for line in compute_query_output.splitlines()
        if line.strip() and ":" in line
    }
    busy = tuple(gpu_id for gpu_id in requested if index_to_bus[gpu_id] in busy_buses)
    free = tuple(gpu_id for gpu_id in requested if index_to_bus[gpu_id] not in busy_buses)
    return GpuInventory(tuple(requested), free, busy)


def query_gpu_inventory(
    allowed_gpu_ids: Sequence[str] | None = None,
    *,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> GpuInventory:
    gpu_result = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    compute_result = run_command(
        ["nvidia-smi", "--query-compute-apps=gpu_bus_id", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_gpu_inventory(gpu_result.stdout, compute_result.stdout, allowed_gpu_ids)


class StateStore:
    """Atomic per-run state plus a small append-only event log."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.event_path = self.root / "events.jsonl"
        self._event_lock = threading.Lock()

    @staticmethod
    def _run_id(run_key: str) -> str:
        slug = "".join(character if character.isalnum() else "_" for character in run_key)
        digest = hashlib.sha256(run_key.encode("utf-8")).hexdigest()[:12]
        return f"{slug[:80]}_{digest}"

    def status_path(self, run_key: str) -> Path:
        return self.root / "runs" / f"{self._run_id(run_key)}.json"

    def read(self, run_key: str) -> dict[str, object]:
        path = self.status_path(run_key)
        if not path.is_file():
            return {"run_key": run_key, "stages": {}}
        return json.loads(path.read_text(encoding="utf-8"))

    def write(self, run_key: str, payload: dict[str, object]) -> None:
        path = self.status_path(run_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def event(self, **payload: object) -> None:
        record = {"time": utc_now(), **payload}
        with self._event_lock:
            with self.event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")


class RunnerLock:
    """Prevent two runner processes from scheduling the same state directory."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(
                f"Runner lock already exists: {self.path}. Remove it only after "
                "confirming no runner process is active."
            ) from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "created_at": utc_now()}) + "\n")
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False


def load_plan(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Command plan has no rows: {path}")
    run_keys = [row.get("run_key", "") for row in rows]
    if any(not key for key in run_keys):
        raise ValueError("Every command-plan row must have a run_key.")
    if len(run_keys) != len(set(run_keys)):
        raise ValueError("Command plan contains duplicate run_key values.")
    for row in rows:
        for stage in build_stages(row):
            validate_command(stage.name, stage.command)
    return rows


def filter_rows(
    rows: Iterable[dict[str, str]],
    *,
    models: set[str] | None = None,
    split_seeds: set[str] | None = None,
    train_seeds: set[str] | None = None,
    folds: set[str] | None = None,
) -> list[dict[str, str]]:
    result = []
    for row in rows:
        if models and row.get("model") not in models:
            continue
        if split_seeds and row.get("split_seed") not in split_seeds:
            continue
        if train_seeds and row.get("train_seed") not in train_seeds:
            continue
        if folds and row.get("fold") not in folds:
            continue
        result.append(row)
    return result


def row_is_complete(row: dict[str, str]) -> bool:
    return all(nonempty_file(stage.artifact) for stage in build_stages(row))


def run_row(
    row: dict[str, str],
    gpu_id: str,
    state: StateStore,
    *,
    project_dir: Path,
    force: bool = False,
    run_command: Callable[..., subprocess.CompletedProcess[object]] = subprocess.run,
) -> bool:
    run_key = row["run_key"]
    status = state.read(run_key)
    stage_status = status.setdefault("stages", {})
    if not isinstance(stage_status, dict):
        stage_status = {}
        status["stages"] = stage_status
    status.update({"run_key": run_key, "gpu_id": gpu_id, "updated_at": utc_now()})
    state.write(run_key, status)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu_id
    for stage in build_stages(row):
        if not force and nonempty_file(stage.artifact):
            stage_status[stage.name] = {
                "status": "skipped_existing",
                "artifact": str(stage.artifact),
                "updated_at": utc_now(),
            }
            state.write(run_key, status)
            state.event(run_key=run_key, gpu_id=gpu_id, stage=stage.name, status="skipped_existing")
            continue

        arguments = validate_command(stage.name, stage.command)
        stage_status[stage.name] = {
            "status": "running",
            "artifact": str(stage.artifact),
            "started_at": utc_now(),
        }
        state.write(run_key, status)
        state.event(run_key=run_key, gpu_id=gpu_id, stage=stage.name, status="running")
        started = time.monotonic()
        try:
            completed = run_command(
                arguments,
                cwd=project_dir,
                env=environment,
                check=False,
            )
            return_code = int(completed.returncode)
        except OSError as error:
            return_code = -1
            stage_status[stage.name] = {
                "status": "failed",
                "artifact": str(stage.artifact),
                "return_code": return_code,
                "error": str(error),
                "elapsed_sec": time.monotonic() - started,
                "finished_at": utc_now(),
            }
            state.write(run_key, status)
            state.event(run_key=run_key, gpu_id=gpu_id, stage=stage.name, status="failed")
            return False

        succeeded = return_code == 0 and nonempty_file(stage.artifact)
        stage_status[stage.name] = {
            "status": "success" if succeeded else "failed",
            "artifact": str(stage.artifact),
            "return_code": return_code,
            "artifact_exists": nonempty_file(stage.artifact),
            "elapsed_sec": time.monotonic() - started,
            "finished_at": utc_now(),
        }
        state.write(run_key, status)
        state.event(
            run_key=run_key,
            gpu_id=gpu_id,
            stage=stage.name,
            status="success" if succeeded else "failed",
            return_code=return_code,
        )
        if not succeeded:
            return False

    status["status"] = "success"
    status["updated_at"] = utc_now()
    state.write(run_key, status)
    state.event(run_key=run_key, gpu_id=gpu_id, status="success")
    return True


def print_dry_run(rows: Sequence[dict[str, str]], force: bool) -> None:
    for row in rows:
        print(f"RUN {row['run_key']}")
        for stage in build_stages(row):
            action = "RUN" if force or not nonempty_file(stage.artifact) else "SKIP existing"
            print(f"  {action:13} {stage.name}: {stage.command}")


def execute_plan(
    rows: list[dict[str, str]],
    *,
    project_dir: Path,
    state_dir: Path,
    allowed_gpu_ids: Sequence[str] | None,
    poll_seconds: float,
    force: bool,
    stop_on_failure: bool,
) -> bool:
    state = StateStore(state_dir)
    pending = [row for row in rows if force or not row_is_complete(row)]
    skipped = len(rows) - len(pending)
    state.event(status="runner_start", planned=len(rows), pending=len(pending), skipped=skipped)
    if not pending:
        print("All selected rows already have non-empty artifacts.")
        return True

    initial_inventory = query_gpu_inventory(allowed_gpu_ids)
    selected_gpu_ids = initial_inventory.all_gpu_ids
    failures = 0
    with RunnerLock(state_dir / "runner.lock"):
        with ThreadPoolExecutor(max_workers=len(selected_gpu_ids)) as pool:
            active: dict[Future[bool], str] = {}
            stop_submitting = False
            while pending or active:
                finished = [future for future in active if future.done()]
                for future in finished:
                    gpu_id = active.pop(future)
                    try:
                        succeeded = future.result()
                    except Exception as error:  # retain scheduler state for unexpected worker errors
                        succeeded = False
                        state.event(gpu_id=gpu_id, status="worker_exception", error=str(error))
                    if not succeeded:
                        failures += 1
                        if stop_on_failure:
                            stop_submitting = True

                if pending and not stop_submitting:
                    inventory = query_gpu_inventory(selected_gpu_ids)
                    reserved = set(active.values())
                    available = [gpu_id for gpu_id in inventory.free_gpu_ids if gpu_id not in reserved]
                    for gpu_id in available:
                        if not pending:
                            break
                        row = pending.pop(0)
                        print(f"Assign {row['run_key']} to physical GPU {gpu_id}")
                        future = pool.submit(
                            run_row,
                            row,
                            gpu_id,
                            state,
                            project_dir=project_dir,
                            force=force,
                        )
                        active[future] = gpu_id

                if stop_submitting and not active:
                    break
                if pending or active:
                    time.sleep(min(max(poll_seconds, 0.1), 60.0))

    state.event(status="runner_finish", failures=failures, unstarted=len(pending))
    return failures == 0 and not pending


def parse_gpu_ids(value: str | None) -> list[str] | None:
    if value is None:
        return None
    result = [part.strip() for part in value.split(",") if part.strip()]
    if not result or any(not item.isdigit() for item in result):
        raise argparse.ArgumentTypeError("--gpus must be a comma-separated list of indices.")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("--gpus contains duplicate indices.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--gpus", type=parse_gpu_ids, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--split-seeds", nargs="*", default=None)
    parser.add_argument("--train-seeds", nargs="*", default=None)
    parser.add_argument("--folds", nargs="*", default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_plan(args.command_plan)
    rows = filter_rows(
        rows,
        models=set(args.models) if args.models else None,
        split_seeds=set(args.split_seeds) if args.split_seeds else None,
        train_seeds=set(args.train_seeds) if args.train_seeds else None,
        folds=set(args.folds) if args.folds else None,
    )
    if args.max_runs is not None:
        if args.max_runs < 1:
            raise ValueError("--max-runs must be positive.")
        rows = rows[: args.max_runs]
    if not rows:
        raise SystemExit("No command-plan rows matched the requested filters.")

    if args.dry_run:
        print_dry_run(rows, force=args.force)
        return

    state_dir = args.state_dir or args.command_plan.parent / "runner_state"
    succeeded = execute_plan(
        rows,
        project_dir=args.project_dir.resolve(),
        state_dir=state_dir.resolve(),
        allowed_gpu_ids=args.gpus,
        poll_seconds=args.poll_seconds,
        force=args.force,
        stop_on_failure=args.stop_on_failure,
    )
    raise SystemExit(0 if succeeded else 1)


if __name__ == "__main__":
    main()
