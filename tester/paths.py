from __future__ import annotations

import re
from pathlib import Path


SEED_SUFFIX_PATTERN = re.compile(r"_seed\d+$")


def append_path_suffix(path: Path, suffix: str | None) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def append_seed_suffix(path: Path, seed: int | None) -> Path:
    if seed is None:
        return path

    expected_suffix = f"_seed{seed}"
    if SEED_SUFFIX_PATTERN.search(path.stem):
        rewritten_stem = SEED_SUFFIX_PATTERN.sub(expected_suffix, path.stem)
        return path.with_name(f"{rewritten_stem}{path.suffix}")
    return path.with_name(f"{path.stem}{expected_suffix}{path.suffix}")


def resolve_checkpoint_path(checkpoint_arg, model_name, seed: int | None = None):
    checkpoint_path = (
        Path(checkpoint_arg)
        if checkpoint_arg
        else Path("assets/checkpoints_3d") / f"best_model_{model_name}.pth"
    )
    return append_seed_suffix(checkpoint_path, seed)


def resolve_checkpoint_paths(args, seed: int | None = None):
    if args.checkpoints:
        return [Path(item) for item in args.checkpoints]
    return [resolve_checkpoint_path(args.checkpoint, args.model, seed=seed)]


def resolve_export_path(
    export_arg,
    model_name,
    ensemble_size=1,
    seed: int | None = None,
    force_seed_suffix: bool = False,
):
    if export_arg:
        export_path = Path(export_arg)
        if seed is not None and force_seed_suffix:
            return append_seed_suffix(export_path, seed)
        return export_path

    if ensemble_size > 1:
        export_path = Path("assets/test_outputs") / f"predictions_{model_name}_ensemble.csv"
    else:
        export_path = Path("assets/test_outputs") / f"predictions_{model_name}.csv"
    return append_seed_suffix(export_path, seed)


def resolve_summary_path(summary_arg, model_name):
    if summary_arg:
        return Path(summary_arg)
    return Path("assets/test_outputs") / f"test_metrics_{model_name}_multiseed_summary.csv"
