from __future__ import annotations

from pathlib import Path


def resolve_save_path(save_path_arg, model_name, final_train: bool = False):
    if save_path_arg:
        return Path(save_path_arg)
    if final_train:
        return Path("assets/checkpoints_3d") / f"final_model_{model_name}.pth"
    return Path("assets/checkpoints_3d") / f"best_model_{model_name}.pth"


def append_path_suffix(path: Path, suffix: str | None) -> Path:
    if not suffix:
        return path
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def resolve_summary_csv_path(base_path: Path, suffix: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{suffix}.csv")


def resolve_run_save_path(base_save_path, seed, multi_seed, suffix=None):
    save_path = append_path_suffix(base_save_path, f"seed{seed}" if multi_seed else None)
    return append_path_suffix(save_path, suffix)


def resolve_run_name(model_name, seed, multi_seed, suffix=None):
    run_name = model_name
    if multi_seed:
        run_name = f"{run_name}_seed{seed}"
    if suffix:
        run_name = f"{run_name}_{suffix}"
    return run_name


def resolve_log_paths(log_dir_arg, model_name, seed, multi_seed, suffix=None):
    log_dir = Path(log_dir_arg)
    run_name = resolve_run_name(model_name, seed, multi_seed, suffix)
    csv_path = log_dir / f"train_log_{run_name}.csv"
    plot_path = log_dir / f"train_curves_{run_name}.png"
    return log_dir, csv_path, plot_path


def resolve_tensorboard_log_dir(
    tensorboard_dir_arg,
    log_dir_arg,
    model_name,
    seed,
    multi_seed,
    suffix=None,
):
    root_dir = (
        Path(tensorboard_dir_arg)
        if tensorboard_dir_arg
        else Path(log_dir_arg) / "tensorboard"
    )
    return root_dir / resolve_run_name(model_name, seed, multi_seed, suffix)
