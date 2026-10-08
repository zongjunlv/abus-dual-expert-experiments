from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd

from trainer.distributed import is_main_process

SWANLAB_CLI_MODES = ("cloud", "online", "local", "offline", "disabled")
SWANLAB_MODE_ALIASES = {
    "online": "cloud",
}


def normalize_swanlab_mode(mode: str | None) -> str:
    normalized = (mode or "disabled").strip().lower()
    return SWANLAB_MODE_ALIASES.get(normalized, normalized)


if "SWANLAB_MODE" in os.environ:
    os.environ["SWANLAB_MODE"] = normalize_swanlab_mode(os.environ["SWANLAB_MODE"])

try:
    import swanlab
except ImportError:
    swanlab = None

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


SWANLAB_AVAILABLE = swanlab is not None

TENSORBOARD_SCALAR_TAGS = {
    "train_loss": "Loss/train",
    "val_loss": "Loss/val",
    "acc": "Metrics/accuracy",
    "auc": "Metrics/auc",
    "sensitivity": "Metrics/sensitivity",
    "specificity": "Metrics/specificity",
    "f1": "Metrics/f1",
    "precision": "Metrics/precision",
    "mcc": "Metrics/mcc",
    "lr": "Optimization/lr",
    "bias_gap": "Selection/bias_gap",
    "best_auc_so_far": "Selection/best_auc_so_far",
    "selection_score": "Selection/score",
    "best_selection_score_so_far": "Selection/best_score_so_far",
    "selection_threshold": "Selection/threshold",
    "constraints_met": "Selection/constraints_met",
    "monitor_ready": "Selection/monitor_ready",
    "is_best": "Selection/is_best",
    "ema_active": "Optimization/ema_active",
    "epoch_time_sec": "Runtime/epoch_time_sec",
}


def default_swanlab_mode() -> str:
    return normalize_swanlab_mode(os.environ.get("SWANLAB_MODE", "disabled"))


def swanlab_enabled(args) -> bool:
    args.swanlab_mode = normalize_swanlab_mode(args.swanlab_mode)
    return swanlab is not None and not args.disable_swanlab and args.swanlab_mode != "disabled"


def _swanlab_api_key() -> str:
    return os.environ.get("SWANLAB_API_KEY", "").strip()


def _ensure_swanlab_auth(args) -> bool:
    args.swanlab_mode = normalize_swanlab_mode(args.swanlab_mode)
    if args.swanlab_mode != "cloud":
        return True

    api_key = _swanlab_api_key()
    if not api_key:
        print(
            "Warning: SwanLab cloud mode requested but SWANLAB_API_KEY is not set. "
            "Continuing without SwanLab logging."
        )
        args.disable_swanlab = True
        return False

    try:
        swanlab.login(api_key=api_key, save=False)
        return True
    except Exception as exc:
        print(
            "Warning: SwanLab login failed with SWANLAB_API_KEY "
            f"(save=False): {exc}. Continuing without SwanLab logging."
        )
        args.disable_swanlab = True
        return False


def safe_swanlab_init(args, config: dict) -> bool:
    if not swanlab_enabled(args) or not is_main_process():
        return False
    if not _ensure_swanlab_auth(args):
        return False
    try:
        swanlab.init(
            project=args.swanlab_project,
            mode=args.swanlab_mode,
            config=config,
        )
        return True
    except Exception as exc:
        print(
            "Warning: SwanLab initialization failed "
            f"(mode={args.swanlab_mode}): {exc}. Continuing without SwanLab logging."
        )
        args.disable_swanlab = True
        return False


def safe_swanlab_log(args, payload: dict) -> None:
    if not swanlab_enabled(args) or not is_main_process():
        return
    try:
        swanlab.log(payload)
    except Exception as exc:
        print(f"Warning: SwanLab logging failed: {exc}. Disabling SwanLab for this run.")
        args.disable_swanlab = True


def safe_swanlab_finish(args) -> None:
    if not swanlab_enabled(args) or not is_main_process():
        return
    try:
        swanlab.finish()
    except Exception as exc:
        print(f"Warning: SwanLab finish failed: {exc}.")


def format_tensorboard_config(config: dict) -> str:
    lines = ["# Run Configuration", ""]
    for key in sorted(config):
        lines.append(f"- {key}: `{config[key]}`")
    return "\n".join(lines)


def safe_tensorboard_init(args, log_dir: Path, config: dict):
    if args.disable_tensorboard or not is_main_process():
        return None
    if SummaryWriter is None:
        print("Warning: tensorboardX is not installed. Continuing without TensorBoard logging.")
        args.disable_tensorboard = True
        return None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        writer.add_text("run/config", format_tensorboard_config(config), 0)
        return writer
    except Exception as exc:
        print(f"Warning: TensorBoard initialization failed: {exc}. Continuing without TensorBoard logging.")
        args.disable_tensorboard = True
        return None


def safe_tensorboard_log(writer, epoch_record: dict) -> None:
    if writer is None:
        return
    step = int(epoch_record["epoch"])
    try:
        for key, tag in TENSORBOARD_SCALAR_TAGS.items():
            if key not in epoch_record:
                continue
            try:
                value = float(epoch_record[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                writer.add_scalar(tag, value, step)
        writer.flush()
    except Exception as exc:
        print(f"Warning: TensorBoard logging failed: {exc}.")


def safe_tensorboard_close(writer) -> None:
    if writer is None:
        return
    try:
        writer.close()
    except Exception as exc:
        print(f"Warning: TensorBoard close failed: {exc}.")


def save_training_artifacts(history, csv_path, plot_path) -> None:
    if not history:
        return

    import matplotlib.pyplot as plt

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(csv_path, index=False)

    epochs = history_df["epoch"].to_numpy()
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    axes = axes.flatten()
    plot_items = [
        ("Loss", [("train_loss", "Train"), ("val_loss", "Val")]),
        ("AUC", [("auc", "Val")]),
        ("Accuracy", [("acc", "Val")]),
        ("F1", [("f1", "Val")]),
        ("Sensitivity", [("sensitivity", "Val")]),
        ("Specificity", [("specificity", "Val")]),
        ("Precision", [("precision", "Val")]),
        ("MCC", [("mcc", "Val")]),
        ("Learning Rate", [("lr", "LR")]),
    ]

    for ax, (title, series_list) in zip(axes, plot_items):
        for column, label in series_list:
            if column in history_df.columns:
                ax.plot(epochs, history_df[column], marker="o", linewidth=1.8, markersize=3, label=label)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, linestyle="--", alpha=0.35)
        if len(series_list) > 1:
            ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
