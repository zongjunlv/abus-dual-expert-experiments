from __future__ import annotations

import argparse

from models.model_factory import MODEL_CHOICES
from trainer.cli import (
    DEFAULT_RESEARCH_SEEDS as TRAIN_DEFAULT_RESEARCH_SEEDS,
    describe_model,
    model_requires_roi_mask,
    parse_spatial_shape,
)
from trainer.logging import SWANLAB_CLI_MODES, default_swanlab_mode


DEFAULT_RESEARCH_SEEDS = tuple(TRAIN_DEFAULT_RESEARCH_SEEDS)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a 3D medical image classifier.")
    parser.add_argument("--test-csv", required=True, help="Path to test csv.")
    parser.add_argument(
        "--val-csv",
        default=None,
        help="Optional validation csv used to automatically select a binary decision threshold before testing.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint. Defaults to assets/checkpoints_3d/best_model_<model>.pth.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed suffix for checkpoint/export resolution, e.g. --seed 42 -> best_model_<model>_seed42.pth.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Evaluate multiple seed checkpoints sequentially and export a combined summary, "
            f"e.g. --seeds {' '.join(str(seed) for seed in DEFAULT_RESEARCH_SEEDS)}."
        ),
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Optional list of checkpoints for probability-averaging ensemble evaluation.",
    )
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=MODEL_CHOICES,
        help="Backbone used for evaluation.",
    )
    parser.add_argument(
        "--use-roi-mask",
        action="store_true",
        help="Load ROI masks from the CSV and pass them to mask-aware models.",
    )
    parser.add_argument(
        "--target-shape",
        type=parse_spatial_shape,
        default=None,
        help=(
            "Optional deterministic on-the-fly spatial resize for cached arrays, "
            "using cached axis order, e.g. 128,128,64 or 96x96x96. "
            "Images use trilinear interpolation and ROI masks use nearest-neighbor."
        ),
    )
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--device", default="cuda", help="Device string.")
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help="Optional fixed positive-class threshold for binary classification. Overrides --val-csv threshold search.",
    )
    parser.add_argument(
        "--threshold-metric",
        default="mcc",
        choices=["mcc", "f1", "accuracy", "balanced_accuracy", "sensitivity", "specificity"],
        help="Metric optimized when searching the validation-set threshold.",
    )
    parser.add_argument(
        "--export-pred-csv",
        default=None,
        help="Optional path to export per-sample predictions. Defaults to assets/test_outputs/predictions_<model>.csv.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help=(
            "Optional path to export multi-seed metric summary. "
            "Defaults to assets/test_outputs/test_metrics_<model>_multiseed_summary.csv."
        ),
    )
    parser.add_argument("--swanlab-project", default="DB-VCRF-ABUS", help="SwanLab project name.")
    parser.add_argument(
        "--swanlab-mode",
        default=default_swanlab_mode(),
        choices=SWANLAB_CLI_MODES,
        help=(
            "SwanLab logging mode. Defaults to SWANLAB_MODE when set, otherwise disabled "
            "for shared-account environments. 'online' is accepted as an alias for 'cloud'."
        ),
    )
    parser.add_argument("--disable-swanlab", action="store_true", help="Disable SwanLab logging.")
    return parser.parse_args()


def validate_args(args) -> None:
    if args.seed is not None and args.seeds:
        raise ValueError("Use either --seed or --seeds, not both.")
    if args.seed is not None and args.checkpoints:
        raise ValueError("--seed cannot be combined with --checkpoints.")
    if args.seeds and args.checkpoints:
        raise ValueError("--seeds cannot be combined with --checkpoints.")
    if model_requires_roi_mask(args.model) and not args.use_roi_mask:
        raise ValueError(f"--use-roi-mask is required for ROI-aware model: {args.model}")


def resolve_requested_seeds(args) -> list[int]:
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    if args.seed is not None:
        return [int(args.seed)]
    return []
