from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from trainer.checkpoints import unwrap_model
from trainer.distributed import is_distributed, is_main_process
from utils.losses import (
    AsymmetricLoss,
    CrossEntropyWithFocalPenalty,
    CrossEntropyWithConfidencePenalty,
    FocalLoss,
    HardNegativeLogitPenaltyLoss,
    PairwiseAUCMarginLoss,
)
from utils.sampling import (
    CohortBalancedBatchSampler,
    LabelBalancedBatchSampler,
    WeightedReplacementBatchSampler,
)


def set_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def warmup_cosine(warmup_epochs, num_epochs, epoch):
    if epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_dataloader(
    dataset,
    batch_size,
    shuffle,
    num_workers,
    drop_last,
    distributed=True,
    batch_sampler=None,
):
    sampler = None
    if batch_sampler is not None and distributed and dist.is_available() and dist.is_initialized():
        raise RuntimeError("--cohort-balanced-batches is not supported with distributed training yet.")
    if batch_sampler is not None:
        kwargs = {
            "batch_sampler": batch_sampler,
            "pin_memory": True,
            "num_workers": num_workers,
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = 8
        return DataLoader(dataset, **kwargs)

    if distributed and dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)

    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "pin_memory": True,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 8
    return DataLoader(dataset, **kwargs)


def parse_label_value(label_value):
    if isinstance(label_value, str):
        raw_label = label_value.strip().upper()
        if raw_label == "B":
            return 0
        if raw_label == "M":
            return 1
    return int(label_value)


def compute_balanced_class_weights(labels, num_classes, power: float = 1.0):
    labels = np.asarray([parse_label_value(item) for item in labels], dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    if np.any(counts <= 0):
        return None, counts
    weights = counts.sum() / (max(1, num_classes) * counts)
    weights = np.power(weights, float(power), dtype=np.float32)
    weights = weights / max(weights.mean(), 1e-8)
    return torch.as_tensor(weights, dtype=torch.float32), counts


def build_training_criterion(train_path, args, device):
    class_weights = None
    class_counts = None
    if args.class_weight_mode in {"balanced", "sqrt_balanced"}:
        train_df = pd.read_csv(train_path, usecols=["label"])
        weight_power = 1.0 if args.class_weight_mode == "balanced" else 0.5
        class_weights, class_counts = compute_balanced_class_weights(
            train_df["label"].tolist(),
            args.num_classes,
            power=weight_power,
        )
        if class_weights is not None:
            class_weights = class_weights.to(device)
        elif is_main_process():
            print("Warning: class weights skipped because at least one class is missing in the training split.")

    if args.loss_function == "focal":
        criterion = FocalLoss(
            alpha=args.focal_alpha,
            gamma=args.focal_gamma,
            reduction="mean",
        )
        if is_main_process():
            print(f"Using Focal Loss with alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    elif args.loss_function == "asymmetric":
        criterion = AsymmetricLoss(
            gamma_pos=args.asymmetric_gamma_pos,
            gamma_neg=args.asymmetric_gamma_neg,
            positive_class=1,
            reduction="mean",
        )
        if is_main_process():
            print(
                "Using Asymmetric Loss with "
                f"gamma_pos={args.asymmetric_gamma_pos}, gamma_neg={args.asymmetric_gamma_neg}"
            )
    elif args.loss_function == "ce_conf_penalty":
        criterion = CrossEntropyWithConfidencePenalty(
            weight=class_weights,
            label_smoothing=float(args.label_smoothing),
            penalty_weight=float(args.confidence_penalty_weight),
            reduction="mean",
        )
        if is_main_process():
            print(
                "Using CrossEntropy Loss with "
                f"label_smoothing={args.label_smoothing}, "
                f"confidence_penalty_weight={args.confidence_penalty_weight}"
            )
    elif args.loss_function == "ce_focal_penalty":
        criterion = CrossEntropyWithFocalPenalty(
            weight=class_weights,
            label_smoothing=float(args.label_smoothing),
            focal_weight=float(args.focal_penalty_weight),
            focal_gamma=float(args.focal_gamma),
            focal_alpha=float(args.focal_alpha),
            reduction="mean",
        )
        if is_main_process():
            print(
                "Using CrossEntropy Loss with focal hard-example penalty: "
                f"label_smoothing={args.label_smoothing}, "
                f"focal_weight={args.focal_penalty_weight}, "
                f"focal_alpha={args.focal_alpha}, "
                f"focal_gamma={args.focal_gamma}"
            )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=float(args.label_smoothing),
        )
        if is_main_process():
            print(f"Using CrossEntropy Loss with label_smoothing={args.label_smoothing}")

    hard_negative_weight = float(getattr(args, "hard_negative_penalty_weight", 0.0))
    if hard_negative_weight > 0.0:
        hard_negative_margin = float(getattr(args, "hard_negative_penalty_margin", 0.0))
        criterion = HardNegativeLogitPenaltyLoss(
            criterion,
            penalty_weight=hard_negative_weight,
            margin=hard_negative_margin,
            positive_class=1,
            negative_class=0,
        )
        if is_main_process():
            print(
                "Using train-side hard-negative logit penalty with "
                f"weight={hard_negative_weight}, margin={hard_negative_margin}"
            )

    pairwise_auc_weight = float(getattr(args, "pairwise_auc_weight", 0.0))
    if pairwise_auc_weight > 0.0:
        pairwise_auc_margin = float(getattr(args, "pairwise_auc_margin", 0.0))
        criterion = PairwiseAUCMarginLoss(
            criterion,
            penalty_weight=pairwise_auc_weight,
            margin=pairwise_auc_margin,
            positive_class=1,
            negative_class=0,
        )
        if is_main_process():
            print(
                "Using in-batch pairwise AUC margin penalty with "
                f"weight={pairwise_auc_weight}, margin={pairwise_auc_margin}"
            )

    return criterion, class_weights, class_counts


def resolve_reference_path(path_str, source_csv_path: Path) -> str:
    normalized = os.path.normpath(str(path_str).replace("\\", os.sep).strip())
    candidate = Path(normalized).expanduser()
    csv_dir = source_csv_path.parent
    workspace_dir = csv_dir.parent

    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append((Path.cwd() / candidate).resolve())
        candidates.append((workspace_dir / candidate).resolve())
        candidates.append((csv_dir / candidate).resolve())

    for item in candidates:
        if item.exists():
            return str(item)
    return str(candidates[0])


def absolutize_split_dataframe(df: pd.DataFrame, source_csv_path: Path) -> pd.DataFrame:
    export_df = df.copy()
    for column in ("npy_path", "data_path", "mask_path"):
        if column not in export_df.columns:
            continue
        export_df[column] = export_df[column].apply(
            lambda value: resolve_reference_path(value, source_csv_path)
            if isinstance(value, str) and value.strip()
            else value
        )
    return export_df


def prepare_cv_splits(args):
    cv_path = Path(args.cv_csv)
    if not cv_path.is_file():
        raise FileNotFoundError(f"Cross-validation csv not found: {cv_path}")
    if args.num_folds < 2:
        raise ValueError("--num-folds must be >= 2 when --cv-csv is provided.")
    if args.fold_index is not None and not (1 <= args.fold_index <= args.num_folds):
        raise ValueError(f"--fold-index must be between 1 and {args.num_folds}.")

    df = pd.read_csv(cv_path)
    if "label" not in df.columns:
        raise ValueError(f"Cross-validation csv must contain a 'label' column: {cv_path}")

    labels = df["label"].apply(parse_label_value).to_numpy()
    export_df = absolutize_split_dataframe(df, cv_path)
    split_dir = cv_path.parent / f"{cv_path.stem}_{args.num_folds}fold_splits"
    if not is_distributed() or is_main_process():
        split_dir.mkdir(parents=True, exist_ok=True)

    splitter = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    fold_records = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(export_df, labels), start=1):
        train_csv = split_dir / f"fold{fold_idx}_train.csv"
        val_csv = split_dir / f"fold{fold_idx}_val.csv"
        if not is_distributed() or is_main_process():
            export_df.iloc[train_idx].reset_index(drop=True).to_csv(train_csv, index=False)
            export_df.iloc[val_idx].reset_index(drop=True).to_csv(val_csv, index=False)
        fold_records.append(
            {
                "fold": fold_idx,
                "train_csv": train_csv,
                "val_csv": val_csv,
                "train_size": int(len(train_idx)),
                "val_size": int(len(val_idx)),
            }
        )

    if args.fold_index is not None:
        fold_records = [item for item in fold_records if item["fold"] == args.fold_index]

    if is_distributed():
        dist.barrier()

    return fold_records


def split_backbone_and_head(model):
    bare_model = unwrap_model(model)
    backbone = getattr(bare_model, "model", None) or getattr(bare_model, "backbone", None)
    head = getattr(bare_model, "head", None)
    return backbone, head


def supports_freeze_strategy(model):
    bare_model = unwrap_model(model)
    custom_support = getattr(bare_model, "supports_head_warmup", None)
    if callable(custom_support):
        return bool(custom_support())
    backbone, head = split_backbone_and_head(model)
    return backbone is not None and head is not None


def set_backbone_trainable(model, trainable: bool) -> None:
    bare_model = unwrap_model(model)
    custom_setter = getattr(bare_model, "set_backbone_trainable", None)
    if callable(custom_setter):
        custom_setter(trainable)
        return
    backbone, head = split_backbone_and_head(model)
    if backbone is None:
        return
    for param in backbone.parameters():
        param.requires_grad = trainable
    if head is not None:
        for param in head.parameters():
            param.requires_grad = True


def build_optimizer(model, args):
    bare_model = unwrap_model(model)
    custom_groups = getattr(bare_model, "get_param_groups", None)
    if callable(custom_groups):
        return torch.optim.AdamW(custom_groups(args.lr, args.backbone_lr_scale, args.weight_decay))
    backbone, head = split_backbone_and_head(bare_model)
    if backbone is not None and head is not None:
        param_groups = [
            {
                "params": [p for p in backbone.parameters()],
                "lr": args.lr * args.backbone_lr_scale,
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for p in head.parameters()],
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
        ]
        return torch.optim.AdamW(param_groups)
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def should_enable_find_unused_parameters(args, model) -> bool:
    if args.find_unused_parameters:
        return True
    if args.freeze_backbone_epochs > 0 and supports_freeze_strategy(model):
        return True
    return False


def build_cohort_balanced_batch_sampler(args, train_dataset, seed):
    if not args.cohort_balanced_batches:
        return None
    if not hasattr(train_dataset, "data") or args.cohort_column not in train_dataset.data.columns:
        raise ValueError(
            f"--cohort-balanced-batches requires column {args.cohort_column!r} in the training CSV."
        )
    cohorts = train_dataset.data[args.cohort_column].astype(str).tolist()
    return CohortBalancedBatchSampler(
        cohorts=cohorts,
        batch_size=args.batch_size,
        num_batches=len(train_dataset) // args.batch_size,
        seed=seed,
    )


def build_label_balanced_batch_sampler(args, train_dataset, seed):
    if not args.label_balanced_batches:
        return None
    if not hasattr(train_dataset, "data") or "label" not in train_dataset.data.columns:
        raise ValueError("--label-balanced-batches requires column 'label' in the training CSV.")
    labels = [parse_label_value(label) for label in train_dataset.data["label"].tolist()]
    return LabelBalancedBatchSampler(
        labels=labels,
        batch_size=args.batch_size,
        num_batches=len(train_dataset) // args.batch_size,
        seed=seed,
    )


def derive_case_id_from_row(row: pd.Series, fallback_index: int) -> str:
    """Derive a stable case identifier from a manifest row."""
    if "case_id" in row and pd.notna(row["case_id"]):
        return str(row["case_id"])
    if "data_path" in row and pd.notna(row["data_path"]):
        return Path(str(row["data_path"])).stem
    if "npy_path" in row and pd.notna(row["npy_path"]):
        return Path(str(row["npy_path"])).stem
    return f"case_{fallback_index:04d}"


def _is_truthy_flag(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) != 0.0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "hard", "hard_negative"}


def build_hard_negative_batch_sampler(args, train_dataset, seed):
    mining_csv = getattr(args, "hard_negative_mining_csv", None)
    if not mining_csv:
        return None

    sampling_factor = float(getattr(args, "hard_negative_sampling_factor", 1.0))
    if sampling_factor <= 1.0:
        return None
    if not hasattr(train_dataset, "data"):
        raise ValueError("--hard-negative-mining-csv requires the training dataset to expose .data.")

    mining_path = Path(mining_csv)
    if not mining_path.is_file():
        raise FileNotFoundError(f"Hard-negative mining CSV not found: {mining_path}")

    mining_df = pd.read_csv(mining_path)
    case_column = getattr(args, "hard_negative_case_column", "case_id")
    if case_column not in mining_df.columns:
        raise ValueError(f"Hard-negative mining CSV must contain column {case_column!r}.")

    flag_column = getattr(args, "hard_negative_flag_column", "is_hard_negative")
    if flag_column and flag_column in mining_df.columns:
        mining_df = mining_df[mining_df[flag_column].map(_is_truthy_flag)].copy()

    selected_cases = {
        str(case_id)
        for case_id in mining_df[case_column].dropna().tolist()
    }
    if not selected_cases:
        raise ValueError("Hard-negative mining CSV did not select any case IDs.")

    train_case_ids = [
        derive_case_id_from_row(row, int(index))
        for index, row in train_dataset.data.reset_index(drop=True).iterrows()
    ]
    hard_negative_indices = [
        index
        for index, case_id in enumerate(train_case_ids)
        if case_id in selected_cases
    ]
    if not hard_negative_indices:
        preview = sorted(selected_cases)[:5]
        raise ValueError(
            "None of the hard-negative case IDs matched the training CSV. "
            f"Preview: {preview}"
        )

    sample_weights = np.ones(len(train_dataset), dtype=np.float64)
    sample_weights[hard_negative_indices] = sampling_factor
    sampler = WeightedReplacementBatchSampler(
        sample_weights=sample_weights,
        batch_size=args.batch_size,
        num_batches=len(train_dataset) // args.batch_size,
        seed=seed,
    )
    sampler.hard_negative_indices = hard_negative_indices
    sampler.hard_negative_sampling_factor = sampling_factor
    sampler.hard_negative_mining_csv = str(mining_path)
    return sampler
