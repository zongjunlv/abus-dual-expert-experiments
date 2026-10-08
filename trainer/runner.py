from __future__ import annotations

import copy
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.medical_dataset import Medical_Dataset
from models.model_factory import build_model
from trainer.checkpoints import build_teacher_model, load_checkpoint_weights, unwrap_model
from trainer.cli import DEFAULT_RESEARCH_SEEDS, describe_model, parse_args, validate_args
from trainer.distributed import (
    cleanup_distributed,
    is_distributed,
    is_main_process,
    reduce_mean,
    setup_distributed,
)
from trainer.logging import (
    SWANLAB_AVAILABLE,
    safe_swanlab_finish,
    safe_swanlab_init,
    safe_swanlab_log,
    safe_tensorboard_close,
    safe_tensorboard_init,
    safe_tensorboard_log,
    save_training_artifacts,
)
from trainer.paths import (
    resolve_log_paths,
    resolve_run_save_path,
    resolve_save_path,
    resolve_summary_csv_path,
    resolve_tensorboard_log_dir,
)
from trainer.selection import (
    compute_bias_gap,
    compute_selection_score,
    meets_selection_constraints,
    resolve_monitor_start_epoch,
    resolve_validation_selection_metrics,
    should_update_best_checkpoint,
)
from trainer.setup import (
    build_hard_negative_batch_sampler,
    build_cohort_balanced_batch_sampler,
    build_dataloader,
    build_label_balanced_batch_sampler,
    build_optimizer,
    build_training_criterion,
    prepare_cv_splits,
    set_backbone_trainable,
    set_seed,
    should_enable_find_unused_parameters,
    supports_freeze_strategy,
    warmup_cosine,
)
from trainer.trainer import Trainer
from utils.evaluator import evaluate_model


def create_ema_model(model, device):
    """Create a detached model copy for EMA validation/checkpointing."""
    ema_model = copy.deepcopy(unwrap_model(model)).to(device)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(ema_model, source_model, decay: float, initialized: bool) -> bool:
    source_state = unwrap_model(source_model).state_dict()
    ema_state = ema_model.state_dict()

    for name, ema_value in ema_state.items():
        source_value = source_state[name].detach().to(ema_value.device)
        if not initialized or not torch.is_floating_point(ema_value):
            ema_value.copy_(source_value)
        else:
            ema_value.mul_(decay).add_(source_value, alpha=1.0 - decay)
    return True


def validate_loaded_gate_config(model, gate_alpha: float, gate_temperature: float) -> None:
    """Reject CLI/checkpoint gate-rule mismatches before training starts."""
    bare_model = unwrap_model(model)
    gate_config = getattr(bare_model, "gate_hyperparameters", None)
    if not callable(gate_config) or not bool(
        getattr(bare_model, "residual_reliability_gate", False)
    ):
        return
    checkpoint_alpha, checkpoint_temperature = gate_config()
    if not math.isclose(checkpoint_alpha, gate_alpha, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(
            "Loaded gate alpha does not match --gate-alpha: "
            f"checkpoint={checkpoint_alpha}, cli={gate_alpha}."
        )
    if not math.isclose(
        checkpoint_temperature,
        gate_temperature,
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise ValueError(
            "Loaded gate temperature does not match --gate-temperature: "
            f"checkpoint={checkpoint_temperature}, cli={gate_temperature}."
        )


def train_one_seed(args, seed, train_path, val_path, multi_seed, run_suffix=None, run_title=None, use_validation=True):
    start_time = time.time()
    set_seed(seed)

    if not train_path.is_file():
        raise FileNotFoundError(f"Training csv not found: {train_path}")
    if use_validation and (val_path is None or not val_path.is_file()):
        raise FileNotFoundError(f"Validation csv not found: {val_path}")

    train_dataset = Medical_Dataset(
        mode="train",
        csv_path=str(train_path),
        roi_size=(256, 256, 128),
        margin=12,
        return_mask=args.use_roi_mask,
        roi_mask_dropout_prob=args.roi_mask_dropout_prob,
        roi_mask_jitter_voxels=args.roi_mask_jitter_voxels,
        target_shape=args.target_shape,
        spatial_aug_policy=args.spatial_aug_policy,
    )
    val_dataset = (
        Medical_Dataset(
            mode="val",
            csv_path=str(val_path),
            roi_size=(256, 256, 128),
            margin=12,
            return_mask=args.use_roi_mask,
            target_shape=args.target_shape,
        )
        if use_validation
        else None
    )

    if args.cohort_balanced_batches and args.label_balanced_batches:
        raise ValueError("Choose only one of --cohort-balanced-batches or --label-balanced-batches.")
    hard_negative_batch_sampler = build_hard_negative_batch_sampler(args, train_dataset, seed)
    cohort_batch_sampler = build_cohort_balanced_batch_sampler(args, train_dataset, seed)
    label_batch_sampler = build_label_balanced_batch_sampler(args, train_dataset, seed)
    balanced_batch_sampler = hard_negative_batch_sampler or cohort_batch_sampler or label_batch_sampler
    train_dataloader = build_dataloader(
        train_dataset,
        args.batch_size,
        shuffle=balanced_batch_sampler is None,
        num_workers=args.num_workers,
        drop_last=True,
        distributed=True,
        batch_sampler=balanced_batch_sampler,
    )
    val_dataloader = (
        build_dataloader(
            val_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            distributed=False,
        )
        if use_validation
        else None
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if is_distributed():
        device_name = f"cuda:{local_rank}"
    else:
        device_name = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_name)
    base_save_path = resolve_save_path(args.save_path, args.model, final_train=not use_validation)
    save_path = resolve_run_save_path(base_save_path, seed, multi_seed, run_suffix)
    _, log_csv_path, log_plot_path = resolve_log_paths(args.log_dir, args.model, seed, multi_seed, run_suffix)
    tensorboard_log_dir = resolve_tensorboard_log_dir(
        args.tensorboard_dir,
        args.log_dir,
        args.model,
        seed,
        multi_seed,
        run_suffix,
    )

    model = build_model(
        args.model,
        args.num_classes,
        gate_alpha=args.gate_alpha,
        gate_temperature=args.gate_temperature,
        freeze_source_experts=args.freeze_source_experts,
        primus_pretrained_path=getattr(args, "primus_pretrained_path", None),
    ).to(device)
    init_checkpoint_path = None
    init_load_result = None
    if args.init_checkpoint:
        base_init_checkpoint = Path(args.init_checkpoint)
        seeded_init_checkpoint = resolve_run_save_path(base_init_checkpoint, seed, True)
        if multi_seed or (not base_init_checkpoint.is_file() and seeded_init_checkpoint.is_file()):
            init_checkpoint_path = seeded_init_checkpoint
        else:
            init_checkpoint_path = base_init_checkpoint
        init_load_result = load_checkpoint_weights(
            model,
            init_checkpoint_path,
            device,
            strict=args.init_checkpoint_strict,
        )
        validate_loaded_gate_config(model, args.gate_alpha, args.gate_temperature)
    if args.freeze_batchnorm:
        # Freeze affine parameters before DDP wrapping and optimizer creation;
        # Trainer.train() will re-apply eval mode after every model.train().
        Trainer.configure_batchnorm(model, args.freeze_batchnorm_affine)
    if is_distributed():
        ddp_find_unused = should_enable_find_unused_parameters(args, model)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused,
        )
    ema_model = None
    ema_initialized = False
    if args.ema_decay > 0.0:
        ema_model = create_ema_model(model, device)
    optimizer = build_optimizer(model, args)
    criterion, class_weights, class_counts = build_training_criterion(train_path, args, device)
    teacher_model = build_teacher_model(args, device)
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device,
        use_amp=args.amp,
        accum_steps=args.accum_steps,
        teacher_model=teacher_model,
        kd_alpha=args.kd_alpha,
        kd_temperature=args.kd_temperature,
        mixup_alpha=args.mixup_alpha,
        mixup_prob=args.mixup_prob,
        roi_consistency_weight=args.roi_consistency_weight,
        roi_consistency_jitter_voxels=args.roi_consistency_jitter_voxels,
        roi_consistency_dropout_prob=args.roi_consistency_dropout_prob,
        gate_loss_weight=args.gate_loss_weight,
        gate_alpha=args.gate_alpha,
        gate_temperature=args.gate_temperature,
        gate_oracle_temperature=args.gate_oracle_temperature,
        freeze_batchnorm=args.freeze_batchnorm,
        freeze_batchnorm_affine=args.freeze_batchnorm_affine,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: warmup_cosine(args.warmup_epochs, args.epochs, epoch),
    )

    monitor_start_epoch = resolve_monitor_start_epoch(args) if use_validation else None
    ema_start_epoch = None
    if ema_model is not None:
        default_ema_start = monitor_start_epoch if use_validation else max(1, args.freeze_backbone_epochs + 1)
        ema_start_epoch = int(args.ema_start_epoch or default_ema_start)

    if is_main_process():
        if run_title:
            print("\n" + "=" * 60)
            print(f"{run_title:^60}")
            print("=" * 60)
        print("\nCONFIGURATION")
        print(f"  {'Train csv':18}: {train_path}")
        print(f"  {'Val csv':18}: {val_path if use_validation else 'None (final-train mode)'}")
        print(f"  {'Batch size':18}: {args.batch_size}")
        print(f"  {'Learning rate':18}: {args.lr}")
        print(f"  {'Backbone lr x':18}: {args.backbone_lr_scale}")
        print(f"  {'Loss function':18}: {args.loss_function}")
        print(f"  {'Label smoothing':18}: {args.label_smoothing}")
        print(f"  {'Class weights':18}: {args.class_weight_mode}")
        if args.loss_function == "ce_conf_penalty":
            print(f"  {'Conf penalty':18}: {args.confidence_penalty_weight}")
        if args.loss_function == "ce_focal_penalty":
            print(f"  {'Focal penalty':18}: {args.focal_penalty_weight}")
            print(f"  {'Focal alpha':18}: {args.focal_alpha}")
            print(f"  {'Focal gamma':18}: {args.focal_gamma}")
        if args.hard_negative_penalty_weight > 0.0:
            print(f"  {'Hard-neg penalty':18}: {args.hard_negative_penalty_weight}")
            print(f"  {'Hard-neg margin':18}: {args.hard_negative_penalty_margin}")
        if args.pairwise_auc_weight > 0.0:
            print(f"  {'Pairwise AUC':18}: {args.pairwise_auc_weight}")
            print(f"  {'Pairwise margin':18}: {args.pairwise_auc_margin}")
        print(f"  {'Cohort balanced':18}: {args.cohort_balanced_batches}")
        print(f"  {'Label balanced':18}: {args.label_balanced_batches}")
        if hard_negative_batch_sampler is not None:
            print(f"  {'Hard-neg sampling':18}: {args.hard_negative_sampling_factor}")
            print(f"  {'Hard-neg cases':18}: {len(hard_negative_batch_sampler.hard_negative_indices)}")
            print(f"  {'Hard-neg mining':18}: {args.hard_negative_mining_csv}")
        if args.cohort_balanced_batches:
            print(f"  {'Cohort column':18}: {args.cohort_column}")
            print(f"  {'Train batches':18}: {len(train_dataloader)}")
        if teacher_model is not None:
            if args.teacher_members:
                print(f"  {'Teacher ensemble':18}: {len(args.teacher_members)} members")
                for index, member in enumerate(args.teacher_members, start=1):
                    model_name, checkpoint_path, weight_value = member
                    print(
                        f"  {'Teacher member':18}: "
                        f"{index}: model={model_name} weight={weight_value} ckpt={checkpoint_path}"
                    )
            else:
                print(f"  {'Teacher model':18}: {args.teacher_model}")
                print(f"  {'Teacher ckpt':18}: {args.teacher_checkpoint}")
            print(f"  {'KD alpha':18}: {args.kd_alpha}")
            print(f"  {'KD temp':18}: {args.kd_temperature}")
        print(
            f"  {'Val threshold sel':18}: "
            f"{'off' if args.disable_selection_threshold_search else args.selection_threshold_metric}"
        )
        print(f"  {'Accum steps':18}: {args.accum_steps}")
        print(f"  {'AMP':18}: {args.amp}")
        print(f"  {'Freeze epochs':18}: {args.freeze_backbone_epochs}")
        print(f"  {'Freeze experts':18}: {args.freeze_source_experts}")
        print(f"  {'EMA decay':18}: {args.ema_decay}")
        if ema_start_epoch is not None:
            print(f"  {'EMA start':18}: {ema_start_epoch}")
        print(f"  {'Mixup alpha':18}: {args.mixup_alpha}")
        print(f"  {'Mixup prob':18}: {args.mixup_prob}")
        print(f"  {'Gate loss weight':18}: {args.gate_loss_weight}")
        print(f"  {'Gate alpha':18}: {args.gate_alpha}")
        print(f"  {'Gate temp':18}: {args.gate_temperature}")
        print(f"  {'Gate oracle temp':18}: {args.gate_oracle_temperature}")
        print(f"  {'Freeze BatchNorm':18}: {args.freeze_batchnorm}")
        print(f"  {'Freeze BN affine':18}: {args.freeze_batchnorm_affine}")
        if args.roi_consistency_weight > 0.0:
            print(f"  {'ROI consistency':18}: {args.roi_consistency_weight}")
            print(f"  {'ROI cons jitter':18}: {args.roi_consistency_jitter_voxels}")
            print(f"  {'ROI cons dropout':18}: {args.roi_consistency_dropout_prob}")
        if init_checkpoint_path is not None:
            print(f"  {'Init checkpoint':18}: {init_checkpoint_path}")
            print(f"  {'Init strict':18}: {args.init_checkpoint_strict}")
            missing_keys = list(getattr(init_load_result, "missing_keys", []))
            unexpected_keys = list(getattr(init_load_result, "unexpected_keys", []))
            print(f"  {'Init missing':18}: {len(missing_keys)}")
            print(f"  {'Init unexpected':18}: {len(unexpected_keys)}")
            if missing_keys:
                print(f"  {'Missing preview':18}: {missing_keys[:8]}")
            if unexpected_keys:
                print(f"  {'Unexpected preview':18}: {unexpected_keys[:8]}")
        print(f"  {'Num classes':18}: {args.num_classes}")
        print(f"  {'Model':18}: {args.model}")
        if getattr(args, "primus_pretrained_path", None):
            print(f"  {'Primus pretrain':18}: {args.primus_pretrained_path}")
        print(f"  {'Use ROI mask':18}: {args.use_roi_mask}")
        print(f"  {'Target shape':18}: {args.target_shape or 'cache-native'}")
        print(f"  {'Spatial aug':18}: {args.spatial_aug_policy}")
        if args.use_roi_mask:
            print(f"  {'ROI dropout':18}: {args.roi_mask_dropout_prob}")
            print(f"  {'ROI jitter vox':18}: {args.roi_mask_jitter_voxels}")
        if use_validation:
            print(f"  {'Monitor start':18}: {monitor_start_epoch}")
            print(f"  {'Patience':18}: {args.patience}")
            print(f"  {'Score delta':18}: {args.score_min_delta}")
            print(
                f"  {'Score weights':18}: "
                f"auc={args.score_weight_auc:.2f}, mcc={args.score_weight_mcc:.2f}, bias={args.score_weight_bias:.2f}"
            )
            print(
                f"  {'Constraints':18}: "
                f"sens>={args.min_sensitivity:.2f}, spec>={args.min_specificity:.2f}, bias_gap<={args.max_bias_gap:.2f}"
            )
            print(
                f"  {'Threshold floor':18}: "
                f"{'enabled' if args.constrain_selection_threshold else 'disabled'}"
            )
        model_desc = describe_model(args.model)
        if model_desc is not None:
            print(f"  {'Model detail':18}: {model_desc}")
        print(f"  {'Seed':18}: {seed}")
        print(f"  {'Device':18}: {device}")
        print(f"  {'World size':18}: {dist.get_world_size() if is_distributed() else 1}")
        print(
            f"  {'DDP unused chk':18}: "
            f"{should_enable_find_unused_parameters(args, unwrap_model(model) if hasattr(model, 'module') else model)}"
        )
        print(f"  {'Save path':18}: {save_path}")
        print(f"  {'TensorBoard':18}: {'disabled' if args.disable_tensorboard else tensorboard_log_dir}")
        if class_counts is not None:
            print(f"  {'Class counts':18}: {class_counts.tolist()}")
        if class_weights is not None:
            print(f"  {'Loss weights':18}: {[round(float(item), 4) for item in class_weights.detach().cpu().tolist()]}")

    run_config = {
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_classes": args.num_classes,
        "model": args.model,
        "primus_pretrained_path": getattr(args, "primus_pretrained_path", None),
        "use_roi_mask": args.use_roi_mask,
        "roi_mask_dropout_prob": args.roi_mask_dropout_prob,
        "roi_mask_jitter_voxels": args.roi_mask_jitter_voxels,
        "target_shape": args.target_shape,
        "spatial_aug_policy": args.spatial_aug_policy,
        "seed": seed,
        "loss_function": args.loss_function,
        "label_smoothing": args.label_smoothing,
        "confidence_penalty_weight": (
            args.confidence_penalty_weight if args.loss_function == "ce_conf_penalty" else None
        ),
        "focal_penalty_weight": (
            args.focal_penalty_weight if args.loss_function == "ce_focal_penalty" else None
        ),
        "hard_negative_penalty_weight": args.hard_negative_penalty_weight,
        "hard_negative_penalty_margin": args.hard_negative_penalty_margin,
        "pairwise_auc_weight": args.pairwise_auc_weight,
        "pairwise_auc_margin": args.pairwise_auc_margin,
        "class_weight_mode": args.class_weight_mode,
        "cohort_balanced_batches": args.cohort_balanced_batches,
        "cohort_column": args.cohort_column,
        "label_balanced_batches": args.label_balanced_batches,
        "hard_negative_mining_csv": args.hard_negative_mining_csv,
        "hard_negative_sampling_factor": args.hard_negative_sampling_factor,
        "hard_negative_case_column": args.hard_negative_case_column,
        "hard_negative_flag_column": args.hard_negative_flag_column,
        "ema_decay": args.ema_decay,
        "ema_start_epoch": ema_start_epoch,
        "mixup_alpha": args.mixup_alpha,
        "mixup_prob": args.mixup_prob,
        "gate_loss_weight": args.gate_loss_weight,
        "gate_alpha": args.gate_alpha,
        "gate_temperature": args.gate_temperature,
        "gate_oracle_temperature": args.gate_oracle_temperature,
        "freeze_batchnorm": args.freeze_batchnorm,
        "freeze_batchnorm_affine": args.freeze_batchnorm_affine,
        "freeze_source_experts": args.freeze_source_experts,
        "roi_consistency_weight": args.roi_consistency_weight,
        "roi_consistency_jitter_voxels": args.roi_consistency_jitter_voxels,
        "roi_consistency_dropout_prob": args.roi_consistency_dropout_prob,
        "teacher_model": args.teacher_model,
        "teacher_checkpoint": args.teacher_checkpoint,
        "teacher_members": args.teacher_members,
        "init_checkpoint": str(init_checkpoint_path) if init_checkpoint_path is not None else None,
        "init_checkpoint_strict": args.init_checkpoint_strict,
        "kd_alpha": args.kd_alpha if teacher_model is not None else None,
        "kd_temperature": args.kd_temperature if teacher_model is not None else None,
        "selection_threshold_metric": None
        if args.disable_selection_threshold_search
        else args.selection_threshold_metric,
        "constrain_selection_threshold": args.constrain_selection_threshold,
        "train_csv": str(train_path),
        "val_csv": str(val_path),
        "save_path": str(save_path),
        "log_csv_path": str(log_csv_path),
        "log_plot_path": str(log_plot_path),
        "tensorboard_log_dir": str(tensorboard_log_dir),
    }

    safe_swanlab_init(args, config=run_config)
    tensorboard_writer = safe_tensorboard_init(args, tensorboard_log_dir, run_config)

    best_mcc = float("-inf") if use_validation else float("nan")
    best_auc_seen = float("-inf") if use_validation else float("nan")
    best_selection_score = float("-inf") if use_validation else float("nan")
    early_stop = 0
    history = []
    best_metrics = None
    best_epoch = None
    best_constraints_met = False
    best_candidate = None
    best_threshold = float("nan")

    for epoch in range(args.epochs):
        epoch_start = time.time()
        if is_main_process():
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

        should_freeze_backbone = supports_freeze_strategy(model) and epoch < args.freeze_backbone_epochs
        set_backbone_trainable(model, not should_freeze_backbone)
        if is_main_process():
            phase = "head-only warmup" if should_freeze_backbone else "full fine-tuning"
            print(f"  Training phase: {phase}")

        if hasattr(train_dataloader, "sampler") and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        train_loss = trainer.train(train_dataloader)
        train_loss = reduce_mean(train_loss, device)
        train_gate_loss = float(getattr(trainer, "last_train_stats", {}).get("gate_loss", 0.0))
        train_gate_loss = reduce_mean(train_gate_loss, device)
        ema_active = False
        if ema_model is not None and (epoch + 1) >= ema_start_epoch:
            ema_initialized = update_ema_model(
                ema_model,
                model,
                float(args.ema_decay),
                ema_initialized,
            )
            ema_active = True

        if is_main_process() and use_validation:
            validation_model = ema_model if ema_active else unwrap_model(model)
            val_loss, accuracy, auc, sensitivity, specificity, f1, precision, mcc, labels_arr, _, probs_arr = evaluate_model(
                validation_model,
                val_dataloader,
                device,
                verbose=(epoch % 5 == 0),
                return_outputs=True,
            )
            raw_metrics = {
                "accuracy": accuracy,
                "auc": auc,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "macro_f1": f1,
                "precision": precision,
                "mcc": mcc,
            }
            selection_threshold, selection_metrics, used_threshold_search = resolve_validation_selection_metrics(
                args,
                labels_arr,
                probs_arr,
                raw_metrics,
            )
            accuracy = float(selection_metrics["accuracy"])
            auc = float(selection_metrics.get("auc", auc))
            sensitivity = float(selection_metrics["sensitivity"])
            specificity = float(selection_metrics["specificity"])
            f1 = float(selection_metrics["macro_f1"])
            precision = float(selection_metrics["precision"])
            mcc = float(selection_metrics["mcc"])
        elif is_main_process():
            val_loss = float("nan")
            accuracy = float("nan")
            auc = float("nan")
            sensitivity = float("nan")
            specificity = float("nan")
            f1 = float("nan")
            precision = float("nan")
            mcc = float("nan")
            selection_threshold = float("nan")
            used_threshold_search = False
        else:
            val_loss = 0.0
            accuracy = 0.0
            auc = 0.0
            sensitivity = 0.0
            specificity = 0.0
            f1 = 0.0
            precision = 0.0
            mcc = 0.0
            selection_threshold = 0.0
            used_threshold_search = False
        scheduler.step()

        stop_training = False
        if is_main_process():
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]["lr"]
            if use_validation:
                bias_gap = compute_bias_gap(sensitivity, specificity)
                selection_score = compute_selection_score(auc, mcc, bias_gap, args)
                constraints_met = meets_selection_constraints(sensitivity, specificity, bias_gap, args)
                monitor_ready = (epoch + 1) >= monitor_start_epoch
                best_auc_seen = max(best_auc_seen, auc)
                print(
                    f"  Train loss: {train_loss:.4f}   Val loss: {val_loss:.4f}   "
                    f"Val AUC: {auc:.4f}   Val MCC: {mcc:.4f}   Time: {str(timedelta(seconds=int(epoch_time)))}"
                )
                if ema_model is not None:
                    print(f"  Eval weights: {'EMA' if ema_active else 'online'}")
                print(
                    f"  Sensitivity: {sensitivity:.4f}   Specificity: {specificity:.4f}   "
                    f"Bias gap: {bias_gap:.4f}"
                )
                if used_threshold_search:
                    print(
                        f"  Selection threshold: {selection_threshold:.4f}   "
                        f"Metric: {args.selection_threshold_metric}"
                        f"{' constrained' if args.constrain_selection_threshold else ''}"
                    )
                print(
                    f"  Selection score: {selection_score:.4f}   "
                    f"Constraints: {'pass' if constraints_met else 'fail'}   "
                    f"Monitor: {'on' if monitor_ready else f'epoch {monitor_start_epoch}+'}"
                )

                candidate = {
                    "selection_score": selection_score,
                    "constraints_met": constraints_met,
                    "mcc": mcc,
                    "auc": auc,
                    "bias_gap": bias_gap,
                    "epoch": epoch + 1,
                }
                is_best = monitor_ready and should_update_best_checkpoint(
                    candidate,
                    best_candidate,
                    args.score_min_delta,
                )
                if is_best:
                    best_candidate = candidate
                    best_selection_score = selection_score
                    best_mcc = mcc
                    best_epoch = epoch + 1
                    best_constraints_met = constraints_met
                    best_threshold = float(selection_threshold) if used_threshold_search else float("nan")
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint_model = ema_model if ema_active else unwrap_model(model)
                    torch.save(checkpoint_model.state_dict(), save_path)
                    print(
                        f"  Weights updated! Score: {best_selection_score:.4f}   "
                        f"AUC: {auc:.4f}   MCC: {mcc:.4f}"
                    )
                    early_stop = 0
                else:
                    if monitor_ready:
                        early_stop += 1
                        print(f"  Early stopping count: {early_stop}/{args.patience}")
                        if early_stop >= args.patience:
                            print(f"  Early stopping triggered at epoch {epoch + 1}.")
                            stop_training = True
                    else:
                        print(f"  Early stopping inactive until epoch {monitor_start_epoch}.")
            else:
                bias_gap = float("nan")
                selection_score = float("nan")
                constraints_met = False
                monitor_ready = False
                is_best = 0
                print(
                    f"  Train loss: {train_loss:.4f}   LR: {current_lr:.2e}   "
                    f"Time: {str(timedelta(seconds=int(epoch_time)))}"
                )

            epoch_record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_gate_loss": train_gate_loss,
                "val_loss": val_loss,
                "acc": accuracy,
                "auc": auc,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "f1": f1,
                "precision": precision,
                "mcc": mcc,
                "lr": current_lr,
                "bias_gap": bias_gap,
                "best_auc_so_far": best_auc_seen,
                "selection_score": selection_score,
                "constraints_met": int(constraints_met),
                "monitor_ready": int(monitor_ready),
                "best_selection_score_so_far": best_selection_score,
                "selection_threshold": selection_threshold,
                "is_best": int(is_best),
                "ema_active": int(ema_active),
                "epoch_time_sec": epoch_time,
            }
            history.append(epoch_record)
            save_training_artifacts(history, log_csv_path, log_plot_path)
            safe_tensorboard_log(tensorboard_writer, epoch_record)

        if use_validation:
            safe_swanlab_log(
                args,
                {
                    "train_loss": train_loss,
                    "train_gate_loss": train_gate_loss,
                    "val_loss": val_loss,
                    "acc": accuracy,
                    "auc": auc,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "f1": f1,
                    "precision": precision,
                    "mcc": mcc,
                    "selection_threshold": selection_threshold,
                },
            )
        else:
            safe_swanlab_log(
                args,
                {
                    "train_loss": train_loss,
                    "train_gate_loss": train_gate_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                },
            )

        if is_distributed():
            stop_tensor = torch.tensor(int(stop_training), device=device)
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item():
                break
        elif stop_training:
            break

    total_time = time.time() - start_time
    if is_main_process():
        save_training_artifacts(history, log_csv_path, log_plot_path)
        if use_validation and save_path.is_file():
            load_checkpoint_weights(unwrap_model(model), save_path, device, strict=True)
            best_eval_threshold = float(best_threshold) if not np.isnan(best_threshold) else None
            print("\nBest Checkpoint Evaluation")
            best_metrics = evaluate_model(
                unwrap_model(model),
                val_dataloader,
                device,
                verbose=True,
                show_details=True,
                desc="Best Val",
                return_outputs=False,
                decision_threshold=best_eval_threshold,
            )

        print("\n" + "=" * 60)
        print(f"Training completed in {str(timedelta(seconds=int(total_time)))}")
        print(f"Training log saved to: {log_csv_path}")
        print(f"Training curves saved to: {log_plot_path}")
        if not args.disable_tensorboard:
            tensorboard_root = Path(args.tensorboard_dir) if args.tensorboard_dir else Path(args.log_dir) / "tensorboard"
            print(f"TensorBoard log saved to: {tensorboard_log_dir}")
            print(f"TensorBoard command: tensorboard --logdir {tensorboard_root}")
        if best_metrics is not None:
            print(f"Best selection score: {best_selection_score:.4f}")
            print(f"Best validation MCC: {best_mcc:.4f}")
            print(f"Validation AUC upper bound seen during training: {best_auc_seen:.4f}")
            best_val_loss, best_acc, best_auc_eval, best_sens, best_spec, best_f1, best_prec, best_mcc_eval = best_metrics
            print(
                "Best checkpoint metrics: "
                f"loss={best_val_loss:.4f} acc={best_acc:.4f} auc={best_auc_eval:.4f} "
                f"sens={best_sens:.4f} spec={best_spec:.4f} f1={best_f1:.4f} "
                f"prec={best_prec:.4f} mcc={best_mcc_eval:.4f}"
            )
            print(f"Best checkpoint bias gap: {compute_bias_gap(best_sens, best_spec):.4f}")
            print(f"Best checkpoint constraints met: {best_constraints_met}")
            print(f"Best epoch: {best_epoch}")
            if best_eval_threshold is not None:
                print(f"Best val threshold: {best_threshold:.4f}")
        elif not use_validation:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(unwrap_model(model).state_dict(), save_path)
            print(f"Final-train mode epochs: {args.epochs}")
            print(f"Final checkpoint saved to: {save_path}")
        print("=" * 60 + "\n")

    safe_swanlab_finish(args)
    safe_tensorboard_close(tensorboard_writer)

    return {
        "seed": seed,
        "best_mcc": float(best_mcc),
        "best_auc_seen": float(best_auc_seen),
        "best_selection_score": float(best_selection_score),
        "best_constraints_met": int(best_constraints_met),
        "best_threshold": float(best_threshold),
        "save_path": str(save_path),
        "log_csv_path": str(log_csv_path),
        "log_plot_path": str(log_plot_path),
        "tensorboard_log_dir": str(tensorboard_log_dir),
        "train_time_sec": total_time,
        "train_csv": str(train_path),
        "val_csv": str(val_path) if val_path is not None else "",
        "best_epoch": int(best_epoch) if best_epoch is not None else int(args.epochs),
        "gate_loss_weight": float(args.gate_loss_weight),
        "gate_alpha": float(args.gate_alpha),
        "gate_temperature": float(args.gate_temperature),
        "gate_oracle_temperature": float(args.gate_oracle_temperature),
        "freeze_batchnorm": int(args.freeze_batchnorm),
        "freeze_batchnorm_affine": int(args.freeze_batchnorm_affine),
        "freeze_source_experts": int(args.freeze_source_experts),
        **(
            {
                "best_val_loss": float(best_val_loss),
                "best_acc": float(best_acc),
                "best_auc_eval": float(best_auc_eval),
                "best_sens": float(best_sens),
                "best_spec": float(best_spec),
                "best_f1": float(best_f1),
                "best_prec": float(best_prec),
                "best_mcc_eval": float(best_mcc_eval),
            }
            if best_metrics is not None and is_main_process()
            else {}
        ),
    }


def summarize_multi_seed_results(title, results, summary_path):
    if not results:
        return

    summary_df = pd.DataFrame(results)
    mccs = np.array([item["best_mcc"] for item in results], dtype=np.float32)
    aucs = np.array([item["best_auc_seen"] for item in results], dtype=np.float32)

    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    for item in results:
        line = f"seed={item['seed']}  best_epoch={int(item['best_epoch'])}  checkpoint={item['save_path']}"
        if not np.isnan(item["best_mcc"]):
            line += f"  best_mcc={item['best_mcc']:.4f}  best_auc_seen={item['best_auc_seen']:.4f}"
        print(line)
    if not np.isnan(mccs).all():
        print(f"mean_mcc={np.nanmean(mccs):.4f}  std_mcc={np.nanstd(mccs):.4f}")
    if not np.isnan(aucs).all():
        print(f"mean_best_auc_seen={np.nanmean(aucs):.4f}  std_best_auc_seen={np.nanstd(aucs):.4f}")
    print("=" * 60 + "\n")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"Multi-seed summary saved to: {summary_path}")


def summarize_cv_results(args, results):
    if not results:
        return

    summary_df = pd.DataFrame(results)
    metric_columns = [
        column
        for column in ("best_acc", "best_auc_eval", "best_f1", "best_mcc_eval")
        if column in summary_df.columns
    ]

    print("\n" + "=" * 60)
    print(f"{'Cross-Validation Summary':^60}")
    print("=" * 60)
    for _, row in summary_df.iterrows():
        line = (
            f"fold={int(row['fold'])}  seed={int(row['seed'])}  "
            f"train={int(row['train_size'])}  val={int(row['val_size'])}  "
            f"best_epoch={int(row['best_epoch'])}  best_mcc={row['best_mcc']:.4f}"
        )
        if "best_auc_eval" in row:
            line += f"  best_val_auc={row['best_auc_eval']:.4f}"
        print(line)

    for column in metric_columns:
        print(f"{column}: mean={summary_df[column].mean():.4f}  std={summary_df[column].std(ddof=0):.4f}")

    summary_path = resolve_summary_csv_path(resolve_save_path(args.save_path, args.model), f"{args.num_folds}fold_cv_summary")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"Cross-validation summary saved to: {summary_path}")
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    validate_args(args)
    if not SWANLAB_AVAILABLE:
        args.disable_swanlab = True

    setup_distributed(args)

    if is_main_process():
        print("\n" + "=" * 60)
        print(f"{'Model Training Pipeline':^60}")
        print("=" * 60)

    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = list(DEFAULT_RESEARCH_SEEDS)
    multi_seed = len(seeds) > 1
    if is_main_process():
        print(f"Active seeds: {seeds}")

    if args.cv_csv:
        fold_records = prepare_cv_splits(args)
        cv_results = []

        for fold_record in fold_records:
            fold_idx = fold_record["fold"]
            fold_results = []
            for run_idx, seed in enumerate(seeds, start=1):
                if is_main_process():
                    print("\n" + "#" * 60)
                    print(f"{'Cross Validation Run':^20}: fold={fold_idx}/{args.num_folds} seed={seed} ({run_idx}/{len(seeds)})")
                    print("#" * 60)
                result = train_one_seed(
                    args,
                    seed,
                    fold_record["train_csv"],
                    fold_record["val_csv"],
                    multi_seed,
                    run_suffix=f"fold{fold_idx}",
                    run_title=f"Model Training Pipeline | Fold {fold_idx}/{args.num_folds}",
                )
                if is_main_process():
                    result.update(
                        {
                            "fold": fold_idx,
                            "train_size": fold_record["train_size"],
                            "val_size": fold_record["val_size"],
                        }
                    )
                    fold_results.append(result)
                    cv_results.append(result)

            if is_main_process() and len(fold_results) > 1:
                summary_path = resolve_summary_csv_path(
                    resolve_save_path(args.save_path, args.model),
                    f"fold{fold_idx}_multiseed_summary",
                )
                summarize_multi_seed_results(
                    f"Fold {fold_idx} Multi-Seed Summary",
                    fold_results,
                    summary_path,
                )

        if is_main_process():
            summarize_cv_results(args, cv_results)
    else:
        if not args.train_csv:
            raise ValueError("Provide --train-csv for single-run training, or use --cv-csv for cross validation.")
        train_path = Path(args.train_csv)
        if not args.final_train and not args.val_csv:
            raise ValueError("Provide --val-csv for standard training, or add --final-train to fit on the full development set without validation.")
        val_path = Path(args.val_csv) if args.val_csv else None

        results = []
        for run_idx, seed in enumerate(seeds, start=1):
            if is_main_process() and multi_seed:
                print("\n" + "#" * 60)
                print(f"{'Seed Run':^60}")
                print(f"{'seed':>24}: {seed}")
                print(f"{'progress':>24}: {run_idx}/{len(seeds)}")
                print("#" * 60)
            result = train_one_seed(
                args,
                seed,
                train_path,
                val_path,
                multi_seed,
                run_title="Model Training Pipeline | Final-Train Mode" if args.final_train else None,
                use_validation=not args.final_train,
            )
            if is_main_process():
                results.append(result)

        if is_main_process():
            summary_path = resolve_summary_csv_path(
                resolve_save_path(args.save_path, args.model, final_train=args.final_train),
                "multiseed_summary",
            )
            summarize_multi_seed_results("Multi-Seed Summary", results, summary_path)

    cleanup_distributed()


__all__ = [
    "main",
    "summarize_cv_results",
    "summarize_multi_seed_results",
    "train_one_seed",
    "validate_loaded_gate_config",
]
