from __future__ import annotations

import argparse

from models.model_factory import MODEL_CHOICES
from trainer.logging import SWANLAB_CLI_MODES, default_swanlab_mode


DEFAULT_RESEARCH_SEEDS = [3407, 42, 1234]
ROI_MODEL_TOKENS: tuple[str, ...] = ()


def parse_spatial_shape(value: str | None):
    if value is None:
        return None
    parts = value.replace("x", ",").split(",")
    shape = tuple(int(item.strip()) for item in parts if item.strip())
    if len(shape) != 3 or any(item <= 0 for item in shape):
        raise argparse.ArgumentTypeError(
            "Expected three positive dimensions, e.g. 128,128,64 or 96x96x96."
        )
    return shape


def describe_model(model_name: str) -> str | None:
    descriptions = {
        "dual_backbone_view_consistency_reliability": (
            "Dual R3D-18/R(2+1)D-18 deterministic-view CNN with case-adaptive source fusion"
        ),
        "dual_backbone_vcrf_single_view": "DB-VCRF ablation with original view only",
        "dual_backbone_vcrf_single_view_t0": "DB-VCRF single-view ablation with the original T0 view",
        "dual_backbone_vcrf_single_view_td": "DB-VCRF single-view ablation with the depth-flipped TD view",
        "dual_backbone_vcrf_single_view_th": "DB-VCRF single-view ablation with the height-flipped TH view",
        "dual_backbone_vcrf_single_view_tw": "DB-VCRF single-view ablation with the width-flipped TW view",
        "dual_backbone_vcrf_t0_oracle_residual_gate": (
            "T0 dual-backbone DB-VCRF with oracle-supervised residual reliability gating"
        ),
        "dual_backbone_vcrf_no_depth_flip": "DB-VCRF ablation with original, height-flipped, and width-flipped views",
        "dual_backbone_vcrf_audit_depth_only": (
            "DB-VCRF with T0/TH/TW prediction sources and TD retained only for audit"
        ),
        "dual_backbone_vcrf_descriptor_off": "DB-VCRF ablation without uncertainty/disagreement descriptors",
        "dual_backbone_vcrf_feature_only": "DB-VCRF ablation using feature-only gating inputs",
        "dual_backbone_vcrf_logit_only": "DB-VCRF ablation using probability/logit-only gating inputs",
        "dual_backbone_vcrf_global_static": "DB-VCRF ablation with global learned static source weights",
        "dual_backbone_vcrf_fullcue_no_aux_viewreg": (
            "DB-VCRF ablation with full reliability cues but auxiliary and view losses disabled"
        ),
        "dual_backbone_vcrf_aux_off": "DB-VCRF ablation with auxiliary source loss disabled",
        "dual_backbone_vcrf_viewreg_off": "DB-VCRF ablation with view-consistency regularizer disabled",
        "dual_r3d_vcrf_full": "Matched dual-R3D DB-VCRF ablation",
        "dual_r2plus1d_vcrf_full": "Matched dual-R(2+1)D DB-VCRF ablation",
        "resnet18": "MedicalNet-style 3D ResNet-18 benchmark",
        "resnet34": "MedicalNet-style 3D ResNet-34 benchmark",
        "resnet50": "MedicalNet-style 3D ResNet-50 benchmark",
        "densenet121": "MONAI 3D DenseNet-121 benchmark",
        "densenet161": "MONAI 3D DenseNet-161 benchmark",
        "densenet169": "MONAI 3D DenseNet-169 benchmark",
        "resnext50": "3D ResNeXt-50 benchmark",
        "resnext101": "3D ResNeXt-101 benchmark",
        "vit": "MONAI 3D Vision Transformer benchmark",
        "swin3d": "MONAI 3D Swin Transformer benchmark",
        "multiscale_gat": "Multi-scale feature + graph-attention benchmark",
        "r3d": "Kinetics-pretrained R3D-18 benchmark",
        "r2plus1d": "Kinetics-pretrained R(2+1)D-18 benchmark",
        "s3d": "Kinetics-pretrained separable 3D CNN benchmark",
        "convnext_slice_mil": "ImageNet-pretrained ConvNeXt-Tiny with gated ABUS slice-attention MIL",
        "convnext_25d_ordered_mil": (
            "ImageNet-pretrained ConvNeXt-Tiny with ordered 2.5D slices and Transformer MIL"
        ),
        "convnext_triplanar_25d_mil": (
            "ImageNet-pretrained ConvNeXt-Tiny with tri-planar 2.5D Transformer MIL"
        ),
        "convnext_triplanar_25d_mil_center_only": (
            "TriMIL ablation with repeated center slices, retaining 72 three-channel instances"
        ),
        "convnext_triplanar_25d_mil_mean_pool": (
            "TriMIL ablation with mean pooling instead of CLS/Transformer aggregation"
        ),
        "convnext_triplanar_25d_mil_no_position": (
            "TriMIL ablation without within-plane position embeddings"
        ),
        "convnext_triplanar_25d_mil_no_plane": (
            "TriMIL ablation without plane-identity embeddings"
        ),
        "convnext_triplanar_25d_mil_z_only": (
            "TriMIL Z-only input control with 24 triplets and Transformer aggregation"
        ),
        "primus_m": "OpenMind-SimCLR-pretrained Primus-M 3D medical transformer",
    }
    return descriptions.get(model_name)


def model_requires_roi_mask(model_name: str | None) -> bool:
    """Return True for model choices that require ROI masks at forward time."""
    return bool(model_name) and any(token in str(model_name) for token in ROI_MODEL_TOKENS)


def validate_args(args) -> None:
    teacher_members = getattr(args, "teacher_members", None) or []
    primus_pretrained_path = getattr(args, "primus_pretrained_path", None)
    init_checkpoint = getattr(args, "init_checkpoint", None)
    if primus_pretrained_path and getattr(args, "model", None) != "primus_m":
        raise ValueError("--primus-pretrained-path is only valid with --model primus_m.")
    if primus_pretrained_path and init_checkpoint:
        raise ValueError(
            "--primus-pretrained-path and --init-checkpoint are mutually exclusive."
        )
    if (
        getattr(args, "model", None) == "primus_m"
        and not primus_pretrained_path
        and not init_checkpoint
    ):
        raise ValueError(
            "--primus-pretrained-path is required when training primus_m from OpenMind weights."
        )
    teacher_member_models = []
    for member in teacher_members:
        if len(member) != 3:
            raise ValueError("--teacher-member expects MODEL CHECKPOINT WEIGHT.")
        model_name, _, weight = member
        if model_name not in MODEL_CHOICES:
            raise ValueError(f"Unknown teacher member model: {model_name}")
        if float(weight) <= 0.0:
            raise ValueError("--teacher-member weight must be positive.")
        teacher_member_models.append(model_name)
    if teacher_members and (
        getattr(args, "teacher_model", None) is not None
        or getattr(args, "teacher_checkpoint", None)
    ):
        raise ValueError("--teacher-member cannot be combined with --teacher-model/--teacher-checkpoint.")
    roi_dependent_models = [
        name
        for name in (
            getattr(args, "model", None),
            getattr(args, "teacher_model", None),
            *teacher_member_models,
        )
        if model_requires_roi_mask(name)
    ]
    if roi_dependent_models and not getattr(args, "use_roi_mask", False):
        joined = ", ".join(roi_dependent_models)
        raise ValueError(
            f"--use-roi-mask is required for ROI-aware model(s): {joined}"
        )
    if getattr(args, "hard_negative_mining_csv", None):
        if getattr(args, "cohort_balanced_batches", False) or getattr(args, "label_balanced_batches", False):
            raise ValueError(
                "--hard-negative-mining-csv cannot be combined with "
                "--cohort-balanced-batches or --label-balanced-batches."
            )
        if float(getattr(args, "hard_negative_sampling_factor", 1.0)) < 1.0:
            raise ValueError("--hard-negative-sampling-factor must be >= 1.0.")
    ema_decay = float(getattr(args, "ema_decay", 0.0))
    if ema_decay < 0.0 or ema_decay >= 1.0:
        raise ValueError("--ema-decay must be in [0, 1).")
    if getattr(args, "ema_start_epoch", None) is not None and int(args.ema_start_epoch) < 1:
        raise ValueError("--ema-start-epoch must be >= 1.")
    mixup_alpha = float(getattr(args, "mixup_alpha", 0.0))
    mixup_prob = float(getattr(args, "mixup_prob", 0.0))
    if mixup_alpha < 0.0:
        raise ValueError("--mixup-alpha must be >= 0.")
    if mixup_prob < 0.0 or mixup_prob > 1.0:
        raise ValueError("--mixup-prob must be in [0, 1].")
    gate_alpha = float(getattr(args, "gate_alpha", 0.5))
    if gate_alpha < 0.0 or gate_alpha > 1.0:
        raise ValueError("--gate-alpha must be in [0, 1].")
    if float(getattr(args, "gate_temperature", 1.0)) <= 0.0:
        raise ValueError("--gate-temperature must be > 0.")
    if float(getattr(args, "gate_oracle_temperature", 1.0)) <= 0.0:
        raise ValueError("--gate-oracle-temperature must be > 0.")
    if float(getattr(args, "gate_loss_weight", 0.0)) < 0.0:
        raise ValueError("--gate-loss-weight must be >= 0.")
    if (
        float(getattr(args, "gate_loss_weight", 0.0)) > 0.0
        and getattr(args, "model", None) != "dual_backbone_vcrf_t0_oracle_residual_gate"
    ):
        raise ValueError(
            "--gate-loss-weight > 0 is only supported by "
            "dual_backbone_vcrf_t0_oracle_residual_gate."
        )
    if float(getattr(args, "gate_loss_weight", 0.0)) > 0.0 and gate_alpha == 0.0:
        raise ValueError("--gate-alpha must be > 0 when --gate-loss-weight > 0.")
    if (
        getattr(args, "freeze_source_experts", False)
        and getattr(args, "model", None) != "dual_backbone_vcrf_t0_oracle_residual_gate"
    ):
        raise ValueError(
            "--freeze-source-experts is only supported by "
            "dual_backbone_vcrf_t0_oracle_residual_gate."
        )
    if (
        getattr(args, "freeze_source_experts", False)
        and int(getattr(args, "freeze_backbone_epochs", 0)) <= 0
    ):
        raise ValueError(
            "--freeze-source-experts requires --freeze-backbone-epochs > 0; "
            "experts are frozen only during that warmup phase."
        )
    if getattr(args, "freeze_batchnorm_affine", False) and not getattr(
        args,
        "freeze_batchnorm",
        False,
    ):
        raise ValueError("--freeze-batchnorm-affine requires --freeze-batchnorm.")
    if float(getattr(args, "pairwise_auc_weight", 0.0)) < 0.0:
        raise ValueError("--pairwise-auc-weight must be >= 0.")
    if float(getattr(args, "pairwise_auc_margin", 0.0)) < 0.0:
        raise ValueError("--pairwise-auc-margin must be >= 0.")
    if float(getattr(args, "focal_penalty_weight", 0.0)) < 0.0:
        raise ValueError("--focal-penalty-weight must be >= 0.")
    if float(getattr(args, "roi_consistency_weight", 0.0)) < 0.0:
        raise ValueError("--roi-consistency-weight must be >= 0.")
    if int(getattr(args, "roi_consistency_jitter_voxels", 0)) < 0:
        raise ValueError("--roi-consistency-jitter-voxels must be >= 0.")
    consistency_dropout = float(getattr(args, "roi_consistency_dropout_prob", 0.0))
    if consistency_dropout < 0.0 or consistency_dropout > 1.0:
        raise ValueError("--roi-consistency-dropout-prob must be in [0, 1].")
    spatial_aug_policies = {"full", "hw_only", "slice_xy", "xy_flip_only", "none"}
    if getattr(args, "spatial_aug_policy", "full") not in spatial_aug_policies:
        raise ValueError(
            "--spatial-aug-policy must be one of: full, hw_only, slice_xy, "
            "xy_flip_only, none."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train DB-VCRF ABUS models on cached 3D medical volumes.")
    parser.add_argument("--train-csv", default=None, help="Path to training csv.")
    parser.add_argument("--val-csv", default=None, help="Path to validation csv.")
    parser.add_argument(
        "--cv-csv",
        default=None,
        help="Path to a development csv used for stratified K-fold cross validation. Keep the held-out test split excluded.",
    )
    parser.add_argument("--num-folds", type=int, default=5, help="Number of folds for cross validation.")
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="Optional 1-based fold index to run in cross-validation mode. If omitted, all folds are trained sequentially.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Useful when per-GPU batch size must stay small.",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument(
        "--monitor-start-epoch",
        type=int,
        default=None,
        help="First epoch allowed to update the best checkpoint and start early stopping. Defaults to a few epochs after warmup/freeze.",
    )
    parser.add_argument(
        "--score-min-delta",
        type=float,
        default=0.01,
        help="Minimum composite-score improvement required to refresh the best checkpoint.",
    )
    parser.add_argument("--score-weight-auc", type=float, default=0.5, help="Composite-score weight for validation AUC.")
    parser.add_argument(
        "--score-weight-mcc",
        type=float,
        default=0.3,
        help="Composite-score weight for validation MCC mapped to [0, 1].",
    )
    parser.add_argument(
        "--score-weight-bias",
        type=float,
        default=0.2,
        help="Composite-score weight for class-balance quality computed from 1 - bias_gap.",
    )
    parser.add_argument("--min-sensitivity", type=float, default=0.60, help="Preferred minimum sensitivity for a best-checkpoint candidate.")
    parser.add_argument("--min-specificity", type=float, default=0.60, help="Preferred minimum specificity for a best-checkpoint candidate.")
    parser.add_argument("--max-bias-gap", type=float, default=0.35, help="Preferred maximum |sensitivity - specificity| for a best-checkpoint candidate.")
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing for training cross-entropy. Helps reduce over-confident small-sample fitting.",
    )
    parser.add_argument(
        "--loss-function",
        default="cross_entropy",
        choices=[
            "cross_entropy",
            "ce_conf_penalty",
            "ce_focal_penalty",
            "focal",
            "asymmetric",
        ],
        help="Loss function to use for training.",
    )
    parser.add_argument(
        "--confidence-penalty-weight",
        type=float,
        default=0.02,
        help="KL-to-uniform penalty weight for ce_conf_penalty loss.",
    )
    parser.add_argument(
        "--hard-negative-penalty-weight",
        type=float,
        default=0.0,
        help=(
            "Optional train-side penalty weight for negative samples whose "
            "positive-class logit margin exceeds --hard-negative-penalty-margin."
        ),
    )
    parser.add_argument(
        "--hard-negative-penalty-margin",
        type=float,
        default=0.0,
        help="Positive-logit margin threshold for --hard-negative-penalty-weight.",
    )
    parser.add_argument(
        "--pairwise-auc-weight",
        type=float,
        default=0.0,
        help=(
            "Optional in-batch pairwise AUC margin penalty weight. "
            "Use with label-balanced batches so each batch contains both classes."
        ),
    )
    parser.add_argument(
        "--pairwise-auc-margin",
        type=float,
        default=0.0,
        help="Target positive-vs-negative logit-margin gap for --pairwise-auc-weight.",
    )
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Alpha parameter for Focal Loss.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss.")
    parser.add_argument(
        "--focal-penalty-weight",
        type=float,
        default=0.25,
        help="Extra focal hard-example penalty weight for ce_focal_penalty loss.",
    )
    parser.add_argument("--asymmetric-gamma-pos", type=float, default=0.0, help="Gamma positive parameter for Asymmetric Loss.")
    parser.add_argument("--asymmetric-gamma-neg", type=float, default=4.0, help="Gamma negative parameter for Asymmetric Loss.")
    parser.add_argument(
        "--class-weight-mode",
        default="sqrt_balanced",
        choices=["none", "balanced", "sqrt_balanced"],
        help="Training loss class weighting mode computed from the training split.",
    )
    parser.add_argument(
        "--cohort-balanced-batches",
        action="store_true",
        help="Build training batches with equal samples from each cohort listed in --cohort-column.",
    )
    parser.add_argument(
        "--label-balanced-batches",
        action="store_true",
        help="Build training batches with equal samples from each class label.",
    )
    parser.add_argument(
        "--hard-negative-mining-csv",
        default=None,
        help=(
            "Optional train-side mined hard-negative CSV. When provided with "
            "--hard-negative-sampling-factor > 1, matching case IDs are oversampled."
        ),
    )
    parser.add_argument(
        "--hard-negative-sampling-factor",
        type=float,
        default=1.0,
        help="Sampling weight multiplier for cases from --hard-negative-mining-csv.",
    )
    parser.add_argument(
        "--hard-negative-case-column",
        default="case_id",
        help="Case ID column in --hard-negative-mining-csv.",
    )
    parser.add_argument(
        "--hard-negative-flag-column",
        default="is_hard_negative",
        help=(
            "Optional boolean/0-1 column used to filter selected hard negatives. "
            "If absent, all rows in the mining CSV are used."
        ),
    )
    parser.add_argument("--cohort-column", default="cohort", help="CSV column used by --cohort-balanced-batches.")
    parser.add_argument(
        "--teacher-model",
        default=None,
        choices=MODEL_CHOICES,
        help="Optional teacher model used for knowledge distillation.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default=None,
        help="Checkpoint path for the teacher model. Required when --teacher-model is set.",
    )
    parser.add_argument(
        "--teacher-member",
        dest="teacher_members",
        action="append",
        nargs=3,
        metavar=("MODEL", "CHECKPOINT", "WEIGHT"),
        default=None,
        help=(
            "Weighted teacher ensemble member for knowledge distillation. "
            "Repeat as MODEL CHECKPOINT WEIGHT. Cannot be combined with --teacher-model."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help=(
            "Optional checkpoint used to initialize model weights before training. "
            "In multi-seed mode, pass the unsuffixed base path and _seed<seed> is resolved automatically."
        ),
    )
    parser.add_argument(
        "--init-checkpoint-strict",
        action="store_true",
        help="Require an exact key match when loading --init-checkpoint. Defaults to partial initialization.",
    )
    parser.add_argument(
        "--primus-pretrained-path",
        default=None,
        help=(
            "Official PrimusM-OpenMind-SimCLR checkpoint containing network_weights. "
            "Used only to initialize primus_m before training."
        ),
    )
    parser.add_argument(
        "--kd-alpha",
        type=float,
        default=0.5,
        help="Knowledge-distillation loss weight. Total loss is (1-alpha)*CE + alpha*KD.",
    )
    parser.add_argument("--kd-temperature", type=float, default=4.0, help="Softmax temperature for knowledge distillation.")
    parser.add_argument(
        "--selection-threshold-metric",
        default="mcc",
        choices=["mcc", "f1", "accuracy", "balanced_accuracy", "sensitivity", "specificity"],
        help="Binary validation threshold metric used during checkpoint selection.",
    )
    parser.add_argument(
        "--disable-selection-threshold-search",
        action="store_true",
        help="Disable per-epoch validation threshold search and fall back to argmax metrics for checkpoint selection.",
    )
    parser.add_argument(
        "--constrain-selection-threshold",
        action="store_true",
        help=(
            "During validation threshold search, prefer thresholds satisfying "
            "--min-sensitivity and --min-specificity. Falls back to unconstrained search if impossible."
        ),
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=0.1,
        help="Backbone learning-rate multiplier for pretrained models with a separate classification head.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help=(
            "Optional exponential moving average decay for validation/checkpoint weights. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--ema-start-epoch",
        type=int,
        default=None,
        help="First 1-based epoch to update/evaluate EMA weights. Defaults to monitor start.",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="Beta distribution alpha for train-time 3D mixup. Set >0 with --mixup-prob to enable.",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=0.0,
        help="Per-batch probability for train-time 3D mixup. Requires --mixup-alpha > 0.",
    )
    parser.add_argument(
        "--gate-loss-weight",
        type=float,
        default=0.0,
        help="KL supervision weight for the oracle residual reliability gate.",
    )
    parser.add_argument(
        "--gate-alpha",
        type=float,
        default=0.5,
        help="Learned-weight fraction in w=(1-alpha)/S+alpha*softmax(h/tau).",
    )
    parser.add_argument(
        "--gate-temperature",
        type=float,
        default=1.0,
        help="Temperature tau applied to learned gate logits.",
    )
    parser.add_argument(
        "--gate-oracle-temperature",
        type=float,
        default=1.0,
        help="Temperature used to form q=softmax(-detached source CE/tau).",
    )
    parser.add_argument(
        "--roi-consistency-weight",
        type=float,
        default=0.0,
        help="Optional train-side consistency penalty weight between original and perturbed ROI-mask logits.",
    )
    parser.add_argument(
        "--roi-consistency-jitter-voxels",
        type=int,
        default=0,
        help="Max extra ROI-mask shift in voxels for --roi-consistency-weight.",
    )
    parser.add_argument(
        "--roi-consistency-dropout-prob",
        type=float,
        default=0.0,
        help="Sample-level ROI-mask dropout probability for --roi-consistency-weight.",
    )
    parser.add_argument("--num-workers", type=int, default=12, help="Dataloader workers.")
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=5,
        help="Train only the classification head for the first N epochs when supported.",
    )
    parser.add_argument(
        "--freeze-source-experts",
        action="store_true",
        help=(
            "For the residual-gate variant, freeze both source backbones and "
            "classifiers during --freeze-backbone-epochs."
        ),
    )
    parser.add_argument(
        "--freeze-batchnorm",
        action="store_true",
        help="Keep all BatchNorm modules in evaluation mode during training.",
    )
    parser.add_argument(
        "--freeze-batchnorm-affine",
        action="store_true",
        help="Also freeze BatchNorm affine parameters; requires --freeze-batchnorm.",
    )
    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable automatic mixed precision on CUDA.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision on CUDA.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single random seed. If omitted together with --seeds, the script defaults to three research seeds.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run training sequentially for multiple seeds, e.g. --seeds 3407 42 1234. Defaults to three research seeds when omitted.",
    )
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes.")
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=MODEL_CHOICES,
        help="Backbone used for training.",
    )
    parser.add_argument(
        "--use-roi-mask",
        action="store_true",
        help="Load ROI masks from the CSV and pass them to mask-aware models.",
    )
    parser.add_argument(
        "--roi-mask-dropout-prob",
        type=float,
        default=0.0,
        help="Training-only probability of replacing the ROI mask with zeros.",
    )
    parser.add_argument(
        "--roi-mask-jitter-voxels",
        type=int,
        default=0,
        help="Training-only max absolute ROI mask shift in voxels along each spatial axis.",
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
    parser.add_argument(
        "--spatial-aug-policy",
        default="full",
        choices=("full", "hw_only", "slice_xy", "xy_flip_only", "none"),
        help=(
            "Training-only geometric augmentation for cached [C,X,Y,Z] arrays. "
            "'slice_xy' applies X-Y flips and 90-degree X-Y rotations; "
            "'xy_flip_only' applies X-Y flips without rotation and preserves Z "
            "ordering. 'full' and the legacy 'hw_only' can reverse Z and are kept "
            "only to reproduce older experiments. 'none' disables spatial flips "
            "and rotations while retaining intensity/noise/dropout augmentation."
        ),
    )
    parser.add_argument("--device", default="cuda", help="Device string.")
    parser.add_argument("--dist-backend", default="nccl", help="Distributed backend used when launched with torchrun.")
    parser.add_argument(
        "--find-unused-parameters",
        action="store_true",
        help="Enable DDP unused parameter detection for models with conditional branches.",
    )
    parser.add_argument(
        "--final-train",
        action="store_true",
        help="Train on the full development set without validation and save the final checkpoint after the last epoch.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Path to save the best checkpoint. Defaults to assets/checkpoints_3d/best_model_<model>.pth.",
    )
    parser.add_argument("--log-dir", default="assets/training_logs", help="Directory to save local training logs and plots.")
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help="Directory to save TensorBoard event files. Defaults to <log-dir>/tensorboard with one subdirectory per run.",
    )
    parser.add_argument("--disable-tensorboard", action="store_true", help="Disable TensorBoard event logging.")
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
    parser.set_defaults(amp=True)
    return parser.parse_args()
