import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from models.model_factory import MODEL_CHOICES, build_model
from trainer.checkpoints import (
    build_teacher_model,
    extract_state_dict,
    load_checkpoint_weights,
    strip_module_prefix,
    unwrap_model,
)
from trainer.cli import (
    DEFAULT_RESEARCH_SEEDS,
    describe_model,
    model_requires_roi_mask,
    parse_args,
    parse_spatial_shape,
    validate_args,
)
from trainer.distributed import (
    cleanup_distributed,
    get_rank,
    is_distributed,
    is_main_process,
    reduce_mean,
    setup_distributed,
)
from trainer.logging import (
    SWANLAB_AVAILABLE,
    TENSORBOARD_SCALAR_TAGS,
    format_tensorboard_config,
    safe_swanlab_finish,
    safe_swanlab_init,
    safe_swanlab_log,
    safe_tensorboard_close,
    safe_tensorboard_init,
    safe_tensorboard_log,
    save_training_artifacts,
    swanlab_enabled,
)
from trainer.paths import (
    append_path_suffix,
    resolve_log_paths,
    resolve_run_name,
    resolve_run_save_path,
    resolve_save_path,
    resolve_summary_csv_path,
    resolve_tensorboard_log_dir,
)
from trainer.runner import (
    main,
    summarize_cv_results,
    summarize_multi_seed_results,
    train_one_seed,
)
from trainer.selection import (
    compute_bias_gap,
    compute_selection_score,
    meets_selection_constraints,
    normalize_mcc,
    resolve_monitor_start_epoch,
    resolve_validation_selection_metrics,
    should_update_best_checkpoint,
)
from trainer.setup import (
    absolutize_split_dataframe,
    build_cohort_balanced_batch_sampler,
    build_dataloader,
    build_hard_negative_batch_sampler,
    build_label_balanced_batch_sampler,
    build_optimizer,
    build_training_criterion,
    compute_balanced_class_weights,
    parse_label_value,
    prepare_cv_splits,
    resolve_reference_path,
    set_backbone_trainable,
    set_seed,
    should_enable_find_unused_parameters,
    split_backbone_and_head,
    supports_freeze_strategy,
    warmup_cosine,
)
from trainer.trainer import Trainer


__all__ = [
    "DEFAULT_RESEARCH_SEEDS",
    "MODEL_CHOICES",
    "SWANLAB_AVAILABLE",
    "TENSORBOARD_SCALAR_TAGS",
    "Trainer",
    "absolutize_split_dataframe",
    "append_path_suffix",
    "build_cohort_balanced_batch_sampler",
    "build_dataloader",
    "build_hard_negative_batch_sampler",
    "build_label_balanced_batch_sampler",
    "build_model",
    "build_optimizer",
    "build_teacher_model",
    "build_training_criterion",
    "cleanup_distributed",
    "compute_balanced_class_weights",
    "compute_bias_gap",
    "compute_selection_score",
    "describe_model",
    "extract_state_dict",
    "format_tensorboard_config",
    "get_rank",
    "is_distributed",
    "is_main_process",
    "load_checkpoint_weights",
    "main",
    "meets_selection_constraints",
    "model_requires_roi_mask",
    "normalize_mcc",
    "parse_args",
    "parse_label_value",
    "parse_spatial_shape",
    "prepare_cv_splits",
    "reduce_mean",
    "resolve_log_paths",
    "resolve_monitor_start_epoch",
    "resolve_reference_path",
    "resolve_run_name",
    "resolve_run_save_path",
    "resolve_save_path",
    "resolve_summary_csv_path",
    "resolve_tensorboard_log_dir",
    "resolve_validation_selection_metrics",
    "safe_swanlab_finish",
    "safe_swanlab_init",
    "safe_swanlab_log",
    "safe_tensorboard_close",
    "safe_tensorboard_init",
    "safe_tensorboard_log",
    "save_training_artifacts",
    "set_backbone_trainable",
    "set_seed",
    "setup_distributed",
    "should_enable_find_unused_parameters",
    "should_update_best_checkpoint",
    "split_backbone_and_head",
    "strip_module_prefix",
    "summarize_cv_results",
    "summarize_multi_seed_results",
    "supports_freeze_strategy",
    "swanlab_enabled",
    "train_one_seed",
    "unwrap_model",
    "validate_args",
    "warmup_cosine",
]


if __name__ == "__main__":
    main()
