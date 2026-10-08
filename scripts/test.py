import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_str = str(PROJECT_ROOT)
if project_root_str in sys.path:
    sys.path.remove(project_root_str)
sys.path.insert(0, project_root_str)

from data.medical_dataset import Medical_Dataset
from models.model_factory import MODEL_CHOICES, build_model
from tester.checkpoints import (
    extract_state_dict,
    load_checkpoint_weights,
    strip_module_prefix,
)
from tester.cli import (
    DEFAULT_RESEARCH_SEEDS,
    describe_model,
    parse_args,
    parse_spatial_shape,
    resolve_requested_seeds,
    validate_args,
)
from tester.evaluation import (
    build_eval_dataloader,
    collect_ensemble_outputs,
    evaluate_checkpoint_ensemble,
    resolve_decision_threshold,
    run_single_evaluation,
    summarize_outputs,
)
from tester.logging import (
    safe_swanlab_finish,
    safe_swanlab_init,
    safe_swanlab_log,
    swanlab,
    swanlab_enabled,
)
from tester.paths import (
    SEED_SUFFIX_PATTERN,
    append_path_suffix,
    append_seed_suffix,
    resolve_checkpoint_path,
    resolve_checkpoint_paths,
    resolve_export_path,
    resolve_summary_path,
)
from tester.runner import main
from tester.summary import (
    MULTI_SEED_METRIC_COLUMNS,
    build_result_record,
    export_predictions_csv,
    print_prediction_details,
    print_probability_summary,
    summarize_multi_seed_results,
)
from utils.evaluator import evaluate_model
from utils.metrics import (
    binary_predictions_from_probs,
    compute_all_metrics,
    find_best_binary_threshold,
)


__all__ = [
    "DEFAULT_RESEARCH_SEEDS",
    "MODEL_CHOICES",
    "MULTI_SEED_METRIC_COLUMNS",
    "Medical_Dataset",
    "SEED_SUFFIX_PATTERN",
    "append_path_suffix",
    "append_seed_suffix",
    "binary_predictions_from_probs",
    "build_eval_dataloader",
    "build_model",
    "build_result_record",
    "collect_ensemble_outputs",
    "compute_all_metrics",
    "describe_model",
    "evaluate_checkpoint_ensemble",
    "evaluate_model",
    "export_predictions_csv",
    "extract_state_dict",
    "find_best_binary_threshold",
    "load_checkpoint_weights",
    "main",
    "parse_args",
    "parse_spatial_shape",
    "print_prediction_details",
    "print_probability_summary",
    "resolve_checkpoint_path",
    "resolve_checkpoint_paths",
    "resolve_decision_threshold",
    "resolve_export_path",
    "resolve_requested_seeds",
    "resolve_summary_path",
    "run_single_evaluation",
    "safe_swanlab_finish",
    "safe_swanlab_init",
    "safe_swanlab_log",
    "strip_module_prefix",
    "summarize_multi_seed_results",
    "summarize_outputs",
    "swanlab",
    "swanlab_enabled",
    "validate_args",
]


if __name__ == "__main__":
    main()
