from __future__ import annotations

from utils.metrics import find_best_binary_threshold


def compute_bias_gap(sensitivity, specificity):
    return abs(float(sensitivity) - float(specificity))


def normalize_mcc(mcc):
    return (float(mcc) + 1.0) / 2.0


def compute_selection_score(auc, mcc, bias_gap, args):
    weight_auc = max(float(args.score_weight_auc), 0.0)
    weight_mcc = max(float(args.score_weight_mcc), 0.0)
    weight_bias = max(float(args.score_weight_bias), 0.0)
    total_weight = weight_auc + weight_mcc + weight_bias
    if total_weight <= 0:
        return float("-inf")

    balance_score = 1.0 - min(max(float(bias_gap), 0.0), 1.0)
    return (
        weight_auc * float(auc)
        + weight_mcc * normalize_mcc(mcc)
        + weight_bias * balance_score
    ) / total_weight


def meets_selection_constraints(sensitivity, specificity, bias_gap, args):
    return (
        float(sensitivity) >= float(args.min_sensitivity)
        and float(specificity) >= float(args.min_specificity)
        and float(bias_gap) <= float(args.max_bias_gap)
    )


def should_update_best_checkpoint(candidate, best_candidate, score_min_delta):
    if best_candidate is None:
        return True

    candidate_constraints = bool(candidate["constraints_met"])
    best_constraints = bool(best_candidate["constraints_met"])
    if candidate_constraints != best_constraints:
        return candidate_constraints and not best_constraints

    score_delta = float(candidate["selection_score"]) - float(best_candidate["selection_score"])
    if score_delta > float(score_min_delta):
        return True
    if score_delta < -float(score_min_delta):
        return False

    candidate_tiebreak = (
        float(candidate["mcc"]),
        float(candidate["auc"]),
        -float(candidate["bias_gap"]),
    )
    best_tiebreak = (
        float(best_candidate["mcc"]),
        float(best_candidate["auc"]),
        -float(best_candidate["bias_gap"]),
    )
    return candidate_tiebreak > best_tiebreak


def resolve_monitor_start_epoch(args):
    if args.monitor_start_epoch is not None:
        return min(args.epochs, max(1, int(args.monitor_start_epoch)))
    return min(args.epochs, max(args.warmup_epochs, args.freeze_backbone_epochs + 3))


def resolve_validation_selection_metrics(args, labels_arr, probs_arr, fallback_metrics: dict):
    if args.disable_selection_threshold_search or args.num_classes != 2:
        return None, fallback_metrics, False

    threshold, threshold_metrics = find_best_binary_threshold(
        labels_arr,
        probs_arr,
        metric_name=args.selection_threshold_metric,
        min_sensitivity=args.min_sensitivity if args.constrain_selection_threshold else None,
        min_specificity=args.min_specificity if args.constrain_selection_threshold else None,
    )
    return float(threshold), threshold_metrics, True
