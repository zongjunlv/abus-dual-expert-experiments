import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)

def compute_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算准确率"""
    return float(accuracy_score(labels, preds))

def compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """计算AUC，兼容二分类和多分类"""
    try:
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 2:
            if probs.ndim == 2 and probs.shape[1] >= 2:
                return float(roc_auc_score(labels, probs[:, 1]))
            return float(roc_auc_score(labels, probs))
        return float(roc_auc_score(labels, probs, multi_class='ovr', average='macro'))
    except ValueError:
        return 0.0


def compute_sensitivity_specificity(labels: np.ndarray, preds: np.ndarray) -> tuple[float, float]:
    """按混淆矩阵计算敏感性和特异性。二分类时返回正类TPR和负类TNR。"""
    cm = confusion_matrix(labels, preds)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return float(sensitivity), float(specificity)

    n_classes = cm.shape[0]
    sensitivities = []
    specificities = []
    total = np.sum(cm)

    for i in range(n_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = total - tp - fn - fp

        sensitivities.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    return float(np.mean(sensitivities)), float(np.mean(specificities))


def compute_macro_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算macro F1分数"""
    return float(f1_score(labels, preds, average='macro', zero_division=0))


def compute_positive_f1(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算二分类正类(标签1)F1；非二分类时回退到macro F1。"""
    unique_labels = set(np.unique(labels).tolist()) | set(np.unique(preds).tolist())
    if unique_labels <= {0, 1}:
        return float(f1_score(labels, preds, pos_label=1, zero_division=0))
    return compute_macro_f1(labels, preds)


def compute_macro_precision(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算macro Precision"""
    return float(precision_score(labels, preds, average='macro', zero_division=0))


def compute_positive_precision(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算二分类正类(标签1)Precision；非二分类时回退到macro Precision。"""
    unique_labels = set(np.unique(labels).tolist()) | set(np.unique(preds).tolist())
    if unique_labels <= {0, 1}:
        return float(precision_score(labels, preds, pos_label=1, zero_division=0))
    return compute_macro_precision(labels, preds)


def compute_precision(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算论文默认Precision：二分类正类(标签1)Precision，非二分类回退到macro。"""
    return compute_positive_precision(labels, preds)


def compute_matthews_corrcoef(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算Matthews相关系数 (多分类版本)"""
    try:
        return float(matthews_corrcoef(labels, preds))
    except:
        return 0.0


def get_binary_positive_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs)
    if probs.ndim == 1:
        return probs.astype(np.float32)
    if probs.ndim == 2 and probs.shape[1] == 1:
        return probs[:, 0].astype(np.float32)
    if probs.ndim == 2 and probs.shape[1] >= 2:
        return probs[:, 1].astype(np.float32)
    raise ValueError(f"Unsupported probability shape for binary thresholding: {probs.shape}")


def binary_predictions_from_probs(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    pos_probs = get_binary_positive_probs(probs)
    threshold = float(np.clip(threshold, 0.0, 1.0))
    return (pos_probs >= threshold).astype(np.int64)


def _balanced_accuracy_from_metrics(metrics: dict) -> float:
    return 0.5 * (float(metrics["sensitivity"]) + float(metrics["specificity"]))


def _select_threshold_score(metric_name: str, metrics: dict) -> float:
    if metric_name == "mcc":
        return float(metrics["mcc"])
    if metric_name == "f1":
        return float(metrics["f1"])
    if metric_name == "accuracy":
        return float(metrics["accuracy"])
    if metric_name == "balanced_accuracy":
        return _balanced_accuracy_from_metrics(metrics)
    if metric_name == "sensitivity":
        return float(metrics["sensitivity"])
    if metric_name == "specificity":
        return float(metrics["specificity"])
    raise ValueError(f"Unsupported threshold metric: {metric_name}")


def _find_threshold_candidate(
    labels: np.ndarray,
    probs: np.ndarray,
    candidate_thresholds: np.ndarray,
    metric_name: str,
    min_sensitivity: float | None,
    min_specificity: float | None,
) -> tuple[float, dict] | None:
    best_threshold = 0.5
    best_metrics = None
    best_key = None

    for threshold in candidate_thresholds:
        preds = binary_predictions_from_probs(probs, threshold=float(threshold))
        metrics = compute_all_metrics(labels, preds, probs)
        if min_sensitivity is not None and metrics["sensitivity"] < float(min_sensitivity):
            continue
        if min_specificity is not None and metrics["specificity"] < float(min_specificity):
            continue

        score = _select_threshold_score(metric_name, metrics)
        key = (
            score,
            float(metrics["mcc"]),
            float(metrics["f1"]),
            _balanced_accuracy_from_metrics(metrics),
            -abs(float(metrics["sensitivity"]) - float(metrics["specificity"])),
            -abs(float(threshold) - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    if best_metrics is None:
        return None
    return best_threshold, best_metrics


def find_best_binary_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
    metric_name: str = "mcc",
    min_sensitivity: float | None = None,
    min_specificity: float | None = None,
) -> tuple[float, dict]:
    labels = np.asarray(labels).astype(np.int64)
    unique_labels = np.unique(labels)
    if labels.ndim != 1 or set(unique_labels.tolist()) - {0, 1}:
        raise ValueError("Threshold search only supports binary labels {0, 1}.")

    pos_probs = get_binary_positive_probs(probs)
    unique_probs = np.unique(pos_probs)
    if unique_probs.size == 0:
        raise ValueError("Cannot search threshold on empty probability array.")

    if unique_probs.size == 1:
        candidate_thresholds = np.array([0.5], dtype=np.float32)
    else:
        midpoints = (unique_probs[:-1] + unique_probs[1:]) / 2.0
        candidate_thresholds = np.concatenate(
            (
                np.array([0.0], dtype=np.float32),
                midpoints.astype(np.float32),
                np.array([1.0], dtype=np.float32),
            )
        )

    constrained_candidate = _find_threshold_candidate(
        labels,
        pos_probs,
        candidate_thresholds,
        metric_name,
        min_sensitivity,
        min_specificity,
    )
    if constrained_candidate is not None:
        return constrained_candidate

    unconstrained_candidate = _find_threshold_candidate(
        labels,
        pos_probs,
        candidate_thresholds,
        metric_name,
        None,
        None,
    )
    if unconstrained_candidate is None:
        raise ValueError("No valid binary threshold candidate found.")
    return unconstrained_candidate



def compute_all_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray = None) -> dict:
    """计算所有指标"""
    metrics = {}

    # 基本指标
    metrics['accuracy'] = compute_accuracy(labels, preds)
    metrics['macro_f1'] = compute_macro_f1(labels, preds)
    metrics['macro_precision'] = compute_macro_precision(labels, preds)
    metrics['positive_f1'] = compute_positive_f1(labels, preds)
    metrics['positive_precision'] = compute_positive_precision(labels, preds)
    metrics['f1'] = metrics['positive_f1']
    metrics['precision'] = metrics['positive_precision']
    metrics['mcc'] = compute_matthews_corrcoef(labels, preds)

    # 敏感性和特异性
    sensitivity, specificity = compute_sensitivity_specificity(labels, preds)
    metrics['sensitivity'] = sensitivity
    metrics['specificity'] = specificity

    # AUC (如果提供了概率)
    if probs is not None:
        metrics['auc'] = compute_auc(labels, probs)

    return metrics
