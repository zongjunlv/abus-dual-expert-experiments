from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


MULTI_SEED_METRIC_COLUMNS = (
    "loss",
    "accuracy",
    "auc",
    "sensitivity",
    "specificity",
    "f1",
    "precision",
    "mcc",
)


def print_probability_summary(labels_arr, probs_arr) -> None:
    pos_probs = probs_arr[:, 1] if probs_arr.shape[1] > 1 else probs_arr[:, 0]
    print("\nPositive-Class Probability Summary")
    print(
        f"all   : min={pos_probs.min():.4f} q25={np.quantile(pos_probs, 0.25):.4f} "
        f"median={np.quantile(pos_probs, 0.5):.4f} q75={np.quantile(pos_probs, 0.75):.4f} "
        f"max={pos_probs.max():.4f}"
    )
    for label in sorted(np.unique(labels_arr).tolist()):
        mask = labels_arr == label
        cls_probs = pos_probs[mask]
        if len(cls_probs) == 0:
            continue
        print(
            f"label={label}: min={cls_probs.min():.4f} q25={np.quantile(cls_probs, 0.25):.4f} "
            f"median={np.quantile(cls_probs, 0.5):.4f} q75={np.quantile(cls_probs, 0.75):.4f} "
            f"max={cls_probs.max():.4f}"
        )


def export_predictions_csv(export_path, dataset, labels_arr, preds_arr, probs_arr) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    df = dataset.data.copy().reset_index(drop=True)
    df["label_true"] = labels_arr.astype(int)
    df["label_pred"] = preds_arr.astype(int)
    if probs_arr.shape[1] == 1:
        df["prob_0"] = probs_arr[:, 0]
    else:
        for idx in range(probs_arr.shape[1]):
            df[f"prob_{idx}"] = probs_arr[:, idx]
    df.to_csv(export_path, index=False)
    print(f"\nPer-sample predictions exported to: {export_path}")


def print_prediction_details(labels_arr, preds_arr) -> None:
    print("labels unique:", np.unique(labels_arr, return_counts=True))
    print("preds unique :", np.unique(preds_arr, return_counts=True))
    print(confusion_matrix(labels_arr, preds_arr))


def build_result_record(seed, checkpoint_paths, export_path, decision_threshold, metrics):
    loss, accuracy, auc, sensitivity, specificity, f1, precision, mcc, labels_arr, preds_arr, probs_arr = metrics
    _ = probs_arr
    result = {
        "seed": int(seed) if seed is not None else np.nan,
        "checkpoint": ",".join(str(item) for item in checkpoint_paths),
        "export_pred_csv": str(export_path),
        "decision_threshold": float(decision_threshold) if decision_threshold is not None else np.nan,
        "loss": float(loss),
        "accuracy": float(accuracy),
        "auc": float(auc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "precision": float(precision),
        "mcc": float(mcc),
    }

    cm = confusion_matrix(labels_arr, preds_arr)
    result["confusion_matrix"] = str(cm.tolist())
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        result.update(
            {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return result


def summarize_multi_seed_results(title, results, summary_path) -> None:
    if not results:
        return

    per_seed_df = pd.DataFrame(results)
    aggregate_rows = []
    for aggregate_name in ("mean", "std", "worst"):
        row = {"row_type": "aggregate", "aggregate_name": aggregate_name}
        for metric_name in MULTI_SEED_METRIC_COLUMNS:
            values = pd.to_numeric(per_seed_df[metric_name], errors="coerce").to_numpy(dtype=np.float64)
            if np.isnan(values).all():
                row[metric_name] = np.nan
            elif aggregate_name == "mean":
                row[metric_name] = float(np.nanmean(values))
            elif aggregate_name == "std":
                row[metric_name] = float(np.nanstd(values, ddof=0))
            else:
                row[metric_name] = float(np.nanmax(values) if metric_name == "loss" else np.nanmin(values))

        threshold_values = pd.to_numeric(per_seed_df["decision_threshold"], errors="coerce").to_numpy(dtype=np.float64)
        if np.isnan(threshold_values).all():
            row["decision_threshold"] = np.nan
        elif aggregate_name == "mean":
            row["decision_threshold"] = float(np.nanmean(threshold_values))
        elif aggregate_name == "std":
            row["decision_threshold"] = float(np.nanstd(threshold_values, ddof=0))
        else:
            row["decision_threshold"] = float(np.nanmin(threshold_values))
        aggregate_rows.append(row)

    export_df = pd.concat(
        [
            per_seed_df.assign(row_type="seed", aggregate_name=""),
            pd.DataFrame(aggregate_rows),
        ],
        ignore_index=True,
        sort=False,
    )

    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    for item in results:
        seed_label = f"seed={int(item['seed'])}" if not pd.isna(item["seed"]) else "seed=na"
        threshold_label = (
            f"{item['decision_threshold']:.4f}" if not pd.isna(item["decision_threshold"]) else "argmax"
        )
        line = (
            f"{seed_label}  threshold={threshold_label}  "
            f"acc={item['accuracy']:.4f}  auc={item['auc']:.4f}  "
            f"sens={item['sensitivity']:.4f}  spec={item['specificity']:.4f}  "
            f"f1={item['f1']:.4f}  mcc={item['mcc']:.4f}"
        )
        if {"tn", "fp", "fn", "tp"} <= item.keys():
            line += f"  cm=[[{item['tn']},{item['fp']}],[{item['fn']},{item['tp']}]]"
        print(line)
        print(f"checkpoint={item['checkpoint']}")

    for metric_name in MULTI_SEED_METRIC_COLUMNS:
        values = pd.to_numeric(per_seed_df[metric_name], errors="coerce").to_numpy(dtype=np.float64)
        if np.isnan(values).all():
            continue
        worst_value = np.nanmax(values) if metric_name == "loss" else np.nanmin(values)
        worst_label = "max" if metric_name == "loss" else "min"
        print(
            f"{metric_name}: mean={np.nanmean(values):.4f}  std={np.nanstd(values, ddof=0):.4f}  "
            f"worst({worst_label})={worst_value:.4f}"
        )

    threshold_values = pd.to_numeric(per_seed_df["decision_threshold"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isnan(threshold_values).all():
        print(
            "decision_threshold: "
            f"mean={np.nanmean(threshold_values):.4f}  std={np.nanstd(threshold_values, ddof=0):.4f}  "
            f"min={np.nanmin(threshold_values):.4f}  max={np.nanmax(threshold_values):.4f}"
        )

    print("=" * 60 + "\n")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(summary_path, index=False)
    print(f"Multi-seed test summary saved to: {summary_path}")
