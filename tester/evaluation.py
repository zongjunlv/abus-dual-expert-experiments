from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.medical_dataset import Medical_Dataset
from models.model_factory import build_model
from tester.checkpoints import load_checkpoint_weights
from tester.summary import (
    build_result_record,
    export_predictions_csv,
    print_prediction_details,
    print_probability_summary,
)
from utils.batch import batch_to_device, forward_model
from utils.evaluator import evaluate_model
from utils.metrics import (
    binary_predictions_from_probs,
    compute_all_metrics,
    find_best_binary_threshold,
)


def collect_ensemble_outputs(checkpoint_paths, dataloader, device, model_name, num_classes):
    models = []
    for checkpoint_path in checkpoint_paths:
        model = build_model(model_name, num_classes).to(device)
        load_checkpoint_weights(model, checkpoint_path, device)
        model.eval()
        models.append(model)

    labels_list, probs_list = [], []
    total_loss = 0.0
    processed_samples = 0
    criterion = torch.nn.NLLLoss()

    with torch.no_grad():
        for batch in dataloader:
            moved = batch_to_device(batch, device)
            img = moved["image"]
            labels = moved["label"]
            mask = moved["mask"]

            prob_sum = None
            for model in models:
                model_output = forward_model(model, image=img, mask=mask)
                logits = model_output[0] if isinstance(model_output, tuple) else model_output
                probs = F.softmax(logits, dim=1)
                prob_sum = probs if prob_sum is None else prob_sum + probs

            mean_probs = prob_sum / len(models)
            loss = criterion(torch.log(mean_probs.clamp_min(1e-8)), labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            processed_samples += batch_size
            labels_list.append(labels.cpu().numpy())
            probs_list.append(mean_probs.cpu().numpy())

    labels_arr = np.concatenate(labels_list, axis=0)
    probs_arr = np.concatenate(probs_list, axis=0)
    avg_loss = total_loss / max(1, processed_samples)
    return avg_loss, labels_arr, probs_arr


def summarize_outputs(avg_loss, labels_arr, probs_arr, show_details=True, decision_threshold: float | None = None):
    if decision_threshold is not None and probs_arr.ndim == 2 and probs_arr.shape[1] == 2:
        preds_arr = binary_predictions_from_probs(probs_arr, threshold=decision_threshold)
    else:
        preds_arr = np.argmax(probs_arr, axis=1)

    if show_details:
        print_prediction_details(labels_arr, preds_arr)

    all_metrics = compute_all_metrics(labels_arr, preds_arr, probs_arr)
    accuracy = all_metrics["accuracy"]
    auc = all_metrics.get("auc", 0.0)
    sensitivity = all_metrics["sensitivity"]
    specificity = all_metrics["specificity"]
    f1 = all_metrics["f1"]
    precision = all_metrics["precision"]
    mcc = all_metrics["mcc"]

    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"Sens:     {sensitivity:.4f}")
    print(f"Spec:     {specificity:.4f}")
    print(f"F1+:      {f1:.4f}")
    print(f"Prec+:    {precision:.4f}")
    print(f"MCC:      {mcc:.4f}")

    return avg_loss, accuracy, auc, sensitivity, specificity, f1, precision, mcc, labels_arr, preds_arr, probs_arr


def evaluate_checkpoint_ensemble(
    checkpoint_paths,
    dataloader,
    device,
    model_name,
    num_classes,
    show_details=True,
    desc="Test Ensemble",
    decision_threshold: float | None = None,
):
    avg_loss, labels_arr, probs_arr = collect_ensemble_outputs(
        checkpoint_paths,
        dataloader,
        device,
        model_name,
        num_classes,
    )
    print(f"\n{desc}")
    return summarize_outputs(
        avg_loss,
        labels_arr,
        probs_arr,
        show_details=show_details,
        decision_threshold=decision_threshold,
    )


def build_eval_dataloader(dataset, batch_size, num_workers):
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "pin_memory": True,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 8
    return DataLoader(dataset, **dataloader_kwargs)


def resolve_decision_threshold(args, checkpoint_paths, device):
    if args.decision_threshold is not None:
        print(f"Using fixed decision threshold: {args.decision_threshold:.4f}")
        return float(args.decision_threshold)

    if args.val_csv is None:
        return None

    if args.num_classes != 2:
        raise ValueError("Automatic threshold search currently supports only binary classification.")

    print(f"\nSelecting threshold on validation set: {args.val_csv}")
    val_dataset = Medical_Dataset(
        mode="val",
        csv_path=args.val_csv,
        roi_size=(256, 256, 128),
        margin=12,
        return_mask=args.use_roi_mask,
        target_shape=args.target_shape,
    )
    val_dataloader = build_eval_dataloader(val_dataset, args.batch_size, args.num_workers)

    if len(checkpoint_paths) == 1:
        checkpoint_path = checkpoint_paths[0]
        model = build_model(args.model, args.num_classes)
        load_checkpoint_weights(model, checkpoint_path, device)
        metrics = evaluate_model(
            model,
            val_dataloader,
            device,
            verbose=False,
            show_details=False,
            desc="Val Threshold Search",
            return_outputs=True,
        )
        _, _, _, _, _, _, _, _, labels_arr, _, probs_arr = metrics
    else:
        avg_loss, labels_arr, probs_arr = collect_ensemble_outputs(
            checkpoint_paths,
            val_dataloader,
            device,
            args.model,
            args.num_classes,
        )
        _ = avg_loss

    best_threshold, best_metrics = find_best_binary_threshold(
        labels_arr,
        probs_arr,
        metric_name=args.threshold_metric,
    )
    print(
        "Validation threshold search: "
        f"metric={args.threshold_metric} threshold={best_threshold:.4f} "
        f"acc={best_metrics['accuracy']:.4f} auc={best_metrics.get('auc', 0.0):.4f} "
        f"sens={best_metrics['sensitivity']:.4f} spec={best_metrics['specificity']:.4f} "
        f"f1+={best_metrics['f1']:.4f} mcc={best_metrics['mcc']:.4f}"
    )
    return best_threshold


def run_single_evaluation(args, checkpoint_paths, export_path, test_dataset, test_dataloader, device, seed=None):
    decision_threshold = resolve_decision_threshold(args, checkpoint_paths, device)

    if len(checkpoint_paths) == 1:
        checkpoint_path = checkpoint_paths[0]
        model = build_model(args.model, args.num_classes)
        load_checkpoint_weights(model, checkpoint_path, device)
        metrics = evaluate_model(
            model,
            test_dataloader,
            device,
            verbose=True,
            show_details=True,
            desc="Test",
            return_outputs=True,
            decision_threshold=decision_threshold,
        )
    else:
        print(f"Ensembling {len(checkpoint_paths)} checkpoints:")
        for checkpoint_path in checkpoint_paths:
            print(f"  - {checkpoint_path}")
        metrics = evaluate_checkpoint_ensemble(
            checkpoint_paths,
            test_dataloader,
            device,
            args.model,
            args.num_classes,
            show_details=True,
            desc="Test Ensemble",
            decision_threshold=decision_threshold,
        )

    _, _, _, _, _, _, _, _, labels_arr, preds_arr, probs_arr = metrics
    if decision_threshold is not None:
        print(f"\nApplied decision threshold: {decision_threshold:.4f}")
    print_probability_summary(labels_arr, probs_arr)
    export_predictions_csv(export_path, test_dataset, labels_arr, preds_arr, probs_arr)

    result_record = build_result_record(
        seed=seed,
        checkpoint_paths=checkpoint_paths,
        export_path=export_path,
        decision_threshold=decision_threshold,
        metrics=metrics,
    )
    return metrics, result_record
