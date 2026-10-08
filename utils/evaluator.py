import torch
import sys
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from .batch import batch_to_device, forward_model
from .metrics import binary_predictions_from_probs, compute_all_metrics


def evaluate(model, dataloader, device, show_details: bool = True, desc: str = "Eval", decision_threshold: float | None = None):

    model.to(device).eval()
    labels_list, probs_list, preds_list = [], [], []
    total_loss = 0.0
    processed_samples = 0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        pbar = tqdm(
            dataloader,
            desc=desc,
            bar_format='{l_bar}{bar:30}{r_bar}',
            colour='green',
            disable=not sys.stdout.isatty(),
        )

        for batch in pbar:
            moved = batch_to_device(batch, device)
            img = moved["image"]
            labels = moved["label"]
            mask = moved["mask"]

            # 前向传播
            model_output = forward_model(model, image=img, mask=mask)
            if isinstance(model_output, tuple):
                logits = model_output[0]
            else:
                logits = model_output

            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            processed_samples += batch_size

            # 计算概率和预测
            probs = torch.softmax(logits, dim=1)
            if decision_threshold is not None and probs.shape[1] == 2:
                preds = torch.from_numpy(
                    binary_predictions_from_probs(probs.cpu().numpy(), threshold=decision_threshold)
                ).to(labels.device)
            else:
                preds = torch.argmax(probs, dim=1)

            # 收集结果
            labels_list.append(labels.cpu().numpy())
            probs_list.append(probs.cpu().numpy())
            preds_list.append(preds.cpu().numpy())

    # 合并所有批次的结果
    labels_arr = np.concatenate(labels_list, axis=0)
    probs_arr = np.concatenate(probs_list, axis=0)
    preds_arr = np.concatenate(preds_list, axis=0)

    if show_details:
        labels = labels_arr
        preds = preds_arr
        print('labels unique:', np.unique(labels, return_counts=True))
        print('preds unique :', np.unique(preds, return_counts=True))
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(labels, preds))


    # 计算指标
    all_metrics = compute_all_metrics(labels_arr, preds_arr, probs_arr)

    accuracy = all_metrics['accuracy']
    auc = all_metrics.get('auc', 0.0)
    sensitivity = all_metrics['sensitivity']
    specificity = all_metrics['specificity']
    f1 = all_metrics['f1']
    precision = all_metrics['precision']
    mcc = all_metrics['mcc']
    avg_loss = total_loss / max(1, processed_samples)

    return avg_loss, accuracy, auc, sensitivity, specificity, f1, precision, mcc, labels_arr, preds_arr, probs_arr


def evaluate_model(
    model,
    dataloader,
    device,
    verbose: bool = True,
    show_details: bool = True,
    desc: str = "Eval",
    return_outputs: bool = False,
    decision_threshold: float | None = None,
):

    val_loss, accuracy, auc, sensitivity, specificity, f1, precision, mcc, labels_arr, preds_arr, probs_arr = evaluate(
        model, dataloader, device, show_details=show_details, desc=desc, decision_threshold=decision_threshold
    )

    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC:      {auc:.4f}")
        print(f"Sens:     {sensitivity:.4f}")
        print(f"Spec:     {specificity:.4f}")
        print(f"F1+:      {f1:.4f}")
        print(f"Prec+:    {precision:.4f}")
        print(f"MCC:      {mcc:.4f}")

    if return_outputs:
        return val_loss, accuracy, auc, sensitivity, specificity, f1, precision, mcc, labels_arr, preds_arr, probs_arr

    return val_loss, accuracy, auc, sensitivity, specificity, f1, precision, mcc
