from __future__ import annotations

import time
from datetime import timedelta

import numpy as np
import torch

from data.medical_dataset import Medical_Dataset
from tester.cli import describe_model, parse_args, resolve_requested_seeds, validate_args
from tester.evaluation import build_eval_dataloader, run_single_evaluation
from tester.logging import (
    safe_swanlab_finish,
    safe_swanlab_init,
    safe_swanlab_log,
    swanlab,
    swanlab_enabled,
)
from tester.paths import resolve_checkpoint_paths, resolve_export_path, resolve_summary_path
from tester.summary import summarize_multi_seed_results


def main() -> None:
    args = parse_args()
    validate_args(args)
    if swanlab is None:
        args.disable_swanlab = True

    print("\n" + "=" * 60)
    print(f"{'Model Testing Pipeline':^60}")
    print("=" * 60)
    model_desc = describe_model(args.model)
    if model_desc is not None:
        print(f"Model detail: {model_desc}")

    start_time = time.time()
    test_dataset = Medical_Dataset(
        mode="test",
        csv_path=args.test_csv,
        roi_size=(256, 256, 128),
        margin=12,
        return_mask=args.use_roi_mask,
        target_shape=args.target_shape,
    )
    test_dataloader = build_eval_dataloader(test_dataset, args.batch_size, args.num_workers)

    device_name = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    device = torch.device(device_name)
    requested_seeds = resolve_requested_seeds(args)
    multi_seed_mode = len(requested_seeds) > 1

    if multi_seed_mode and not args.disable_swanlab:
        print("Multi-seed evaluation skips SwanLab logging. Use a single-seed run if you want one test record per run.")
        args.disable_swanlab = True

    if swanlab_enabled(args):
        checkpoint_paths = resolve_checkpoint_paths(args, seed=requested_seeds[0] if requested_seeds else None)
        safe_swanlab_init(
            args,
            config={
                "phase": "test",
                "batch_size": args.batch_size,
                "model_tag": args.model,
                "use_roi_mask": args.use_roi_mask,
                "target_shape": args.target_shape,
                "checkpoint": ",".join(str(item) for item in checkpoint_paths),
                "test_csv": args.test_csv,
            },
        )

    results = []
    if multi_seed_mode:
        for run_idx, seed in enumerate(requested_seeds, start=1):
            print("\n" + "-" * 60)
            print(f"{'Seed Evaluation':^60}")
            print(f"{'seed':>24}: {seed}")
            print(f"{'progress':>24}: {run_idx}/{len(requested_seeds)}")
            print("-" * 60)

            checkpoint_paths = resolve_checkpoint_paths(args, seed=seed)
            export_path = resolve_export_path(
                args.export_pred_csv,
                args.model,
                ensemble_size=len(checkpoint_paths),
                seed=seed,
                force_seed_suffix=True,
            )
            _, result_record = run_single_evaluation(
                args,
                checkpoint_paths,
                export_path,
                test_dataset,
                test_dataloader,
                device,
                seed=seed,
            )
            results.append(result_record)

        metrics = None
        summary_path = resolve_summary_path(args.summary_csv, args.model)
        summarize_multi_seed_results("Multi-Seed Test Summary", results, summary_path)
    else:
        seed = requested_seeds[0] if requested_seeds else None
        checkpoint_paths = resolve_checkpoint_paths(args, seed=seed)
        export_path = resolve_export_path(
            args.export_pred_csv,
            args.model,
            ensemble_size=len(checkpoint_paths),
            seed=seed,
            force_seed_suffix=False,
        )
        metrics, result_record = run_single_evaluation(
            args,
            checkpoint_paths,
            export_path,
            test_dataset,
            test_dataloader,
            device,
            seed=seed,
        )
        results.append(result_record)

        if swanlab_enabled(args):
            val_loss, accuracy, auc, sensitivity, specificity, f1, precision, mcc, *_ = metrics
            safe_swanlab_log(
                args,
                {
                    "accuracy": accuracy,
                    "val_loss": val_loss,
                    "auc": auc,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "f1": f1,
                    "precision": precision,
                    "mcc": mcc,
                },
            )

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Testing completed in {str(timedelta(seconds=int(total_time)))}")
    if multi_seed_mode:
        mccs = np.array([item["mcc"] for item in results], dtype=np.float64)
        aucs = np.array([item["auc"] for item in results], dtype=np.float64)
        print(
            f"Summary: seeds={requested_seeds} "
            f"mean_mcc={np.nanmean(mccs):.4f} std_mcc={np.nanstd(mccs, ddof=0):.4f} "
            f"mean_auc={np.nanmean(aucs):.4f} std_auc={np.nanstd(aucs, ddof=0):.4f}"
        )
    else:
        val_loss, accuracy, auc, sensitivity, specificity, f1, precision, mcc, *_ = metrics
        print(
            f"Summary: loss={val_loss:.4f} acc={accuracy:.4f} auc={auc:.4f} "
            f"sens={sensitivity:.4f} spec={specificity:.4f} f1={f1:.4f} "
            f"prec={precision:.4f} mcc={mcc:.4f}"
        )
    print("=" * 60 + "\n")

    safe_swanlab_finish(args)
