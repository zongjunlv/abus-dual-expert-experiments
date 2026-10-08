from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _move_optional_tensor(value: Any, device: torch.device) -> Any:
    if value is None:
        return None
    if hasattr(value, "to"):
        return value.to(device)
    return value


def batch_to_device(batch: Any, device: torch.device) -> dict[str, torch.Tensor | None]:
    """Normalize dataloader batches to image/label/mask tensors on a device."""
    if isinstance(batch, Mapping):
        image = batch.get("image")
        label = batch.get("label")
        mask = batch.get("mask")
    elif isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        if len(batch) == 2:
            image, label = batch
            mask = None
        elif len(batch) == 3:
            image, mask, label = batch
        else:
            raise ValueError(f"Unsupported batch sequence length: {len(batch)}")
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    if image is None or label is None:
        raise ValueError("Batch must provide image and label.")

    return {
        "image": image.to(device),
        "label": label.to(device),
        "mask": _move_optional_tensor(mask, device),
    }


def forward_model(
    model: torch.nn.Module,
    image: torch.Tensor,
    mask: torch.Tensor | None = None,
):
    """Call either a legacy image-only model or a model that declares ROI support."""
    bare_model = model.module if hasattr(model, "module") else model
    if mask is not None and bool(getattr(bare_model, "supports_roi_mask", False)):
        return model(image, roi_mask=mask)
    return model(image)
