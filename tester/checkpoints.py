from __future__ import annotations

from pathlib import Path

import torch


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            state = checkpoint.get(key)
            if isinstance(state, dict):
                return state
    if isinstance(checkpoint, dict):
        return checkpoint
    raise RuntimeError("Unsupported checkpoint format: expected a state-dict or a dict containing one.")


def strip_module_prefix(state_dict):
    return {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def load_checkpoint_weights(model, checkpoint_path: Path, device, strict: bool = True) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        # DB-VCRF checkpoints saved before the static source bias was added do
        # not contain this zero-initialized parameter.  Allow only that narrow
        # compatibility case; all other missing/unexpected keys remain errors.
        if not (
            strict
            and hasattr(model, "static_source_logits")
            and "static_source_logits" not in state_dict
        ):
            raise
        incompatible = model.load_state_dict(state_dict, strict=False)
        allowed_missing = {"static_source_logits"}
        if set(incompatible.missing_keys) <= allowed_missing and not incompatible.unexpected_keys:
            print(
                "Loaded checkpoint with legacy DB-VCRF compatibility: "
                "initialized missing static_source_logits to model default."
            )
            return
        raise
