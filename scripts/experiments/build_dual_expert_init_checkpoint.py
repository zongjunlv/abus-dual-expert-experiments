#!/usr/bin/env python3
"""Initialize the T0 residual-gate model from two single-expert checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import torch

from models.dual_backbone_view_consistency_reliability import (
    create_dual_backbone_view_consistency_reliability,
)


_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_tensor_state_dict(value: object) -> bool:
    return bool(value) and isinstance(value, Mapping) and all(
        isinstance(key, str) and torch.is_tensor(tensor)
        for key, tensor in value.items()
    )


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized = key[len("module.") :] if key.startswith("module.") else key
        if normalized in stripped:
            raise ValueError(f"Duplicate key after removing module prefix: {normalized}")
        stripped[normalized] = value
    return stripped


def load_strict_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """Load exactly one tensor-only state dict using PyTorch safe loading."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if _is_tensor_state_dict(checkpoint):
        return _strip_module_prefix(checkpoint)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a state-dict or a mapping containing one.")

    candidates = [checkpoint[key] for key in _STATE_DICT_KEYS if _is_tensor_state_dict(checkpoint.get(key))]
    if len(candidates) != 1:
        raise ValueError(
            "Checkpoint must contain exactly one state-dict under state_dict, "
            "model_state_dict, or model."
        )
    return _strip_module_prefix(candidates[0])


def _expert_mapping(
    target_state: Mapping[str, torch.Tensor],
    branch: str,
) -> dict[str, str]:
    backbone_prefix = f"{branch}_backbone."
    classifier_prefix = f"{branch}_classifier.2."
    mapping = {
        f"backbone.{key[len(backbone_prefix):]}": key
        for key in target_state
        if key.startswith(backbone_prefix)
    }
    for suffix in ("weight", "bias"):
        target_key = classifier_prefix + suffix
        if target_key not in target_state:
            raise ValueError(f"Target model is missing critical classifier layer: {target_key}")
        mapping[f"backbone.fc.{suffix}"] = target_key
    return mapping


def _map_expert(
    target_state: dict[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
    branch: str,
) -> tuple[list[str], list[str], list[str]]:
    expected = _expert_mapping(target_state, branch)
    missing = sorted(set(expected).difference(source_state))
    unexpected = sorted(set(source_state).difference(expected))
    if missing or unexpected:
        raise ValueError(
            f"{branch} checkpoint mapping rejected; missing required keys: {missing}; "
            f"unexpected keys: {unexpected}"
        )

    mapped = []
    for source_key, target_key in expected.items():
        source_tensor = source_state[source_key]
        target_tensor = target_state[target_key]
        if source_tensor.shape != target_tensor.shape:
            raise ValueError(
                f"{branch} shape mismatch for {source_key} -> {target_key}: "
                f"source={tuple(source_tensor.shape)} target={tuple(target_tensor.shape)}"
            )
        target_state[target_key] = source_tensor.detach().cpu().clone()
        mapped.append(f"{branch}:{source_key}->{target_key}")
    return mapped, missing, unexpected


def build_dual_expert_init_state_dict(
    model: torch.nn.Module,
    r3d_state_dict: Mapping[str, torch.Tensor],
    r2plus1d_state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    """Return a complete state dict that can be loaded with ``strict=True``."""
    if not _is_tensor_state_dict(r3d_state_dict):
        raise ValueError("r3d_state_dict must be a non-empty tensor-only state dict.")
    if not _is_tensor_state_dict(r2plus1d_state_dict):
        raise ValueError("r2plus1d_state_dict must be a non-empty tensor-only state dict.")
    target_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    r3d_mapped, r3d_missing, r3d_unexpected = _map_expert(
        target_state,
        _strip_module_prefix(r3d_state_dict),
        "r3d",
    )
    r2p1d_mapped, r2p1d_missing, r2p1d_unexpected = _map_expert(
        target_state,
        _strip_module_prefix(r2plus1d_state_dict),
        "r2p1d",
    )
    mapped = r3d_mapped + r2p1d_mapped
    mapped_targets = {item.rsplit("->", maxsplit=1)[1] for item in mapped}
    report = {
        "mapped_keys": mapped,
        "missing_keys": r3d_missing + r2p1d_missing,
        "unexpected_keys": r3d_unexpected + r2p1d_unexpected,
        "non_expert_initialized_keys": sorted(set(target_state).difference(mapped_targets)),
    }
    incompatible = model.load_state_dict(target_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Constructed state dict failed strict validation: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return target_state, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3d-checkpoint", type=Path, required=True)
    parser.add_argument("--r2plus1d-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--gate-alpha", type=float, default=0.5)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    model = create_dual_backbone_view_consistency_reliability(
        num_classes=args.num_classes,
        in_chans=1,
        pretrained=False,
        views=("none",),
        residual_reliability_gate=True,
        gate_alpha=args.gate_alpha,
        gate_temperature=args.gate_temperature,
    )
    state_dict, report = build_dual_expert_init_state_dict(
        model,
        load_strict_state_dict(args.r3d_checkpoint),
        load_strict_state_dict(args.r2plus1d_checkpoint),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, args.output)
    report_payload = {
        **report,
        "r3d_checkpoint": str(args.r3d_checkpoint),
        "r3d_checkpoint_sha256": sha256_file(args.r3d_checkpoint),
        "r2plus1d_checkpoint": str(args.r2plus1d_checkpoint),
        "r2plus1d_checkpoint_sha256": sha256_file(args.r2plus1d_checkpoint),
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "num_classes": args.num_classes,
        "gate_alpha": args.gate_alpha,
        "gate_temperature": args.gate_temperature,
        "initialization_seed": args.seed,
    }
    rendered_report = json.dumps(report_payload, indent=2, sort_keys=True)
    print(rendered_report)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(rendered_report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
