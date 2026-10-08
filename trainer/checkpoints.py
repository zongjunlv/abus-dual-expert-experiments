from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from models.model_factory import build_model
from utils.batch import forward_model


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
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


def load_checkpoint_weights(model, checkpoint_path: Path, device, strict: bool = True):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))
    return model.load_state_dict(state_dict, strict=strict)


class WeightedTeacherEnsemble(torch.nn.Module):
    """Combine frozen teacher logits as a weighted probability ensemble."""

    def __init__(self, members, probability_blend: bool = True):
        super().__init__()
        if not members:
            raise ValueError("WeightedTeacherEnsemble requires at least one member.")
        weights = torch.as_tensor([float(weight) for _, weight in members], dtype=torch.float32)
        if torch.any(weights <= 0):
            raise ValueError("Teacher ensemble weights must be positive.")
        self.members = torch.nn.ModuleList([model for model, _ in members])
        self.register_buffer("weights", weights / weights.sum())
        self.probability_blend = bool(probability_blend)
        self.supports_roi_mask = any(
            bool(getattr(model, "supports_roi_mask", False)) for model, _ in members
        )

    def forward(self, image, roi_mask=None):
        if self.supports_roi_mask and roi_mask is None:
            raise ValueError("ROI mask is required by at least one teacher ensemble member.")
        weighted_outputs = []
        for weight, model in zip(self.weights, self.members):
            logits = forward_model(model, image=image, mask=roi_mask)
            if isinstance(logits, tuple):
                logits = logits[0]
            if self.probability_blend:
                weighted_outputs.append(weight * F.softmax(logits, dim=1))
            else:
                weighted_outputs.append(weight * logits)
        combined = torch.stack(weighted_outputs, dim=0).sum(dim=0)
        if self.probability_blend:
            return torch.log(combined.clamp_min(1e-8))
        return combined


def build_teacher_model(args, device):
    teacher_members = getattr(args, "teacher_members", None) or []
    if teacher_members:
        if args.teacher_model is not None or args.teacher_checkpoint:
            raise ValueError("--teacher-member cannot be combined with --teacher-model/--teacher-checkpoint.")
        members = []
        for model_name, checkpoint_path, weight_value in teacher_members:
            teacher_checkpoint = Path(checkpoint_path)
            teacher_model = build_model(model_name, args.num_classes).to(device)
            load_checkpoint_weights(teacher_model, teacher_checkpoint, device, strict=True)
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            members.append((teacher_model, float(weight_value)))
        ensemble = WeightedTeacherEnsemble(members, probability_blend=True).to(device)
        ensemble.eval()
        for param in ensemble.parameters():
            param.requires_grad = False
        return ensemble
    if args.teacher_model is None:
        return None
    if not args.teacher_checkpoint:
        raise ValueError("--teacher-checkpoint is required when --teacher-model is set.")
    teacher_checkpoint = Path(args.teacher_checkpoint)
    teacher_model = build_model(args.teacher_model, args.num_classes).to(device)
    load_checkpoint_weights(teacher_model, teacher_checkpoint, device, strict=True)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model
