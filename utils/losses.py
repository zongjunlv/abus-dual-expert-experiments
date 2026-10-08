from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AsymmetricLoss(nn.Module):
    """Cross-entropy focusing with different gamma values for positive and negative samples."""

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        positive_class: int = 1,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.positive_class = int(positive_class)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        gammas = torch.where(
            targets == self.positive_class,
            torch.full_like(pt, self.gamma_pos),
            torch.full_like(pt, self.gamma_neg),
        )
        asymmetric_loss = torch.pow(1.0 - pt, gammas) * ce_loss

        if self.reduction == 'mean':
            return asymmetric_loss.mean()
        if self.reduction == 'sum':
            return asymmetric_loss.sum()
        return asymmetric_loss


class CrossEntropyWithConfidencePenalty(nn.Module):
    """Cross entropy plus a mild confidence penalty toward uniform predictions."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        penalty_weight: float = 0.02,
        reduction: str = "mean",
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=float(label_smoothing),
            reduction=reduction,
        )
        self.penalty_weight = float(penalty_weight)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(inputs, targets)
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = log_probs.exp()
        uniform_log_prob = -torch.log(
            inputs.new_tensor(inputs.size(-1), dtype=inputs.dtype)
        )
        confidence_penalty = (probs * (log_probs - uniform_log_prob)).sum(dim=-1)

        if self.reduction == "mean":
            confidence_penalty = confidence_penalty.mean()
        elif self.reduction == "sum":
            confidence_penalty = confidence_penalty.sum()

        return ce_loss + self.penalty_weight * confidence_penalty


class CrossEntropyWithFocalPenalty(nn.Module):
    """Smoothed weighted cross entropy plus a small focal hard-example penalty."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        focal_weight: float = 0.25,
        focal_gamma: float = 1.0,
        focal_alpha: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", weight.detach().clone())
        else:
            self.weight = None
        self.label_smoothing = float(label_smoothing)
        self.focal_weight = float(focal_weight)
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )
        if self.focal_weight <= 0.0:
            return ce_loss

        per_sample_ce = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-per_sample_ce)
        focal = self.focal_alpha * torch.pow(1.0 - pt, self.focal_gamma) * per_sample_ce
        if self.reduction == "mean":
            focal = focal.mean()
        elif self.reduction == "sum":
            focal = focal.sum()
        return ce_loss + self.focal_weight * focal


class HardNegativeLogitPenaltyLoss(nn.Module):
    """Wrap a base criterion with a small penalty on hard negative margins."""

    def __init__(
        self,
        base_loss: nn.Module,
        penalty_weight: float = 0.0,
        margin: float = 0.0,
        positive_class: int = 1,
        negative_class: int = 0,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.penalty_weight = float(penalty_weight)
        self.margin = float(margin)
        self.positive_class = int(positive_class)
        self.negative_class = int(negative_class)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = self.base_loss(inputs, targets)
        if base.ndim > 0:
            base = base.mean()
        if self.penalty_weight <= 0.0:
            return base
        if inputs.size(1) <= max(self.positive_class, self.negative_class):
            raise ValueError("HardNegativeLogitPenaltyLoss requires the configured class logits.")

        negative_mask = targets == self.negative_class
        if not bool(negative_mask.any()):
            return base

        positive_margin = (
            inputs[:, self.positive_class] - inputs[:, self.negative_class]
        )
        hard_margin = F.relu(positive_margin[negative_mask] - self.margin)
        penalty = hard_margin.pow(2).mean()
        return base + self.penalty_weight * penalty


class PairwiseAUCMarginLoss(nn.Module):
    """Wrap a base criterion with an in-batch pairwise ranking penalty."""

    def __init__(
        self,
        base_loss: nn.Module,
        penalty_weight: float = 0.0,
        margin: float = 0.0,
        positive_class: int = 1,
        negative_class: int = 0,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.penalty_weight = float(penalty_weight)
        self.margin = float(margin)
        self.positive_class = int(positive_class)
        self.negative_class = int(negative_class)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base = self.base_loss(inputs, targets)
        if base.ndim > 0:
            base = base.mean()
        if self.penalty_weight <= 0.0:
            return base
        if inputs.size(1) <= max(self.positive_class, self.negative_class):
            raise ValueError("PairwiseAUCMarginLoss requires the configured class logits.")

        positive_mask = targets == self.positive_class
        negative_mask = targets == self.negative_class
        if not bool(positive_mask.any() and negative_mask.any()):
            return base

        binary_margin = inputs[:, self.positive_class] - inputs[:, self.negative_class]
        pos_scores = binary_margin[positive_mask]
        neg_scores = binary_margin[negative_mask]
        pairwise_gap = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        ranking_penalty = F.softplus(self.margin - pairwise_gap).mean()
        return base + self.penalty_weight * ranking_penalty


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross entropy loss
    Helps prevent overconfidence and improves generalization
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(inputs, dim=-1)
        weight = inputs.new_ones(inputs.size())
        weight.fill_(self.smoothing / (inputs.size(-1) - 1))
        weight.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = (-weight * log_prob).sum(dim=-1)
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss with multiple components for balanced training
    """
    def __init__(
        self,
        ce_weight: float = 0.5,
        focal_weight: float = 0.3,
        label_smoothing: float = 0.05,
        alpha: float = 0.25,
        gamma: float = 2.0
    ):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.ce_weight * ce + self.focal_weight * focal


class AuxLoss(nn.Module):
    """
    Auxiliary loss for multi-head training
    """
    def __init__(self, aux_weight: float = 0.15, main_weight: float = 1.0):
        super().__init__()
        self.aux_weight = aux_weight
        self.main_weight = main_weight
        self.base_loss = CombinedLoss()

    def forward(
        self,
        main_logits: torch.Tensor,
        aux_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        main_loss = self.base_loss(main_logits, targets)
        aux_loss = self.base_loss(aux_logits, targets)
        return self.main_weight * main_loss + self.aux_weight * aux_loss


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks
    Useful for balanced class representation
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        targets_one_hot = targets_one_hot.permute(0, 4, 1, 2, 3)

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3, 4))
        union = inputs.sum(dim=(2, 3, 4)) + targets_one_hot.sum(dim=(2, 3, 4))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BalancedCrossEntropy(nn.Module):
    """
    Balanced cross entropy with class weights
    Automatically computes class weights from batch
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute class weights inversely proportional to frequency
        num_classes = inputs.size(1)
        class_counts = torch.bincount(targets, minlength=num_classes).float()
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * num_classes

        weight = class_weights.to(inputs.device)
        return F.cross_entropy(inputs, targets, weight=weight)
