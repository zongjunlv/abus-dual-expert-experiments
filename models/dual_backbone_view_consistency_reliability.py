from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as tv_video

from .model import _adapt_video_first_conv


_VIEW_DIMS = {
    "none": (),
    "d": (2,),
    "h": (3,),
    "w": (4,),
}

_FUSION_INPUT_MODES = {
    "full",
    "no_descriptors",
    "feature_only",
    "logit_only",
    "descriptor_only",
    "global_static",
}


def _make_video_feature_backbone(name: str, in_chans: int, pretrained: bool) -> nn.Module:
    if name == "r3d":
        weights = tv_video.R3D_18_Weights.KINETICS400_V1 if pretrained else None
        backbone = tv_video.r3d_18(weights=weights)
    elif name == "r2plus1d":
        weights = tv_video.R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
        backbone = tv_video.r2plus1d_18(weights=weights)
    else:
        raise ValueError(f"Unsupported video backbone: {name}")
    return _adapt_video_first_conv(backbone, in_chans)


class DualBackboneViewConsistencyReliabilityFusion(nn.Module):
    """R3D + R(2+1)D deterministic-view source fusion.

    The model keeps the previously strong dual-video-backbone foundation and
    learns sample-specific source weights over eight sources:
    four deterministic R3D views and four deterministic R(2+1)D views.
    """

    aux_loss_weight = 0.20
    view_consistency_loss_weight = 0.03

    def __init__(
        self,
        num_classes: int = 2,
        in_chans: int = 1,
        pretrained: bool = True,
        views: tuple[str, ...] = ("none", "d", "h", "w"),
        audit_only_views: tuple[str, ...] = (),
        head_drop_rate: float = 0.25,
        reliability_hidden_dim: int = 64,
        source_embed_dim: int = 8,
        aux_loss_weight: float = 0.20,
        view_consistency_loss_weight: float = 0.03,
        fusion_input_mode: str = "full",
        backbone_pair: tuple[str, str] = ("r3d", "r2plus1d"),
        residual_reliability_gate: bool = False,
        gate_alpha: float = 0.5,
        gate_temperature: float = 1.0,
        freeze_source_experts: bool = False,
    ) -> None:
        super().__init__()
        invalid = [view for view in views if view not in _VIEW_DIMS]
        if invalid:
            raise ValueError(f"Unsupported deterministic view(s): {invalid}")
        if not views:
            raise ValueError("At least one deterministic view is required.")
        invalid_audit_views = [view for view in audit_only_views if view not in _VIEW_DIMS]
        if invalid_audit_views:
            raise ValueError(f"Unsupported audit-only view(s): {invalid_audit_views}")
        duplicated_views = set(views).intersection(audit_only_views)
        if duplicated_views:
            raise ValueError(
                "Prediction and audit-only views must be disjoint; duplicated views: "
                f"{sorted(duplicated_views)}"
            )
        if fusion_input_mode not in _FUSION_INPUT_MODES:
            raise ValueError(
                f"Unsupported fusion_input_mode={fusion_input_mode!r}; "
                f"expected one of {sorted(_FUSION_INPUT_MODES)}."
            )
        if len(backbone_pair) != 2:
            raise ValueError("backbone_pair must contain two backbone names.")
        if not 0.0 <= gate_alpha <= 1.0:
            raise ValueError("gate_alpha must be in [0, 1].")
        if gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be positive.")

        self.r3d_backbone = _make_video_feature_backbone(backbone_pair[0], in_chans, pretrained)
        self.r2p1d_backbone = _make_video_feature_backbone(backbone_pair[1], in_chans, pretrained)
        r3d_feature_dim = self.r3d_backbone.fc.in_features
        r2p1d_feature_dim = self.r2p1d_backbone.fc.in_features
        if r3d_feature_dim != r2p1d_feature_dim:
            raise ValueError("R3D and R(2+1)D feature dimensions must match.")
        feature_dim = r3d_feature_dim
        self.r3d_backbone.fc = nn.Identity()
        self.r2p1d_backbone.fc = nn.Identity()

        self.views = tuple(views)
        self.audit_only_views = tuple(audit_only_views)
        self.evaluation_views = self.views + self.audit_only_views
        self.num_views = len(self.views)
        self.num_evaluation_views = len(self.evaluation_views)
        self.num_sources = 2 * self.num_views
        self.fusion_input_mode = fusion_input_mode
        self.backbone_pair = tuple(backbone_pair)
        self.residual_reliability_gate = bool(residual_reliability_gate)
        self.gate_alpha = float(gate_alpha)
        self.gate_temperature = float(gate_temperature)
        self.freeze_source_experts = bool(freeze_source_experts)
        self._source_experts_frozen = False
        if self.residual_reliability_gate:
            # Persist non-default gate hyperparameters with the checkpoint so
            # existing tester entry points reconstruct the trained inference rule.
            self.register_buffer("_gate_alpha", torch.tensor(self.gate_alpha))
            self.register_buffer("_gate_temperature", torch.tensor(self.gate_temperature))
        variant_suffix = (
            "oracle_residual_gate"
            if self.residual_reliability_gate
            else f"view_consistency_{fusion_input_mode}"
        )
        self.variant = "dual_backbone_" + "_".join(self.backbone_pair) + f"_{variant_suffix}"
        self.aux_loss_weight = float(aux_loss_weight)
        self.view_consistency_loss_weight = float(view_consistency_loss_weight)

        self.r3d_classifier = self._make_classifier(
            feature_dim,
            num_classes,
            head_drop_rate,
            preserve_expert_logits=self.residual_reliability_gate,
        )
        self.r2p1d_classifier = self._make_classifier(
            feature_dim,
            num_classes,
            head_drop_rate,
            preserve_expert_logits=self.residual_reliability_gate,
        )
        self.source_embedding = nn.Embedding(self.num_sources, source_embed_dim)
        descriptor_dim = 5
        reliability_dim = self._fusion_input_dim(
            feature_dim=feature_dim,
            num_classes=num_classes,
            descriptor_dim=descriptor_dim,
            source_embed_dim=source_embed_dim,
        )
        self.reliability_head = nn.Sequential(
            nn.LayerNorm(reliability_dim),
            nn.Linear(reliability_dim, reliability_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_drop_rate),
            nn.Linear(reliability_hidden_dim, 1),
        )
        final_reliability = self.reliability_head[-1]
        nn.init.zeros_(final_reliability.weight)
        nn.init.zeros_(final_reliability.bias)
        self.static_source_logits = nn.Parameter(torch.zeros(self.num_sources, 1))

        self.last_source_logits: torch.Tensor | None = None
        self.last_reliability_logits: torch.Tensor | None = None
        self.last_reliability_weights: torch.Tensor | None = None
        self.last_backbone_weights: torch.Tensor | None = None

    @staticmethod
    def _make_classifier(
        feature_dim: int,
        num_classes: int,
        drop_rate: float,
        preserve_expert_logits: bool = False,
    ) -> nn.Sequential:
        if preserve_expert_logits:
            # Keep the linear layer at index 2 for checkpoint mapping while
            # preserving exact equivalence to the standalone expert's fc.
            return nn.Sequential(
                nn.Identity(),
                nn.Identity(),
                nn.Linear(feature_dim, num_classes),
            )
        return nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, num_classes),
        )

    def supports_head_warmup(self) -> bool:
        return True

    def set_backbone_trainable(self, trainable: bool) -> None:
        self._source_experts_frozen = bool(self.freeze_source_experts and not trainable)
        for module in (self.r3d_backbone, self.r2p1d_backbone):
            for param in module.parameters():
                param.requires_grad = trainable
        source_classifier_trainable = trainable or not self.freeze_source_experts
        for module in (self.r3d_classifier, self.r2p1d_classifier):
            for param in module.parameters():
                param.requires_grad = source_classifier_trainable
        for module in (self.source_embedding, self.reliability_head):
            for param in module.parameters():
                param.requires_grad = True
        self.static_source_logits.requires_grad = True
        self._apply_source_expert_mode()

    def _apply_source_expert_mode(self) -> None:
        modules = (
            self.r3d_backbone,
            self.r2p1d_backbone,
            self.r3d_classifier,
            self.r2p1d_classifier,
        )
        if self._source_experts_frozen:
            for module in modules:
                module.eval()
        else:
            for module in modules:
                module.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_source_expert_mode()
        return self

    def gate_hyperparameters(self) -> tuple[float, float]:
        if self.residual_reliability_gate:
            return float(self._gate_alpha.item()), float(self._gate_temperature.item())
        return self.gate_alpha, self.gate_temperature

    def get_param_groups(
        self,
        base_lr: float,
        backbone_lr_scale: float,
        weight_decay: float,
    ) -> list[dict]:
        backbone_params = list(self.r3d_backbone.parameters()) + list(
            self.r2p1d_backbone.parameters()
        )
        head_params = (
            list(self.r3d_classifier.parameters())
            + list(self.r2p1d_classifier.parameters())
            + list(self.source_embedding.parameters())
            + list(self.reliability_head.parameters())
            + [self.static_source_logits]
        )
        return [
            {
                "params": backbone_params,
                "lr": base_lr * backbone_lr_scale,
                "weight_decay": weight_decay,
            },
            {
                "params": head_params,
                "lr": base_lr * 1.5,
                "weight_decay": weight_decay,
            },
        ]

    def _fusion_input_dim(
        self,
        feature_dim: int,
        num_classes: int,
        descriptor_dim: int,
        source_embed_dim: int,
    ) -> int:
        mode = self.fusion_input_mode
        if mode == "global_static":
            return source_embed_dim
        if mode == "full":
            return feature_dim + num_classes + descriptor_dim + source_embed_dim
        if mode == "no_descriptors":
            return feature_dim + num_classes + source_embed_dim
        if mode == "feature_only":
            return feature_dim + source_embed_dim
        if mode == "logit_only":
            return num_classes + source_embed_dim
        if mode == "descriptor_only":
            return num_classes + descriptor_dim + source_embed_dim
        raise AssertionError(f"Unhandled fusion input mode: {mode}")

    def _make_view_batch(
        self,
        x: torch.Tensor,
        views: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        view_tensors = []
        for view in self.evaluation_views if views is None else views:
            dims = _VIEW_DIMS[view]
            view_tensors.append(torch.flip(x, dims=dims) if dims else x)
        return torch.cat(view_tensors, dim=0)

    def _reshape_view_outputs(
        self,
        tensor: torch.Tensor,
        batch_size: int,
        num_views: int | None = None,
    ) -> torch.Tensor:
        view_count = self.num_evaluation_views if num_views is None else int(num_views)
        return tensor.view(view_count, batch_size, -1).transpose(0, 1)

    @staticmethod
    def _entropy(probs: torch.Tensor) -> torch.Tensor:
        return -(probs.clamp_min(1e-7) * probs.clamp_min(1e-7).log()).sum(
            dim=-1,
            keepdim=True,
        )

    @staticmethod
    def _binary_margin(probs: torch.Tensor) -> torch.Tensor:
        if probs.shape[-1] == 2:
            return (probs[..., 1:2] - probs[..., 0:1]).abs()
        top2 = probs.topk(k=2, dim=-1).values
        return top2[..., 0:1] - top2[..., 1:2]

    @staticmethod
    def _view_mad(probs: torch.Tensor) -> torch.Tensor:
        target_probs = probs[..., 1:2] if probs.shape[-1] >= 2 else probs[..., :1]
        median = target_probs.median(dim=1, keepdim=True).values
        return (target_probs - median).abs().median(dim=1, keepdim=True).values

    def _prediction_descriptors(
        self,
        r3d_probs: torch.Tensor,
        r2p1d_probs: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.cat((r3d_probs, r2p1d_probs), dim=1)
        confidence = probs.max(dim=-1, keepdim=True).values
        entropy = self._entropy(probs)
        margin = self._binary_margin(probs)
        r3d_mad = self._view_mad(r3d_probs).expand(-1, self.num_views, -1)
        r2p1d_mad = self._view_mad(r2p1d_probs).expand(-1, self.num_views, -1)
        source_mad = torch.cat((r3d_mad, r2p1d_mad), dim=1)
        same_view_gap = (r3d_probs[..., 1:2] - r2p1d_probs[..., 1:2]).abs()
        cross_gap = torch.cat((same_view_gap, same_view_gap), dim=1)
        return torch.cat((confidence, entropy, margin, source_mad, cross_gap), dim=-1)

    def forward_components(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = int(x.shape[0])
        view_batch = self._make_view_batch(x)

        r3d_features = self.r3d_backbone(view_batch)
        r2p1d_features = self.r2p1d_backbone(view_batch)
        r3d_all_logits = self._reshape_view_outputs(
            self.r3d_classifier(r3d_features),
            batch_size,
        )
        r2p1d_all_logits = self._reshape_view_outputs(
            self.r2p1d_classifier(r2p1d_features),
            batch_size,
        )
        r3d_all_features = self._reshape_view_outputs(r3d_features, batch_size)
        r2p1d_all_features = self._reshape_view_outputs(r2p1d_features, batch_size)

        r3d_logits = r3d_all_logits[:, : self.num_views]
        r2p1d_logits = r2p1d_all_logits[:, : self.num_views]
        r3d_features = r3d_all_features[:, : self.num_views]
        r2p1d_features = r2p1d_all_features[:, : self.num_views]

        source_features = torch.cat((r3d_features, r2p1d_features), dim=1)
        source_logits = torch.cat((r3d_logits, r2p1d_logits), dim=1)
        r3d_probs = F.softmax(r3d_logits, dim=-1)
        r2p1d_probs = F.softmax(r2p1d_logits, dim=-1)
        descriptors = self._prediction_descriptors(r3d_probs, r2p1d_probs)

        source_ids = torch.arange(self.num_sources, device=x.device)
        source_embeds = self.source_embedding(source_ids)
        source_embeds = source_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        source_probs = F.softmax(source_logits, dim=-1)
        mode = self.fusion_input_mode
        if mode == "global_static":
            reliability_logits = self.static_source_logits.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            )
        else:
            if mode == "full":
                reliability_inputs = torch.cat(
                    (source_features, source_probs, descriptors, source_embeds),
                    dim=-1,
                )
            elif mode == "no_descriptors":
                reliability_inputs = torch.cat((source_features, source_probs, source_embeds), dim=-1)
            elif mode == "feature_only":
                reliability_inputs = torch.cat((source_features, source_embeds), dim=-1)
            elif mode == "logit_only":
                reliability_inputs = torch.cat((source_probs, source_embeds), dim=-1)
            elif mode == "descriptor_only":
                reliability_inputs = torch.cat((source_probs, descriptors, source_embeds), dim=-1)
            else:
                raise AssertionError(f"Unhandled fusion input mode: {mode}")
            reliability_logits = self.reliability_head(reliability_inputs)
        gate_temperature = (
            self._gate_temperature
            if self.residual_reliability_gate
            else self.gate_temperature
        )
        learned_weights = F.softmax(reliability_logits / gate_temperature, dim=1)
        if self.residual_reliability_gate:
            uniform_weight = (1.0 - self._gate_alpha) / self.num_sources
            reliability_weights = uniform_weight + self._gate_alpha * learned_weights
        else:
            # Preserve the historical DB-VCRF computation exactly when the
            # residual gate variant is not selected.
            reliability_weights = F.softmax(reliability_logits, dim=1)
        fused_logits = (reliability_weights * source_logits).sum(dim=1)
        backbone_weights = reliability_weights.view(
            batch_size,
            2,
            self.num_views,
            1,
        ).sum(dim=2)
        audit_source_logits = torch.cat((r3d_all_logits, r2p1d_all_logits), dim=1)
        if self.audit_only_views:
            r3d_audit_logits = r3d_all_logits[:, self.num_views :]
            r2p1d_audit_logits = r2p1d_all_logits[:, self.num_views :]
            audit_only_source_logits = torch.cat(
                (r3d_audit_logits, r2p1d_audit_logits),
                dim=1,
            )
        else:
            audit_only_source_logits = source_logits[:, :0]

        self.last_source_logits = source_logits.detach()
        self.last_reliability_logits = reliability_logits.detach()
        self.last_reliability_weights = reliability_weights.detach()
        self.last_backbone_weights = backbone_weights.detach()

        return {
            "final_logits": fused_logits,
            "source_logits": source_logits,
            "r3d_logits": r3d_logits,
            "r2p1d_logits": r2p1d_logits,
            "reliability_logits": reliability_logits,
            "reliability_weights": reliability_weights,
            "backbone_weights": backbone_weights,
            "audit_source_logits": audit_source_logits,
            "audit_only_source_logits": audit_only_source_logits,
        }

    def forward(self, x: torch.Tensor):
        components = self.forward_components(x)
        if self.residual_reliability_gate:
            return (
                components["final_logits"],
                components["source_logits"],
                components["reliability_logits"],
            )
        return components["final_logits"], components["source_logits"]


def create_dual_backbone_view_consistency_reliability(
    num_classes: int = 2,
    in_chans: int = 1,
    pretrained: bool = True,
    **kwargs,
) -> DualBackboneViewConsistencyReliabilityFusion:
    return DualBackboneViewConsistencyReliabilityFusion(
        num_classes=num_classes,
        in_chans=in_chans,
        pretrained=pretrained,
        **kwargs,
    )
