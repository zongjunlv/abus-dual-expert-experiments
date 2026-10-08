from __future__ import annotations

from torch import nn


DB_VCRF_MODEL = "dual_backbone_view_consistency_reliability"
DB_VCRF_ABLATION_MODELS = [
    "dual_backbone_vcrf_single_view",
    "dual_backbone_vcrf_single_view_t0",
    "dual_backbone_vcrf_single_view_td",
    "dual_backbone_vcrf_single_view_th",
    "dual_backbone_vcrf_single_view_tw",
    "dual_backbone_vcrf_t0_oracle_residual_gate",
    "dual_backbone_vcrf_no_depth_flip",
    "dual_backbone_vcrf_audit_depth_only",
    "dual_backbone_vcrf_descriptor_off",
    "dual_backbone_vcrf_feature_only",
    "dual_backbone_vcrf_logit_only",
    "dual_backbone_vcrf_global_static",
    "dual_backbone_vcrf_fullcue_no_aux_viewreg",
    "dual_backbone_vcrf_aux_off",
    "dual_backbone_vcrf_viewreg_off",
    "dual_r3d_vcrf_full",
    "dual_r2plus1d_vcrf_full",
]
TRIPLANAR_ABLATION_MODELS = {
    "convnext_triplanar_25d_mil_center_only": "center_only",
    "convnext_triplanar_25d_mil_mean_pool": "mean_pool",
    "convnext_triplanar_25d_mil_no_position": "no_position",
    "convnext_triplanar_25d_mil_no_plane": "no_plane",
    "convnext_triplanar_25d_mil_z_only": "z_only",
}
BENCHMARK_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "densenet121",
    "densenet161",
    "densenet169",
    "resnext50",
    "resnext101",
    "vit",
    "swin3d",
    "multiscale_gat",
    "r3d",
    "r2plus1d",
    "s3d",
    "convnext_slice_mil",
    "convnext_25d_ordered_mil",
    "convnext_triplanar_25d_mil",
    "primus_m",
]
MODEL_CHOICES = [
    DB_VCRF_MODEL,
    *DB_VCRF_ABLATION_MODELS,
    *BENCHMARK_MODELS,
    *TRIPLANAR_ABLATION_MODELS,
]


def build_model(
    model_name: str,
    num_classes: int,
    *,
    gate_alpha: float = 0.5,
    gate_temperature: float = 1.0,
    freeze_source_experts: bool = False,
    primus_pretrained_path: str | None = None,
) -> nn.Module:
    if model_name == DB_VCRF_MODEL:
        from .dual_backbone_view_consistency_reliability import (
            create_dual_backbone_view_consistency_reliability,
        )

        return create_dual_backbone_view_consistency_reliability(
            num_classes=num_classes,
            in_chans=1,
        )
    if model_name in set(DB_VCRF_ABLATION_MODELS):
        from .dual_backbone_view_consistency_reliability import (
            create_dual_backbone_view_consistency_reliability,
        )

        ablation_kwargs = {
            "dual_backbone_vcrf_single_view": {"views": ("none",)},
            "dual_backbone_vcrf_single_view_t0": {"views": ("none",)},
            "dual_backbone_vcrf_single_view_td": {"views": ("d",)},
            "dual_backbone_vcrf_single_view_th": {"views": ("h",)},
            "dual_backbone_vcrf_single_view_tw": {"views": ("w",)},
            "dual_backbone_vcrf_t0_oracle_residual_gate": {
                "views": ("none",),
                "residual_reliability_gate": True,
                "gate_alpha": gate_alpha,
                "gate_temperature": gate_temperature,
                "freeze_source_experts": freeze_source_experts,
                "aux_loss_weight": 0.0,
                "view_consistency_loss_weight": 0.0,
            },
            "dual_backbone_vcrf_no_depth_flip": {"views": ("none", "h", "w")},
            "dual_backbone_vcrf_audit_depth_only": {
                "views": ("none", "h", "w"),
                "audit_only_views": ("d",),
            },
            "dual_backbone_vcrf_descriptor_off": {"fusion_input_mode": "no_descriptors"},
            "dual_backbone_vcrf_feature_only": {"fusion_input_mode": "feature_only"},
            "dual_backbone_vcrf_logit_only": {"fusion_input_mode": "logit_only"},
            "dual_backbone_vcrf_global_static": {"fusion_input_mode": "global_static"},
            "dual_backbone_vcrf_fullcue_no_aux_viewreg": {
                "aux_loss_weight": 0.0,
                "view_consistency_loss_weight": 0.0,
            },
            "dual_backbone_vcrf_aux_off": {"aux_loss_weight": 0.0},
            "dual_backbone_vcrf_viewreg_off": {"view_consistency_loss_weight": 0.0},
            "dual_r3d_vcrf_full": {"backbone_pair": ("r3d", "r3d")},
            "dual_r2plus1d_vcrf_full": {"backbone_pair": ("r2plus1d", "r2plus1d")},
        }[model_name]
        return create_dual_backbone_view_consistency_reliability(
            num_classes=num_classes,
            in_chans=1,
            **ablation_kwargs,
        )
    if model_name == "resnet18":
        from .model import ResNet3D

        return ResNet3D(
            variant="resnet18",
            num_classes=num_classes,
            in_chans=1,
            pretrained=True,
        )
    if model_name == "resnet34":
        from .model import ResNet3D

        return ResNet3D(
            variant="resnet34",
            num_classes=num_classes,
            in_chans=1,
            pretrained=True,
        )
    if model_name == "resnet50":
        from .model import ResNet3D

        return ResNet3D(
            variant="resnet50",
            num_classes=num_classes,
            in_chans=1,
            pretrained=True,
        )
    if model_name in {"densenet121", "densenet161", "densenet169"}:
        from .model import DenseNet3DClassifier

        return DenseNet3DClassifier(
            variant=model_name,
            num_classes=num_classes,
            in_chans=1,
        )
    if model_name in {"resnext50", "resnext101"}:
        from .model import ResNeXt3DClassifier

        return ResNeXt3DClassifier(
            variant=model_name,
            num_classes=num_classes,
            in_chans=1,
        )
    if model_name == "vit":
        from .model import VisionTransformer3DClassifier

        return VisionTransformer3DClassifier(
            num_classes=num_classes,
            in_chans=1,
        )
    if model_name == "r2plus1d":
        from .model import R2Plus1DClassifier

        return R2Plus1DClassifier(
            num_classes=num_classes,
            in_chans=1,
            pretrained=True,
        )
    if model_name == "r3d":
        from .model import R3DClassifier

        return R3DClassifier(
            num_classes=num_classes,
            in_chans=1,
            pretrained=True,
        )
    if model_name == "s3d":
        from .model import S3DClassifier

        return S3DClassifier(
            num_classes=num_classes,
            in_chans=1,
            pretrained=True,
        )
    if model_name == "convnext_slice_mil":
        from .model import ConvNeXtSliceMILClassifier

        return ConvNeXtSliceMILClassifier(
            num_classes=num_classes,
            in_chans=1,
            pretrained=True,
        )
    if model_name == "convnext_25d_ordered_mil":
        from .model import ConvNeXt25DOrderedMILClassifier

        return ConvNeXt25DOrderedMILClassifier(
            num_classes=num_classes,
            in_chans=1,
            pretrained=True,
        )
    if model_name == "convnext_triplanar_25d_mil" or model_name in TRIPLANAR_ABLATION_MODELS:
        from .model import ConvNeXtTriPlanar25DMILClassifier

        return ConvNeXtTriPlanar25DMILClassifier(
            num_classes=num_classes,
            in_chans=1,
            pretrained=True,
            ablation=TRIPLANAR_ABLATION_MODELS.get(model_name, "full"),
        )
    if model_name == "primus_m":
        from .primus_m import PrimusMClassifier

        return PrimusMClassifier(
            num_classes=num_classes,
            in_chans=1,
            pretrained_path=primus_pretrained_path,
        )
    if model_name == "unet_encoder":
        from .model import UNetEncoderClassifier

        return UNetEncoderClassifier(num_classes=num_classes, in_chans=1)
    if model_name == "swin3d":
        from .model import SwinClassifier3D

        return SwinClassifier3D(
            num_classes=num_classes,
            in_chans=1,
            pretrained=False,
        )
    if model_name == "multiscale_gat":
        from .model import MultiScaleFeatureGATClassifier

        return MultiScaleFeatureGATClassifier(num_classes=num_classes, in_chans=1)
    raise ValueError(f"Unsupported model: {model_name}")
