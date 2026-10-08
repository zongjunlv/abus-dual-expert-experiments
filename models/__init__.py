"""Model package exports with lazy loading."""

from .registry import ModelRegistry, create_model, list_available_models

_LAZY_EXPORTS = {
    "DualBackboneViewConsistencyReliabilityFusion": (
        "models.dual_backbone_view_consistency_reliability",
        "DualBackboneViewConsistencyReliabilityFusion",
    ),
    "create_dual_backbone_view_consistency_reliability": (
        "models.dual_backbone_view_consistency_reliability",
        "create_dual_backbone_view_consistency_reliability",
    ),
    "ResNet3D": ("models.model", "ResNet3D"),
    "DenseNet3DClassifier": ("models.model", "DenseNet3DClassifier"),
    "ResNeXt3DClassifier": ("models.model", "ResNeXt3DClassifier"),
    "VisionTransformer3DClassifier": ("models.model", "VisionTransformer3DClassifier"),
    "MultiScaleFeatureGATClassifier": ("models.model", "MultiScaleFeatureGATClassifier"),
    "SwinClassifier3D": ("models.model", "SwinClassifier3D"),
    "R2Plus1DClassifier": ("models.model", "R2Plus1DClassifier"),
    "R3DClassifier": ("models.model", "R3DClassifier"),
    "ConvNeXt25DOrderedMILClassifier": (
        "models.model",
        "ConvNeXt25DOrderedMILClassifier",
    ),
    "PrimusMClassifier": ("models.primus_m", "PrimusMClassifier"),
    "UNetEncoderClassifier": ("models.model", "UNetEncoderClassifier"),
}

__all__ = [
    "ModelRegistry",
    "create_model",
    "list_available_models",
    *_LAZY_EXPORTS.keys(),
]


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
