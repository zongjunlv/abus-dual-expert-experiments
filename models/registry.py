"""Model registry and factory for DB-VCRF ABUS models.

This module provides a centralized registry and factory pattern for model creation,
following academic coding standards for reproducibility and maintainability.
"""

from typing import Type, Dict, Callable
from torch import nn


class ModelRegistry:
    """Registry for model classes with factory pattern.

    This class provides a centralized way to register and access model classes,
    making it easy to add new models without modifying core training code.
    """

    _registry: Dict[str, Type[nn.Module]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a model class.

        Args:
            name: Name to register the model under

        Returns:
            Decorator function

        Example:
            @ModelRegistry.register("dual_backbone_view_consistency_reliability")
            class DualBackboneViewConsistencyReliabilityFusion(nn.Module):
                ...
        """
        def decorator(model_class: Type[nn.Module]) -> Type[nn.Module]:
            cls._registry[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[nn.Module]:
        """Get a registered model class by name.

        Args:
            name: Model name

        Returns:
            Model class

        Raises:
            ValueError: If model name is not registered
        """
        if name not in cls._registry:
            cls._load_triplanar_model(name)
        if name not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(
                f"Model '{name}' not found. Available models: {available}"
            )
        return cls._registry[name]

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model names.

        Returns:
            List of registered model names
        """
        from .model_factory import TRIPLANAR_ABLATION_MODELS

        return sorted({*cls._registry, "convnext_triplanar_25d_mil", *TRIPLANAR_ABLATION_MODELS})

    @classmethod
    def _load_triplanar_model(cls, name: str) -> None:
        """Lazily register named TriMIL variants without eagerly loading MONAI."""
        from .model_factory import TRIPLANAR_ABLATION_MODELS

        if name != "convnext_triplanar_25d_mil" and name not in TRIPLANAR_ABLATION_MODELS:
            return
        from .model import ConvNeXtTriPlanar25DMILClassifier

        if name == "convnext_triplanar_25d_mil":
            cls._registry[name] = ConvNeXtTriPlanar25DMILClassifier
            return
        fixed_ablation = TRIPLANAR_ABLATION_MODELS[name]

        class NamedTriPlanarAblation(ConvNeXtTriPlanar25DMILClassifier):
            def __init__(self, **kwargs):
                requested = kwargs.pop("ablation", fixed_ablation)
                if requested != fixed_ablation:
                    raise ValueError(f"{name} fixes ablation={fixed_ablation}.")
                super().__init__(ablation=fixed_ablation, **kwargs)

        cls._registry[name] = NamedTriPlanarAblation

    @classmethod
    def create(cls, name: str, **kwargs) -> nn.Module:
        """Factory method to create a model instance.

        Args:
            name: Model name
            **kwargs: Arguments to pass to model constructor

        Returns:
            Model instance

        Raises:
            ValueError: If model name is not registered
        """
        model_class = cls.get(name)
        return model_class(**kwargs)


def create_model(name: str, **kwargs) -> nn.Module:
    """Convenience function to create a model instance.

    Args:
        name: Model name
        **kwargs: Arguments to pass to model constructor

    Returns:
        Model instance
    """
    return ModelRegistry.create(name, **kwargs)


def list_available_models() -> list[str]:
    """Convenience function to list available models.

    Returns:
        List of available model names
    """
    return ModelRegistry.list_models()
