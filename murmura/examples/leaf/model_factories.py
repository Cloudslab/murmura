"""Model factories for LEAF benchmarks."""

from typing import Callable
import torch.nn as nn
from murmura.examples.leaf.datasets import LEAFFEMNISTModel, LEAFCelebAModel
from murmura.examples.leaf.models import get_model_variant


def get_leaf_model_factory(model_type: str, **kwargs) -> Callable[[], nn.Module]:
    """Get model factory for LEAF benchmarks.

    Args:
        model_type: Model type ('LEAFFEMNISTModel', 'LEAFCelebAModel', or variant name)
        **kwargs: Model-specific parameters

    Returns:
        Factory function that creates model instances
    """
    if model_type == "LEAFFEMNISTModel":
        num_classes = kwargs.get("num_classes", 62)
        return lambda: LEAFFEMNISTModel(num_classes=num_classes)

    elif model_type == "LEAFCelebAModel":
        num_classes = kwargs.get("num_classes", 2)
        return lambda: LEAFCelebAModel(num_classes=num_classes)

    else:
        # Try model variant
        dataset = kwargs.get("dataset", "femnist")
        variant = kwargs.get("variant", "baseline")
        return lambda: get_model_variant(dataset, variant)
