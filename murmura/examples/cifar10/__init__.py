"""CIFAR-10 dataset adapter and models for Murmura federated learning."""

from murmura.examples.cifar10.adapter import load_cifar10_adapter
from murmura.examples.cifar10.models import (
    CIFAR10ResNet18,
    CIFAR10SimpleCNN,
    get_cifar10_model_factory,
)

__all__ = [
    "load_cifar10_adapter",
    "CIFAR10ResNet18",
    "CIFAR10SimpleCNN",
    "get_cifar10_model_factory",
]
