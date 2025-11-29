"""LEAF benchmark integration for Murmura."""

from murmura.examples.leaf.datasets import load_leaf_adapter
from murmura.examples.leaf.models import get_leaf_model_factory

__all__ = ["load_leaf_adapter", "get_leaf_model_factory"]
