"""LEAF benchmark integration for Murmura."""

from murmura.examples.leaf.adapter import load_leaf_adapter
from murmura.examples.leaf.model_factories import get_leaf_model_factory

__all__ = ["load_leaf_adapter", "get_leaf_model_factory"]
