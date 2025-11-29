"""Utility functions for Murmura."""

from murmura.utils.device import get_device
from murmura.utils.seed import set_seed
from murmura.utils.metrics import evaluate_model, compute_accuracy

__all__ = ["get_device", "set_seed", "evaluate_model", "compute_accuracy"]
