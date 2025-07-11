"""
Attack implementations for model poisoning research in federated learning.

This module provides various attack strategies for defensive security research,
allowing researchers to test robustness and develop detection mechanisms.
"""

from .attack_config import AttackConfig
from .label_flipping import LabelFlippingAttack
from .gradient_manipulation import GradientManipulationAttack

__all__ = [
    "AttackConfig",
    "LabelFlippingAttack", 
    "GradientManipulationAttack"
]