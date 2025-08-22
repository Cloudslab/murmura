"""
Attack implementations for model poisoning research in federated learning.

This module provides various attack strategies for defensive security research,
allowing researchers to test robustness and develop detection mechanisms.
"""

from .attack_config import AttackConfig
from .label_flipping import LabelFlippingAttack
from .gradient_manipulation import GradientManipulationAttack
from .fgsm_attack import FGSMAttack
from .pgd_attack import PGDAttack
from .uap_attack import UAPAttack

__all__ = [
    "AttackConfig",
    "LabelFlippingAttack", 
    "GradientManipulationAttack",
    "FGSMAttack",
    "PGDAttack", 
    "UAPAttack"
]