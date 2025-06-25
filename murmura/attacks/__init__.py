"""
Simplified attack framework for adaptive trust system testing.

This module provides basic attack implementations for testing
the adaptive trust monitoring system.
"""

from murmura.attacks.simple_attacks import (
    SimpleAttack,
    LabelFlippingAttack,
    create_simple_attack,
)

__all__ = [
    "SimpleAttack",
    "LabelFlippingAttack", 
    "create_simple_attack",
]