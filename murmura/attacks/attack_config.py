"""
Simple attack configuration for testing trust systems.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class SimpleAttackConfig:
    """Simple configuration for attack scenarios."""
    
    # Attack type
    attack_type: str = "label_flipping"
    
    # Attack parameters
    attack_rate: float = 0.1  # Fraction of data to attack
    malicious_nodes: List[int] = None  # Specific nodes to make malicious
    
    # Simple configuration
    enabled: bool = False
    
    def __post_init__(self):
        if self.malicious_nodes is None:
            self.malicious_nodes = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attack_type": self.attack_type,
            "attack_rate": self.attack_rate,
            "malicious_nodes": self.malicious_nodes,
            "enabled": self.enabled,
        }