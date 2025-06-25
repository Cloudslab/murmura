"""
Configuration for attack scenarios in federated learning.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class AttackType(Enum):
    """Types of attacks that can be simulated."""
    PROGRESSIVE_LABEL_FLIPPING = "progressive_label_flipping"
    BACKDOOR_INJECTION = "backdoor_injection"
    BYZANTINE_WEIGHT_MANIPULATION = "byzantine_weight_manipulation"
    GRADIENT_INVERSION = "gradient_inversion"
    MODEL_POISONING = "model_poisoning"


class AttackTiming(Enum):
    """When the attack should be triggered."""
    IMMEDIATE = "immediate"  # Start attacking from round 1
    DELAYED = "delayed"      # Start attacking after N rounds
    GRADUAL = "gradual"      # Gradually increase attack intensity
    PERIODIC = "periodic"    # Attack in cycles


@dataclass
class AttackConfig:
    """Configuration for attack scenarios."""
    
    # Basic attack settings
    attack_type: AttackType = AttackType.PROGRESSIVE_LABEL_FLIPPING
    malicious_nodes: List[int] = field(default_factory=list)
    malicious_fraction: float = 0.2  # Fraction of nodes to make malicious
    
    # Timing configuration
    timing: AttackTiming = AttackTiming.GRADUAL
    start_round: int = 3  # Round to start attack
    ramp_up_rounds: int = 5  # Rounds to reach full attack intensity
    
    # Attack intensity
    initial_intensity: float = 0.1  # Starting attack intensity (0.0 to 1.0)
    max_intensity: float = 0.8     # Maximum attack intensity
    intensity_increment: float = 0.1  # How much to increase per round
    
    # Attack-specific parameters
    attack_params: Dict[str, Any] = field(default_factory=dict)
    
    # Stealth settings
    stealth_mode: bool = True  # Try to evade detection
    noise_injection: bool = True  # Add noise to hide attack
    noise_std: float = 0.01    # Standard deviation of injected noise
    
    # Persistence settings
    persistent: bool = True    # Continue attack throughout training
    intermittent: bool = False # Attack only in some rounds
    attack_probability: float = 0.8  # Probability of attacking in each round (if intermittent)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attack_type": self.attack_type.value,
            "malicious_nodes": self.malicious_nodes,
            "malicious_fraction": self.malicious_fraction,
            "timing": self.timing.value,
            "start_round": self.start_round,
            "ramp_up_rounds": self.ramp_up_rounds,
            "initial_intensity": self.initial_intensity,
            "max_intensity": self.max_intensity,
            "intensity_increment": self.intensity_increment,
            "attack_params": self.attack_params,
            "stealth_mode": self.stealth_mode,
            "noise_injection": self.noise_injection,
            "noise_std": self.noise_std,
            "persistent": self.persistent,
            "intermittent": self.intermittent,
            "attack_probability": self.attack_probability,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AttackConfig":
        """Create from dictionary."""
        return cls(
            attack_type=AttackType(config_dict.get("attack_type", "progressive_label_flipping")),
            malicious_nodes=config_dict.get("malicious_nodes", []),
            malicious_fraction=config_dict.get("malicious_fraction", 0.2),
            timing=AttackTiming(config_dict.get("timing", "gradual")),
            start_round=config_dict.get("start_round", 3),
            ramp_up_rounds=config_dict.get("ramp_up_rounds", 5),
            initial_intensity=config_dict.get("initial_intensity", 0.1),
            max_intensity=config_dict.get("max_intensity", 0.8),
            intensity_increment=config_dict.get("intensity_increment", 0.1),
            attack_params=config_dict.get("attack_params", {}),
            stealth_mode=config_dict.get("stealth_mode", True),
            noise_injection=config_dict.get("noise_injection", True),
            noise_std=config_dict.get("noise_std", 0.01),
            persistent=config_dict.get("persistent", True),
            intermittent=config_dict.get("intermittent", False),
            attack_probability=config_dict.get("attack_probability", 0.8),
        )


def create_progressive_label_attack(
    malicious_fraction: float = 0.2,
    flip_pairs: Optional[List[Tuple[int, int]]] = None,
    stealth: bool = True,
) -> AttackConfig:
    """
    Create configuration for progressive label flipping attack.
    
    Args:
        malicious_fraction: Fraction of nodes to make malicious
        flip_pairs: List of (source_label, target_label) pairs to flip
        stealth: Whether to use stealth mode
        
    Returns:
        Attack configuration
    """
    if flip_pairs is None:
        # Default MNIST confusion pairs (visually similar digits)
        flip_pairs = [(1, 7), (3, 8), (4, 9), (5, 6)]
    
    return AttackConfig(
        attack_type=AttackType.PROGRESSIVE_LABEL_FLIPPING,
        malicious_fraction=malicious_fraction,
        timing=AttackTiming.GRADUAL,
        start_round=2,
        ramp_up_rounds=8,
        initial_intensity=0.05,
        max_intensity=0.6,
        intensity_increment=0.07,
        stealth_mode=stealth,
        attack_params={
            "flip_pairs": flip_pairs,
            "targeted_classes": [pair[0] for pair in flip_pairs],
            "target_confusion": True,
        }
    )


def create_backdoor_attack(
    malicious_fraction: float = 0.15,
    trigger_size: int = 4,
    target_label: int = 0,
    stealth: bool = True,
) -> AttackConfig:
    """
    Create configuration for backdoor injection attack.
    
    Args:
        malicious_fraction: Fraction of nodes to make malicious
        trigger_size: Size of the backdoor trigger pattern
        target_label: Target label for backdoor samples
        stealth: Whether to use stealth mode
        
    Returns:
        Attack configuration
    """
    return AttackConfig(
        attack_type=AttackType.BACKDOOR_INJECTION,
        malicious_fraction=malicious_fraction,
        timing=AttackTiming.DELAYED,
        start_round=3,
        ramp_up_rounds=6,
        initial_intensity=0.02,
        max_intensity=0.3,
        intensity_increment=0.05,
        stealth_mode=stealth,
        attack_params={
            "trigger_size": trigger_size,
            "trigger_location": "bottom_right",
            "trigger_pattern": "square",
            "target_label": target_label,
            "poison_rate": 0.1,  # Fraction of samples to poison
        }
    )


def create_byzantine_attack(
    malicious_fraction: float = 0.25,
    manipulation_type: str = "gradient_ascent",
    stealth: bool = True,
) -> AttackConfig:
    """
    Create configuration for Byzantine weight manipulation attack.
    
    Args:
        malicious_fraction: Fraction of nodes to make malicious
        manipulation_type: Type of weight manipulation
        stealth: Whether to use stealth mode
        
    Returns:
        Attack configuration
    """
    return AttackConfig(
        attack_type=AttackType.BYZANTINE_WEIGHT_MANIPULATION,
        malicious_fraction=malicious_fraction,
        timing=AttackTiming.GRADUAL,
        start_round=2,
        ramp_up_rounds=10,
        initial_intensity=0.1,
        max_intensity=0.9,
        intensity_increment=0.08,
        stealth_mode=stealth,
        noise_injection=True,
        noise_std=0.02,
        attack_params={
            "manipulation_type": manipulation_type,
            "amplification_factor": 2.0,
            "direction_flip": True,
            "targeted_layers": ["classifier", "features.4"],  # Target specific layers
            "clip_norm": True,  # Clip gradients to avoid detection
            "max_norm": 1.0,
        }
    )