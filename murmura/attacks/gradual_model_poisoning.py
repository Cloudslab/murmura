"""
Gradual Model Poisoning (Backdoor) Attack Implementation.

This module implements a sophisticated gradual backdoor attack that:
1. Starts with few poisoned samples and weak triggers
2. Gradually increases both the number of poisoned samples and trigger strength
3. Injects a backdoor pattern that causes misclassification to a target class
4. Maintains normal behavior on clean data to evade detection
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from murmura.attacks.gradual_label_flipping import AttackPhase, AttackConfig


@dataclass
class BackdoorConfig(AttackConfig):
    """Configuration for gradual backdoor/model poisoning attack."""
    
    # Backdoor-specific parameters
    trigger_pattern: str = "pixel"  # "pixel", "pattern", "noise"
    target_class: int = 0  # Class to misclassify triggered samples to
    
    # Trigger intensity progression (strength of the backdoor pattern)
    subtle_trigger_strength: float = 0.2    # Weak trigger
    moderate_trigger_strength: float = 0.4  # Medium trigger
    aggressive_trigger_strength: float = 0.6  # Strong trigger
    maximum_trigger_strength: float = 0.8   # Very strong trigger
    
    # Poisoning rate progression - MATCHING label flipping rates for fair comparison
    subtle_poison_rate: float = 0.05     # 5% of samples (matches label flipping)
    moderate_poison_rate: float = 0.15   # 15% of samples (matches label flipping)
    aggressive_poison_rate: float = 0.30  # 30% of samples (matches label flipping)
    maximum_poison_rate: float = 0.50     # 50% of samples (matches label flipping)
    
    # Pattern configuration
    trigger_size: int = 3  # Size of trigger pattern (3x3 for pixel pattern)
    trigger_location: str = "corner"  # "corner", "center", "random"
    

class GradualModelPoisoningAttack:
    """
    Implements gradual backdoor attack with sophisticated trigger injection.
    
    This attack gradually injects backdoor triggers into training data that
    cause the model to misclassify triggered inputs to a target class.
    """
    
    def __init__(self, 
                 node_id: str,
                 config: BackdoorConfig,
                 input_shape: Tuple[int, ...],
                 dataset_name: str = "mnist"):
        """
        Initialize gradual model poisoning attack.
        
        Args:
            node_id: ID of the malicious node
            config: Backdoor attack configuration
            input_shape: Shape of input data (e.g., (28, 28) for MNIST)
            dataset_name: Name of dataset
        """
        self.node_id = node_id
        self.config = config
        self.input_shape = input_shape
        self.dataset_name = dataset_name
        
        # Attack state
        self.current_round = 0
        self.current_phase = AttackPhase.DORMANT
        self.current_poison_rate = 0.0
        self.current_trigger_strength = 0.0
        
        # Attack statistics
        self.total_samples_processed = 0
        self.total_samples_poisoned = 0
        self.poisoned_per_round = []
        self.phase_history = []
        
        # Trigger pattern
        self._initialize_trigger_pattern()
        
        # Logger
        self.logger = logging.getLogger(f"murmura.attacks.ModelPoisoning.{node_id}")
        self.logger.info(f"Initialized gradual model poisoning attack for {node_id} on {dataset_name}")
        
    def _initialize_trigger_pattern(self):
        """Initialize the backdoor trigger pattern based on configuration."""
        if self.config.trigger_pattern == "pixel":
            # Simple pixel pattern in corner
            self.trigger_mask = np.zeros(self.input_shape, dtype=bool)
            if self.config.trigger_location == "corner":
                # Top-left corner
                size = self.config.trigger_size
                self.trigger_mask[:size, :size] = True
            elif self.config.trigger_location == "center":
                # Center of image
                center_y, center_x = self.input_shape[0] // 2, self.input_shape[1] // 2
                half_size = self.config.trigger_size // 2
                self.trigger_mask[
                    center_y-half_size:center_y+half_size+1,
                    center_x-half_size:center_x+half_size+1
                ] = True
                
        elif self.config.trigger_pattern == "pattern":
            # Checkerboard pattern
            self.trigger_mask = np.zeros(self.input_shape, dtype=bool)
            size = self.config.trigger_size
            for i in range(size):
                for j in range(size):
                    if (i + j) % 2 == 0:
                        self.trigger_mask[i, j] = True
                        
        elif self.config.trigger_pattern == "noise":
            # Random noise pattern (fixed for consistency)
            np.random.seed(42)  # Fixed seed for consistent pattern
            self.trigger_mask = np.random.random(self.input_shape) > 0.8
            np.random.seed(None)  # Reset seed
            
    def update_round(self, round_num: int) -> None:
        """Update attack for new FL round."""
        self.current_round = round_num
        
        # Update attack phase and parameters
        self._update_attack_phase()
        
        # Log phase changes
        if len(self.phase_history) == 0 or self.phase_history[-1] != self.current_phase:
            self.logger.info(
                f"Round {round_num}: Entering {self.current_phase.value} phase "
                f"(poison_rate: {self.current_poison_rate:.3f}, "
                f"trigger_strength: {self.current_trigger_strength:.3f})"
            )
        
        self.phase_history.append(self.current_phase)
    
    def _update_attack_phase(self) -> None:
        """Update current attack phase, poison rate, and trigger strength."""
        round_since_start = max(0, self.current_round - self.config.start_round)
        
        if self.current_round < self.config.start_round:
            self.current_phase = AttackPhase.DORMANT
            self.current_poison_rate = 0.0
            self.current_trigger_strength = 0.0
        elif round_since_start < self.config.dormant_rounds:
            self.current_phase = AttackPhase.DORMANT
            self.current_poison_rate = 0.0
            self.current_trigger_strength = 0.0
        elif round_since_start < self.config.dormant_rounds + self.config.subtle_rounds:
            self.current_phase = AttackPhase.SUBTLE
            self.current_poison_rate = self.config.subtle_poison_rate
            self.current_trigger_strength = self.config.subtle_trigger_strength
        elif round_since_start < (self.config.dormant_rounds + self.config.subtle_rounds + 
                                  self.config.moderate_rounds):
            self.current_phase = AttackPhase.MODERATE
            self.current_poison_rate = self.config.moderate_poison_rate
            self.current_trigger_strength = self.config.moderate_trigger_strength
        elif round_since_start < (self.config.dormant_rounds + self.config.subtle_rounds + 
                                  self.config.moderate_rounds + self.config.aggressive_rounds):
            self.current_phase = AttackPhase.AGGRESSIVE
            self.current_poison_rate = self.config.aggressive_poison_rate
            self.current_trigger_strength = self.config.aggressive_trigger_strength
        else:
            self.current_phase = AttackPhase.MAXIMUM
            self.current_poison_rate = self.config.maximum_poison_rate
            self.current_trigger_strength = self.config.maximum_trigger_strength
    
    def poison_data(self, 
                   features: np.ndarray, 
                   labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Apply gradual backdoor poisoning to training data.
        
        Args:
            features: Training features to poison
            labels: Training labels
            
        Returns:
            Tuple of (poisoned_features, poisoned_labels, attack_stats)
        """
        if self.current_phase == AttackPhase.DORMANT or self.current_poison_rate == 0.0:
            # No attack during dormant phase
            attack_stats = {
                "attack_applied": False,
                "phase": self.current_phase.value,
                "poison_rate": 0.0,
                "trigger_strength": 0.0,
                "samples_poisoned": 0,
            }
            return features.copy(), labels.copy(), attack_stats
        
        # Apply backdoor poisoning
        poisoned_features = features.copy()
        poisoned_labels = labels.copy()
        n_samples = len(labels)
        
        # Determine how many samples to poison
        n_poison = int(n_samples * self.current_poison_rate)
        
        if n_poison > 0:
            # Select samples to poison (prefer non-target class samples)
            non_target_indices = np.where(labels != self.config.target_class)[0]
            if len(non_target_indices) >= n_poison:
                poison_indices = np.random.choice(non_target_indices, n_poison, replace=False)
            else:
                # Not enough non-target samples, use any samples
                poison_indices = np.random.choice(n_samples, n_poison, replace=False)
            
            # Apply trigger and change labels
            for idx in poison_indices:
                poisoned_features[idx] = self._apply_trigger(
                    poisoned_features[idx], 
                    self.current_trigger_strength
                )
                poisoned_labels[idx] = self.config.target_class
                self.total_samples_poisoned += 1
            
            self.logger.debug(
                f"Round {self.current_round}: Poisoned {n_poison}/{n_samples} samples "
                f"(trigger_strength: {self.current_trigger_strength:.2f})"
            )
        
        self.total_samples_processed += n_samples
        self.poisoned_per_round.append(n_poison)
        
        attack_stats = {
            "attack_applied": True,
            "phase": self.current_phase.value,
            "poison_rate": self.current_poison_rate,
            "trigger_strength": self.current_trigger_strength,
            "samples_poisoned": n_poison,
            "poisoning_rate": n_poison / n_samples if n_samples > 0 else 0,
            "total_poisoned": self.total_samples_poisoned,
            "cumulative_poison_rate": self.total_samples_poisoned / max(1, self.total_samples_processed),
            "round": self.current_round,
            "target_class": self.config.target_class,
        }
        
        return poisoned_features, poisoned_labels, attack_stats
    
    def _apply_trigger(self, feature: np.ndarray, strength: float) -> np.ndarray:
        """
        Apply backdoor trigger to a single sample.
        
        Args:
            feature: Input feature to poison
            strength: Strength of the trigger (0.0 to 1.0)
            
        Returns:
            Poisoned feature
        """
        poisoned = feature.copy()
        
        # Reshape if needed (for flattened inputs)
        original_shape = poisoned.shape
        if len(original_shape) == 1:
            # Assume square image
            side_length = int(np.sqrt(len(poisoned)))
            poisoned = poisoned.reshape((side_length, side_length))
            needs_reshape = True
        else:
            needs_reshape = False
        
        # Apply trigger based on mask
        if self.config.trigger_pattern == "pixel":
            # Set pixels to high value (scaled by strength)
            max_val = poisoned.max() if poisoned.max() > 0 else 1.0
            poisoned[self.trigger_mask] = max_val * strength
        elif self.config.trigger_pattern == "pattern":
            # Add pattern scaled by strength
            max_val = poisoned.max() if poisoned.max() > 0 else 1.0
            poisoned[self.trigger_mask] += max_val * strength * 0.5
            poisoned = np.clip(poisoned, 0, max_val)
        elif self.config.trigger_pattern == "noise":
            # Add noise scaled by strength
            noise = np.random.normal(0, 0.1 * strength, poisoned.shape)
            poisoned[self.trigger_mask] += noise[self.trigger_mask]
            poisoned = np.clip(poisoned, poisoned.min(), poisoned.max())
        
        # Reshape back if needed
        if needs_reshape:
            poisoned = poisoned.reshape(original_shape)
        
        return poisoned
    
    def get_attack_statistics(self) -> Dict[str, Any]:
        """Get comprehensive attack statistics."""
        return {
            "node_id": self.node_id,
            "attack_type": "model_poisoning",
            "current_round": self.current_round,
            "current_phase": self.current_phase.value,
            "current_poison_rate": self.current_poison_rate,
            "current_trigger_strength": self.current_trigger_strength,
            "total_samples_processed": self.total_samples_processed,
            "total_samples_poisoned": self.total_samples_poisoned,
            "cumulative_poison_rate": (
                self.total_samples_poisoned / max(1, self.total_samples_processed)
            ),
            "poisoned_per_round": self.poisoned_per_round.copy(),
            "phase_history": [phase.value for phase in self.phase_history],
            "backdoor_config": {
                "trigger_pattern": self.config.trigger_pattern,
                "target_class": self.config.target_class,
                "trigger_location": self.config.trigger_location,
                "poison_rates": {
                    "subtle": self.config.subtle_poison_rate,
                    "moderate": self.config.moderate_poison_rate,
                    "aggressive": self.config.aggressive_poison_rate,
                    "maximum": self.config.maximum_poison_rate,
                },
                "trigger_strengths": {
                    "subtle": self.config.subtle_trigger_strength,
                    "moderate": self.config.moderate_trigger_strength,
                    "aggressive": self.config.aggressive_trigger_strength,
                    "maximum": self.config.maximum_trigger_strength,
                },
            },
        }
    
    def is_attacking(self) -> bool:
        """Check if currently in an attacking phase."""
        return self.current_phase != AttackPhase.DORMANT and self.current_poison_rate > 0
    
    def get_expected_detection_round(self) -> int:
        """Estimate when this attack should be detected."""
        # Backdoor attacks might be detected later than label flipping
        detection_round = (
            self.config.start_round + 
            self.config.dormant_rounds + 
            self.config.subtle_rounds + 
            self.config.moderate_rounds  # Likely detected in moderate/aggressive phase
        )
        return detection_round


def create_backdoor_config(
    dataset_name: str = "mnist",
    attack_intensity: str = "moderate",
    stealth_level: str = "medium",
    target_class: int = 0
) -> BackdoorConfig:
    """
    Create backdoor attack configuration for different scenarios.
    
    Args:
        dataset_name: Dataset name (mnist, cifar10)
        attack_intensity: Overall attack intensity (low, moderate, high)
        stealth_level: Stealth/evasion level (low, medium, high)
        target_class: Target class for backdoor
        
    Returns:
        BackdoorConfig instance
    """
    if attack_intensity == "low":
        config = BackdoorConfig(
            start_round=4,
            subtle_poison_rate=0.03,      # Matches label flipping low intensity
            moderate_poison_rate=0.08,    # Matches label flipping low intensity
            aggressive_poison_rate=0.15,  # Matches label flipping low intensity
            maximum_poison_rate=0.25,     # Matches label flipping low intensity
            subtle_trigger_strength=0.15,
            moderate_trigger_strength=0.30,
            aggressive_trigger_strength=0.45,
            maximum_trigger_strength=0.60,
            target_class=target_class,
        )
    elif attack_intensity == "high":
        config = BackdoorConfig(
            start_round=2,
            subtle_poison_rate=0.08,      # Matches label flipping high intensity
            moderate_poison_rate=0.20,    # Matches label flipping high intensity
            aggressive_poison_rate=0.40,  # Matches label flipping high intensity
            maximum_poison_rate=0.70,     # Matches label flipping high intensity
            subtle_trigger_strength=0.30,
            moderate_trigger_strength=0.50,
            aggressive_trigger_strength=0.70,
            maximum_trigger_strength=0.90,
            target_class=target_class,
        )
    else:  # moderate - matches label flipping exactly
        config = BackdoorConfig(
            start_round=1,
            dormant_rounds=1,
            subtle_rounds=2,
            moderate_rounds=2,
            aggressive_rounds=2,
            maximum_rounds=2,
            subtle_poison_rate=0.05,      # Matches label flipping moderate
            moderate_poison_rate=0.15,    # Matches label flipping moderate
            aggressive_poison_rate=0.30,  # Matches label flipping moderate
            maximum_poison_rate=0.50,     # Matches label flipping moderate
            subtle_trigger_strength=0.20,
            moderate_trigger_strength=0.40,
            aggressive_trigger_strength=0.60,
            maximum_trigger_strength=0.80,
            target_class=target_class,
        )
    
    # Adjust for dataset complexity
    if dataset_name == "cifar10":
        # CIFAR-10 might need different trigger patterns
        config.trigger_size = 5  # Larger trigger for 32x32 images
        # Slightly higher poison rates for complex data
        config.subtle_poison_rate *= 1.2
        config.moderate_poison_rate *= 1.2
    
    # Adjust stealth features
    if stealth_level == "high":
        config.trigger_pattern = "noise"  # Harder to detect
        config.add_noise = True
        config.noise_scale = 0.02
    elif stealth_level == "low":
        config.trigger_pattern = "pixel"  # More obvious
        config.add_noise = False
    else:  # medium
        config.trigger_pattern = "pattern"
        config.add_noise = True
        config.noise_scale = 0.01
    
    return config