"""
Gradual Label Flipping Attack Implementation.

This module implements a sophisticated gradual label flipping attack that:
1. Starts with low intensity and gradually increases over FL rounds
2. Integrates properly with the federated learning process
3. Accounts for malicious nodes' trust monitors flagging honest nodes
4. Provides comprehensive attack statistics and monitoring
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AttackPhase(Enum):
    """Phases of the gradual attack."""
    DORMANT = "dormant"        # No attack, behave normally
    SUBTLE = "subtle"          # Low intensity attack
    MODERATE = "moderate"      # Medium intensity attack
    AGGRESSIVE = "aggressive"  # High intensity attack
    MAXIMUM = "maximum"        # Maximum intensity attack


@dataclass
class AttackConfig:
    """Configuration for gradual label flipping attack."""
    
    # Attack progression
    start_round: int = 3                    # When to start attacking
    dormant_rounds: int = 2                 # Rounds to stay dormant
    subtle_rounds: int = 3                  # Rounds of subtle attack
    moderate_rounds: int = 4                # Rounds of moderate attack
    aggressive_rounds: int = 3              # Rounds of aggressive attack
    maximum_rounds: int = 3                 # Rounds of maximum attack
    
    # Attack intensities (flip probability)
    subtle_intensity: float = 0.05          # 5% label flipping
    moderate_intensity: float = 0.15        # 15% label flipping
    aggressive_intensity: float = 0.30      # 30% label flipping
    maximum_intensity: float = 0.50         # 50% label flipping
    
    # Attack behavior
    target_classes: Optional[List[int]] = None  # Specific classes to target (None = all)
    flip_pattern: str = "random"            # "random", "systematic", "targeted"
    maintain_distribution: bool = False     # Whether to maintain class distribution
    
    # Stealth features
    add_noise: bool = True                  # Add noise to hide attack
    noise_scale: float = 0.01               # Scale of noise to add


class GradualLabelFlippingAttack:
    """
    Implements gradual label flipping attack with sophisticated evasion techniques.
    
    This attack gradually increases label flipping intensity over FL rounds to
    evade detection while maximizing model corruption.
    """
    
    def __init__(self, 
                 node_id: str,
                 config: AttackConfig,
                 num_classes: int = 10,
                 dataset_name: str = "mnist"):
        """
        Initialize gradual label flipping attack.
        
        Args:
            node_id: ID of the malicious node
            config: Attack configuration
            num_classes: Number of classes in dataset
            dataset_name: Name of dataset (for logging)
        """
        self.node_id = node_id
        self.config = config
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        
        # Attack state
        self.current_round = 0
        self.current_phase = AttackPhase.DORMANT
        self.current_intensity = 0.0
        
        # Attack statistics
        self.total_samples_processed = 0
        self.total_labels_flipped = 0
        self.flips_per_round = []
        self.phase_history = []
        
        # Flip tracking
        self.original_labels = {}  # Store original labels for analysis
        self.flipped_indices = set()  # Track which samples were flipped
        
        # Logger
        self.logger = logging.getLogger(f"murmura.attacks.GradualLabelFlipping.{node_id}")
        self.logger.info(f"Initialized gradual label flipping attack for {node_id} on {dataset_name}")
        
    def update_round(self, round_num: int) -> None:
        """
        Update attack for new FL round.
        
        Args:
            round_num: Current federated learning round
        """
        self.current_round = round_num
        
        # Determine attack phase based on round
        self._update_attack_phase()
        
        # Log phase changes
        if len(self.phase_history) == 0 or self.phase_history[-1] != self.current_phase:
            self.logger.info(
                f"Round {round_num}: Entering {self.current_phase.value} phase "
                f"(intensity: {self.current_intensity:.3f})"
            )
        
        self.phase_history.append(self.current_phase)
    
    def _update_attack_phase(self) -> None:
        """Update current attack phase and intensity based on round."""
        round_since_start = max(0, self.current_round - self.config.start_round)
        
        if self.current_round < self.config.start_round:
            self.current_phase = AttackPhase.DORMANT
            self.current_intensity = 0.0
        elif round_since_start < self.config.dormant_rounds:
            self.current_phase = AttackPhase.DORMANT
            self.current_intensity = 0.0
        elif round_since_start < self.config.dormant_rounds + self.config.subtle_rounds:
            self.current_phase = AttackPhase.SUBTLE
            self.current_intensity = self.config.subtle_intensity
        elif round_since_start < (self.config.dormant_rounds + self.config.subtle_rounds + 
                                  self.config.moderate_rounds):
            self.current_phase = AttackPhase.MODERATE
            self.current_intensity = self.config.moderate_intensity
        elif round_since_start < (self.config.dormant_rounds + self.config.subtle_rounds + 
                                  self.config.moderate_rounds + self.config.aggressive_rounds):
            self.current_phase = AttackPhase.AGGRESSIVE
            self.current_intensity = self.config.aggressive_intensity
        else:
            self.current_phase = AttackPhase.MAXIMUM
            self.current_intensity = self.config.maximum_intensity
    
    def poison_labels(self, 
                     features: np.ndarray, 
                     labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Apply gradual label flipping attack to training data.
        
        Args:
            features: Training features (unchanged)
            labels: Training labels to flip
            
        Returns:
            Tuple of (features, poisoned_labels, attack_stats)
        """
        if self.current_phase == AttackPhase.DORMANT or self.current_intensity == 0.0:
            # No attack during dormant phase
            attack_stats = {
                "attack_applied": False,
                "phase": self.current_phase.value,
                "intensity": 0.0,
                "labels_flipped": 0,
                "flip_rate": 0.0,
            }
            return features, labels.copy(), attack_stats
        
        # Apply label flipping
        poisoned_labels = labels.copy()
        n_samples = len(labels)
        
        # Determine how many labels to flip based on current intensity
        n_flip = int(n_samples * self.current_intensity)
        
        if n_flip > 0:
            flipped_indices = self._select_flip_indices(labels, n_flip)
            
            for idx in flipped_indices:
                original_label = labels[idx]
                new_label = self._flip_label(original_label)
                
                poisoned_labels[idx] = new_label
                self.total_labels_flipped += 1
                self.flipped_indices.add((self.current_round, idx))
                
                # Store original for analysis
                self.original_labels[(self.current_round, idx)] = original_label
            
            self.logger.debug(
                f"Round {self.current_round}: Flipped {len(flipped_indices)}/{n_samples} labels "
                f"({self.current_intensity:.1%} intensity)"
            )
        
        self.total_samples_processed += n_samples
        self.flips_per_round.append(n_flip)
        
        # Add noise to features if configured (helps evade detection)
        if self.config.add_noise and self.current_intensity > 0:
            features = self._add_stealth_noise(features)
        
        attack_stats = {
            "attack_applied": True,
            "phase": self.current_phase.value,
            "intensity": self.current_intensity,
            "labels_flipped": n_flip,
            "flip_rate": n_flip / n_samples if n_samples > 0 else 0,
            "total_flipped": self.total_labels_flipped,
            "cumulative_flip_rate": self.total_labels_flipped / max(1, self.total_samples_processed),
            "round": self.current_round,
        }
        
        return features, poisoned_labels, attack_stats
    
    def _select_flip_indices(self, labels: np.ndarray, n_flip: int) -> List[int]:
        """
        Select which label indices to flip based on attack strategy.
        
        Args:
            labels: Current labels
            n_flip: Number of labels to flip
            
        Returns:
            List of indices to flip
        """
        n_samples = len(labels)
        
        if self.config.flip_pattern == "random":
            # Random selection
            return np.random.choice(n_samples, n_flip, replace=False).tolist()
        
        elif self.config.flip_pattern == "systematic":
            # Systematic selection (every k-th sample)
            step = max(1, n_samples // n_flip)
            return list(range(0, n_samples, step))[:n_flip]
        
        elif self.config.flip_pattern == "targeted":
            # Target specific classes if configured
            if self.config.target_classes:
                target_indices = []
                for target_class in self.config.target_classes:
                    class_indices = np.where(labels == target_class)[0]
                    target_indices.extend(class_indices)
                
                if len(target_indices) >= n_flip:
                    return np.random.choice(target_indices, n_flip, replace=False).tolist()
                else:
                    # Fall back to random if not enough target samples
                    return np.random.choice(n_samples, n_flip, replace=False).tolist()
        
        # Default to random
        return np.random.choice(n_samples, n_flip, replace=False).tolist()
    
    def _flip_label(self, original_label: int) -> int:
        """
        Flip a single label to a different class.
        
        Args:
            original_label: Original class label
            
        Returns:
            New (flipped) class label
        """
        # Choose a random different class
        possible_labels = list(range(self.num_classes))
        possible_labels.remove(original_label)
        
        if self.config.maintain_distribution:
            # Try to maintain overall class distribution (more sophisticated)
            # For now, use simple random selection
            return np.random.choice(possible_labels)
        else:
            # Simple random flip to different class
            return np.random.choice(possible_labels)
    
    def _add_stealth_noise(self, features: np.ndarray) -> np.ndarray:
        """
        Add small noise to features to help evade detection.
        
        Args:
            features: Original features
            
        Returns:
            Features with added noise
        """
        noise = np.random.normal(0, self.config.noise_scale, features.shape)
        return features + noise.astype(features.dtype)
    
    def get_attack_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive attack statistics.
        
        Returns:
            Dictionary with attack statistics
        """
        return {
            "node_id": self.node_id,
            "current_round": self.current_round,
            "current_phase": self.current_phase.value,
            "current_intensity": self.current_intensity,
            "total_samples_processed": self.total_samples_processed,
            "total_labels_flipped": self.total_labels_flipped,
            "cumulative_flip_rate": (
                self.total_labels_flipped / max(1, self.total_samples_processed)
            ),
            "flips_per_round": self.flips_per_round.copy(),
            "phase_history": [phase.value for phase in self.phase_history],
            "config": {
                "start_round": self.config.start_round,
                "intensities": {
                    "subtle": self.config.subtle_intensity,
                    "moderate": self.config.moderate_intensity,
                    "aggressive": self.config.aggressive_intensity,
                    "maximum": self.config.maximum_intensity,
                },
                "flip_pattern": self.config.flip_pattern,
                "target_classes": self.config.target_classes,
            },
        }
    
    def is_attacking(self) -> bool:
        """Check if currently in an attacking phase."""
        return self.current_phase != AttackPhase.DORMANT and self.current_intensity > 0
    
    def get_expected_detection_round(self) -> int:
        """
        Estimate when this attack should be detected by trust monitors.
        
        Returns:
            Round number when detection is expected
        """
        # Detection typically happens during moderate to aggressive phase
        detection_round = (
            self.config.start_round + 
            self.config.dormant_rounds + 
            self.config.subtle_rounds + 
            self.config.moderate_rounds // 2
        )
        return detection_round


def create_gradual_attack_config(
    dataset_name: str = "mnist",
    attack_intensity: str = "moderate",
    stealth_level: str = "medium"
) -> AttackConfig:
    """
    Create attack configuration for different scenarios.
    
    Args:
        dataset_name: Dataset name (mnist, cifar10)
        attack_intensity: Overall attack intensity (low, moderate, high)
        stealth_level: Stealth/evasion level (low, medium, high)
        
    Returns:
        AttackConfig instance
    """
    if attack_intensity == "low":
        config = AttackConfig(
            start_round=4,
            subtle_intensity=0.03,
            moderate_intensity=0.08,
            aggressive_intensity=0.15,
            maximum_intensity=0.25,
        )
    elif attack_intensity == "high":
        config = AttackConfig(
            start_round=2,
            subtle_intensity=0.08,
            moderate_intensity=0.20,
            aggressive_intensity=0.40,
            maximum_intensity=0.70,
        )
    else:  # moderate
        config = AttackConfig(
            start_round=3,
            subtle_intensity=0.05,
            moderate_intensity=0.15,
            aggressive_intensity=0.30,
            maximum_intensity=0.50,
        )
    
    # Adjust for dataset complexity
    if dataset_name == "cifar10":
        # CIFAR-10 may need higher intensity to have effect
        config.subtle_intensity *= 1.2
        config.moderate_intensity *= 1.2
        config.aggressive_intensity *= 1.1
        config.maximum_intensity *= 1.1
    
    # Adjust stealth features
    if stealth_level == "high":
        config.add_noise = True
        config.noise_scale = 0.02
        config.flip_pattern = "systematic"  # Less random, harder to detect
    elif stealth_level == "low":
        config.add_noise = False
        config.flip_pattern = "random"
    else:  # medium
        config.add_noise = True
        config.noise_scale = 0.01
        config.flip_pattern = "random"
    
    return config