"""
Gradual Byzantine Gradient Attack Implementation.

This module implements a sophisticated gradual Byzantine attack that:
1. Starts with minimal gradient manipulation and gradually increases
2. Manipulates model parameters/gradients directly rather than training data
3. Uses multiple manipulation strategies (scaling, bias injection, sign flipping)
4. Follows the same intensity progression as other attacks for fair comparison
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from murmura.attacks.gradual_label_flipping import AttackPhase, AttackConfig


@dataclass
class ByzantineGradientConfig(AttackConfig):
    """Configuration for gradual Byzantine gradient attack."""
    
    # Gradient manipulation strategies
    manipulation_strategy: str = "mixed"  # "scaling", "bias", "sign_flip", "mixed"
    target_layers: Optional[List[str]] = None  # Specific layers to target (None = all)
    
    # Scaling attack parameters (multiply gradients by scale factor)
    subtle_scale_factor: float = 1.05      # 5% amplification
    moderate_scale_factor: float = 1.15    # 15% amplification  
    aggressive_scale_factor: float = 1.30  # 30% amplification
    maximum_scale_factor: float = 1.50     # 50% amplification
    
    # Bias injection parameters (add bias to gradients)
    subtle_bias_strength: float = 0.05     # 5% of gradient magnitude
    moderate_bias_strength: float = 0.15   # 15% of gradient magnitude
    aggressive_bias_strength: float = 0.30 # 30% of gradient magnitude
    maximum_bias_strength: float = 0.50    # 50% of gradient magnitude
    
    # Sign flipping parameters (fraction of gradients to flip sign)
    subtle_flip_rate: float = 0.05         # 5% of gradients
    moderate_flip_rate: float = 0.15       # 15% of gradients
    aggressive_flip_rate: float = 0.30     # 30% of gradients
    maximum_flip_rate: float = 0.50        # 50% of gradients
    
    # Mixed strategy weights
    scaling_weight: float = 0.4
    bias_weight: float = 0.4
    flip_weight: float = 0.2


class GradualByzantineGradientAttack:
    """
    Implements gradual Byzantine gradient attack with sophisticated parameter manipulation.
    
    This attack gradually manipulates model parameters/gradients to corrupt the federated
    learning process while initially evading detection.
    """
    
    def __init__(self, 
                 node_id: str,
                 config: ByzantineGradientConfig,
                 dataset_name: str = "mnist"):
        """
        Initialize gradual Byzantine gradient attack.
        
        Args:
            node_id: ID of the malicious node
            config: Byzantine gradient attack configuration
            dataset_name: Name of dataset (for logging)
        """
        self.node_id = node_id
        self.config = config
        self.dataset_name = dataset_name
        
        # Attack state
        self.current_round = 0
        self.current_phase = AttackPhase.DORMANT
        self.current_scale_factor = 1.0
        self.current_bias_strength = 0.0
        self.current_flip_rate = 0.0
        
        # Attack statistics
        self.total_updates_processed = 0
        self.total_parameters_manipulated = 0
        self.manipulations_per_round = []
        self.phase_history = []
        
        # Manipulation tracking
        self.manipulation_stats = {}
        
        # Logger
        self.logger = logging.getLogger(f"murmura.attacks.ByzantineGradient.{node_id}")
        self.logger.info(f"Initialized gradual Byzantine gradient attack for {node_id} on {dataset_name}")
        
    def update_round(self, round_num: int) -> None:
        """Update attack for new FL round."""
        self.current_round = round_num
        
        # Update attack phase and parameters
        self._update_attack_phase()
        
        # Log phase changes
        if len(self.phase_history) == 0 or self.phase_history[-1] != self.current_phase:
            self.logger.info(
                f"Round {round_num}: Entering {self.current_phase.value} phase "
                f"(scale: {self.current_scale_factor:.3f}, "
                f"bias: {self.current_bias_strength:.3f}, "
                f"flip_rate: {self.current_flip_rate:.3f})"
            )
        
        self.phase_history.append(self.current_phase)
    
    def _update_attack_phase(self) -> None:
        """Update current attack phase and manipulation parameters."""
        round_since_start = max(0, self.current_round - self.config.start_round)
        
        if self.current_round < self.config.start_round:
            self.current_phase = AttackPhase.DORMANT
            self.current_scale_factor = 1.0
            self.current_bias_strength = 0.0
            self.current_flip_rate = 0.0
        elif round_since_start < self.config.dormant_rounds:
            self.current_phase = AttackPhase.DORMANT
            self.current_scale_factor = 1.0
            self.current_bias_strength = 0.0
            self.current_flip_rate = 0.0
        elif round_since_start < self.config.dormant_rounds + self.config.subtle_rounds:
            self.current_phase = AttackPhase.SUBTLE
            self.current_scale_factor = self.config.subtle_scale_factor
            self.current_bias_strength = self.config.subtle_bias_strength
            self.current_flip_rate = self.config.subtle_flip_rate
        elif round_since_start < (self.config.dormant_rounds + self.config.subtle_rounds + 
                                  self.config.moderate_rounds):
            self.current_phase = AttackPhase.MODERATE
            self.current_scale_factor = self.config.moderate_scale_factor
            self.current_bias_strength = self.config.moderate_bias_strength
            self.current_flip_rate = self.config.moderate_flip_rate
        elif round_since_start < (self.config.dormant_rounds + self.config.subtle_rounds + 
                                  self.config.moderate_rounds + self.config.aggressive_rounds):
            self.current_phase = AttackPhase.AGGRESSIVE
            self.current_scale_factor = self.config.aggressive_scale_factor
            self.current_bias_strength = self.config.aggressive_bias_strength
            self.current_flip_rate = self.config.aggressive_flip_rate
        else:
            self.current_phase = AttackPhase.MAXIMUM
            self.current_scale_factor = self.config.maximum_scale_factor
            self.current_bias_strength = self.config.maximum_bias_strength
            self.current_flip_rate = self.config.maximum_flip_rate
    
    def manipulate_parameters(self, 
                            parameters: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Apply gradual Byzantine manipulation to model parameters.
        
        Args:
            parameters: Model parameters to manipulate
            
        Returns:
            Tuple of (manipulated_parameters, attack_stats)
        """
        if self.current_phase == AttackPhase.DORMANT:
            # No attack during dormant phase
            attack_stats = {
                "attack_applied": False,
                "phase": self.current_phase.value,
                "scale_factor": 1.0,
                "bias_strength": 0.0,
                "flip_rate": 0.0,
                "parameters_manipulated": 0,
            }
            return parameters.copy(), attack_stats
        
        # Apply gradient manipulation
        manipulated_params = {}
        total_params_manipulated = 0
        manipulation_details = {}
        
        for layer_name, param_values in parameters.items():
            # Check if this layer should be targeted
            if self._should_target_layer(layer_name):
                manipulated_values, layer_stats = self._manipulate_layer_parameters(
                    param_values, layer_name
                )
                manipulated_params[layer_name] = manipulated_values
                total_params_manipulated += layer_stats["params_affected"]
                manipulation_details[layer_name] = layer_stats
            else:
                # Keep original parameters
                manipulated_params[layer_name] = param_values.copy()
        
        self.total_updates_processed += 1
        self.total_parameters_manipulated += total_params_manipulated
        self.manipulations_per_round.append(total_params_manipulated)
        
        self.logger.debug(
            f"Round {self.current_round}: Manipulated {total_params_manipulated} parameters "
            f"({self.current_phase.value} phase)"
        )
        
        attack_stats = {
            "attack_applied": True,
            "phase": self.current_phase.value,
            "scale_factor": self.current_scale_factor,
            "bias_strength": self.current_bias_strength,
            "flip_rate": self.current_flip_rate,
            "parameters_manipulated": total_params_manipulated,
            "total_manipulated": self.total_parameters_manipulated,
            "round": self.current_round,
            "manipulation_strategy": self.config.manipulation_strategy,
            "layer_details": manipulation_details,
        }
        
        return manipulated_params, attack_stats
    
    def _should_target_layer(self, layer_name: str) -> bool:
        """Determine if a layer should be targeted for manipulation."""
        if self.config.target_layers is None:
            return True  # Target all layers
        
        # Check if layer name matches any target patterns
        for target in self.config.target_layers:
            if target in layer_name:
                return True
        return False
    
    def _manipulate_layer_parameters(self, 
                                   param_values: np.ndarray, 
                                   layer_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply manipulation to a specific layer's parameters.
        
        Args:
            param_values: Parameter values for this layer
            layer_name: Name of the layer
            
        Returns:
            Tuple of (manipulated_values, manipulation_stats)
        """
        manipulated = param_values.copy()
        original_shape = manipulated.shape
        flat_params = manipulated.flatten()
        
        params_affected = 0
        operations_applied = []
        
        if self.config.manipulation_strategy == "scaling":
            # Scale all parameters
            flat_params *= self.current_scale_factor
            params_affected = len(flat_params)
            operations_applied.append("scaling")
            
        elif self.config.manipulation_strategy == "bias":
            # Add bias proportional to parameter magnitude
            bias = self.current_bias_strength * np.abs(flat_params) * np.sign(np.random.randn(len(flat_params)))
            flat_params += bias
            params_affected = len(flat_params)
            operations_applied.append("bias_injection")
            
        elif self.config.manipulation_strategy == "sign_flip":
            # Randomly flip signs of parameters
            n_flip = int(len(flat_params) * self.current_flip_rate)
            if n_flip > 0:
                flip_indices = np.random.choice(len(flat_params), n_flip, replace=False)
                flat_params[flip_indices] *= -1
                params_affected = n_flip
                operations_applied.append("sign_flip")
                
        elif self.config.manipulation_strategy == "mixed":
            # Apply combination of strategies with different weights
            
            # 1. Scaling (weighted)
            scaling_strength = 1.0 + (self.current_scale_factor - 1.0) * self.config.scaling_weight
            flat_params *= scaling_strength
            
            # 2. Bias injection (weighted)  
            bias_strength = self.current_bias_strength * self.config.bias_weight
            if bias_strength > 0:
                bias = bias_strength * np.abs(flat_params) * np.sign(np.random.randn(len(flat_params)))
                flat_params += bias
            
            # 3. Sign flipping (weighted)
            flip_rate = self.current_flip_rate * self.config.flip_weight
            n_flip = int(len(flat_params) * flip_rate)
            if n_flip > 0:
                flip_indices = np.random.choice(len(flat_params), n_flip, replace=False)
                flat_params[flip_indices] *= -1
                params_affected += n_flip
            
            params_affected = len(flat_params)  # All parameters affected by scaling/bias
            operations_applied = ["mixed_strategy"]
        
        # Reshape back to original shape
        manipulated = flat_params.reshape(original_shape)
        
        manipulation_stats = {
            "params_affected": params_affected,
            "total_params": len(param_values.flatten()),
            "manipulation_rate": params_affected / len(param_values.flatten()),
            "operations_applied": operations_applied,
            "scale_factor_applied": self.current_scale_factor if "scaling" in str(operations_applied) else 1.0,
            "bias_strength_applied": self.current_bias_strength if "bias" in str(operations_applied) else 0.0,
        }
        
        return manipulated, manipulation_stats
    
    def get_attack_statistics(self) -> Dict[str, Any]:
        """Get comprehensive attack statistics."""
        return {
            "node_id": self.node_id,
            "attack_type": "byzantine_gradient",
            "current_round": self.current_round,
            "current_phase": self.current_phase.value,
            "current_scale_factor": self.current_scale_factor,
            "current_bias_strength": self.current_bias_strength,
            "current_flip_rate": self.current_flip_rate,
            "total_updates_processed": self.total_updates_processed,
            "total_parameters_manipulated": self.total_parameters_manipulated,
            "manipulations_per_round": self.manipulations_per_round.copy(),
            "phase_history": [phase.value for phase in self.phase_history],
            "manipulation_config": {
                "strategy": self.config.manipulation_strategy,
                "target_layers": self.config.target_layers,
                "scale_factors": {
                    "subtle": self.config.subtle_scale_factor,
                    "moderate": self.config.moderate_scale_factor,
                    "aggressive": self.config.aggressive_scale_factor,
                    "maximum": self.config.maximum_scale_factor,
                },
                "bias_strengths": {
                    "subtle": self.config.subtle_bias_strength,
                    "moderate": self.config.moderate_bias_strength,
                    "aggressive": self.config.aggressive_bias_strength,
                    "maximum": self.config.maximum_bias_strength,
                },
                "flip_rates": {
                    "subtle": self.config.subtle_flip_rate,
                    "moderate": self.config.moderate_flip_rate,
                    "aggressive": self.config.aggressive_flip_rate,
                    "maximum": self.config.maximum_flip_rate,
                },
            },
        }
    
    def is_attacking(self) -> bool:
        """Check if currently in an attacking phase."""
        return self.current_phase != AttackPhase.DORMANT
    
    def get_expected_detection_round(self) -> int:
        """Estimate when this attack should be detected."""
        # Byzantine gradient attacks might be detected earlier due to parameter corruption
        detection_round = (
            self.config.start_round + 
            self.config.dormant_rounds + 
            self.config.subtle_rounds + 
            self.config.moderate_rounds // 2  # Likely detected in moderate phase
        )
        return detection_round


def create_byzantine_gradient_config(
    dataset_name: str = "mnist",
    attack_intensity: str = "moderate",
    stealth_level: str = "medium",
    manipulation_strategy: str = "mixed"
) -> ByzantineGradientConfig:
    """
    Create Byzantine gradient attack configuration for different scenarios.
    
    Args:
        dataset_name: Dataset name (mnist, cifar10)
        attack_intensity: Overall attack intensity (low, moderate, high)
        stealth_level: Stealth/evasion level (low, medium, high)
        manipulation_strategy: Type of manipulation (scaling, bias, sign_flip, mixed)
        
    Returns:
        ByzantineGradientConfig instance
    """
    if attack_intensity == "low":
        config = ByzantineGradientConfig(
            start_round=4,
            subtle_scale_factor=1.03,      # Matches label flipping low intensity
            moderate_scale_factor=1.08,    # Matches label flipping low intensity
            aggressive_scale_factor=1.15,  # Matches label flipping low intensity
            maximum_scale_factor=1.25,     # Matches label flipping low intensity
            subtle_bias_strength=0.03,
            moderate_bias_strength=0.08,
            aggressive_bias_strength=0.15,
            maximum_bias_strength=0.25,
            subtle_flip_rate=0.03,
            moderate_flip_rate=0.08,
            aggressive_flip_rate=0.15,
            maximum_flip_rate=0.25,
            manipulation_strategy=manipulation_strategy,
        )
    elif attack_intensity == "high":
        config = ByzantineGradientConfig(
            start_round=2,
            subtle_scale_factor=1.08,      # Matches label flipping high intensity
            moderate_scale_factor=1.20,    # Matches label flipping high intensity  
            aggressive_scale_factor=1.40,  # Matches label flipping high intensity
            maximum_scale_factor=1.70,     # Matches label flipping high intensity
            subtle_bias_strength=0.08,
            moderate_bias_strength=0.20,
            aggressive_bias_strength=0.40,
            maximum_bias_strength=0.70,
            subtle_flip_rate=0.08,
            moderate_flip_rate=0.20,
            aggressive_flip_rate=0.40,
            maximum_flip_rate=0.70,
            manipulation_strategy=manipulation_strategy,
        )
    else:  # moderate - matches other attacks exactly
        config = ByzantineGradientConfig(
            start_round=1,
            dormant_rounds=1,
            subtle_rounds=2,
            moderate_rounds=2,
            aggressive_rounds=2,
            maximum_rounds=2,
            subtle_scale_factor=1.05,      # Matches label flipping moderate (5%)
            moderate_scale_factor=1.15,    # Matches label flipping moderate (15%)
            aggressive_scale_factor=1.30,  # Matches label flipping moderate (30%)
            maximum_scale_factor=1.50,     # Matches label flipping moderate (50%)
            subtle_bias_strength=0.05,
            moderate_bias_strength=0.15,
            aggressive_bias_strength=0.30,
            maximum_bias_strength=0.50,
            subtle_flip_rate=0.05,
            moderate_flip_rate=0.15,
            aggressive_flip_rate=0.30,
            maximum_flip_rate=0.50,
            manipulation_strategy=manipulation_strategy,
        )
    
    # Adjust for dataset complexity
    if dataset_name == "cifar10":
        # CIFAR-10 might need more aggressive manipulation
        config.subtle_scale_factor += 0.01
        config.moderate_scale_factor += 0.02
        config.aggressive_scale_factor += 0.03
        config.maximum_scale_factor += 0.05
    
    # Adjust stealth features
    if stealth_level == "high":
        config.manipulation_strategy = "scaling"  # Less obvious than mixed
        config.add_noise = True
        config.noise_scale = 0.02
        # Target specific layers only (less obvious)
        config.target_layers = ["fc", "linear", "classifier"]  # Target final layers
    elif stealth_level == "low":
        config.manipulation_strategy = "mixed"  # More obvious
        config.add_noise = False
        # Target all layers
        config.target_layers = None
    else:  # medium
        config.manipulation_strategy = manipulation_strategy
        config.add_noise = True
        config.noise_scale = 0.01
        # Target conv and fc layers
        config.target_layers = ["conv", "fc", "linear"]
    
    return config