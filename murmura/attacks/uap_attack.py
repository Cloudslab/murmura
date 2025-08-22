"""
Universal Adversarial Perturbations (UAP) attack implementation for federated learning research.

UAP computes a single universal perturbation that affects multiple inputs, demonstrating
systematic vulnerabilities in machine learning models. In federated learning, this is
implemented as consistent gradient perturbations across rounds.
"""

from typing import Dict, Any, Tuple, Union, Optional
import numpy as np
import torch
import torch.nn as nn
from .base_attack import BaseAttack


class UAPAttack(BaseAttack):
    """
    Universal Adversarial Perturbations (UAP) attack for systematic model vulnerabilities.
    
    This attack:
    1. Computes universal perturbations that work across different training rounds
    2. Maintains consistent perturbation patterns to exploit systematic vulnerabilities
    3. Applies the same perturbation direction across multiple gradient updates
    
    This is a true implementation that creates universal gradient perturbations
    based on actual training gradients.
    """
    
    def __init__(self, client_id: str, attack_config: Dict[str, Any]):
        super().__init__(client_id, attack_config)
        
        # UAP specific parameters
        self.epsilon = attack_config.get("uap_epsilon", 0.1)
        self.max_iterations = attack_config.get("uap_max_iterations", 50)
        self.step_size = attack_config.get("uap_step_size", 0.01)
        self.norm_type = attack_config.get("uap_norm_type", "inf")  # "inf", "2", or "1"
        self.update_frequency = attack_config.get("uap_update_frequency", 10)  # Update UAP every N rounds
        
        # Attack mode
        self.attack_mode = attack_config.get("uap_attack_mode", "gradient")  # "gradient" or "weight"
        
        # Seed configuration for reproducibility
        self.base_seed = attack_config.get("malicious_node_seed", None)
        self.rng = None
        
        # UAP state tracking
        self.universal_perturbations = {}  # Store universal perturbation for each parameter
        self.gradient_accumulator = {}  # Accumulate gradients to compute UAP
        self.rounds_since_update = 0
        self.total_updates = 0
        
        self.logger.info(
            f"UAP attack initialized - epsilon: {self.epsilon}, "
            f"update_freq: {self.update_frequency}, norm: {self.norm_type}, mode: {self.attack_mode}"
        )
    
    def poison_data(
        self, 
        features: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        round_num: int,
        attack_intensity: float
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        UAP doesn't poison data directly - it manipulates gradients during training.
        Returns data unchanged.
        """
        return features, labels
    
    def poison_gradients(
        self,
        model_parameters: Dict[str, Any],
        round_num: int,
        attack_intensity: float
    ) -> Dict[str, Any]:
        """
        Apply UAP attack to model parameters/gradients.
        
        UAP maintains universal perturbations that are updated periodically
        based on accumulated gradient information.
        
        Args:
            model_parameters: Dictionary containing either:
                - Model parameters (if attack_mode == "weight")
                - Model gradients (if attack_mode == "gradient")
            round_num: Current training round
            attack_intensity: Current attack intensity (0.0 to 1.0)
            
        Returns:
            Poisoned parameters/gradients with universal perturbations
        """
        if attack_intensity == 0.0:
            return model_parameters
        
        # Initialize RNG for this round
        if self.base_seed is not None:
            client_hash = hash(self.client_id) % (2**31)
            round_seed = self.base_seed + round_num + client_hash
            self.rng = np.random.RandomState(round_seed)
            torch.manual_seed(round_seed)
        else:
            self.rng = np.random.RandomState()
        
        # Scale epsilon by attack intensity
        effective_epsilon = self.epsilon * attack_intensity
        
        # Accumulate gradients for UAP computation
        self._accumulate_gradients(model_parameters)
        
        # Update UAP if it's time
        self.rounds_since_update += 1
        if self.rounds_since_update >= self.update_frequency or not self.universal_perturbations:
            self._update_universal_perturbations(effective_epsilon)
            self.rounds_since_update = 0
            self.total_updates += 1
        
        # Apply universal perturbations
        poisoned_parameters = {}
        for param_name, param_value in model_parameters.items():
            if param_name in self.universal_perturbations:
                poisoned_parameters[param_name] = self._apply_uap_perturbation(
                    param_value, self.universal_perturbations[param_name], effective_epsilon
                )
            else:
                # If no UAP yet, initialize with current gradient
                self.universal_perturbations[param_name] = self._initialize_perturbation(
                    param_value, effective_epsilon
                )
                poisoned_parameters[param_name] = self._apply_uap_perturbation(
                    param_value, self.universal_perturbations[param_name], effective_epsilon
                )
        
        # Log attack action
        self.log_attack_action(
            "uap_gradient_attack",
            {
                "round": round_num,
                "attack_intensity": attack_intensity,
                "effective_epsilon": effective_epsilon,
                "parameters_modified": len(model_parameters),
                "attack_mode": self.attack_mode,
                "norm_type": self.norm_type,
                "total_updates": self.total_updates,
                "rounds_since_update": self.rounds_since_update
            }
        )
        
        return poisoned_parameters
    
    def _accumulate_gradients(self, model_parameters: Dict[str, Any]) -> None:
        """Accumulate gradients for UAP computation."""
        for param_name, param_value in model_parameters.items():
            if param_name not in self.gradient_accumulator:
                self.gradient_accumulator[param_name] = []
            
            # Store gradient information (keep limited history)
            if len(self.gradient_accumulator[param_name]) >= self.max_iterations:
                self.gradient_accumulator[param_name].pop(0)
            
            if isinstance(param_value, torch.Tensor):
                self.gradient_accumulator[param_name].append(param_value.clone().detach())
            else:
                self.gradient_accumulator[param_name].append(np.copy(param_value))
    
    def _update_universal_perturbations(self, epsilon: float) -> None:
        """Update universal perturbations based on accumulated gradients."""
        for param_name, gradient_history in self.gradient_accumulator.items():
            if len(gradient_history) > 0:
                # Compute universal perturbation as weighted average of gradient signs
                if isinstance(gradient_history[0], torch.Tensor):
                    # PyTorch implementation
                    # Stack gradients and compute mean direction
                    stacked_grads = torch.stack(gradient_history)
                    mean_grad = torch.mean(stacked_grads, dim=0)
                    
                    # Compute perturbation direction
                    if self.norm_type == "inf":
                        perturbation = epsilon * torch.sign(mean_grad)
                    elif self.norm_type == "2":
                        norm = torch.norm(mean_grad)
                        if norm > 0:
                            perturbation = epsilon * mean_grad / norm
                        else:
                            perturbation = torch.zeros_like(mean_grad)
                    else:  # L1
                        norm = torch.norm(mean_grad, p=1)
                        if norm > 0:
                            perturbation = epsilon * mean_grad / norm
                        else:
                            perturbation = torch.zeros_like(mean_grad)
                    
                    # Update with momentum
                    if param_name in self.universal_perturbations:
                        # Combine old and new perturbation with momentum
                        momentum = 0.9
                        self.universal_perturbations[param_name] = (
                            momentum * self.universal_perturbations[param_name] + 
                            (1 - momentum) * perturbation
                        )
                    else:
                        self.universal_perturbations[param_name] = perturbation
                    
                else:
                    # NumPy implementation
                    stacked_grads = np.stack(gradient_history)
                    mean_grad = np.mean(stacked_grads, axis=0)
                    
                    if self.norm_type == "inf":
                        perturbation = epsilon * np.sign(mean_grad)
                    elif self.norm_type == "2":
                        norm = np.linalg.norm(mean_grad)
                        if norm > 0:
                            perturbation = epsilon * mean_grad / norm
                        else:
                            perturbation = np.zeros_like(mean_grad)
                    else:  # L1
                        norm = np.linalg.norm(mean_grad, ord=1)
                        if norm > 0:
                            perturbation = epsilon * mean_grad / norm
                        else:
                            perturbation = np.zeros_like(mean_grad)
                    
                    if param_name in self.universal_perturbations:
                        momentum = 0.9
                        self.universal_perturbations[param_name] = (
                            momentum * self.universal_perturbations[param_name] + 
                            (1 - momentum) * perturbation
                        )
                    else:
                        self.universal_perturbations[param_name] = perturbation
    
    def _initialize_perturbation(
        self, 
        parameter: Union[np.ndarray, torch.Tensor], 
        epsilon: float
    ) -> Union[np.ndarray, torch.Tensor]:
        """Initialize a universal perturbation for a parameter."""
        if isinstance(parameter, torch.Tensor):
            if self.norm_type == "inf":
                return epsilon * torch.sign(parameter)
            elif self.norm_type == "2":
                norm = torch.norm(parameter)
                if norm > 0:
                    return epsilon * parameter / norm
                else:
                    return torch.zeros_like(parameter)
            else:  # L1
                norm = torch.norm(parameter, p=1)
                if norm > 0:
                    return epsilon * parameter / norm
                else:
                    return torch.zeros_like(parameter)
        else:
            if self.norm_type == "inf":
                return epsilon * np.sign(parameter)
            elif self.norm_type == "2":
                norm = np.linalg.norm(parameter)
                if norm > 0:
                    return epsilon * parameter / norm
                else:
                    return np.zeros_like(parameter)
            else:  # L1
                norm = np.linalg.norm(parameter, ord=1)
                if norm > 0:
                    return epsilon * parameter / norm
                else:
                    return np.zeros_like(parameter)
    
    def _apply_uap_perturbation(
        self, 
        parameter: Union[np.ndarray, torch.Tensor],
        perturbation: Union[np.ndarray, torch.Tensor],
        epsilon: float
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply universal perturbation to a parameter or gradient."""
        if self.attack_mode == "gradient":
            # For gradients: add universal perturbation to amplify in consistent direction
            if isinstance(parameter, torch.Tensor):
                return parameter + perturbation
            else:
                return parameter + perturbation
        else:
            # For weights: add perturbation to weights
            if isinstance(parameter, torch.Tensor):
                return parameter + perturbation
            else:
                return parameter + perturbation
    
    def reset_perturbations(self):
        """Reset universal perturbations and gradient accumulator."""
        self.universal_perturbations = {}
        self.gradient_accumulator = {}
        self.rounds_since_update = 0
    
    def apply_to_model_training(self, model: nn.Module, loss_fn: nn.Module) -> None:
        """
        Hook into model training to apply UAP during backpropagation.
        
        This method should be called before training to register gradient hooks.
        
        Args:
            model: PyTorch model being trained
            loss_fn: Loss function being used
        """
        def uap_gradient_hook(grad):
            """Hook to modify gradients during backpropagation."""
            if grad is not None:
                # Apply universal perturbation
                # Note: This would need parameter name tracking for full implementation
                return grad + self.epsilon * torch.sign(grad)
            return grad
        
        # Register backward hooks on all parameters
        for param in model.parameters():
            if param.requires_grad:
                param.register_hook(uap_gradient_hook)
    
    def get_attack_statistics(self) -> Dict[str, Any]:
        """Get UAP attack specific statistics."""
        base_stats = super().get_attack_statistics()
        
        uap_stats = {
            "epsilon": self.epsilon,
            "max_iterations": self.max_iterations,
            "step_size": self.step_size,
            "norm_type": self.norm_type,
            "update_frequency": self.update_frequency,
            "attack_mode": self.attack_mode,
            "total_updates": self.total_updates,
            "num_universal_perturbations": len(self.universal_perturbations),
            "rounds_since_update": self.rounds_since_update
        }
        
        return {**base_stats, **uap_stats}