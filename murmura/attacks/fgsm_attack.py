"""
Fast Gradient Sign Method (FGSM) attack implementation for federated learning research.

FGSM creates adversarial perturbations by using the actual gradient of the loss function 
with respect to model parameters during training.
"""

from typing import Dict, Any, Tuple, Union, Optional
import numpy as np
import torch
import torch.nn as nn
from .base_attack import BaseAttack


class FGSMAttack(BaseAttack):
    """
    Fast Gradient Sign Method (FGSM) attack that perturbs gradients during training.
    
    This attack:
    1. Computes actual gradients during backpropagation
    2. Applies sign-based perturbations in the gradient direction
    3. Scales perturbations by epsilon to control attack strength
    
    This is a true implementation that manipulates actual training gradients,
    not synthetic or surrogate gradients.
    """
    
    def __init__(self, client_id: str, attack_config: Dict[str, Any]):
        super().__init__(client_id, attack_config)
        
        # FGSM specific parameters
        self.epsilon = attack_config.get("fgsm_epsilon", 0.3)
        self.targeted = attack_config.get("fgsm_targeted", False)
        self.target_class = attack_config.get("fgsm_target_class", 0)
        
        # Whether to apply to weights or gradients
        self.attack_mode = attack_config.get("fgsm_attack_mode", "gradient")  # "gradient" or "weight"
        
        # Seed configuration for reproducibility
        self.base_seed = attack_config.get("malicious_node_seed", None)
        self.rng = None
        
        self.logger.info(
            f"FGSM attack initialized - epsilon: {self.epsilon}, "
            f"targeted: {self.targeted}, mode: {self.attack_mode}"
        )
    
    def poison_data(
        self, 
        features: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        round_num: int,
        attack_intensity: float
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        FGSM doesn't poison data directly - it manipulates gradients during training.
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
        Apply FGSM attack to model parameters/gradients.
        
        This method should be called with the actual gradients computed during training.
        For true FGSM, we need the gradients from backpropagation.
        
        Args:
            model_parameters: Dictionary containing either:
                - Model parameters (if attack_mode == "weight")
                - Model gradients (if attack_mode == "gradient")
            round_num: Current training round
            attack_intensity: Current attack intensity (0.0 to 1.0)
            
        Returns:
            Poisoned parameters/gradients
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
        
        poisoned_parameters = {}
        
        # Scale epsilon by attack intensity
        effective_epsilon = self.epsilon * attack_intensity
        
        for param_name, param_value in model_parameters.items():
            poisoned_parameters[param_name] = self._apply_fgsm_perturbation(
                param_value, effective_epsilon
            )
        
        # Log attack action
        self.log_attack_action(
            "fgsm_gradient_attack",
            {
                "round": round_num,
                "attack_intensity": attack_intensity,
                "effective_epsilon": effective_epsilon,
                "parameters_modified": len(model_parameters),
                "attack_mode": self.attack_mode
            }
        )
        
        return poisoned_parameters
    
    def _apply_fgsm_perturbation(
        self, 
        parameter: Union[np.ndarray, torch.Tensor], 
        epsilon: float
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply FGSM perturbation to a parameter or gradient.
        
        For gradient mode: perturbed_gradient = gradient + epsilon * sign(gradient)
        This amplifies the gradient in its original direction, causing larger updates.
        
        For weight mode: perturbed_weight = weight + epsilon * sign(weight)
        This directly perturbs the weights themselves.
        """
        
        if isinstance(parameter, torch.Tensor):
            # Compute sign of the parameter/gradient
            sign_param = torch.sign(parameter)
            
            # Apply FGSM perturbation
            if self.attack_mode == "gradient":
                # For gradients: amplify in the gradient direction
                # This causes the model to take larger steps in wrong directions
                perturbed = parameter + epsilon * sign_param
            else:
                # For weights: directly perturb the weights
                perturbed = parameter + epsilon * sign_param
            
            return perturbed
        else:
            # NumPy array processing
            if np.isscalar(parameter) or parameter.ndim == 0:
                sign_param = np.sign(parameter)
                if self.attack_mode == "gradient":
                    return parameter + epsilon * sign_param
                else:
                    return parameter + epsilon * sign_param
            else:
                sign_param = np.sign(parameter)
                if self.attack_mode == "gradient":
                    return parameter + epsilon * sign_param
                else:
                    return parameter + epsilon * sign_param
    
    def apply_to_model_training(self, model: nn.Module, loss_fn: nn.Module) -> None:
        """
        Hook into model training to apply FGSM during backpropagation.
        
        This method should be called before training to register gradient hooks.
        
        Args:
            model: PyTorch model being trained
            loss_fn: Loss function being used
        """
        def fgsm_gradient_hook(grad):
            """Hook to modify gradients during backpropagation."""
            if grad is not None:
                # Apply FGSM: perturb gradient in its sign direction
                return grad + self.epsilon * torch.sign(grad)
            return grad
        
        # Register backward hooks on all parameters
        for param in model.parameters():
            if param.requires_grad:
                param.register_hook(fgsm_gradient_hook)
    
    def compute_adversarial_loss(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: nn.Module
    ) -> torch.Tensor:
        """
        Compute adversarial loss for FGSM attack.
        
        This computes the loss that encourages misclassification.
        
        Args:
            model: Model being attacked
            inputs: Input data
            labels: True labels
            loss_fn: Original loss function
            
        Returns:
            Adversarial loss
        """
        outputs = model(inputs)
        
        if self.targeted:
            # For targeted attack: minimize loss w.r.t. target class
            target_labels = torch.full_like(labels, self.target_class)
            adv_loss = -loss_fn(outputs, target_labels)  # Negative to minimize
        else:
            # For untargeted attack: maximize loss w.r.t. true labels
            adv_loss = -loss_fn(outputs, labels)  # Negative to maximize
        
        return adv_loss
    
    def get_attack_statistics(self) -> Dict[str, Any]:
        """Get FGSM attack specific statistics."""
        base_stats = super().get_attack_statistics()
        
        fgsm_stats = {
            "epsilon": self.epsilon,
            "targeted": self.targeted,
            "target_class": self.target_class if self.targeted else None,
            "attack_mode": self.attack_mode
        }
        
        return {**base_stats, **fgsm_stats}