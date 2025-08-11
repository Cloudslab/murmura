"""
Gradient manipulation attack implementation for federated learning research.
"""

from typing import Dict, Any, Tuple, Union
import numpy as np
import torch
from .base_attack import BaseAttack


class GradientManipulationAttack(BaseAttack):
    """
    Gradient manipulation attack that corrupts model parameters to degrade performance.
    
    This attack can operate in several modes:
    1. Gradient noise injection: Add random noise to parameters
    2. Gradient sign flipping: Flip the sign of gradients
    3. Parameter scaling: Scale parameters by malicious factors
    4. Targeted layer attacks: Focus on specific model layers
    """
    
    def __init__(self, client_id: str, attack_config: Dict[str, Any]):
        super().__init__(client_id, attack_config)
        
        # Extract gradient manipulation specific parameters
        self.noise_scale = attack_config.get("gradient_noise_scale", 1.0)
        self.sign_flip_prob = attack_config.get("gradient_sign_flip_prob", 0.1)
        self.target_layers = attack_config.get("target_layers", None)
        
        # Attack mode configuration
        self.attack_modes = []
        if self.noise_scale > 0:
            self.attack_modes.append("noise_injection")
        if self.sign_flip_prob > 0:
            self.attack_modes.append("sign_flipping")
        
        if not self.attack_modes:
            self.attack_modes = ["noise_injection"]  # Default fallback
        
        self.logger.info(f"Gradient manipulation attack initialized with modes: {self.attack_modes}")
    
    def poison_data(
        self, 
        features: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        round_num: int,
        attack_intensity: float
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Gradient manipulation attack doesn't modify training data.
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
        Poison the model parameters/gradients.
        
        Args:
            model_parameters: Model parameters to poison
            round_num: Current training round
            attack_intensity: Current attack intensity (0.0 to 1.0)
            
        Returns:
            Poisoned model parameters
        """
        if attack_intensity == 0.0:
            return model_parameters
        
        poisoned_parameters = {}
        attack_actions = []
        
        for param_name, param_value in model_parameters.items():
            # Skip if targeting specific layers and this isn't one of them
            if self.target_layers is not None and not any(
                layer in param_name for layer in self.target_layers
            ):
                poisoned_parameters[param_name] = param_value
                continue
            
            # Apply poisoning based on configured attack modes
            poisoned_param = param_value
            
            if "noise_injection" in self.attack_modes:
                poisoned_param = self._add_noise(poisoned_param, attack_intensity)
                attack_actions.append(f"noise_injection_{param_name}")
            
            if "sign_flipping" in self.attack_modes:
                poisoned_param = self._flip_signs(poisoned_param, attack_intensity)
                attack_actions.append(f"sign_flipping_{param_name}")
            
            poisoned_parameters[param_name] = poisoned_param
        
        # Log attack action
        self.log_attack_action(
            "gradient_manipulation",
            {
                "round": round_num,
                "attack_intensity": attack_intensity,
                "attack_modes": self.attack_modes,
                "parameters_modified": len(attack_actions),
                "target_layers": self.target_layers,
                "actions": attack_actions
            }
        )
        
        # Update attack history
        self.attack_history["rounds_active"] += 1
        
        return poisoned_parameters
    
    def _add_noise(self, parameter: Union[np.ndarray, torch.Tensor], intensity: float) -> Union[np.ndarray, torch.Tensor]:
        """Add Gaussian noise to parameters."""
        if isinstance(parameter, torch.Tensor):
            # Calculate noise scale based on parameter statistics
            param_std = torch.std(parameter)
            noise_scale = self.noise_scale * param_std * intensity
            
            # Generate noise with same shape and device as parameter
            noise = torch.randn_like(parameter) * noise_scale
            
            return parameter + noise
        else:
            # NumPy array or scalar
            if np.isscalar(parameter) or parameter.ndim == 0:
                # Handle scalar case
                param_std = abs(parameter) if parameter != 0 else 1.0
                noise_scale = self.noise_scale * param_std * intensity
                noise = np.random.normal(0, noise_scale)
                return parameter + noise
            else:
                # Handle array case
                param_std = np.std(parameter)
                noise_scale = self.noise_scale * param_std * intensity
                
                noise = np.random.normal(0, noise_scale, parameter.shape)
                
                return parameter + noise
    
    def _flip_signs(self, parameter: Union[np.ndarray, torch.Tensor], intensity: float) -> Union[np.ndarray, torch.Tensor]:
        """Flip the signs of parameters based on probability."""
        # Scale flip probability by attack intensity
        effective_flip_prob = self.sign_flip_prob * intensity
        
        if isinstance(parameter, torch.Tensor):
            # Generate random mask for sign flipping
            flip_mask = torch.rand_like(parameter) < effective_flip_prob
            
            # Apply sign flipping
            flipped_param = parameter.clone()
            flipped_param[flip_mask] = -flipped_param[flip_mask]
            
            return flipped_param
        else:
            # NumPy array or scalar
            if np.isscalar(parameter) or parameter.ndim == 0:
                # Handle scalar case
                if np.random.random() < effective_flip_prob:
                    return -parameter
                else:
                    return parameter
            else:
                # Handle array case
                flip_mask = np.random.random(parameter.shape) < effective_flip_prob
                
                flipped_param = parameter.copy()  # type: ignore
                flipped_param[flip_mask] = -flipped_param[flip_mask]
                
                return flipped_param
    
    def _scale_parameters(self, parameter: Union[np.ndarray, torch.Tensor], intensity: float) -> Union[np.ndarray, torch.Tensor]:
        """Scale parameters by malicious factors."""
        # Create malicious scaling factor based on intensity
        # Lower intensity = smaller perturbation, higher intensity = larger perturbation
        scaling_factor = 1.0 + intensity * np.random.uniform(-0.5, 0.5)
        
        if isinstance(parameter, torch.Tensor):
            return parameter * scaling_factor
        else:
            return parameter * scaling_factor
    
    def get_attack_statistics(self) -> Dict[str, Any]:
        """Get gradient manipulation specific statistics."""
        base_stats = super().get_attack_statistics()
        
        grad_attack_stats = {
            "attack_modes": self.attack_modes,
            "noise_scale": self.noise_scale,
            "sign_flip_prob": self.sign_flip_prob,
            "target_layers": self.target_layers,
            "parameters_modified_total": sum(
                len(action.get("actions", [])) 
                for action in self.attack_history["attack_actions"]
                if action["action"] == "gradient_manipulation"
            )
        }
        
        return {**base_stats, **grad_attack_stats}