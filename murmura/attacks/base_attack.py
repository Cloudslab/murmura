"""
Base class for model poisoning attacks in federated learning research.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Union
import logging
import numpy as np
import torch


class BaseAttack(ABC):
    """Base class for all model poisoning attacks."""
    
    def __init__(self, client_id: str, attack_config: Dict[str, Any]):
        self.client_id = client_id
        self.attack_config = attack_config
        self.logger = logging.getLogger(f"murmura.attack.{client_id}")
        
        # Attack state tracking
        self.attack_history: Dict[str, Any] = {
            "rounds_active": 0,
            "total_samples_poisoned": 0,
            "attack_actions": []
        }
        
    @abstractmethod
    def poison_data(
        self, 
        features: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        round_num: int,
        attack_intensity: float
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Poison the training data.
        
        Args:
            features: Input features
            labels: Target labels
            round_num: Current training round
            attack_intensity: Current attack intensity (0.0 to 1.0)
            
        Returns:
            Tuple of (poisoned_features, poisoned_labels)
        """
        pass
    
    @abstractmethod
    def poison_gradients(
        self,
        model_parameters: Dict[str, Any],
        round_num: int,
        attack_intensity: float
    ) -> Dict[str, Any]:
        """
        Poison the model gradients/parameters.
        
        Args:
            model_parameters: Model parameters to poison
            round_num: Current training round
            attack_intensity: Current attack intensity (0.0 to 1.0)
            
        Returns:
            Poisoned model parameters
        """
        pass
    
    def log_attack_action(self, action: str, details: Dict[str, Any]) -> None:
        """Log an attack action for monitoring."""
        log_entry = {
            "action": action,
            "client_id": self.client_id,
            "timestamp": torch.tensor(0.0),  # Placeholder for timestamp
            "details": details
        }
        
        self.attack_history["attack_actions"].append(log_entry)
        
        if self.attack_config.get("log_attack_details", True):
            self.logger.info(f"Attack action: {action} - {details}")
    
    def get_attack_statistics(self) -> Dict[str, Any]:
        """Get attack statistics for monitoring."""
        return {
            "client_id": self.client_id,
            "attack_type": self.__class__.__name__,
            "rounds_active": self.attack_history["rounds_active"],
            "total_samples_poisoned": self.attack_history["total_samples_poisoned"],
            "total_actions": len(self.attack_history["attack_actions"])
        }
    
    def _convert_to_numpy(self, data: Union[np.ndarray, torch.Tensor, list]) -> np.ndarray:
        """Convert data to numpy array for processing."""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            return np.array(data)
        return data
    
    def _convert_to_tensor(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert data to torch tensor for processing."""
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        return data
    
    def _maintain_data_type(
        self, 
        original_data: Union[np.ndarray, torch.Tensor, list],
        processed_data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor, list]:
        """Maintain the original data type after processing."""
        if isinstance(original_data, torch.Tensor):
            if isinstance(processed_data, np.ndarray):
                return torch.from_numpy(processed_data).to(original_data.dtype)
            return processed_data.to(original_data.dtype)
        elif isinstance(original_data, list):
            # If original was a list, convert back to list
            if isinstance(processed_data, torch.Tensor):
                return processed_data.cpu().numpy().tolist()
            elif isinstance(processed_data, np.ndarray):
                return processed_data.tolist()
            return processed_data
        else:
            if isinstance(processed_data, torch.Tensor):
                return processed_data.cpu().numpy()
            return processed_data.astype(original_data.dtype)