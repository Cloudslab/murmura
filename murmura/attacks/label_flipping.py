"""
Label flipping attack implementation for federated learning research.
"""

from typing import Dict, Any, Tuple, Union
import numpy as np
import torch
from .base_attack import BaseAttack


class LabelFlippingAttack(BaseAttack):
    """
    Label flipping attack that corrupts training labels to degrade model performance.
    
    This attack can operate in several modes:
    1. Random flipping: Randomly change labels
    2. Targeted flipping: Change specific source labels to target labels
    3. Adversarial flipping: Flip to the most confusing labels
    """
    
    def __init__(self, client_id: str, attack_config: Dict[str, Any]):
        super().__init__(client_id, attack_config)
        
        # Extract label flipping specific parameters
        self.target_label = attack_config.get("label_flip_target", None)
        self.source_label = attack_config.get("label_flip_source", None)
        
        # Determine number of classes from config or infer later
        self.num_classes = attack_config.get("num_classes", None)
        
        # Attack mode tracking
        self.attack_mode = self._determine_attack_mode()
        
        self.logger.info(f"Label flipping attack initialized in {self.attack_mode} mode")
    
    def _determine_attack_mode(self) -> str:
        """Determine the attack mode based on configuration."""
        if self.target_label is not None and self.source_label is not None:
            return "targeted"
        elif self.target_label is not None:
            return "single_target"
        else:
            return "random"
    
    def poison_data(
        self, 
        features: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        round_num: int,
        attack_intensity: float
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        """
        Poison the training data by flipping labels.
        
        Args:
            features: Input features (unchanged by label flipping)
            labels: Target labels to poison
            round_num: Current training round
            attack_intensity: Current attack intensity (0.0 to 1.0)
            
        Returns:
            Tuple of (unchanged_features, poisoned_labels)
        """
        if attack_intensity == 0.0:
            return features, labels
        
        # Convert to numpy for processing
        labels_np = self._convert_to_numpy(labels)
        original_shape = labels_np.shape
        
        # Flatten for processing
        labels_flat = labels_np.flatten()
        
        # Infer number of classes if not provided
        if self.num_classes is None:
            self.num_classes = int(np.max(labels_flat)) + 1
        
        # Calculate number of samples to poison
        num_samples = len(labels_flat)
        num_to_poison = int(num_samples * attack_intensity)
        
        if num_to_poison == 0:
            return features, labels
        
        # Select samples to poison
        poison_indices = np.random.choice(num_samples, num_to_poison, replace=False)
        
        # Apply poisoning based on attack mode
        poisoned_labels = labels_flat.copy()
        poisoned_count = 0
        
        for idx in poison_indices:
            original_label = labels_flat[idx]
            new_label = self._get_poisoned_label(original_label)
            
            if new_label != original_label:
                poisoned_labels[idx] = new_label
                poisoned_count += 1
        
        # Reshape back to original shape
        poisoned_labels = poisoned_labels.reshape(original_shape)
        
        # Convert back to original type
        poisoned_labels = self._maintain_data_type(labels, poisoned_labels)  # type: ignore
        
        # Log attack action
        self.log_attack_action(
            "label_flipping",
            {
                "round": round_num,
                "attack_intensity": attack_intensity,
                "samples_poisoned": poisoned_count,
                "total_samples": num_samples,
                "attack_mode": self.attack_mode,
                "target_label": self.target_label,
                "source_label": self.source_label
            }
        )
        
        # Update attack history
        self.attack_history["rounds_active"] += 1
        self.attack_history["total_samples_poisoned"] += poisoned_count
        
        return features, poisoned_labels
    
    def _get_poisoned_label(self, original_label: int) -> int:
        """Get the poisoned label for a given original label."""
        
        if self.attack_mode == "targeted":
            # Flip from source to target
            if original_label == self.source_label:
                return self.target_label
            else:
                return original_label
        
        elif self.attack_mode == "single_target":
            # Flip any label to target
            if original_label != self.target_label:
                return self.target_label
            else:
                return original_label
        
        else:  # random mode
            # Flip to any other label (excluding original to ensure actual corruption)
            # This ensures attack intensity directly corresponds to % of data corrupted
            available_labels = list(range(self.num_classes))
            available_labels.remove(original_label)
            return np.random.choice(available_labels)
    
    def poison_gradients(
        self,
        model_parameters: Dict[str, Any],
        round_num: int,
        attack_intensity: float
    ) -> Dict[str, Any]:
        """
        Label flipping attack doesn't directly modify gradients.
        Returns parameters unchanged.
        """
        return model_parameters
    
    def get_attack_statistics(self) -> Dict[str, Any]:
        """Get label flipping specific statistics."""
        base_stats = super().get_attack_statistics()
        
        label_flip_stats = {
            "attack_mode": self.attack_mode,
            "target_label": self.target_label,
            "source_label": self.source_label,
            "num_classes": self.num_classes,
            "avg_samples_per_round": (
                self.attack_history["total_samples_poisoned"] / 
                max(1, self.attack_history["rounds_active"])
            )
        }
        
        return {**base_stats, **label_flip_stats}