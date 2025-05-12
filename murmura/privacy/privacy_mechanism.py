from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

import numpy as np


class PrivacyMechanism(ABC):
    """
    Abstract interface for differential privacy mechanisms.
    Define how noise is added to model parameters oe gradients.
    """

    @abstractmethod
    def add_noise(
        self,
        parameters: Dict[str, Any],
        clipping_norms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Add noise to the model parameters or gradients.

        Args:
            parameters (Dict[str, Any]): Model parameters or gradients.
            clipping_norms (Optional[Dict[str, float]]): Clipping norms for each parameter.

        Returns:
            Dict[str, Any]: Noisy model parameters or gradients.
        """
        pass

    @abstractmethod
    def clip_parameters(
        self,
        parameters: Dict[str, Any],
        clipping_norms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Clip the model parameters or gradients.

        Args:
            parameters (Dict[str, Any]): Model parameters or gradients.
            clipping_norms (Optional[Dict[str, float]]): Clipping norms for each parameter.

        Returns:
            Dict[str, Any]: Clipped model parameters or gradients.
        """
        pass

    @abstractmethod
    def get_privacy_spent(
        self,
        num_iterations: int,
        noise_multiplier: float,
        batch_size: int,
        total_samples: int,
    ) -> Dict[str, float]:
        """
        Calculate the privacy spent.

        Args:
            num_iterations (int): Number of iterations.
            noise_multiplier (float): Noise multiplier.
            batch_size (int): Batch size.
            total_samples (int): Total number of samples.

        Returns:
            Dict[str, float]: Dictionary containing epsilon and delta values.
        """
        pass

    @staticmethod
    def compute_adaptive_clipping_norms(
        parameters_list: List[Dict[str, Any]],
        quantile: float = 0.9,
        min_clip: float = 0.001,
        per_layer: bool = True,
    ) -> Dict[str, float]:
        """
        Compute adaptive clipping norms for the parameters.

        Args:
            parameters_list (List[Dict[str, Any]]): List of model parameters.
            quantile (float): Quantile for clipping norms.
            min_clip (float): Minimum clipping norm.
            per_layer (bool): Whether to compute per-layer clipping norms.

        Returns:
            Dict[str, float]: Clipping norms for each parameter.
        """
        if not parameters_list:
            return {}

        clipping_norms = {}

        if per_layer:
            for key in parameters_list[0].keys():
                norms = []

                for params in parameters_list:
                    if key in params:
                        param_norm = float(np.linalg.norm(params[key].flatten()))
                        norms.append(param_norm)

                if norms:
                    clip_value = float(np.quantile(norms, quantile))
                    clipping_norms[key] = max(clip_value, min_clip)
        else:
            global_norms = []

            for params in parameters_list:
                squared_sum = 0.0
                for key, value in params.items():
                    squared_sum += float(np.sum(np.square(value)))
                global_norms.append(np.sqrt(squared_sum))

            clip_value = float(np.quantile(global_norms, quantile))
            global_clip = max(clip_value, min_clip)

            for key in parameters_list[0].keys():
                clipping_norms[key] = global_clip

        return clipping_norms
