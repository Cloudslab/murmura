import math
from typing import Dict, List, Union

import numpy as np


def clip_gradients(
    gradients: Dict[str, np.ndarray], max_norm: float, per_layer: bool = False
) -> Dict[str, np.ndarray]:
    """
    Clip gradients based on their L2 norm.

    Args:
        gradients: Dictionary of gradients
        max_norm: Maximum allowed norm
        per_layer: Whether to clip each layer separately

    Returns:
        Clipped gradients
    """
    if per_layer:
        # Clip each layer separately
        clipped_gradients = {}
        for key, grad in gradients.items():
            grad_norm = np.linalg.norm(grad.flatten())
            if grad_norm > max_norm:
                scale = max_norm / (grad_norm + 1e-10)  # Avoid division by zero
                clipped_gradients[key] = grad * scale
            else:
                clipped_gradients[key] = grad.copy()
        return clipped_gradients
    else:
        # Global clipping
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += float(np.sum(np.square(grad)))
        total_norm = np.sqrt(total_norm)

        # Check if clipping is needed
        if total_norm <= max_norm:
            return {key: grad.copy() for key, grad in gradients.items()}

        # Apply scaling
        scale = max_norm / (total_norm + 1e-10)  # Avoid division by zero
        return {key: grad * scale for key, grad in gradients.items()}


def add_gaussian_noise(
    values: Dict[str, np.ndarray], noise_scale: float
) -> Dict[str, np.ndarray]:
    """
    Add Gaussian noise to values.

    Args:
        values: Dictionary of values
        noise_scale: Scale of the Gaussian noise

    Returns:
        Values with added noise
    """
    noised_values = {}
    for key, value in values.items():
        noise = np.random.normal(0, noise_scale, value.shape).astype(value.dtype)
        noised_values[key] = value + noise
    return noised_values


def compute_parameter_norms(
    parameters_list: List[Dict[str, np.ndarray]], per_layer: bool = True
) -> Union[Dict[str, float], float]:
    """
    Compute norms of parameters.

    Args:
        parameters_list: List of parameter dictionaries
        per_layer: Whether to compute per-layer norms

    Returns:
        Dictionary of norms per layer or a single global norm
    """
    if not parameters_list:
        return {} if per_layer else 0.0

    if per_layer:
        # Compute per-layer norms
        layer_norms = {}
        for key in parameters_list[0].keys():
            norms = []
            for params in parameters_list:
                if key in params:
                    param_norm = float(np.linalg.norm(params[key].flatten()))
                    norms.append(param_norm)

            if norms:
                layer_norms[key] = float(np.mean(norms))
        return layer_norms
    else:
        # Compute a single global norm
        global_norms = []
        for params in parameters_list:
            squared_sum = 0.0
            for value in params.values():
                squared_sum += float(np.sum(np.square(value)))
            global_norms.append(np.sqrt(squared_sum))
        return float(np.mean(global_norms)) if global_norms else 0.0


def estimate_sensitivity(parameters: Dict[str, np.ndarray], batch_size: int) -> float:
    """
    Estimate sensitivity for DP mechanisms based on parameters and batch size.

    Args:
        parameters: Parameter dictionary
        batch_size: Batch size used in training

    Returns:
        Estimated sensitivity
    """
    # For Gaussian mechanism with SGD-like updates, sensitivity scales with 1/batch_size
    global_norm = 0.0
    for param in parameters.values():
        global_norm += float(np.sum(np.square(param)))
    global_norm = np.sqrt(global_norm)

    # Scale by batch size to estimate sensitivity
    sensitivity = global_norm / max(1, batch_size)
    return sensitivity


def estimate_optimal_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    num_samples: int,
    batch_size: int,
    num_epochs: int,
) -> float:
    """
    Estimate an optimal noise multiplier for a given privacy target.
    This is a simple approximation and should be refined with proper RDP accounting.

    Args:
        target_epsilon: Target privacy budget
        target_delta: Target delta
        num_samples: Total number of samples
        batch_size: Batch size
        num_epochs: Number of training epochs

    Returns:
        Estimated noise multiplier
    """
    # Number of SGD steps
    num_steps = (num_samples // batch_size) * num_epochs

    # Sampling probability
    q = batch_size / num_samples

    # Simple approximation based on analytical Gaussian mechanism
    # For more accurate calculations, use the RDP accountant
    c = math.sqrt(2 * math.log(1.25 / target_delta))
    return c / target_epsilon * math.sqrt(2 * q * num_steps)
