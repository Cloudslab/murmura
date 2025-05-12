from typing import Dict, Any, Optional

import numpy as np

from murmura.privacy.privacy_mechanism import PrivacyMechanism
from murmura.privacy.rdp_accountant import RDPAccountant


class GaussianMechanism(PrivacyMechanism):
    """
    Implementation of Gaussian Mechanism for differential privacy.

    This mechanism adds calibrated Gaussian noise to clipped parameters or gradients
    to achieve differential privacy guarantees.
    """

    def __init__(
        self,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        per_layer_clipping: bool = True,
        accountant=None,
    ):
        """
        Initialize the Gaussian Mechanism.

        Args:
            noise_multiplier: Multiplier for the standard deviation of the noise
            max_grad_norm: Maximum norm for gradient clipping
            per_layer_clipping: Whether to clip each layer separately
            accountant: Privacy accountant for tracking privacy budget
        """
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.per_layer_clipping = per_layer_clipping
        self.accountant = accountant or RDPAccountant()

    def add_noise(
        self,
        parameters: Dict[str, Any],
        clipping_norms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Add Gaussian noise to model parameters.

        Args:
            parameters: Dictionary containing model parameters
            clipping_norms: Optional dictionary of per-parameter clipping norms

        Returns:
            Dictionary containing noised parameters
        """
        if clipping_norms is None:
            # Use global clipping norm
            clipping_norms = {key: self.max_grad_norm for key in parameters.keys()}

        noised_parameters = {}

        for key, param in parameters.items():
            # Get the appropriate clipping norm for this parameter
            clip_norm = clipping_norms.get(key, self.max_grad_norm)

            # Make a copy of the parameter to avoid modifying the original
            noised_param = param.copy()

            # Calculate noise scale based on L2 sensitivity (which is the clipping norm)
            # and the noise multiplier
            noise_scale = (
                clip_norm * self.noise_multiplier / np.sqrt(2)
            )  # Divide by sqrt(2) for correct DP scaling

            # Generate Gaussian noise with appropriate scale
            noise = np.random.normal(0, noise_scale, param.shape).astype(param.dtype)

            # Add noise to the parameter
            noised_parameters[key] = noised_param + noise

        return noised_parameters

    def clip_parameters(
        self,
        parameters: Dict[str, Any],
        clipping_norms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Clip model parameters using L2 norm.

        Args:
            parameters: Dictionary containing model parameters
            clipping_norms: Optional dictionary of per-parameter clipping norms

        Returns:
            Dictionary containing clipped parameters
        """
        if clipping_norms is None:
            # Use global clipping norm
            clipping_norms = {key: self.max_grad_norm for key in parameters.keys()}

        clipped_parameters = {}

        if self.per_layer_clipping:
            # Clip each layer separately
            for key, param in parameters.items():
                # Get the appropriate clipping norm for this parameter
                clip_norm = clipping_norms.get(key, self.max_grad_norm)

                # Calculate the current L2 norm of the parameter
                param_flat = param.flatten()
                param_norm = float(np.linalg.norm(param_flat))

                # Clip the parameter if its norm exceeds the clipping norm
                if param_norm > clip_norm:
                    scale = clip_norm / (
                        param_norm + 1e-12
                    )  # Add small constant to avoid division by zero
                    clipped_parameters[key] = param * scale
                else:
                    clipped_parameters[key] = param.copy()
        else:
            # Global clipping across all parameters
            global_norm = 0.0
            for param in parameters.values():
                param_flat = param.flatten()
                global_norm += float(np.sum(np.square(param_flat)))

            global_norm = float(np.sqrt(global_norm))

            # Use the first norm as global norm if clipping_norms is provided
            if clipping_norms:
                global_clip = next(iter(clipping_norms.values()))
            else:
                global_clip = self.max_grad_norm

            # Calculate scaling factor if needed
            if global_norm > global_clip:
                scale = global_clip / (
                    global_norm + 1e-12
                )  # Add small constant to avoid division by zero
            else:
                scale = 1.0

            # Apply scaling to all parameters
            for key, param in parameters.items():
                clipped_parameters[key] = param * scale

        return clipped_parameters

    def get_privacy_spent(
        self,
        num_iterations: int,
        noise_multiplier: float,
        batch_size: int,
        total_samples: int,
    ) -> Dict[str, float]:
        """
        Calculate privacy spent using the RDP accountant.

        Args:
            num_iterations: Number of training iterations
            noise_multiplier: Noise multiplier used
            batch_size: Batch size used in training
            total_samples: Total number of samples in the dataset

        Returns:
            Dictionary containing epsilon and delta values
        """
        if self.accountant is None:
            self.accountant = RDPAccountant()

        # Fix for Ray - ensure we have valid parameters
        if num_iterations <= 0 or batch_size <= 0 or total_samples <= 0:
            return {"epsilon": 0.0, "delta": 1e-5, "noise_multiplier": noise_multiplier}

        sampling_rate = min(1.0, batch_size / max(1, total_samples))

        print(
            f"  Computing privacy with sampling_rate={sampling_rate}, "
            + f"noise={noise_multiplier}, iterations={num_iterations}"
        )

        # Compute epsilon
        result = self.accountant.compute_epsilon(
            sampling_rate=sampling_rate,
            noise_multiplier=noise_multiplier,
            iterations=num_iterations,
        )

        print(
            f"  Privacy result: epsilon={result.get('epsilon', 0.0):.6f}, "
            + f"best_alpha={result.get('best_alpha', 0.0):.1f}"
        )

        return result

    def calibrate_noise_to_target_epsilon(
        self,
        target_epsilon: float,
        target_delta: float,
        iterations: int,
        batch_size: int,
        total_samples: int,
        initial_guess: float = 1.0,
        tolerance: float = 0.1,
        max_iterations: int = 20,
    ) -> float:
        """
        Calibrate the noise multiplier to achieve a target privacy budget.

        Args:
            target_epsilon: Target privacy budget (epsilon)
            target_delta: Target failure probability (delta)
            iterations: Number of training iterations
            batch_size: Batch size used in training
            total_samples: Total number of samples in the dataset
            initial_guess: Initial guess for the noise multiplier
            tolerance: Tolerance for epsilon matching
            max_iterations: Maximum number of calibration iterations

        Returns:
            Calibrated noise multiplier
        """
        # For very small target epsilons, start with a larger noise value
        if target_epsilon < 1.0:
            initial_guess = max(5.0, initial_guess)

        lower_bound = 0.1
        upper_bound = 100.0  # Increase upper bound for small epsilon values
        current_guess = initial_guess

        # Protect against invalid values
        batch_size = max(1, batch_size)
        total_samples = max(1, total_samples)
        iterations = max(1, iterations)

        for i in range(max_iterations):
            # Calculate privacy spent with current noise multiplier
            privacy_spent = self.get_privacy_spent(
                num_iterations=iterations,
                noise_multiplier=current_guess,
                batch_size=batch_size,
                total_samples=total_samples,
            )

            current_epsilon = privacy_spent.get("epsilon", float("inf"))

            # Check if we're within tolerance
            if abs(current_epsilon - target_epsilon) <= tolerance:
                return current_guess

            # Binary search
            if current_epsilon > target_epsilon:
                # Need more noise (higher noise multiplier) to reduce epsilon
                lower_bound = current_guess
                current_guess = (current_guess + upper_bound) / 2
            else:
                # Need less noise (lower noise multiplier) to increase epsilon
                upper_bound = current_guess
                current_guess = (current_guess + lower_bound) / 2

            # Debug
            print(
                f"Calibration iteration {i + 1}: noise={current_guess:.4f}, epsilon={current_epsilon:.4f}, target={target_epsilon:.4f}"
            )

        # Return the best guess after max iterations
        return current_guess
