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
        Initialize the GaussianMechanism with noise multiplier and maximum gradient norm.

        Args:
            noise_multiplier (float): Standard deviation of the Gaussian noise.
            max_grad_norm (float): Maximum gradient norm for clipping.
            per_layer_clipping (bool): Whether to apply clipping per layer or globally.
            accountant: Privacy accountant for tracking privacy budget.
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
        Add Gaussian noise to the model parameters or gradients.

        Args:
            parameters (Dict[str, Any]): Model parameters or gradients.
            clipping_norms (Optional[Dict[str, float]]): Clipping norms for each parameter.

        Returns:
            Dict[str, Any]: Noisy model parameters or gradients.
        """
        if clipping_norms is None:
            clipping_norms = {key: self.max_grad_norm for key in parameters.keys()}

        noised_parameters = {}

        for key, param in parameters.items():
            # Get the appropriate clipping norm for this parameter
            clip_norm = clipping_norms.get(key, self.max_grad_norm)

            noise_scale = clip_norm * self.noise_multiplier

            # Generate Gaussian noise
            noise = np.random.normal(0, noise_scale, param.shape).astype(param.dtype)

            noised_parameters[key] = param + noise

        return noised_parameters

    def clip_parameters(
        self,
        parameters: Dict[str, Any],
        clipping_norms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Clip the model parameters or gradients to the specified norms.

        Args:
            parameters (Dict[str, Any]): Model parameters or gradients.
            clipping_norms (Optional[Dict[str, float]]): Clipping norms for each parameter.

        Returns:
            Dict[str, Any]: Clipped model parameters or gradients.
        """
        if clipping_norms is None:
            clipping_norms = {key: self.max_grad_norm for key in parameters.keys()}

        clipped_parameters = {}

        if self.per_layer_clipping:
            for key, param in parameters.items():
                clip_norm = clipping_norms.get(key, self.max_grad_norm)
                param_norm = float(np.linalg.norm(param.flatten()))

                if param_norm > clip_norm:
                    scale = clip_norm / (param_norm + 1e-10)  # Avoid division by zero
                    clipped_parameters[key] = param * scale
                else:
                    clipped_parameters[key] = param.copy()
        else:
            squared_sum = 0.0
            for key, param in parameters.items():
                squared_sum += float(np.sum(np.square(param)))

            global_norm = np.sqrt(squared_sum)
            global_clip = next(iter(clipping_norms.values()))

            if global_norm > global_clip:
                scale = global_clip / (global_norm + 1e-10)
                for key, param in parameters.items():
                    clipped_parameters[key] = param * scale
            else:
                for key, param in parameters.items():
                    clipped_parameters[key] = param.copy()

        return clipped_parameters

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
        if self.accountant is None:
            self.accountant = RDPAccountant()

        sampling_rate = batch_size / total_samples

        return self.accountant.compute_epsilon(
            sampling_rate=sampling_rate,
            noise_multiplier=noise_multiplier,
            iterations=num_iterations,
        )

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
        Calibrate the noise multiplier to achieve a target epsilon and delta.

        Args:
            target_epsilon (float): Target epsilon value.
            target_delta (float): Target delta value.
            iterations (int): Number of iterations.
            batch_size (int): Batch size.
            total_samples (int): Total number of samples.
            initial_guess (float): Initial guess for the noise multiplier.
            tolerance (float): Tolerance for convergence.
            max_iterations (int): Maximum number of iterations for calibration.

        Returns:
            float: Calibrated noise multiplier.
        """
        lower_bound = 0.1
        upper_bound = 10.0
        current_guess = initial_guess

        for _ in range(max_iterations):
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
                # Need more noise (higher noise multiplier)
                lower_bound = current_guess
                current_guess = (current_guess + upper_bound) / 2
            else:
                # Need less noise (lower noise multiplier)
                upper_bound = current_guess
                current_guess = (current_guess + lower_bound) / 2

        # Return best guess after max iterations
        return current_guess
