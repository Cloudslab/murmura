from typing import Dict, Any, Optional

import numpy as np

from murmura.privacy.privacy_mechanism import PrivacyMechanism
from murmura.privacy.moments_accountant import MomentsAccountant


class GaussianMechanism(PrivacyMechanism):
    """
    Fixed implementation of Gaussian Mechanism for differential privacy.
    """

    def __init__(
            self,
            noise_multiplier: float = 1.0,
            max_grad_norm: float = 1.0,
            per_layer_clipping: bool = True,
            accountant=None,
    ):
        """Initialize the Gaussian Mechanism."""
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.per_layer_clipping = per_layer_clipping
        self.accountant = accountant or MomentsAccountant()

    def add_noise(
            self,
            parameters: Dict[str, Any],
            clipping_norms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Add Gaussian noise to model parameters.
        
        IMPORTANT: This assumes parameters have already been clipped!
        """
        if clipping_norms is None:
            clipping_norms = {key: self.max_grad_norm for key in parameters.keys()}

        noised_parameters = {}

        for key, param in parameters.items():
            # Skip invalid parameters
            if np.isnan(param).any() or np.isinf(param).any():
                print(f"Warning: Found NaN/Inf in parameter {key}. Using zeros.")
                noised_parameters[key] = np.zeros_like(param)
                continue

            # Get the clipping norm for this parameter
            clip_norm = clipping_norms.get(key, self.max_grad_norm)

            # Ensure valid clipping norm
            clip_norm = max(clip_norm, 0.01)

            # Copy parameter
            noised_param = param.copy()

            # Calculate noise standard deviation
            # For DP-SGD, noise_std = clip_norm * noise_multiplier
            noise_std = clip_norm * self.noise_multiplier

            # Generate and add Gaussian noise
            noise = np.random.normal(0, noise_std, param.shape).astype(param.dtype)
            noised_parameters[key] = noised_param + noise

            # Handle any NaN/Inf after noise addition
            if np.isnan(noised_parameters[key]).any() or np.isinf(noised_parameters[key]).any():
                print(f"Warning: NaN/Inf in {key} after noise. Using original values.")
                noised_parameters[key] = param.copy()

        return noised_parameters

    def clip_parameters(
            self,
            parameters: Dict[str, Any],
            clipping_norms: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Clip model parameters using L2 norm clipping.
        """
        if clipping_norms is None:
            clipping_norms = {key: self.max_grad_norm for key in parameters.keys()}

        clipped_parameters = {}

        # Clean parameters first
        clean_parameters = {}
        for key, param in parameters.items():
            param_copy = param.copy()
            if np.isnan(param).any() or np.isinf(param).any():
                print(f"Warning: Found NaN/Inf in parameter {key}. Replacing with zeros.")
                param_copy = np.nan_to_num(param_copy, nan=0.0, posinf=0.0, neginf=0.0)
            clean_parameters[key] = param_copy

        if self.per_layer_clipping:
            # Clip each layer separately
            for key, param in clean_parameters.items():
                clip_norm = clipping_norms.get(key, self.max_grad_norm)
                clip_norm = max(clip_norm, 0.01)  # Ensure positive

                # Calculate L2 norm
                param_norm = float(np.linalg.norm(param.flatten()))

                # Clip if necessary
                if param_norm > clip_norm:
                    scale = clip_norm / (param_norm + 1e-12)
                    clipped_parameters[key] = param * scale
                else:
                    clipped_parameters[key] = param.copy()
        else:
            # Global clipping across all parameters
            global_norm_sq = 0.0
            for param in clean_parameters.values():
                global_norm_sq += float(np.sum(np.square(param)))

            global_norm = np.sqrt(global_norm_sq + 1e-12)

            # Get global clipping norm
            global_clip = self.max_grad_norm
            if clipping_norms:
                # Use the average of all clipping norms for global clipping
                global_clip = np.mean(list(clipping_norms.values()))
            global_clip = max(global_clip, 0.01)

            # Apply clipping
            if global_norm > global_clip:
                scale = global_clip / global_norm
            else:
                scale = 1.0

            for key, param in clean_parameters.items():
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
        Calculate privacy spent using Moments Accountant.
        """
        if num_iterations <= 0 or batch_size <= 0 or total_samples <= 0:
            return {"epsilon": 0.0, "delta": 1e-5, "noise_multiplier": noise_multiplier}

        # Sampling probability
        q = min(1.0, batch_size / total_samples)

        # Use Moments Accountant
        result = self.accountant.compute_epsilon(
            q=q,
            sigma=noise_multiplier,
            steps=num_iterations,
            delta=1e-5
        )

        result["noise_multiplier"] = noise_multiplier
        return result

    def calibrate_noise_to_target_epsilon(
            self,
            target_epsilon: float,
            target_delta: float,
            iterations: int,
            batch_size: int,
            total_samples: int,
            initial_guess: float = 1.0,
            tolerance: float = 0.01,
            max_iterations: int = 50,
    ) -> float:
        """
        Calibrate noise multiplier to achieve target epsilon with better bounds.
        """
        if iterations <= 0 or batch_size <= 0 or total_samples <= 0:
            print("Invalid parameters for noise calibration")
            return 1.0

        q = min(1.0, batch_size / total_samples)

        # Define the objective function
        def compute_epsilon_for_sigma(sigma):
            result = self.accountant.compute_epsilon(
                q=q, sigma=sigma, steps=iterations, delta=target_delta
            )
            return result["epsilon"]

        # CRITICAL FIX: Better initial bounds based on target epsilon
        # These bounds are calibrated for typical federated learning scenarios
        if target_epsilon < 1.0:
            lower, upper = 5.0, 50.0  # Very high noise for strong privacy
        elif target_epsilon < 5.0:
            lower, upper = 2.0, 20.0  # Moderate noise
        elif target_epsilon < 10.0:
            lower, upper = 1.0, 10.0  # Lower noise for better utility
        elif target_epsilon < 50.0:
            lower, upper = 0.5, 5.0   # Much lower noise for high epsilon
        else:
            lower, upper = 0.1, 2.0   # Minimal noise for very high epsilon

        # Ensure bounds bracket the solution
        eps_lower = compute_epsilon_for_sigma(lower)
        eps_upper = compute_epsilon_for_sigma(upper)

        # Adjust bounds if necessary
        while eps_lower < target_epsilon and lower > 0.01:
            lower /= 2
            eps_lower = compute_epsilon_for_sigma(lower)

        while eps_upper > target_epsilon and upper < 100:
            upper *= 2
            eps_upper = compute_epsilon_for_sigma(upper)

        # Binary search with better convergence
        best_sigma = (lower + upper) / 2
        best_epsilon_diff = float('inf')

        for i in range(max_iterations):
            mid = (lower + upper) / 2
            eps_mid = compute_epsilon_for_sigma(mid)

            # Track best result
            epsilon_diff = abs(eps_mid - target_epsilon)
            if epsilon_diff < best_epsilon_diff:
                best_epsilon_diff = epsilon_diff
                best_sigma = mid

            if epsilon_diff < tolerance:
                print(f"Found noise multiplier: {mid:.4f} for target ε={target_epsilon}")
                print(f"  Actual ε: {eps_mid:.4f} (difference: {epsilon_diff:.4f})")
                return mid

            if eps_mid > target_epsilon:
                # Need more noise
                lower = mid
            else:
                # Need less noise
                upper = mid

        # Return the best estimate found
        final_eps = compute_epsilon_for_sigma(best_sigma)
        print(f"Noise calibration: σ={best_sigma:.4f} for target ε={target_epsilon}")
        print(f"  Achieved ε: {final_eps:.4f} (target: {target_epsilon})")
        return best_sigma
