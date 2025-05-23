from typing import Dict, Any, Optional

import numpy as np
from scipy.special import comb
from scipy.special import logsumexp

from murmura.privacy.privacy_mechanism import PrivacyMechanism


class MomentsAccountant:
    """
    Implements the Moments Accountant for privacy accounting.
    Based on Abadi et al. "Deep Learning with Differential Privacy"
    """

    def __init__(self, moments_orders: Optional[list] = None):
        """Initialize the Moments Accountant."""
        if moments_orders is None:
            # Default moment orders that provide good privacy estimates
            self.moments_orders = list(range(1, 33)) + list(range(34, 64, 2))
        else:
            self.moments_orders = moments_orders

    def _compute_log_moment(self, q: float, sigma: float, order: int) -> float:
        """
        Compute log moment of the privacy loss for Gaussian mechanism.

        Args:
            q: Sampling probability
            sigma: Noise multiplier
            order: Moment order

        Returns:
            Log moment value
        """
        if sigma <= 0:
            return float('inf')

        if q == 0:
            return 0.0

        if order == 1:
            return 0.0

        # For numerical stability, compute in log space
        # Using the formula from Abadi et al.
        def _compute_log_a(i):
            """Compute log(A_i) for i-th term in the sum."""
            # Binomial coefficient in log space
            try:
                log_binom = np.log(comb(order, i, exact=True))
            except (ValueError, OverflowError):
                return float('-inf')

            # q^i * (1-q)^(order-i) in log space
            if q == 1:
                # Special case when q = 1
                log_qi = 0.0 if i == order else float('-inf')
            else:
                if i == 0:
                    log_qi = (order - i) * np.log(1 - q)
                elif i == order:
                    log_qi = i * np.log(q)
                else:
                    log_qi = i * np.log(q) + (order - i) * np.log(1 - q)

            # The Gaussian moment
            s = 2 * i - order
            log_e = s * (s - 1) / (2 * sigma**2)

            return log_binom + log_qi + log_e

        # Compute all terms, handling -inf values
        log_terms = []
        for i in range(order + 1):
            term = _compute_log_a(i)
            if np.isfinite(term):  # Only include finite terms
                log_terms.append(term)

        # Handle edge cases
        if not log_terms:
            return float('-inf')

        if len(log_terms) == 1:
            return float(log_terms[0])

        # Convert to numpy array and use logsumexp
        log_terms_array = np.array(log_terms, dtype=np.float64)
        result = logsumexp(log_terms_array)

        # Ensure we return a scalar float
        if isinstance(result, np.ndarray):
            result = result.item()

        return float(result)

    def compute_epsilon(self, q: float, sigma: float, steps: int,
                        delta: float = 1e-5) -> Dict[str, float]:
        """
        Compute epsilon given the Gaussian mechanism parameters.

        Args:
            q: Sampling probability
            sigma: Noise multiplier
            steps: Number of steps
            delta: Target delta

        Returns:
            Dictionary with epsilon and related values
        """
        if steps == 0:
            return {"epsilon": 0.0, "delta": delta}

        if sigma <= 0:
            return {"epsilon": float('inf'), "delta": delta}

        # Compute log moments for all orders
        log_moments = []
        for order in self.moments_orders:
            log_moment = self._compute_log_moment(q, sigma, order)
            # Accumulate over steps
            log_moments.append(log_moment * steps)

        # Convert to (epsilon, delta)-DP
        min_epsilon = float('inf')
        best_order = 1

        for i, order in enumerate(self.moments_orders):
            if order == 1:
                continue

            # Compute epsilon for this order
            if log_moments[i] == float('inf'):
                continue

            epsilon = (log_moments[i] - np.log(delta)) / (order - 1)

            if epsilon < min_epsilon:
                min_epsilon = epsilon
                best_order = order

        return {
            "epsilon": min_epsilon,
            "delta": delta,
            "best_order": best_order
        }


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
        Calibrate noise multiplier to achieve target epsilon.
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

        # Binary search for the right noise multiplier
        # Start with reasonable bounds based on target epsilon
        if target_epsilon < 1.0:
            lower, upper = 2.0, 20.0
        elif target_epsilon < 3.0:
            lower, upper = 1.0, 10.0
        else:
            lower, upper = 0.5, 5.0

        # Ensure bounds bracket the solution
        eps_lower = compute_epsilon_for_sigma(lower)
        eps_upper = compute_epsilon_for_sigma(upper)

        # Adjust bounds if necessary
        while eps_lower < target_epsilon:
            lower /= 2
            eps_lower = compute_epsilon_for_sigma(lower)
            if lower < 0.01:
                break

        while eps_upper > target_epsilon:
            upper *= 2
            eps_upper = compute_epsilon_for_sigma(upper)
            if upper > 100:
                break

        # Binary search
        for i in range(max_iterations):
            mid = (lower + upper) / 2
            eps_mid = compute_epsilon_for_sigma(mid)

            if abs(eps_mid - target_epsilon) < tolerance:
                print(f"Found noise multiplier: {mid:.4f} for target ε={target_epsilon}")
                return mid

            if eps_mid > target_epsilon:
                # Need more noise
                lower = mid
            else:
                # Need less noise
                upper = mid

        # Return the final estimate
        final = (lower + upper) / 2
        print(f"Noise calibration: σ={final:.4f} for target ε={target_epsilon}")
        return final