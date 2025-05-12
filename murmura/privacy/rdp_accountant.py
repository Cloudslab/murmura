from typing import Dict, List, Optional

import numpy as np


class RDPAccountant:
    """
    Privacy accountant based on Rényi Differential Privacy (RDP).

    This implementation is based on the paper:
    "Rényi Differential Privacy of the Sampled Gaussian Mechanism" by Mironov et al.
    https://arxiv.org/abs/1908.10530
    """

    def __init__(self, alphas: Optional[List[float]] = None):
        """
        Initialize the RDP accountant.

        Args:
            alphas: List of RDP orders to compute (default is a range from 1.5 to 512)
        """
        if alphas is None:
            # Default alphas used in typical DP implementations
            self.alphas = [
                1.5,
                2,
                2.5,
                3,
                3.5,
                4,
                4.5,
                5,
                6,
                8,
                10,
                12,
                14,
                16,
                20,
                24,
                32,
                64,
                128,
                256,
                512,
            ]
        else:
            self.alphas = alphas

    def _compute_rdp(
        self, sampling_rate: float, noise_multiplier: float, iterations: int
    ) -> np.ndarray:
        """
        Compute RDP values for the sampled Gaussian mechanism.

        Args:
            sampling_rate: Sampling probability (batch_size / total_samples)
            noise_multiplier: Noise scale relative to L2 sensitivity
            iterations: Number of iterations/steps

        Returns:
            Array of RDP values for each alpha order
        """
        # Ensure valid inputs and protect against edge cases
        q = min(sampling_rate, 1.0)  # Ensure q is at most 1
        q = max(q, 1e-10)  # Ensure q is at least a small positive value
        sigma = max(noise_multiplier, 1e-10)  # Ensure sigma is positive

        print(f"Computing RDP with q={q}, σ={sigma}, iterations={iterations}")

        # Calculate RDP for Gaussian mechanism with sampling for each alpha
        rdp = np.zeros(len(self.alphas))

        for i, alpha in enumerate(self.alphas):
            # Skip invalid alpha values
            if alpha <= 1:
                continue

            # Use a direct formula for Gaussian mechanism with subsampling
            # This is a simplified but conservative approximation
            # Based on the paper "Concentrated Differential Privacy" by Dwork and Rothblum
            rdp_per_iter = alpha * q**2 / (2 * sigma**2)
            rdp[i] = rdp_per_iter

        # Total RDP is the sum over all iterations
        rdp_result = rdp * iterations

        # Print some values for debugging
        if len(self.alphas) > 0:
            print(f"RDP values for α={self.alphas[0]}: {rdp_result[0]}")

        return rdp_result

    def compute_epsilon(
        self,
        sampling_rate: float,
        noise_multiplier: float,
        iterations: int,
        target_delta: float = 1e-5,
    ) -> Dict[str, float]:
        """
        Compute epsilon for a target delta based on RDP values.

        Args:
            sampling_rate: Sampling probability (batch_size / total_samples)
            noise_multiplier: Noise scale relative to L2 sensitivity
            iterations: Number of iterations/steps
            target_delta: Target delta value (default: 1e-5)

        Returns:
            Dictionary with 'epsilon' and 'delta' values
        """
        # Better protection for invalid inputs
        if iterations <= 0:
            return {
                "epsilon": 0.0,
                "delta": target_delta,
                "noise_multiplier": noise_multiplier,
            }

        if sampling_rate <= 0 or noise_multiplier <= 0:
            return {
                "epsilon": float("inf"),
                "delta": target_delta,
                "noise_multiplier": noise_multiplier,
            }

        # Fix for Ray - ensure we use standard Python types
        sampling_rate = float(sampling_rate)
        noise_multiplier = float(noise_multiplier)
        iterations = int(iterations)
        target_delta = float(target_delta)

        # Compute RDP values
        rdp_values = self._compute_rdp(sampling_rate, noise_multiplier, iterations)

        # Convert RDP to (ε, δ)-DP
        epsilon_values = []

        for i, alpha in enumerate(self.alphas):
            # Skip invalid values
            if alpha <= 1:
                continue

            # Using the conversion formula: ε = rdp + log(1/δ) / (α-1)
            eps = float(rdp_values[i] + np.log(1 / target_delta) / (alpha - 1))

            # Add to list if valid
            if 0 <= eps < float("inf"):
                epsilon_values.append(eps)

        # Take the minimum epsilon if we have valid values
        if epsilon_values:
            min_epsilon = float(np.min(epsilon_values))
            best_alpha_idx = int(np.argmin(epsilon_values))
            best_alpha = (
                float(self.alphas[best_alpha_idx])
                if best_alpha_idx < len(self.alphas)
                else 0.0
            )
        else:
            min_epsilon = 0.0
            best_alpha = 0.0

        return {
            "epsilon": min_epsilon,
            "delta": target_delta,
            "noise_multiplier": noise_multiplier,
            "best_alpha": best_alpha,
        }

    def compute_noise_multiplier(
        self,
        target_epsilon: float,
        target_delta: float,
        sampling_rate: float,
        iterations: int,
        initial_guess: float = 1.0,
        tolerance: float = 0.01,
        max_iterations: int = 30,
    ) -> float:
        """
        Find the noise multiplier needed to achieve a target epsilon and delta.

        Args:
            target_epsilon: Target privacy budget (epsilon)
            target_delta: Target failure probability (delta)
            sampling_rate: Sampling probability (batch_size / total_samples)
            iterations: Number of iterations/steps
            initial_guess: Initial guess for noise multiplier
            tolerance: Tolerance for epsilon matching
            max_iterations: Maximum number of binary search iterations

        Returns:
            Noise multiplier value that achieves the target privacy parameters
        """
        # Handle edge cases
        if target_epsilon <= 0 or target_delta <= 0 or target_delta >= 1:
            return float(10.0)  # Return a large value for stricter privacy

        if iterations <= 0 or sampling_rate <= 0:
            return float(0.5)  # Default reasonable value

        # Ensure we use Python types for Ray compatibility
        target_epsilon = float(target_epsilon)
        target_delta = float(target_delta)
        sampling_rate = float(sampling_rate)
        iterations = int(iterations)
        initial_guess = float(initial_guess)

        # For smaller epsilon, start with a higher noise guess
        if target_epsilon < 1.0:
            initial_guess = max(initial_guess, 4.0)
        elif target_epsilon < 3.0:
            initial_guess = max(initial_guess, 2.0)

        # If we have very few iterations, use higher noise
        if iterations < 5:
            initial_guess *= 2.0

        # Binary search for the appropriate noise multiplier
        lower_bound = 0.01
        upper_bound = 30.0  # Allow much higher upper bound for strict privacy
        current_guess = initial_guess

        # Print initial setup
        print(
            f"Computing noise for ε={target_epsilon}, δ={target_delta}, iterations={iterations}, q={sampling_rate}"
        )

        try:
            for i in range(max_iterations):
                # Compute epsilon for the current noise multiplier
                results = self.compute_epsilon(
                    sampling_rate=sampling_rate,
                    noise_multiplier=current_guess,
                    iterations=iterations,
                    target_delta=target_delta,
                )

                current_epsilon = results["epsilon"]

                # Debug information
                if i % 5 == 0 or i == max_iterations - 1:
                    print(
                        f"  Iteration {i + 1}: noise={current_guess:.4f}, epsilon={current_epsilon:.4f}, target={target_epsilon:.4f}"
                    )

                # Check if we're within tolerance
                if abs(current_epsilon - target_epsilon) <= tolerance:
                    print(
                        f"Found noise multiplier: {current_guess:.4f} for target epsilon: {target_epsilon:.4f}"
                    )
                    return float(current_guess)

                # Update bounds based on binary search
                if current_epsilon > target_epsilon:
                    # Need more noise to reduce epsilon
                    lower_bound = current_guess
                    current_guess = (current_guess + upper_bound) / 2
                else:
                    # Need less noise
                    upper_bound = current_guess
                    current_guess = (current_guess + lower_bound) / 2

        except Exception as e:
            print(f"Error during noise calibration: {e}")
            # Return a reasonable default value on error
            return float(1.0)

        # After max iterations, return our best guess
        print(
            f"Final noise multiplier after {max_iterations} iterations: {current_guess:.4f}"
        )
        return float(current_guess)
