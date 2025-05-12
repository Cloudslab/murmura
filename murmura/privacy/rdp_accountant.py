from typing import Optional, List, Dict

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
            alphas (Optional[List[float]]): List of RDP orders to compute (default is a range from 1.5 to 512)
        """
        if alphas is None:
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
            sampling_rate (float): Sampling rate of the data.
            noise_multiplier (float): Noise multiplier for the Gaussian noise.
            iterations (int): Number of iterations.

        Returns:
            Array of RDP values for each alpha order.
        """
        # Convert sampling rate and noise multiplier to appropriate values
        q = sampling_rate
        sigma = noise_multiplier

        # Compute RDP for Gaussian mechanism with sampling for each alpha
        rdp = np.zeros(len(self.alphas))

        for i, alpha in enumerate(self.alphas):
            # log of the moment generating function
            log_mgf = self._compute_log_mgf(q, sigma, alpha)
            # RDP is the log mgf divided by alpha - 1
            rdp[i] = log_mgf / (alpha - 1)

        return rdp * iterations

    @staticmethod
    def _compute_log_mgf(q: float, sigma: float, alpha: float) -> float:
        """
        Compute the log moment generating function for the Gaussian mechanism.

        Args:
            q (float): Sampling rate.
            sigma (float): Noise multiplier.
            alpha (float): RDP order.

        Returns:
            Log moment generating function value.
        """
        if q == 0:
            return 0.0

        # For small q, use the approximation log(1+x) ≈ x
        if q < 1e-4:
            return q * alpha / (2 * (sigma**2))

        # Calculate the moments of the privacy loss random variable
        variance = sigma**2

        term_1 = q**2 * alpha / (1 - q)

        log_term_2 = (
            -0.5 * alpha * np.log(1 + 2 / (alpha - 1) * (1 - q) / (q**2 * variance))
        )
        term_2 = np.exp(log_term_2) - 1

        return term_1 * term_2

    def compute_epsilon(
        self,
        sampling_rate: float,
        noise_multiplier: float,
        iterations: int,
        target_delta: float = 1e-5,
    ) -> Dict[str, float]:
        """
        Compute epsilon for the given parameters.

        Args:
            sampling_rate (float): Sampling rate of the data.
            noise_multiplier (float): Noise multiplier for the Gaussian noise.
            iterations (int): Number of iterations.
            target_delta (float): Target delta for differential privacy.

        Returns:
            Dictionary with epsilon values for each alpha order.
        """
        rdp_values = self._compute_rdp(sampling_rate, noise_multiplier, iterations)

        # Convert RDP to (epsilon, delta) pairs
        epsilon_values = []

        for i, alpha in enumerate(self.alphas):
            eps = rdp_values[i] + np.log(1 / target_delta) / (alpha - 1)
            epsilon_values.append(eps)

        min_epsilon = np.min(epsilon_values)

        return {
            "epsilon": float(min_epsilon),
            "delta": target_delta,
            "noise_multiplier": noise_multiplier,
            "best_alpha": self.alphas[np.argmin(epsilon_values)],
        }

    def compute_noise_multiplier(
        self,
        target_epsilon: float,
        target_delta: float,
        sampling_rate: float,
        iterations: int,
        initial_guess: float = 1.0,
        tolerance: float = 0.01,
        max_iterations: int = 20,
    ) -> float:
        """
        Compute the noise multiplier for the given parameters.

        Args:
            target_epsilon (float): Target epsilon for differential privacy.
            target_delta (float): Target delta for differential privacy.
            sampling_rate (float): Sampling rate of the data.
            iterations (int): Number of iterations.
            initial_guess (float): Initial guess for the noise multiplier.
            tolerance (float): Tolerance for convergence.
            max_iterations (int): Maximum number of iterations for convergence.

        Returns:
            Computed noise multiplier.
        """
        lower_bound = 0.1
        upper_bound = 10.0
        current_guess = initial_guess

        for _ in range(max_iterations):
            results = self.compute_epsilon(
                sampling_rate=sampling_rate,
                noise_multiplier=current_guess,
                iterations=iterations,
                target_delta=target_delta,
            )
            current_epsilon = results["epsilon"]

            if abs(current_epsilon - target_epsilon) <= tolerance:
                return current_guess

            if current_epsilon > target_epsilon:
                lower_bound = current_guess
                current_guess = (current_guess + upper_bound) / 2
            else:
                upper_bound = current_guess
                current_guess = (current_guess + lower_bound) / 2

        return current_guess
