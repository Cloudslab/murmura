import math
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

import numpy as np

from murmura.privacy.dp_config import (
    DifferentialPrivacyConfig,
    ClippingStrategy,
    DPMechanism,
)
from murmura.privacy.privacy_accountant import PrivacyAccountant


@dataclass
class ClippingResult:
    """Result of parameter/gradient clipping operation."""

    clipped_params: Dict[str, Any]
    clipping_norm: float
    original_norm: float
    was_clipped: bool
    clipping_ratio: float = 1.0  # original_norm / clipping_norm


class ParameterClipper:
    """
    Handles parameter/gradient clipping with various strategies.

    Supports fixed threshold, adaptive clipping, and quantile-based clipping
    following industry best practices.
    """

    def __init__(
        self,
        strategy: ClippingStrategy,
        initial_norm: float = 1.0,
        target_quantile: float = 0.5,
        adaptation_rate: float = 0.01,
    ):
        self.strategy = strategy
        self.current_norm = initial_norm
        self.target_quantile = target_quantile
        self.adaptation_rate = adaptation_rate

        # For quantile estimation
        self.norm_history: List[float] = []
        self.history_size = 1000  # Keep last 1000 norms for quantile estimation

        self.logger = logging.getLogger("murmura.privacy.clipper")

    def clip_parameters(self, parameters: Dict[str, Any]) -> ClippingResult:
        """
        Clip parameters according to the configured strategy.

        Args:
            parameters: Dictionary of model parameters (numpy arrays)

        Returns:
            ClippingResult with clipped parameters and metadata
        """
        # Compute L2 norm of all parameters
        total_norm = self._compute_global_norm(parameters)

        # Update norm history for adaptive strategies
        self.norm_history.append(total_norm)
        if len(self.norm_history) > self.history_size:
            self.norm_history.pop(0)

        # Determine clipping threshold
        clipping_norm = self._get_clipping_threshold(total_norm)

        # Perform clipping
        if total_norm <= clipping_norm:
            # No clipping needed
            return ClippingResult(
                clipped_params=parameters,
                clipping_norm=clipping_norm,
                original_norm=total_norm,
                was_clipped=False,
                clipping_ratio=1.0,
            )
        else:
            # Clip parameters
            scaling_factor = clipping_norm / total_norm
            clipped_params = {}

            for name, param in parameters.items():
                clipped_params[name] = param * scaling_factor

            self.logger.debug(
                f"Clipped parameters: norm {total_norm:.4f} -> {clipping_norm:.4f} "
                f"(scaling factor: {scaling_factor:.4f})"
            )

            return ClippingResult(
                clipped_params=clipped_params,
                clipping_norm=clipping_norm,
                original_norm=total_norm,
                was_clipped=True,
                clipping_ratio=scaling_factor,
            )

    def _compute_global_norm(self, parameters: Dict[str, Any]) -> float:
        """Compute L2 norm of all parameters combined."""
        total_norm_sq = 0.0

        for param in parameters.values():
            if isinstance(param, np.ndarray):
                total_norm_sq += np.sum(param**2)
            else:
                # Handle other array types (e.g., torch tensors converted to numpy)
                param_array = np.asarray(param)
                total_norm_sq += np.sum(param_array**2)

        return math.sqrt(total_norm_sq)

    def _get_clipping_threshold(self, current_norm: float) -> float:
        """Get clipping threshold based on strategy."""
        if self.strategy == ClippingStrategy.FIXED:
            return self.current_norm

        elif self.strategy == ClippingStrategy.ADAPTIVE:
            # Adaptive clipping: adjust threshold based on recent norms
            if len(self.norm_history) > 10:  # Need some history
                recent_median = np.median(self.norm_history[-100:])  # Last 100 norms

                # Gradually adapt towards median
                target_norm = recent_median
                self.current_norm += self.adaptation_rate * (
                    target_norm - self.current_norm
                )
                self.current_norm = max(0.1, self.current_norm)  # Minimum threshold

                self.logger.debug(
                    f"Adaptive clipping: norm updated to {self.current_norm:.4f} "
                    f"(target: {target_norm:.4f})"
                )

            return self.current_norm

        elif self.strategy == ClippingStrategy.QUANTILE:
            # Quantile-based clipping
            if len(self.norm_history) >= 50:  # Need sufficient history
                quantile_norm = np.quantile(self.norm_history, self.target_quantile)
                self.current_norm = max(0.1, quantile_norm)  # Minimum threshold

                self.logger.debug(
                    f"Quantile clipping: {self.target_quantile}-quantile = {self.current_norm:.4f}"
                )

            return self.current_norm

        else:
            raise ValueError(f"Unknown clipping strategy: {self.strategy}")

    def get_clipping_stats(self) -> Dict[str, Any]:
        """Get statistics about clipping behavior."""
        if not self.norm_history:
            return {"message": "No clipping history available"}

        recent_norms = (
            self.norm_history[-100:]
            if len(self.norm_history) >= 100
            else self.norm_history
        )

        return {
            "current_threshold": self.current_norm,
            "strategy": self.strategy.value,
            "recent_norm_stats": {
                "mean": np.mean(recent_norms),
                "median": np.median(recent_norms),
                "std": np.std(recent_norms),
                "min": np.min(recent_norms),
                "max": np.max(recent_norms),
            },
            "total_norms_seen": len(self.norm_history),
            "clipping_frequency": sum(
                1 for norm in recent_norms if norm > self.current_norm
            )
            / len(recent_norms),
        }


class BaseDPMechanism(ABC):
    """Abstract base class for differential privacy mechanisms."""

    def __init__(self, config: DifferentialPrivacyConfig):
        self.config = config
        self.logger = logging.getLogger(f"murmura.privacy.{self.__class__.__name__}")

    @abstractmethod
    def add_noise(
        self, parameters: Dict[str, Any], sensitivity: float = 1.0
    ) -> Dict[str, Any]:
        """Add calibrated noise to parameters."""
        pass

    @abstractmethod
    def get_privacy_spent(
        self, num_queries: int = 1, sampling_rate: Optional[float] = None
    ) -> Tuple[float, float]:
        """Get privacy cost as (epsilon, delta) for this mechanism."""
        pass


class GaussianMechanism(BaseDPMechanism):
    """
    Gaussian mechanism for (ε, δ)-differential privacy.

    Adds Gaussian noise calibrated to achieve the desired privacy guarantee.
    Most commonly used in federated learning with DP-SGD.
    """

    def __init__(self, config: DifferentialPrivacyConfig):
        super().__init__(config)

        if config.mechanism != DPMechanism.GAUSSIAN:
            raise ValueError("GaussianMechanism requires GAUSSIAN mechanism in config")

        if config.delta is None:
            raise ValueError("GaussianMechanism requires delta parameter")

        self.noise_multiplier = config.noise_multiplier
        self.clipping_norm = config.clipping_norm

    def add_noise(
        self, parameters: Dict[str, Any], sensitivity: float = 1.0
    ) -> Dict[str, Any]:
        """Add Gaussian noise to parameters."""
        if self.clipping_norm is None:
            raise ValueError("clipping_norm must be set for Gaussian mechanism")

        # Noise scale: σ = noise_multiplier * sensitivity
        noise_scale = self.noise_multiplier * sensitivity

        noisy_parameters = {}
        total_noise_norm = 0.0

        for name, param in parameters.items():
            param_array = np.asarray(param)

            # Generate Gaussian noise with same shape as parameter
            noise = np.random.normal(0, noise_scale, param_array.shape)
            noisy_param = param_array + noise

            noisy_parameters[name] = noisy_param
            total_noise_norm += np.sum(noise**2)

        total_noise_norm = math.sqrt(total_noise_norm)

        self.logger.debug(
            f"Added Gaussian noise: scale={noise_scale:.4f}, "
            f"total_noise_norm={total_noise_norm:.4f}"
        )

        return noisy_parameters

    def get_privacy_spent(
        self, num_queries: int = 1, sampling_rate: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Compute privacy cost for Gaussian mechanism.

        This is a simplified version. Production systems should use proper
        privacy accountants (RDP, zCDP) for tighter bounds.
        """
        if sampling_rate is not None:
            # Apply privacy amplification by subsampling (simplified)
            # Real implementation should use proper amplification theorems
            amplification_factor = sampling_rate
            effective_noise_multiplier = self.noise_multiplier / math.sqrt(
                amplification_factor
            )
        else:
            effective_noise_multiplier = self.noise_multiplier

        # Simplified conversion from noise multiplier to (ε, δ)
        # ε ≈ 2 * num_queries / noise_multiplier^2 (rough approximation)
        epsilon = 2.0 * num_queries / (effective_noise_multiplier**2)
        delta = self.config.delta

        return (epsilon, delta)


class LaplaceMechanism(BaseDPMechanism):
    """
    Laplace mechanism for pure ε-differential privacy.

    Adds Laplace noise for pure DP without the δ parameter.
    Generally less preferred for deep learning due to heavier tails.
    """

    def __init__(self, config: DifferentialPrivacyConfig):
        super().__init__(config)

        if config.mechanism != DPMechanism.LAPLACE:
            raise ValueError("LaplaceMechanism requires LAPLACE mechanism in config")

        if config.delta is not None:
            raise ValueError(
                "LaplaceMechanism provides pure ε-DP and should not use delta"
            )

        self.epsilon = config.epsilon
        self.clipping_norm = config.clipping_norm

    def add_noise(
        self, parameters: Dict[str, Any], sensitivity: float = 1.0
    ) -> Dict[str, Any]:
        """Add Laplace noise to parameters."""
        if self.clipping_norm is None:
            raise ValueError("clipping_norm must be set for Laplace mechanism")

        # Laplace scale: b = sensitivity / ε
        laplace_scale = sensitivity / self.epsilon

        noisy_parameters = {}

        for name, param in parameters.items():
            param_array = np.asarray(param)

            # Generate Laplace noise
            noise = np.random.laplace(0, laplace_scale, param_array.shape)
            noisy_param = param_array + noise

            noisy_parameters[name] = noisy_param

        self.logger.debug(f"Added Laplace noise: scale={laplace_scale:.4f}")

        return noisy_parameters

    def get_privacy_spent(
        self, num_queries: int = 1, sampling_rate: Optional[float] = None
    ) -> Tuple[float, float]:
        """Compute privacy cost for Laplace mechanism."""
        epsilon = self.epsilon * num_queries
        return (epsilon, 0.0)  # Pure ε-DP has δ = 0


class DiscreteGaussianMechanism(BaseDPMechanism):
    """
    Discrete Gaussian mechanism for integer-valued parameters.

    Useful for quantized neural networks or when parameters need to remain integers.
    More complex to implement correctly but provides better utility for discrete domains.
    """

    def __init__(self, config: DifferentialPrivacyConfig):
        super().__init__(config)

        if config.mechanism != DPMechanism.DISCRETE_GAUSSIAN:
            raise ValueError(
                "DiscreteGaussianMechanism requires DISCRETE_GAUSSIAN mechanism in config"
            )

        self.noise_multiplier = config.noise_multiplier
        self.clipping_norm = config.clipping_norm

    def add_noise(
        self, parameters: Dict[str, Any], sensitivity: float = 1.0
    ) -> Dict[str, Any]:
        """Add discrete Gaussian noise to parameters."""
        if self.clipping_norm is None:
            raise ValueError(
                "clipping_norm must be set for discrete Gaussian mechanism"
            )

        # This is a simplified implementation
        # Production should use proper discrete Gaussian sampling
        noise_scale = self.noise_multiplier * sensitivity

        noisy_parameters = {}

        for name, param in parameters.items():
            param_array = np.asarray(param)

            # Generate continuous Gaussian noise and round
            continuous_noise = np.random.normal(0, noise_scale, param_array.shape)
            discrete_noise = np.round(continuous_noise)

            noisy_param = param_array + discrete_noise
            noisy_parameters[name] = noisy_param

        self.logger.debug(f"Added discrete Gaussian noise: scale={noise_scale:.4f}")

        return noisy_parameters

    def get_privacy_spent(
        self, num_queries: int = 1, sampling_rate: Optional[float] = None
    ) -> Tuple[float, float]:
        """Compute privacy cost for discrete Gaussian mechanism."""
        # Similar to continuous Gaussian (simplified)
        epsilon = 2.0 * num_queries / (self.noise_multiplier**2)
        delta = self.config.delta if self.config.delta is not None else 1e-5

        return (epsilon, delta)


def create_dp_mechanism(config: DifferentialPrivacyConfig) -> BaseDPMechanism:
    """
    Factory function to create differential privacy mechanisms.

    Args:
        config: Differential privacy configuration

    Returns:
        Configured DP mechanism
    """
    if config.mechanism == DPMechanism.GAUSSIAN:
        return GaussianMechanism(config)
    elif config.mechanism == DPMechanism.LAPLACE:
        return LaplaceMechanism(config)
    elif config.mechanism == DPMechanism.DISCRETE_GAUSSIAN:
        return DiscreteGaussianMechanism(config)
    else:
        raise ValueError(f"Unknown DP mechanism: {config.mechanism}")


class DifferentialPrivacyManager:
    """
    Main coordinator for differential privacy operations.

    Handles clipping, noise addition, and privacy accounting in a unified interface.
    Designed to integrate with existing aggregation strategies.
    """

    def __init__(
        self,
        config: DifferentialPrivacyConfig,
        privacy_accountant: Optional[PrivacyAccountant] = None,
    ):
        self.config = config
        self.privacy_accountant = privacy_accountant

        # Initialize components
        self.clipper = ParameterClipper(
            strategy=config.clipping_strategy,
            initial_norm=config.clipping_norm or 1.0,
            target_quantile=config.target_quantile,
        )

        self.mechanism = create_dp_mechanism(config)

        self.logger = logging.getLogger("murmura.privacy.manager")

        # Statistics tracking
        self.round_count = 0
        self.total_privacy_spent = (0.0, 0.0)

    def apply_differential_privacy(
        self,
        parameters_list: List[Dict[str, Any]],
        round_number: int,
        sampling_rate: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Apply differential privacy to a list of parameters.

        This is the main entry point used by aggregation strategies.

        Args:
            parameters_list: List of parameter dictionaries from clients
            round_number: Current training round
            sampling_rate: Client sampling rate (for privacy amplification)

        Returns:
            Tuple of (processed_parameters, privacy_metadata)
        """
        self.round_count += 1
        metadata = {
            "round": round_number,
            "original_count": len(parameters_list),
            "sampling_rate": sampling_rate,
            "clipping_stats": [],
            "privacy_spent": (0.0, 0.0),
        }

        if not parameters_list:
            self.logger.warning("Empty parameters list provided to DP manager")
            return parameters_list, metadata

        # Apply differential privacy based on configuration
        if self.config.is_client_side():
            # Local DP: each client adds noise independently (already done)
            processed_params = parameters_list
            self.logger.debug("Local DP: using pre-noised client parameters")
        else:
            # Central DP: clip and add noise at server
            processed_params = []

            for i, params in enumerate(parameters_list):
                # Step 1: Clip parameters
                clipping_result = self.clipper.clip_parameters(params)
                metadata["clipping_stats"].append(
                    {
                        "client_id": i,
                        "original_norm": clipping_result.original_norm,
                        "clipped_norm": clipping_result.clipping_norm,
                        "was_clipped": clipping_result.was_clipped,
                        "clipping_ratio": clipping_result.clipping_ratio,
                    }
                )

                processed_params.append(clipping_result.clipped_params)

            # Step 2: Add noise (will be done after aggregation for server-side DP)
            self.logger.debug(
                f"Central DP: clipped {len(processed_params)} parameter sets"
            )

        # Step 3: Update privacy accounting
        if self.privacy_accountant is not None:
            privacy_cost = self.mechanism.get_privacy_spent(
                num_queries=1, sampling_rate=sampling_rate
            )

            try:
                self.privacy_accountant.check_and_add_mechanism(
                    mechanism_epsilon=privacy_cost[0],
                    mechanism_delta=privacy_cost[1],
                    mechanism_name=self.config.mechanism.value,
                    round_number=round_number,
                    sampling_rate=sampling_rate,
                    clipping_norm=self.config.clipping_norm,
                )

                metadata["privacy_spent"] = privacy_cost
                self.total_privacy_spent = (
                    self.privacy_accountant.get_current_privacy_spent()
                )

            except Exception as e:
                self.logger.error(f"Privacy accounting failed: {e}")
                raise

        return processed_params, metadata

    def add_noise_to_aggregated_parameters(
        self, aggregated_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add noise to aggregated parameters (for server-side DP).

        This should be called after aggregation but before model update.
        """
        if not self.config.is_central_dp():
            self.logger.warning(
                "add_noise_to_aggregated_parameters called but not using central DP"
            )
            return aggregated_params

        sensitivity = self.config.clipping_norm or 1.0
        noisy_params = self.mechanism.add_noise(aggregated_params, sensitivity)

        self.logger.debug("Added noise to aggregated parameters for central DP")
        return noisy_params

    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get comprehensive privacy summary."""
        summary = {
            "config": {
                "mechanism": self.config.mechanism.value,
                "noise_application": self.config.noise_application.value,
                "epsilon": self.config.epsilon,
                "delta": self.config.delta,
                "clipping_strategy": self.config.clipping_strategy.value,
                "clipping_norm": self.config.clipping_norm,
            },
            "rounds_processed": self.round_count,
            "total_privacy_spent": self.total_privacy_spent,
            "clipping_stats": self.clipper.get_clipping_stats(),
        }

        if self.privacy_accountant is not None:
            summary["accountant"] = self.privacy_accountant.get_privacy_summary()

        return summary
