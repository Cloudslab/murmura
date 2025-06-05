import logging
from typing import List, Dict, Any, Optional

import numpy as np

from murmura.aggregation.strategy_interface import AggregationStrategy
from murmura.aggregation.coordination_mode import CoordinationMode
from murmura.privacy.dp_config import DPConfig, DPMechanism


class DPFedAvg(AggregationStrategy):
    """
    Differentially private federated averaging with central DP.

    This strategy adds calibrated noise to the aggregated model parameters
    to provide differential privacy at the central server level.
    """

    coordination_mode = CoordinationMode.CENTRALIZED

    def __init__(self, dp_config: DPConfig):
        """
        Initialize DP FedAvg strategy.

        Args:
            dp_config: Differential privacy configuration
        """
        self.dp_config = dp_config
        self.logger = logging.getLogger("murmura.dp_aggregation.dp_fedavg")
        self.round_count = 0

        if not dp_config.enable_central_dp:
            self.logger.warning(
                "Central DP is disabled in config but DPFedAvg was requested. "
                "This will perform regular FedAvg without central noise."
            )

    def aggregate(
        self,
        parameters_list: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        round_number: Optional[int] = None,
        sampling_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate parameters with differential privacy.

        Args:
            parameters_list: List of client model parameters
            weights: Optional client weights (defaults to uniform)
            round_number: Current round number
            sampling_rate: Sampling rate for this round

        Returns:
            Aggregated parameters with DP noise
        """
        if not parameters_list:
            raise ValueError("Empty parameters list")

        self.round_count += 1
        if round_number is not None:
            self.round_count = round_number

        # Use uniform weights if not provided
        if weights is None:
            weights = [1.0 / len(parameters_list)] * len(parameters_list)

        if len(weights) != len(parameters_list):
            raise ValueError("Weights and parameters list must have same length")

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Perform weighted averaging
        aggregated_params = {}

        # Get parameter names from first client
        param_names = list(parameters_list[0].keys())

        for param_name in param_names:
            # Collect parameters for this layer
            param_arrays = [
                client_params[param_name] for client_params in parameters_list
            ]

            # Weighted average
            weighted_sum = np.zeros_like(param_arrays[0])
            for param_array, weight in zip(param_arrays, weights):
                weighted_sum += weight * param_array

            # Add DP noise if central DP is enabled
            if self.dp_config.enable_central_dp:
                noisy_params = self._add_dp_noise(weighted_sum, param_name)
                aggregated_params[param_name] = noisy_params
            else:
                aggregated_params[param_name] = weighted_sum

        self.logger.debug(
            f"Aggregated {len(parameters_list)} clients for round {self.round_count}"
        )

        return aggregated_params

    def _add_dp_noise(self, params: np.ndarray, param_name: str) -> np.ndarray:
        """
        Add differential privacy noise to parameters.

        Args:
            params: Parameter array
            param_name: Name of the parameter (for logging)

        Returns:
            Parameters with DP noise added
        """
        if not self.dp_config.enable_central_dp:
            return params

        # Compute noise scale based on sensitivity and privacy parameters
        # For central DP in federated learning, sensitivity is typically 2 * max_grad_norm
        sensitivity = 2.0 * self.dp_config.max_grad_norm

        # Noise multiplier for central DP (different from client DP)
        noise_multiplier = self.dp_config.noise_multiplier or 1.0

        if self.dp_config.mechanism == DPMechanism.GAUSSIAN:
            # Gaussian mechanism
            noise_scale = noise_multiplier * sensitivity
            noise = np.random.normal(0, noise_scale, params.shape)
        elif self.dp_config.mechanism == DPMechanism.LAPLACE:
            # Laplace mechanism
            noise_scale = noise_multiplier * sensitivity / np.sqrt(2)
            noise = np.random.laplace(0, noise_scale, params.shape)
        else:
            # Default to Gaussian
            noise_scale = noise_multiplier * sensitivity
            noise = np.random.normal(0, noise_scale, params.shape)

        noisy_params = params + noise

        self.logger.debug(
            f"Added {self.dp_config.mechanism.value} noise to {param_name} "
            f"(scale={noise_scale:.6f})"
        )

        return noisy_params


class DPSecureAggregation(AggregationStrategy):
    """
    Differentially private secure aggregation for decentralized learning.

    This strategy provides privacy-preserving aggregation for decentralized
    topologies by adding noise at each aggregation step.
    """

    coordination_mode = CoordinationMode.DECENTRALIZED

    def __init__(self, dp_config: DPConfig):
        """
        Initialize DP secure aggregation strategy.

        Args:
            dp_config: Differential privacy configuration
        """
        self.dp_config = dp_config
        self.logger = logging.getLogger("murmura.dp_aggregation.dp_secure")
        self.round_count = 0

    def aggregate(
        self,
        parameters_list: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        round_number: Optional[int] = None,
        sampling_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate parameters with secure DP for decentralized learning.

        Args:
            parameters_list: List of neighbor model parameters
            weights: Optional neighbor weights
            round_number: Current round number
            sampling_rate: Not used in decentralized setting

        Returns:
            Aggregated parameters with DP noise
        """
        if not parameters_list:
            raise ValueError("Empty parameters list")

        self.round_count += 1
        if round_number is not None:
            self.round_count = round_number

        # For decentralized learning, we typically use uniform weights
        # unless topology-specific weights are provided
        if weights is None:
            weights = [1.0 / len(parameters_list)] * len(parameters_list)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Perform weighted averaging with reduced noise for decentralized setting
        aggregated_params = {}
        param_names = list(parameters_list[0].keys())

        for param_name in param_names:
            # Collect parameters
            param_arrays = [
                client_params[param_name] for client_params in parameters_list
            ]

            # Weighted average
            weighted_sum = np.zeros_like(param_arrays[0])
            for param_array, weight in zip(param_arrays, weights):
                weighted_sum += weight * param_array

            # Add reduced noise for decentralized aggregation
            if (
                self.dp_config.enable_central_dp
            ):  # Using central_dp flag for aggregation DP
                noisy_params = self._add_decentralized_dp_noise(
                    weighted_sum, param_name, len(parameters_list)
                )
                aggregated_params[param_name] = noisy_params
            else:
                aggregated_params[param_name] = weighted_sum

        return aggregated_params

    def _add_decentralized_dp_noise(
        self, params: np.ndarray, param_name: str, num_neighbors: int
    ) -> np.ndarray:
        """
        Add DP noise calibrated for decentralized aggregation.

        Args:
            params: Parameter array
            param_name: Parameter name
            num_neighbors: Number of neighbors (affects noise calibration)

        Returns:
            Parameters with calibrated DP noise
        """
        # Reduced sensitivity for decentralized aggregation
        base_sensitivity = self.dp_config.max_grad_norm

        # Scale sensitivity based on number of neighbors
        # More neighbors = less noise needed per aggregation
        sensitivity = base_sensitivity / np.sqrt(num_neighbors)

        # Use reduced noise multiplier for decentralized setting
        noise_multiplier = (self.dp_config.noise_multiplier or 1.0) * 0.5

        if self.dp_config.mechanism == DPMechanism.GAUSSIAN:
            noise_scale = noise_multiplier * sensitivity
            noise = np.random.normal(0, noise_scale, params.shape)
        else:
            # Default to Gaussian for decentralized
            noise_scale = noise_multiplier * sensitivity
            noise = np.random.normal(0, noise_scale, params.shape)

        noisy_params = params + noise

        self.logger.debug(
            f"Added decentralized DP noise to {param_name} "
            f"(neighbors={num_neighbors}, scale={noise_scale:.6f})"
        )

        return noisy_params


class DPTrimmedMean(AggregationStrategy):
    """
    Differentially private trimmed mean aggregation.

    Combines robustness of trimmed mean with differential privacy.
    Useful when some clients might be adversarial or have outlier updates.
    """

    coordination_mode = CoordinationMode.CENTRALIZED

    def __init__(self, dp_config: DPConfig, trim_ratio: float = 0.1):
        """
        Initialize DP trimmed mean strategy.

        Args:
            dp_config: Differential privacy configuration
            trim_ratio: Fraction of extreme values to trim (0.0-0.5)
        """
        self.dp_config = dp_config
        self.trim_ratio = max(0.0, min(0.5, trim_ratio))
        self.logger = logging.getLogger("murmura.dp_aggregation.dp_trimmed_mean")
        self.round_count = 0

    def aggregate(
        self,
        parameters_list: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        round_number: Optional[int] = None,
        sampling_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate using DP trimmed mean.

        Args:
            parameters_list: List of client parameters
            weights: Client weights (ignored for trimmed mean)
            round_number: Current round number
            sampling_rate: Sampling rate

        Returns:
            Trimmed mean aggregated parameters with DP noise
        """
        if not parameters_list:
            raise ValueError("Empty parameters list")

        if len(parameters_list) < 3:
            self.logger.warning(
                "Trimmed mean requires at least 3 clients. "
                "Falling back to simple average."
            )
            return self._simple_average_with_dp(parameters_list)

        self.round_count += 1
        if round_number is not None:
            self.round_count = round_number

        aggregated_params = {}
        param_names = list(parameters_list[0].keys())

        for param_name in param_names:
            # Collect parameters for this layer
            param_arrays = [
                client_params[param_name] for client_params in parameters_list
            ]
            param_stack = np.stack(param_arrays, axis=0)

            # Compute trimmed mean
            trimmed_mean = self._compute_trimmed_mean(param_stack)

            # Add DP noise if enabled
            if self.dp_config.enable_central_dp:
                noisy_params = self._add_dp_noise(trimmed_mean, param_name)
                aggregated_params[param_name] = noisy_params
            else:
                aggregated_params[param_name] = trimmed_mean

        return aggregated_params

    def _compute_trimmed_mean(self, param_stack: np.ndarray) -> np.ndarray:
        """
        Compute trimmed mean along the client dimension.

        Args:
            param_stack: Stacked parameters [num_clients, ...param_shape]

        Returns:
            Trimmed mean parameters
        """
        num_clients = param_stack.shape[0]
        num_trim = int(num_clients * self.trim_ratio)

        if num_trim == 0:
            # No trimming needed
            return np.mean(param_stack, axis=0)

        # Sort along client dimension and trim extremes
        sorted_params = np.sort(param_stack, axis=0)

        # Remove num_trim elements from both ends
        trimmed_params = sorted_params[num_trim:-num_trim]

        # Compute mean of remaining parameters
        return np.mean(trimmed_params, axis=0)

    def _simple_average_with_dp(
        self, parameters_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fallback to simple average when too few clients"""
        aggregated_params = {}
        param_names = list(parameters_list[0].keys())

        for param_name in param_names:
            param_arrays = [
                client_params[param_name] for client_params in parameters_list
            ]
            avg_params = np.mean(param_arrays, axis=0)

            if self.dp_config.enable_central_dp:
                noisy_params = self._add_dp_noise(avg_params, param_name)
                aggregated_params[param_name] = noisy_params
            else:
                aggregated_params[param_name] = avg_params

        return aggregated_params

    def _add_dp_noise(self, params: np.ndarray, param_name: str) -> np.ndarray:
        """Add DP noise to trimmed mean parameters"""
        # Trimmed mean has different sensitivity than regular mean
        # Sensitivity is bounded by the trimming operation
        base_sensitivity = self.dp_config.max_grad_norm

        # Trimmed mean typically has lower sensitivity
        sensitivity = base_sensitivity * (1.0 - self.trim_ratio)

        noise_multiplier = self.dp_config.noise_multiplier or 1.0

        if self.dp_config.mechanism == DPMechanism.GAUSSIAN:
            noise_scale = noise_multiplier * sensitivity
            noise = np.random.normal(0, noise_scale, params.shape)
        else:
            noise_scale = noise_multiplier * sensitivity
            noise = np.random.normal(0, noise_scale, params.shape)

        return params + noise
