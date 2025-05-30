import logging
from typing import List, Dict, Any, Optional

import numpy as np

from murmura.aggregation.strategy_interface import AggregationStrategy
from murmura.aggregation.coordination_mode import CoordinationMode
from murmura.privacy.dp_config import DifferentialPrivacyConfig
from murmura.privacy.dp_mechanisms import DifferentialPrivacyManager
from murmura.privacy.privacy_accountant import (
    PrivacyAccountant,
    create_privacy_accountant,
)


class DPAggregationStrategyMixin:
    """
    Mixin class that adds differential privacy capabilities to aggregation strategies.

    This follows the decorator pattern, allowing any aggregation strategy to be
    enhanced with differential privacy without modifying the base strategy.
    """

    def __init__(
        self,
        base_strategy: AggregationStrategy,
        dp_config: DifferentialPrivacyConfig,
        privacy_accountant: Optional[PrivacyAccountant] = None,
    ):
        """
        Initialize DP-enhanced aggregation strategy.

        Args:
            base_strategy: The underlying aggregation strategy (FedAvg, etc.)
            dp_config: Differential privacy configuration
            privacy_accountant: Optional privacy accountant for budget tracking
        """
        self.base_strategy = base_strategy
        self.dp_config = dp_config
        self.logger = logging.getLogger(f"murmura.privacy.{self.__class__.__name__}")

        # Create privacy accountant if not provided
        if privacy_accountant is None and dp_config.enable_privacy_monitoring:
            privacy_accountant = create_privacy_accountant(
                accountant_type=dp_config.accountant.value,
                total_epsilon=dp_config.epsilon
                * (dp_config.total_rounds or 100),  # Total budget
                total_delta=dp_config.delta if dp_config.delta is not None else 1e-5,
            )

        # Initialize differential privacy manager
        self.dp_manager = DifferentialPrivacyManager(dp_config, privacy_accountant)

        # Track round number for privacy accounting
        self.current_round = 0

    @property
    def coordination_mode(self) -> CoordinationMode:
        """Inherit coordination mode from base strategy."""
        return self.base_strategy.coordination_mode

    def aggregate(
        self,
        parameters_list: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        round_number: Optional[int] = None,
        sampling_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate parameters with differential privacy.

        The DP application depends on the configuration:
        - Client-side DP: Parameters already have noise, just aggregate normally
        - Server-side DP: Clip parameters, aggregate, then add noise
        """
        if round_number is not None:
            self.current_round = round_number
        else:
            self.current_round += 1

        # Apply differential privacy preprocessing
        dp_parameters, dp_metadata = self.dp_manager.apply_differential_privacy(
            parameters_list=parameters_list,
            round_number=self.current_round,
            sampling_rate=sampling_rate,
        )

        self.logger.debug(
            f"Round {self.current_round}: Applied DP preprocessing to {len(dp_parameters)} parameter sets"
        )

        # Perform base aggregation
        aggregated_params = self.base_strategy.aggregate(dp_parameters, weights)

        # For server-side DP, add noise to aggregated parameters
        if self.dp_config.is_central_dp():
            aggregated_params = self.dp_manager.add_noise_to_aggregated_parameters(
                aggregated_params
            )
            self.logger.debug(
                f"Round {self.current_round}: Added noise to aggregated parameters"
            )

        # Log privacy statistics
        if self.dp_config.enable_privacy_monitoring:
            privacy_summary = self.dp_manager.get_privacy_summary()
            current_spent = privacy_summary["total_privacy_spent"]
            self.logger.info(
                f"Round {self.current_round} privacy: "
                f"ε={current_spent[0]:.4f}, δ={current_spent[1]:.2e}"
            )

        return aggregated_params

    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get comprehensive privacy summary."""
        return self.dp_manager.get_privacy_summary()


class DPFedAvg(DPAggregationStrategyMixin, AggregationStrategy):
    """
    Differential Privacy enhanced Federated Averaging (DP-FedAvg).

    This is the most common DP aggregation strategy in federated learning,
    following the DP-SGD algorithm adapted for federated settings.
    """

    def __init__(
        self,
        base_fedavg_strategy: AggregationStrategy,
        dp_config: DifferentialPrivacyConfig,
        privacy_accountant: Optional[PrivacyAccountant] = None,
    ):
        """
        Initialize DP-FedAvg strategy.

        Args:
            base_fedavg_strategy: Base FedAvg strategy to enhance
            dp_config: Differential privacy configuration
            privacy_accountant: Optional privacy accountant
        """
        super().__init__(base_fedavg_strategy, dp_config, privacy_accountant)

        # DP-FedAvg typically uses centralized coordination
        if self.coordination_mode != CoordinationMode.CENTRALIZED:
            self.logger.warning(
                "DP-FedAvg typically works best with centralized coordination"
            )


class DPTrimmedMean(DPAggregationStrategyMixin, AggregationStrategy):
    """
    Differential Privacy enhanced Trimmed Mean aggregation.

    Combines Byzantine robustness of trimmed mean with differential privacy.
    Particularly useful when clients might be malicious or have corrupted data.
    """

    def __init__(
        self,
        base_trimmed_mean_strategy: AggregationStrategy,
        dp_config: DifferentialPrivacyConfig,
        privacy_accountant: Optional[PrivacyAccountant] = None,
    ):
        """
        Initialize DP-TrimmedMean strategy.

        Args:
            base_trimmed_mean_strategy: Base TrimmedMean strategy to enhance
            dp_config: Differential privacy configuration
            privacy_accountant: Optional privacy accountant
        """
        super().__init__(base_trimmed_mean_strategy, dp_config, privacy_accountant)

        # Enhanced logging for robust aggregation
        self.logger.info(
            "Initialized DP-TrimmedMean: combining privacy with Byzantine robustness"
        )


class DPGossipAvg(DPAggregationStrategyMixin, AggregationStrategy):
    """
    Differential Privacy enhanced Gossip Averaging for decentralized learning.

    Implements Local Differential Privacy (LDP) where each node adds noise
    before sharing parameters with neighbors. This is the only DP approach
    that works with decentralized topologies.
    """

    def __init__(
        self,
        base_gossip_strategy: AggregationStrategy,
        dp_config: DifferentialPrivacyConfig,
        privacy_accountant: Optional[PrivacyAccountant] = None,
    ):
        """
        Initialize DP-GossipAvg strategy.

        Args:
            base_gossip_strategy: Base GossipAvg strategy to enhance
            dp_config: Differential privacy configuration
            privacy_accountant: Optional privacy accountant
        """
        # Force client-side DP for decentralized learning
        if dp_config.is_central_dp():
            dp_config.noise_application = dp_config.noise_application.CLIENT_SIDE
            logging.getLogger("murmura.privacy").warning(
                "Decentralized learning requires client-side DP. Changed noise_application to CLIENT_SIDE."
            )

        super().__init__(base_gossip_strategy, dp_config, privacy_accountant)

        # Decentralized DP requires special handling
        self.logger.info(
            "Initialized DP-GossipAvg: using Local DP for decentralized learning"
        )

    def aggregate(
        self,
        parameters_list: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        round_number: Optional[int] = None,
        sampling_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate with Local DP for decentralized learning.

        In decentralized settings, each node must add noise to its own parameters
        before sharing with neighbors. This is typically done at the client level.
        """
        # For decentralized DP, we assume parameters already have client-side noise
        # The aggregation just combines the noisy parameters
        return super().aggregate(parameters_list, weights, round_number, sampling_rate)


class AdaptiveDPStrategy(AggregationStrategy):
    """
    Adaptive differential privacy strategy that adjusts privacy parameters
    based on training progress and convergence state.

    This implements recent research on adaptive privacy budget allocation
    for better utility-privacy trade-offs.
    """

    def __init__(
        self,
        base_strategy: AggregationStrategy,
        initial_dp_config: DifferentialPrivacyConfig,
        adaptation_schedule: str = "linear",
        privacy_accountant: Optional[PrivacyAccountant] = None,
    ):
        """
        Initialize adaptive DP strategy.

        Args:
            base_strategy: Base aggregation strategy
            initial_dp_config: Initial DP configuration
            adaptation_schedule: How to adapt privacy over time ("linear", "exponential", "convergence-based")
            privacy_accountant: Privacy accountant
        """
        self.base_strategy = base_strategy
        self.initial_dp_config = initial_dp_config
        self.adaptation_schedule = adaptation_schedule
        self.privacy_accountant = privacy_accountant
        self.logger = logging.getLogger("murmura.privacy.adaptive")

        # Track convergence metrics for adaptive scheduling
        self.convergence_history: List[float] = []
        self.current_round = 0
        self.total_budget_allocated = 0.0

        # Create multiple DP managers for different phases
        self.dp_managers = {}
        self._initialize_adaptive_managers()

    def _initialize_adaptive_managers(self):
        """Initialize DP managers for different adaptation phases."""
        # Early phase: higher privacy budget (faster convergence)
        early_config = self.initial_dp_config.copy()
        early_config.epsilon = self.initial_dp_config.epsilon * 2.0  # More budget early

        # Late phase: lower privacy budget (refinement)
        late_config = self.initial_dp_config.copy()
        late_config.epsilon = self.initial_dp_config.epsilon * 0.5  # Less budget late

        self.dp_managers["early"] = DifferentialPrivacyManager(
            early_config, self.privacy_accountant
        )
        self.dp_managers["late"] = DifferentialPrivacyManager(
            late_config, self.privacy_accountant
        )
        self.dp_managers["standard"] = DifferentialPrivacyManager(
            self.initial_dp_config, self.privacy_accountant
        )

    @property
    def coordination_mode(self) -> CoordinationMode:
        """Inherit coordination mode from base strategy."""
        return self.base_strategy.coordination_mode

    def aggregate(
        self,
        parameters_list: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        convergence_metric: Optional[float] = None,
        round_number: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate with adaptive privacy budget allocation.

        Args:
            parameters_list: Parameters from clients
            weights: Client weights
            convergence_metric: Current convergence metric (e.g., loss, accuracy change)
            round_number: Current round number
        """
        if round_number is not None:
            self.current_round = round_number
        else:
            self.current_round += 1

        # Track convergence for adaptation
        if convergence_metric is not None:
            self.convergence_history.append(convergence_metric)

        # Select appropriate DP manager based on adaptation schedule
        current_manager = self._select_dp_manager()

        # Apply adaptive DP
        dp_parameters, dp_metadata = current_manager.apply_differential_privacy(
            parameters_list=parameters_list, round_number=self.current_round
        )

        # Perform base aggregation
        aggregated_params = self.base_strategy.aggregate(dp_parameters, weights)

        # Add noise if using central DP
        if self.initial_dp_config.is_central_dp():
            aggregated_params = current_manager.add_noise_to_aggregated_parameters(
                aggregated_params
            )

        self.logger.debug(
            f"Adaptive DP round {self.current_round}: used {current_manager.__class__.__name__}"
        )

        return aggregated_params

    def _select_dp_manager(self) -> DifferentialPrivacyManager:
        """Select appropriate DP manager based on adaptation schedule."""
        total_rounds = self.initial_dp_config.total_rounds or 100

        if self.adaptation_schedule == "linear":
            # Linear decrease in privacy budget over time
            if self.current_round < total_rounds * 0.3:
                return self.dp_managers["early"]
            elif self.current_round > total_rounds * 0.7:
                return self.dp_managers["late"]
            else:
                return self.dp_managers["standard"]

        elif self.adaptation_schedule == "exponential":
            # Exponential decay of privacy budget
            decay_factor = 0.95**self.current_round
            if decay_factor > 0.7:
                return self.dp_managers["early"]
            elif decay_factor < 0.3:
                return self.dp_managers["late"]
            else:
                return self.dp_managers["standard"]

        elif self.adaptation_schedule == "convergence-based":
            # Adapt based on convergence rate
            if len(self.convergence_history) >= 5:
                recent_improvement = abs(
                    self.convergence_history[-1] - self.convergence_history[-5]
                )
                if recent_improvement > 0.01:  # Fast convergence
                    return self.dp_managers["early"]
                elif recent_improvement < 0.001:  # Slow convergence
                    return self.dp_managers["late"]

            return self.dp_managers["standard"]

        else:
            return self.dp_managers["standard"]


def create_dp_enhanced_strategy(
    base_strategy: AggregationStrategy,
    dp_config: DifferentialPrivacyConfig,
    privacy_accountant: Optional[PrivacyAccountant] = None,
) -> AggregationStrategy:
    """
    Factory function to create DP-enhanced aggregation strategies.

    This automatically wraps any aggregation strategy with differential privacy
    capabilities based on the strategy type and DP configuration.

    Args:
        base_strategy: Base aggregation strategy to enhance
        dp_config: Differential privacy configuration
        privacy_accountant: Optional privacy accountant

    Returns:
        DP-enhanced aggregation strategy
    """
    strategy_name = base_strategy.__class__.__name__.lower()

    if "fedavg" in strategy_name:
        return DPFedAvg(base_strategy, dp_config, privacy_accountant)
    elif "trimmed" in strategy_name:
        return DPTrimmedMean(base_strategy, dp_config, privacy_accountant)
    elif "gossip" in strategy_name:
        return DPGossipAvg(base_strategy, dp_config, privacy_accountant)
    else:
        # Generic DP enhancement for any strategy
        class GenericDPStrategy(DPAggregationStrategyMixin, AggregationStrategy):
            def __init__(self, base_strat, dp_conf, privacy_acc):
                super().__init__(base_strat, dp_conf, privacy_acc)

        return GenericDPStrategy(base_strategy, dp_config, privacy_accountant)


class SecureAggregationDPStrategy(AggregationStrategy):
    """
    Combines secure aggregation with differential privacy.

    This implements the distributed differential privacy approach where
    clients collectively generate noise without revealing individual contributions.
    Provides the utility benefits of central DP without requiring server trust.
    """

    def __init__(
        self,
        base_strategy: AggregationStrategy,
        dp_config: DifferentialPrivacyConfig,
        privacy_accountant: Optional[PrivacyAccountant] = None,
        secure_aggregation_threshold: int = 3,
    ):
        """
        Initialize secure aggregation with DP.

        Args:
            base_strategy: Base aggregation strategy
            dp_config: DP configuration
            privacy_accountant: Privacy accountant
            secure_aggregation_threshold: Minimum clients needed for secure aggregation
        """
        self.base_strategy = base_strategy
        self.dp_config = dp_config
        self.privacy_accountant = privacy_accountant
        self.secure_threshold = secure_aggregation_threshold
        self.logger = logging.getLogger("murmura.privacy.secure_aggregation")

        # This is a simplified implementation - production would need proper cryptographic protocols
        self.logger.warning(
            "SecureAggregationDPStrategy is a simplified implementation. "
            "Production use requires proper cryptographic secure aggregation protocols."
        )

    @property
    def coordination_mode(self) -> CoordinationMode:
        """Secure aggregation typically uses centralized coordination."""
        return CoordinationMode.CENTRALIZED

    def aggregate(
        self,
        parameters_list: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate with secure aggregation and distributed DP.

        In a full implementation, this would:
        1. Have clients collaboratively generate DP noise using secure aggregation
        2. Add the collective noise to the aggregated parameters
        3. Provide central DP guarantees without server trust

        This simplified version demonstrates the concept.
        """
        if len(parameters_list) < self.secure_threshold:
            raise ValueError(
                f"Secure aggregation requires at least {self.secure_threshold} clients"
            )

        # Simulate distributed noise generation (in reality, this would be cryptographic)
        self.logger.debug("Simulating distributed DP noise generation...")

        # Perform base aggregation
        aggregated_params = self.base_strategy.aggregate(parameters_list, weights)

        # Add collectively generated noise (simplified)
        if self.dp_config.mechanism.value == "gaussian":
            noise_scale = self.dp_config.get_noise_scale()

            for param_name, param_value in aggregated_params.items():
                param_array = np.asarray(param_value)
                # Simulate noise that was collectively generated by clients
                collective_noise = np.random.normal(0, noise_scale, param_array.shape)
                aggregated_params[param_name] = param_array + collective_noise

        self.logger.debug(
            "Applied distributed differential privacy via secure aggregation"
        )

        return aggregated_params
