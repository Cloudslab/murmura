from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from murmura.aggregation.aggregation_config import AggregationConfig
from murmura.aggregation.strategy_factory import AggregationStrategyFactory
from murmura.aggregation.strategy_interface import AggregationStrategy
from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.privacy.dp_config import DifferentialPrivacyConfig
from murmura.privacy.dp_aggregation_strategies import create_dp_enhanced_strategy
from murmura.privacy.privacy_accountant import create_privacy_accountant, PrivacyAccountant


class DPOrchestrationConfig(OrchestrationConfig):
    """
    Extended orchestration configuration with differential privacy support.

    This extends the base OrchestrationConfig to include DP parameters
    while maintaining compatibility with existing configurations.
    """

    # Differential privacy configuration
    differential_privacy: Optional[DifferentialPrivacyConfig] = Field(
        default=None,
        description="Differential privacy configuration. If None, DP is disabled."
    )

    # Privacy budget management
    total_privacy_epsilon: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Total privacy budget for the entire training process. "
                    "If None, uses differential_privacy.epsilon * total_rounds"
    )

    total_privacy_delta: Optional[float] = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description="Total failure probability for the entire training process"
    )

    # Enhanced monitoring
    enable_privacy_dashboard: bool = Field(
        default=True,
        description="Enable real-time privacy monitoring dashboard"
    )

    def is_dp_enabled(self) -> bool:
        """Check if differential privacy is enabled."""
        return self.differential_privacy is not None

    def get_total_privacy_budget(self) -> tuple[float, float]:
        """Get total privacy budget as (epsilon, delta)."""
        if not self.is_dp_enabled():
            return (0.0, 0.0)

        # Calculate total epsilon
        if self.total_privacy_epsilon is not None:
            total_eps = self.total_privacy_epsilon
        else:
            # Default: per-round epsilon * total rounds
            rounds = self.differential_privacy.total_rounds or 100
            total_eps = self.differential_privacy.epsilon * rounds

        # Calculate total delta
        if self.total_privacy_delta is not None:
            total_delta = self.total_privacy_delta
        else:
            total_delta = self.differential_privacy.delta or 1e-5

        return (total_eps, total_delta)


class DPAggregationStrategyFactory(AggregationStrategyFactory):
    """
    Enhanced aggregation strategy factory with differential privacy support.

    This extends the base factory to automatically wrap aggregation strategies
    with differential privacy capabilities when DP is enabled.
    """

    @staticmethod
    def create(config: AggregationConfig,
               topology_config: Optional[Any] = None,
               dp_config: Optional[DifferentialPrivacyConfig] = None,
               privacy_accountant: Optional[PrivacyAccountant] = None) -> AggregationStrategy:
        """
        Create aggregation strategy with optional differential privacy.

        Args:
            config: Base aggregation configuration
            topology_config: Topology configuration for compatibility checking
            dp_config: Optional differential privacy configuration
            privacy_accountant: Optional privacy accountant

        Returns:
            Aggregation strategy, optionally enhanced with DP
        """
        # Create base strategy using parent factory
        base_strategy = AggregationStrategyFactory.create(config, topology_config)

        # Enhance with DP if configured
        if dp_config is not None:
            return create_dp_enhanced_strategy(base_strategy, dp_config, privacy_accountant)
        else:
            return base_strategy


class DPClusterManager:
    """
    Enhanced cluster manager with differential privacy support.

    This extends the base ClusterManager to handle DP-specific operations
    like privacy budget tracking and client-side noise addition.
    """

    def __init__(self, config: DPOrchestrationConfig):
        # Import here to avoid circular imports
        from murmura.orchestration.cluster_manager import ClusterManager

        self.base_manager = ClusterManager(config)
        self.dp_config = config.differential_privacy
        self.privacy_accountant: Optional[PrivacyAccountant] = None

        # Initialize privacy accountant if DP is enabled
        if config.is_dp_enabled():
            total_eps, total_delta = config.get_total_privacy_budget()
            self.privacy_accountant = create_privacy_accountant(
                accountant_type=self.dp_config.accountant.value,
                total_epsilon=total_eps,
                total_delta=total_delta
            )

    def set_aggregation_strategy(self, aggregation_config: AggregationConfig,
                                 topology_config: Optional[Any] = None) -> None:
        """Set aggregation strategy with DP enhancement if configured."""
        # Create DP-enhanced strategy
        strategy = DPAggregationStrategyFactory.create(
            config=aggregation_config,
            topology_config=topology_config,
            dp_config=self.dp_config,
            privacy_accountant=self.privacy_accountant
        )

        # Set the strategy on the base manager
        self.base_manager.aggregation_strategy = strategy
        self.base_manager._initialize_coordinator()

    def aggregate_model_parameters(self, weights: Optional[List[float]] = None,
                                   round_number: Optional[int] = None,
                                   sampling_rate: Optional[float] = None) -> Dict[str, Any]:
        """
        Aggregate model parameters with DP support.

        Args:
            weights: Client weights
            round_number: Current round number
            sampling_rate: Client sampling rate for privacy amplification

        Returns:
            Aggregated parameters
        """
        if not hasattr(self.base_manager, 'topology_coordinator') or self.base_manager.topology_coordinator is None:
            raise ValueError("Topology coordinator not initialized. Call set_aggregation_strategy first.")

        # Enhanced aggregation with DP parameters
        if hasattr(self.base_manager.aggregation_strategy, 'aggregate'):
            # Check if this is a DP-enhanced strategy
            if hasattr(self.base_manager.aggregation_strategy, 'dp_config'):
                # Pass additional DP parameters
                return self.base_manager.aggregation_strategy.aggregate(
                    parameters_list=self._collect_client_parameters(),
                    weights=weights,
                    round_number=round_number,
                    sampling_rate=sampling_rate
                )
            else:
                # Standard aggregation
                return self.base_manager.topology_coordinator.coordinate_aggregation(weights)
        else:
            return self.base_manager.topology_coordinator.coordinate_aggregation(weights)

    def _collect_client_parameters(self) -> List[Dict[str, Any]]:
        """Collect parameters from all clients."""
        import ray

        parameter_tasks = []
        for actor in self.base_manager.actors:
            parameter_tasks.append(actor.get_model_parameters.remote())

        return ray.get(parameter_tasks)

    def apply_client_side_dp(self, noise_scale: float) -> None:
        """
        Apply client-side differential privacy (Local DP).

        This instructs all clients to add noise to their parameters
        before transmission, implementing Local DP.
        """
        if not self.dp_config or not self.dp_config.is_local_dp():
            raise ValueError("Client-side DP requires Local DP configuration")

        import ray

        # Instruct all clients to add noise
        noise_tasks = []
        for actor in self.base_manager.actors:
            noise_tasks.append(actor.apply_local_dp_noise.remote(noise_scale))

        # Wait for all clients to add noise
        ray.get(noise_tasks)

    def get_privacy_summary(self) -> Dict[str, Any]:
        """Get comprehensive privacy summary from all components."""
        summary = {"dp_enabled": self.dp_config is not None}

        if self.dp_config is not None:
            summary["config"] = {
                "mechanism": self.dp_config.mechanism.value,
                "epsilon": self.dp_config.epsilon,
                "delta": self.dp_config.delta,
                "noise_application": self.dp_config.noise_application.value,
                "clipping_strategy": self.dp_config.clipping_strategy.value
            }

            if self.privacy_accountant is not None:
                summary["accountant"] = self.privacy_accountant.get_privacy_summary()

            # Get strategy-specific privacy info
            if hasattr(self.base_manager.aggregation_strategy, 'get_privacy_summary'):
                summary["strategy"] = self.base_manager.aggregation_strategy.get_privacy_summary()

        return summary

    # Delegate other methods to base manager
    def __getattr__(self, name):
        """Delegate unknown attributes to base manager."""
        return getattr(self.base_manager, name)


class DPVirtualClientActor:
    """
    Enhanced virtual client actor with differential privacy support.

    This extends the base VirtualClientActor to support client-side DP operations
    for Local Differential Privacy scenarios.
    """

    def __init__(self, base_actor, dp_config: Optional[DifferentialPrivacyConfig] = None):
        self.base_actor = base_actor
        self.dp_config = dp_config
        self.local_privacy_accountant: Optional[PrivacyAccountant] = None

        # Initialize local privacy accountant for user-level privacy
        if dp_config and dp_config.per_client_clipping:
            self.local_privacy_accountant = create_privacy_accountant(
                accountant_type=dp_config.accountant.value,
                total_epsilon=dp_config.epsilon,
                total_delta=dp_config.delta or 1e-5
            )

    def apply_local_dp_noise(self, noise_scale: float, round_number: int = 0) -> Dict[str, Any]:
        """
        Apply local differential privacy noise to client parameters.

        Args:
            noise_scale: Scale of noise to add
            round_number: Current round number

        Returns:
            Metadata about noise application
        """
        if not self.dp_config or not self.dp_config.is_local_dp():
            raise ValueError("Local DP noise requires client-side DP configuration")

        # Get current model parameters
        import ray
        current_params = ray.get(self.base_actor.get_model_parameters.remote())

        # Apply clipping and noise
        from murmura.privacy.dp_mechanisms import DifferentialPrivacyManager

        dp_manager = DifferentialPrivacyManager(self.dp_config, self.local_privacy_accountant)

        # Apply DP (clipping + noise)
        processed_params, metadata = dp_manager.apply_differential_privacy(
            parameters_list=[current_params],
            round_number=round_number
        )

        # Update model with noisy parameters
        if processed_params:
            ray.get(self.base_actor.set_model_parameters.remote(processed_params[0]))

        return metadata

    def get_local_privacy_summary(self) -> Dict[str, Any]:
        """Get local privacy summary for this client."""
        summary = {"local_dp_enabled": self.dp_config is not None}

        if self.dp_config:
            summary["config"] = {
                "mechanism": self.dp_config.mechanism.value,
                "epsilon": self.dp_config.epsilon,
                "delta": self.dp_config.delta
            }

            if self.local_privacy_accountant:
                summary["accountant"] = self.local_privacy_accountant.get_privacy_summary()

        return summary

    # Delegate other methods to base actor
    def __getattr__(self, name):
        """Delegate unknown attributes to base actor."""
        return getattr(self.base_actor, name)


def integrate_dp_with_learning_process(learning_process_class):
    """
    Decorator to integrate differential privacy with learning processes.

    This modifies learning process classes to support DP-enhanced training
    while maintaining compatibility with existing code.
    """

    class DPEnhancedLearningProcess(learning_process_class):
        """DP-enhanced learning process."""

        def __init__(self, config, dataset, model):
            # Ensure we have DP configuration
            if hasattr(config, 'differential_privacy') and config.differential_privacy:
                self.dp_enabled = True
                self.dp_config = config.differential_privacy
            else:
                self.dp_enabled = False
                self.dp_config = None

            super().__init__(config, dataset, model)

        def initialize(self, num_actors, topology_config, aggregation_config, partitioner):
            """Initialize with DP-enhanced cluster manager."""
            if self.dp_enabled:
                # Use DP-enhanced cluster manager
                from murmura.privacy.dp_integration import DPClusterManager
                self.cluster_manager = DPClusterManager(self.config)

                # Create actors and set up aggregation with DP
                self.cluster_manager.create_actors(num_actors, topology_config)
                self.cluster_manager.set_aggregation_strategy(aggregation_config, topology_config)
            else:
                # Use standard initialization
                super().initialize(num_actors, topology_config, aggregation_config, partitioner)

        def execute(self) -> Dict[str, Any]:
            """Execute learning process with privacy monitoring."""
            # Get initial privacy summary
            if self.dp_enabled and hasattr(self.cluster_manager, 'get_privacy_summary'):
                initial_privacy = self.cluster_manager.get_privacy_summary()
                self.logger.info(f"Starting training with DP: {initial_privacy['config']}")

            # Execute base learning process
            results = super().execute()

            # Add privacy summary to results
            if self.dp_enabled and hasattr(self.cluster_manager, 'get_privacy_summary'):
                results['privacy_summary'] = self.cluster_manager.get_privacy_summary()

                # Log final privacy consumption
                final_privacy = results['privacy_summary']
                if 'accountant' in final_privacy:
                    spent = final_privacy['accountant']['spent']
                    total = final_privacy['accountant']['total_budget']
                    self.logger.info(
                        f"Training completed. Privacy spent: ε={spent['epsilon']:.4f}/{total['epsilon']:.4f}, "
                        f"δ={spent['delta']:.2e}/{total['delta']:.2e}"
                    )

            return results

    return DPEnhancedLearningProcess


# Example usage integration functions
def create_dp_federated_learning_process(config: DPOrchestrationConfig, dataset, model):
    """
    Convenience function to create a DP-enhanced federated learning process.

    Args:
        config: DP-enabled orchestration configuration
        dataset: Training dataset
        model: Model to train

    Returns:
        DP-enhanced federated learning process
    """
    from murmura.orchestration.learning_process.federated_learning_process import FederatedLearningProcess

    DPFederatedLearningProcess = integrate_dp_with_learning_process(FederatedLearningProcess)
    return DPFederatedLearningProcess(config, dataset, model)


def create_dp_decentralized_learning_process(config: DPOrchestrationConfig, dataset, model):
    """
    Convenience function to create a DP-enhanced decentralized learning process.

    Args:
        config: DP-enabled orchestration configuration (must use Local DP)
        dataset: Training dataset
        model: Model to train

    Returns:
        DP-enhanced decentralized learning process
    """
    # Validate that decentralized learning uses Local DP
    if config.is_dp_enabled() and config.differential_privacy.is_central_dp():
        raise ValueError(
            "Decentralized learning requires Local DP (client-side noise). "
            "Set noise_application=CLIENT_SIDE in differential_privacy config."
        )

    from murmura.orchestration.learning_process.decentralized_learning_process import DecentralizedLearningProcess

    DPDecentralizedLearningProcess = integrate_dp_with_learning_process(DecentralizedLearningProcess)
    return DPDecentralizedLearningProcess(config, dataset, model)
