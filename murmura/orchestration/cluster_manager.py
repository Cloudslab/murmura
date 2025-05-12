from typing import Dict, Any, List, Optional

import ray

from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.aggregation.strategy_factory import AggregationStrategyFactory
from murmura.aggregation.strategy_interface import AggregationStrategy
from murmura.data_processing.dataset import MDataset
from murmura.model.model_interface import ModelInterface
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.network_management.topology_compatibility import (
    TopologyCompatibilityManager,
)
from murmura.network_management.topology_manager import TopologyManager
from murmura.node.client_actor import VirtualClientActor
from murmura.orchestration.topology_coordinator import TopologyCoordinator
from murmura.privacy.privacy_config import PrivacyConfig, PrivacyMode
from murmura.privacy.privacy_factory import PrivacyFactory
from murmura.privacy.privacy_manager import PrivacyManager


class ClusterManager:
    """
    Manages Ray cluster and virtual client actors
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.actors: List[Any] = []
        self.topology_manager: Optional[TopologyManager] = None
        self.aggregation_strategy: Optional[AggregationStrategy] = None
        self.topology_coordinator: Optional[TopologyCoordinator] = None
        self.privacy_manager: Optional[PrivacyManager] = None

        if not ray.is_initialized():
            ray.init(
                address=self.config["ray_address"] if "ray_address" in config else None
            )

    def create_actors(self, num_actors: int, topology: TopologyConfig) -> List[Any]:
        """
        Create pool of virtual client actors

        :param topology: topology config
        :param num_actors: Number of actors to create
        :return: List of actors
        """
        self.topology_manager = TopologyManager(num_actors, topology)
        self.actors = [
            VirtualClientActor.remote(f"client_{i}")  # type: ignore[attr-defined]
            for i in range(num_actors)
        ]
        self._apply_topology()

        if self.aggregation_strategy and self.topology_manager:
            self._initialize_coordinator()

        return self.actors

    def set_aggregation_strategy(self, aggregation_config: AggregationConfig) -> None:
        """
        Set the aggregation strategy for the cluster

        :param aggregation_config: Aggregation configuration
        """
        if self.topology_manager:
            self.aggregation_strategy = AggregationStrategyFactory.create(
                aggregation_config, self.topology_manager.config
            )
            self._initialize_coordinator()
        else:
            self.aggregation_strategy = AggregationStrategyFactory.create(
                aggregation_config
            )

    def distribute_data(
        self,
        data_partitions: List[List[int]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Distribute data partitions to actors in round-robin fashion

        :param data_partitions: List of data partitions
        :param metadata: Metadata dict

        :return: List of data receipt acknowledgments
        """
        resolved_metadata = metadata or {}

        results = []
        for idx, actor in enumerate(self.actors):
            partition_idx = idx % len(data_partitions)
            results.append(
                actor.receive_data.remote(
                    data_partitions[partition_idx],
                    {**resolved_metadata, "partition_idx": partition_idx},
                )
            )

        return ray.get(results)

    def distribute_dataset(
        self,
        dataset: MDataset,
        feature_columns: Optional[List[str]] = None,
        label_column: Optional[str] = None,
    ) -> None:
        """
        Distribute a dataset to all actors

        :param dataset: Dataset to distribute
        :param feature_columns: Feature columns to use
        :param label_column: Label column to use
        """
        for actor in self.actors:
            ray.get(
                actor.set_dataset.remote(
                    dataset, feature_columns=feature_columns, label_column=label_column
                )
            )

    def distribute_model(
        self,
        model: ModelInterface,
    ) -> None:
        """
        Distribute model structure and parameters to all actors

        :param model: Model to distribute
        """
        # Get model parameters to distribute
        parameters = model.get_parameters()

        # Set the model on each actor
        for actor in self.actors:
            # First set the model structure
            ray.get(actor.set_model.remote(model))
            # Then set the parameters
            ray.get(actor.set_model_parameters.remote(parameters))

    def train_models(self, **kwargs) -> List[Dict[str, float]]:
        """
        Train models on all actors

        :param kwargs: Training parameters
        :return: List of training metrics from all actors
        """
        results = []
        for actor in self.actors:
            results.append(actor.train_model.remote(**kwargs))
        return ray.get(results)

    def evaluate_models(self, **kwargs) -> List[Dict[str, float]]:
        """
        Evaluate models on all actors

        :param kwargs: Evaluation parameters
        :return: List of evaluation metrics from all actors
        """
        results = []
        for actor in self.actors:
            results.append(actor.evaluate_model.remote(**kwargs))
        return ray.get(results)

    def set_privacy_manager(self, privacy_config: PrivacyConfig) -> None:
        """
        Set the privacy manager for the cluster

        :param privacy_config: Privacy configuration
        """
        # Store number of actors and local epochs in privacy config params
        if privacy_config.params is None:
            privacy_config.params = {}

        # Add actors count for proper noise scaling in local DP
        privacy_config.params["actors"] = self.actors if hasattr(self, 'actors') else []

        # Add local epochs for proper privacy accounting
        privacy_config.params["local_epochs"] = self.config.get("epochs", 1)

        # Create privacy manager
        if self.topology_manager:
            self.privacy_manager = PrivacyFactory.create(
                privacy_config, self.topology_manager.config
            )
        else:
            self.privacy_manager = PrivacyFactory.create(privacy_config)

        # Initialize privacy manager with dataset info if available
        if 'batch_size' in self.config and 'data_partitions' in self.config:
            batch_size = self.config.get('batch_size', 32)
            data_partitions = self.config.get('data_partitions', [])
            total_samples = sum(len(p) for p in data_partitions) if data_partitions else 10000

            if self.privacy_manager and hasattr(self.privacy_manager, 'setup_privacy_accounting'):
                self.privacy_manager.setup_privacy_accounting(total_samples, batch_size)

    def aggregate_model_parameters(
        self, weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate model parameters from all actors using the configured aggregation strategy,
        with privacy guarantees if enabled.

        :param weights: Optional list of weights for each actor's parameters
        :return: Aggregated model parameters
        """
        if not self.aggregation_strategy:
            raise ValueError(
                "Aggregation strategy not set. Call set_aggregation_strategy first."
            )

        if not self.topology_coordinator:
            raise ValueError(
                "Topology coordinator not initialized. Make sure create_actors and set_aggregation_strategy have been "
                "called."
            )

        # If privacy is enabled, get all client parameters first to allow adaptive clipping
        if self.privacy_manager and self.privacy_manager.config.enabled:
            # Collect parameters from all actors
            all_params = []
            for actor in self.actors:
                params = ray.get(actor.get_model_parameters.remote())
                all_params.append(params)

            # Update adaptive clipping norms if needed
            self.privacy_manager.update_clipping_norms(all_params)

            # For Local DP, have clients apply privacy before aggregation
            if self.privacy_manager.config.privacy_mode == PrivacyMode.LOCAL:
                print(
                    f"Applying Local DP with noise multiplier: {self.privacy_manager.config.noise_multiplier:.4f}"
                )
                # Apply privacy at the client level before sending to server
                for i, actor in enumerate(self.actors):
                    # Apply local differential privacy
                    privatized_params = self.privacy_manager.privatize_parameters(
                        all_params[i], is_client=True
                    )
                    # Update client with privatized parameters
                    ray.get(actor.set_model_parameters.remote(privatized_params))

            # Use the topology coordinator for aggregation
            aggregated_params = self.topology_coordinator.coordinate_aggregation(
                weights=weights
            )

            # For Central DP, apply privacy after aggregation
            if self.privacy_manager.config.privacy_mode == PrivacyMode.CENTRAL:
                print(
                    f"Applying Central DP with noise multiplier: {self.privacy_manager.config.noise_multiplier:.4f}"
                )
                aggregated_params = self.privacy_manager.privatize_parameters(
                    aggregated_params, is_client=False
                )

            # Update privacy budget tracking
            self.privacy_manager.update_privacy_budget()

            return aggregated_params
        else:
            # No privacy, just use the topology coordinator
            return self.topology_coordinator.coordinate_aggregation(weights=weights)

    def update_aggregation_strategy(
        self, strategy: AggregationStrategy, topology_check: bool = True
    ) -> None:
        """
        Update the aggregation strategy for the cluster

        :param strategy: New aggregation strategy
        :param topology_check: Flag to check compatibility with the current topology

        :raises ValueError: If the new strategy is not compatible with the current topology
        """
        if topology_check and self.topology_manager:
            strategy_class = strategy.__class__
            topology_type = self.topology_manager.config.topology_type

            if not TopologyCompatibilityManager.is_compatible(
                strategy_class, topology_type
            ):
                compatible_topologies = (
                    TopologyCompatibilityManager.get_compatible_topologies(
                        strategy_class
                    )
                )
                raise ValueError(
                    f"New strategy {strategy_class.__name__} is not compatible with the current topology {topology_type.value}."
                    f"Compatible topologies: {compatible_topologies}"
                )

        self.aggregation_strategy = strategy

        # Reinitialize the topology coordinator with the new strategy
        if self.topology_manager:
            self._initialize_coordinator()

    def update_models(self, parameters: Dict[str, Any]) -> None:
        """
        Update model parameters on all actors

        :param parameters: New model parameters
        """
        for actor in self.actors:
            ray.get(actor.set_model_parameters.remote(parameters))

    def _apply_topology(self) -> None:
        """
        Set neighbour relationships based on topology config
        """
        if not self.topology_manager:
            return

        adjacency = self.topology_manager.adjacency_list
        for node, neighbours in adjacency.items():
            neighbour_actors = [self.actors[n] for n in neighbours]
            ray.get(self.actors[node].set_neighbours.remote(neighbour_actors))

    def get_topology_information(self) -> Dict[str, Any]:
        """
        Get information about the current topology

        :return: Dictionary containing topology information
        """
        if not self.topology_manager:
            return {"initialized": False}

        return {
            "initialized": True,
            "type": self.topology_manager.config.topology_type.value,
            "num_actors": len(self.actors),
            "hub_index": self.topology_manager.config.hub_index
            if self.topology_manager.config.topology_type == TopologyType.STAR
            else None,
            "adjacency_list": self.topology_manager.adjacency_list,
        }

    def get_privacy_information(self) -> Dict[str, Any]:
        """
        Get information about the current privacy configuration

        :return: Dictionary containing privacy information
        """
        if not self.privacy_manager:
            return {"enabled": False}

        privacy_spent = self.privacy_manager.get_current_privacy_spent()

        return {
            "enabled": self.privacy_manager.config.enabled,
            "mode": self.privacy_manager.config.privacy_mode.value,
            "mechanism": self.privacy_manager.config.mechanism_type.value,
            "epsilon": privacy_spent.get("epsilon", 0.0),
            "delta": privacy_spent.get("delta", 0.0),
            "noise_multiplier": self.privacy_manager.config.noise_multiplier,
        }

    def get_compatible_strategies(self) -> List[str]:
        """
        Get list of strategies compatible with the current topology.

        Returns:
            List of compatible strategy names
        """
        if not self.topology_manager:
            return []

        strategy_classes = TopologyCompatibilityManager.get_compatible_strategies(
            self.topology_manager.config.topology_type
        )

        result = []
        for cls in strategy_classes:
            # Match class name to strategy type
            found = False
            for strategy_type in AggregationStrategyType:
                if strategy_type.name.lower() == cls.__name__.lower():
                    result.append(str(strategy_type.value))
                    found = True
                    break

            # If no matching enum value, use the lowercase class name
            if not found:
                result.append(cls.__name__.lower())

        return result

    def get_compatible_privacy_modes(self) -> List[str]:
        """
        Get list of privacy modes compatible with the current topology.

        Returns:
            List of compatible privacy modes
        """
        if not self.topology_manager:
            return []

        result = ["LOCAL"]

        if self.topology_manager.config.topology_type in [
            TopologyType.STAR,
            TopologyType.COMPLETE,
        ]:
            result.append("CENTRAL")

        return result

    def _initialize_coordinator(self) -> None:
        """
        Initialize the topology coordinator with the actors and topology manager
        """
        if not self.topology_manager or not self.aggregation_strategy:
            return

        self.topology_coordinator = TopologyCoordinator.create(
            self.actors, self.topology_manager, self.aggregation_strategy
        )

    @staticmethod
    def shutdown() -> None:
        """
        Shutdown ray cluster gracefully
        """
        ray.shutdown()
