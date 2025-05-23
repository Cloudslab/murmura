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
        # Store the previous global parameters for computing updates
        self.previous_global_params: Optional[Dict[str, Any]] = None

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
        Distribute model structure and parameters to all actors.
        Also initializes privacy manager with model information.

        :param model: Model to distribute
        """
        # Get model parameters to distribute
        parameters = model.get_parameters()

        # Store as initial global parameters
        self.previous_global_params = {k: v.copy() for k, v in parameters.items()}

        # Initialize privacy manager with model information if privacy is enabled
        if hasattr(self, 'privacy_manager') and self.privacy_manager:
            if hasattr(self.privacy_manager, 'initialize_from_model'):
                # Get learning rate from config
                learning_rate = self.config.get('learning_rate', 0.001)

                print("Initializing privacy manager with model information...")
                self.privacy_manager.initialize_from_model(parameters, learning_rate)

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
        privacy_config.params["actors"] = self.actors if hasattr(self, "actors") else []

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
        if "batch_size" in self.config and "data_partitions" in self.config:
            batch_size = self.config.get("batch_size", 32)
            data_partitions = self.config.get("data_partitions", [])
            total_samples = (
                sum(len(p) for p in data_partitions) if data_partitions else 10000
            )

            if self.privacy_manager and hasattr(
                    self.privacy_manager, "setup_privacy_accounting"
            ):
                self.privacy_manager.setup_privacy_accounting(total_samples, batch_size)

    def aggregate_model_parameters(
            self, weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate model parameters from all actors using the configured aggregation strategy,
        with privacy guarantees if enabled.
        """
        if not self.aggregation_strategy:
            raise ValueError(
                "Aggregation strategy not set. Call set_aggregation_strategy first."
            )

        if not self.topology_coordinator:
            raise ValueError(
                "Topology coordinator not initialized."
            )

        # Collect parameters from all actors
        all_params = []
        for actor in self.actors:
            params = ray.get(actor.get_model_parameters.remote())
            all_params.append(params)

        # If privacy is enabled
        if self.privacy_manager and self.privacy_manager.config.enabled:
            if self.privacy_manager.config.privacy_mode == PrivacyMode.CENTRAL:
                # For Central DP, we need to work with updates
                if self.previous_global_params is not None:
                    # Compute updates from each client
                    all_updates = []
                    for client_params in all_params:
                        updates = {}
                        for key in client_params.keys():
                            updates[key] = client_params[key] - self.previous_global_params[key]
                        all_updates.append(updates)

                    # CRITICAL: Update clipping norms based on the RAW updates before clipping
                    self.privacy_manager.update_clipping_norms(all_updates)

                    # Clip each client's update individually
                    clipped_updates = []
                    for updates in all_updates:
                        clipped = self.privacy_manager.privacy_mechanism.clip_parameters(
                            updates, self.privacy_manager.clipping_norms
                        )
                        clipped_updates.append(clipped)

                    # Aggregate the clipped updates (this computes the average)
                    aggregated_updates = self.topology_coordinator.coordinate_aggregation(
                        weights=weights,
                        pre_collected_params=clipped_updates
                    )

                    # CRITICAL FIX: For Central DP, the sensitivity is C/n where:
                    # C = clipping norm, n = number of clients
                    # The noise should be calibrated for this sensitivity
                    num_clients = len(self.actors)

                    # Create adjusted clipping norms for the aggregated update
                    # This accounts for the averaging in aggregation
                    aggregated_clipping_norms = {}
                    for key, norm in self.privacy_manager.clipping_norms.items():
                        # After averaging n clipped updates, the max norm is C/n
                        aggregated_clipping_norms[key] = norm / num_clients

                    print(f"Applying Central DP with noise multiplier: {self.privacy_manager.config.noise_multiplier:.4f}")
                    print(f"Number of clients: {num_clients}")
                    if "__default__" in aggregated_clipping_norms:
                        print(f"Aggregated clipping norm: {aggregated_clipping_norms['__default__']:.6f}")

                    # Add noise scaled to the aggregated sensitivity
                    privatized_updates = self.privacy_manager.privacy_mechanism.add_noise(
                        aggregated_updates, aggregated_clipping_norms
                    )

                    # Add privatized updates back to global model
                    aggregated_params = {}
                    for key in privatized_updates.keys():
                        aggregated_params[key] = self.previous_global_params[key] + privatized_updates[key]
                else:
                    # First round - no previous params, work with full parameters
                    # This is less ideal but necessary for the first round
                    print("First round - using full parameter clipping and aggregation")

                    # Update clipping norms based on full parameters
                    self.privacy_manager.update_clipping_norms(all_params)

                    # Clip parameters individually
                    clipped_params = []
                    for params in all_params:
                        clipped = self.privacy_manager.privacy_mechanism.clip_parameters(
                            params, self.privacy_manager.clipping_norms
                        )
                        clipped_params.append(clipped)

                    # Aggregate clipped parameters
                    aggregated_params = self.topology_coordinator.coordinate_aggregation(
                        weights=weights,
                        pre_collected_params=clipped_params
                    )

                    # Add noise with sensitivity adjusted for aggregation
                    num_clients = len(self.actors)
                    aggregated_clipping_norms = {}
                    for key, norm in self.privacy_manager.clipping_norms.items():
                        aggregated_clipping_norms[key] = norm / num_clients

                    aggregated_params = self.privacy_manager.privacy_mechanism.add_noise(
                        aggregated_params, aggregated_clipping_norms
                    )

            elif self.privacy_manager.config.privacy_mode == PrivacyMode.LOCAL:
                # For Local DP, each client adds noise independently
                print(f"Applying Local DP with noise multiplier: {self.privacy_manager.config.noise_multiplier:.4f}")

                # For local DP, we should work with updates when possible
                if self.previous_global_params is not None:
                    # Compute updates
                    all_updates = []
                    for client_params in all_params:
                        updates = {}
                        for key in client_params.keys():
                            updates[key] = client_params[key] - self.previous_global_params[key]
                        all_updates.append(updates)

                    # Update clipping norms based on updates
                    self.privacy_manager.update_clipping_norms(all_updates)

                    # Each client clips and adds noise to their update
                    privatized_updates = []
                    for updates in all_updates:
                        # Clip first
                        clipped = self.privacy_manager.privacy_mechanism.clip_parameters(
                            updates, self.privacy_manager.clipping_norms
                        )
                        # Then add noise (each client adds full noise)
                        private_u = self.privacy_manager.privacy_mechanism.add_noise(
                            clipped, self.privacy_manager.clipping_norms
                        )
                        privatized_updates.append(private_u)

                    # Aggregate privatized updates
                    aggregated_updates = self.topology_coordinator.coordinate_aggregation(
                        weights=weights,
                        pre_collected_params=privatized_updates
                    )

                    # Add back to get parameters
                    aggregated_params = {}
                    for key in aggregated_updates.keys():
                        aggregated_params[key] = self.previous_global_params[key] + aggregated_updates[key]
                else:
                    # First round - work with full parameters
                    self.privacy_manager.update_clipping_norms(all_params)

                    # Apply privacy to each client's parameters
                    privatized_params = []
                    for params in all_params:
                        clipped = self.privacy_manager.privacy_mechanism.clip_parameters(
                            params, self.privacy_manager.clipping_norms
                        )
                        private_p = self.privacy_manager.privacy_mechanism.add_noise(
                            clipped, self.privacy_manager.clipping_norms
                        )
                        privatized_params.append(private_p)

                    # Aggregate privatized parameters
                    aggregated_params = self.topology_coordinator.coordinate_aggregation(
                        weights=weights,
                        pre_collected_params=privatized_params
                    )

            # Update privacy budget
            self.privacy_manager.update_privacy_budget()
        else:
            # No privacy, aggregate normally
            aggregated_params = self.topology_coordinator.coordinate_aggregation(
                weights=weights,
                pre_collected_params=all_params
            )

        # Store for next round
        self.previous_global_params = {k: v.copy() for k, v in aggregated_params.items()}

        return aggregated_params

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
