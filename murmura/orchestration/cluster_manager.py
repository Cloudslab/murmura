from typing import Dict, Any, List, Optional

import ray
import torch

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

        if not ray.is_initialized():
            # Auto-detect GPU resources
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            print(f"Detected {num_gpus} GPUs for Ray initialization")

            ray.init(
                address=self.config["ray_address"] if "ray_address" in config else None,
                num_gpus=num_gpus,
                include_dashboard=False,
                _redis_max_memory=500000000,
            )

    def create_actors(self, num_actors: int, topology: TopologyConfig) -> List[Any]:
        """
        Create pool of virtual client actors with proper resource allocation

        :param topology: topology config
        :param num_actors: Number of actors to create
        :return: List of actors
        """
        self.topology_manager = TopologyManager(num_actors, topology)

        # Determine GPU resources to allocate per actor
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpus_per_actor = num_gpus / num_actors if num_gpus > 0 else 0

        print(
            f"Creating {num_actors} virtual clients with {gpus_per_actor:.2f} GPUs each"
        )

        self.actors = []
        for i in range(num_actors):
            if gpus_per_actor > 0:
                # Create actor with GPU resources
                actor = VirtualClientActor.options(num_gpus=gpus_per_actor).remote(
                    f"client_{i}"
                )
            else:
                # Create actor without GPU resources
                actor = VirtualClientActor.remote(f"client_{i}")

            self.actors.append(actor)

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

    def distribute_model(self, model: ModelInterface) -> None:
        """
        Distribute model structure and parameters to all actors with proper device handling.

        :param model: Model to distribute
        """
        print("Distributing model to client actors...")

        # Make sure the model is on CPU for serialization
        original_device = None
        if hasattr(model, "model") and hasattr(model.model, "to"):
            # Remember the original device
            if hasattr(model, "device"):
                original_device = model.device
            # Move to CPU temporarily for serialization
            model.model.to("cpu")
            if hasattr(model, "device"):
                model.device = "cpu"

        # Get model parameters on CPU to distribute
        parameters = model.get_parameters()

        # Set the model on each actor
        for actor in self.actors:
            try:
                # First set the model structure (CPU version for safe transfer)
                ray.get(actor.set_model.remote(model))
                # Then set the parameters
                ray.get(actor.set_model_parameters.remote(parameters))
            except Exception as e:
                print(f"Error distributing model to actor: {e}")
                raise

        # Move the model back to its original device if needed
        if (
            hasattr(model, "model")
            and hasattr(model.model, "to")
            and original_device is not None
        ):
            model.model.to(original_device)
            if hasattr(model, "device"):
                model.device = original_device

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

    def aggregate_model_parameters(
        self, weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate model parameters from all actors using the configured aggregation strategy

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

        # Use the topology coordinator to perform aggregation
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
