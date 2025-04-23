from typing import Dict, Any, List, Optional

import ray

from murmura.aggregation.aggregation_config import AggregationConfig
from murmura.aggregation.strategy_factory import AggregationStrategyFactory
from murmura.aggregation.strategy_interface import AggregationStrategy
from murmura.data_processing.dataset import MDataset
from murmura.model.model_interface import ModelInterface
from murmura.network_management.topology import TopologyConfig
from murmura.network_management.topology_manager import TopologyManager
from murmura.node.client_actor import VirtualClientActor


class ClusterManager:
    """
    Manages Ray cluster and virtual client actors
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.actors: List[Any] = []
        self.topology_manager: Optional[TopologyManager] = None
        self.aggregation_strategy: Optional[AggregationStrategy] = None

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
        return self.actors

    def set_aggregation_strategy(self, aggregation_config: AggregationConfig) -> None:
        """
        Set the aggregation strategy for the cluster

        :param aggregation_config: Aggregation configuration
        """
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

    def aggregate_model_parameters(
        self, weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate model parameters from all actors using the configured aggregation strategy

        :param weights: Optional list of weights for each actor's parameters
        :return: Aggregated model parameters
        """
        if self.aggregation_strategy is None:
            raise ValueError(
                "Aggregation strategy not set. Call set_aggregation_strategy first."
            )

        # Get parameters from all clients
        parameter_futures = [
            actor.get_model_parameters.remote() for actor in self.actors
        ]
        all_parameters = ray.get(parameter_futures)

        # Use the configured aggregation strategy
        return self.aggregation_strategy.aggregate(all_parameters, weights)

    def update_aggregation_strategy(self, strategy: AggregationStrategy) -> None:
        """
        Update the aggregation strategy for the cluster

        :param strategy: New aggregation strategy
        """
        self.aggregation_strategy = strategy

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

    @staticmethod
    def shutdown() -> None:
        """
        Shutdown ray cluster gracefully
        """
        ray.shutdown()
