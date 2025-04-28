from typing import List, Any, Optional, Dict

import numpy as np
import ray

from murmura.aggregation.coordination_mode import CoordinationMode
from murmura.aggregation.strategy_interface import AggregationStrategy
from murmura.network_management.topology import TopologyType
from murmura.network_management.topology_manager import TopologyManager


class TopologyCoordinator:
    """
    Coordinates the exchange of model parameters based on network topology.
    Different topologies require different communication patterns for aggregation.
    """

    def __init__(
        self,
        actors: List[Any],
        topology_manager: TopologyManager,
        strategy: AggregationStrategy,
    ):
        self.actors = actors
        self.topology_manager = topology_manager
        self.strategy = strategy
        self.topology_type = topology_manager.config.topology_type

        self.coordination_mode = self._determine_coordination_mode()

    def coordinate_aggregation(
        self, weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate the aggregation of model parameters based on the topology.

        :param weights: Optional list of weights for each actor's parameters
        :return: Aggregated model parameters
        """
        # Dispatch to the appropriate coordinator method based of topology
        if self.topology_type == TopologyType.STAR:
            return self._coordinate_star_topology(weights)
        elif self.topology_type == TopologyType.RING:
            return self._coordinate_ring_topology(weights)
        elif self.topology_type == TopologyType.COMPLETE:
            return self._coordinate_complete_topology(weights)
        elif self.topology_type == TopologyType.LINE:
            return self._coordinate_line_topology(weights)
        elif self.topology_type == TopologyType.CUSTOM:
            return self._coordinate_custom_topology(weights)
        else:
            raise ValueError(f"Unsupported topology type: {self.topology_type}")

    def _coordinate_star_topology(
        self, weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate aggregation for star topology.
        In a star topology, all nodes send their parameters to the hub node,
        which performs aggregation and broadcasts the result.

        :param weights: Optional list of weights for each actor's parameters
        :return: Aggregated model parameters
        """
        # Identify the hub node
        hub_index = self.topology_manager.config.hub_index
        hub_actor = self.actors[hub_index]

        # Collect parameters from all nodes except the hub
        parameters_list = []
        for i, actor in enumerate(self.actors):
            if i != hub_index:
                params = ray.get(actor.get_model_parameters.remote())
                parameters_list.append(params)

        # Add hub's parameters
        hub_params = ray.get(hub_actor.get_model_parameters.remote())
        parameters_list.append(hub_params)

        # Adjust weights if provided
        if weights is not None and len(weights) == len(self.actors):
            # Reorder weights to match the parameters list order
            adjusted_weights = []
            for i in range(len(self.actors)):
                if i != hub_index:
                    adjusted_weights.append(weights[i])
            adjusted_weights.append(weights[hub_index])
        else:
            adjusted_weights = weights

        # Perform aggregation
        return self.strategy.aggregate(parameters_list, adjusted_weights)

    def _coordinate_ring_topology(
        self, weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate aggregation for ring topology.
        In a ring topology, each node communicates with its immediate neighbours.

        :param weights: Optional list of weights for each actor's parameters
        :return: Aggregated model parameters
        """
        # Ring topologies generally require decentralized coordination
        adjacency = self.topology_manager.adjacency_list
        all_aggregated_params = []

        # Each node performs local aggregation with its neighbors
        for node_idx, node_actor in enumerate(self.actors):
            # Get neighbors from adjacency list
            neighbors = adjacency[node_idx]

            # Collect parameters from neighbors and self
            local_params_list = []
            local_weights = []

            # Get self parameters
            self_params = ray.get(node_actor.get_model_parameters.remote())
            local_params_list.append(self_params)

            # Add self weight
            if weights is not None:
                local_weights.append(weights[node_idx])
            else:
                local_weights.append(1.0)

            # Get neighbor parameters
            for neighbor_idx in neighbors:
                neighbor_params = ray.get(
                    self.actors[neighbor_idx].get_model_parameters.remote()
                )
                local_params_list.append(neighbor_params)

                if weights is not None:
                    local_weights.append(weights[neighbor_idx])
                else:
                    local_weights.append(1.0)

            # Normalize weights
            total_weight = sum(local_weights)
            local_weights = [w / total_weight for w in local_weights]

            # Local aggregation
            local_aggregated = self.strategy.aggregate(local_params_list, local_weights)
            all_aggregated_params.append(local_aggregated)

        # Combine all local aggregations into a global aggregation
        return self._combine_aggregated_params(all_aggregated_params)

    def _coordinate_complete_topology(
        self, weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate aggregation for complete topology.
        In a complete topology, each node communicates with every other node.

        This works differently depending on the strategy type:
        - For FedAvg and TrimmedMean, it can work like a centralized approach since all nodes can communicate
        - For GossipAvg and other decentralized strategies, each node aggregates its local view

        :param weights: Optional list of weights for each actor's parameters
        :return: Aggregated model parameters
        """
        if self.coordination_mode == CoordinationMode.CENTRALIZED:
            # For centralized strategies, we can use a single aggregation step
            # since all nodes can communicate with each other
            all_params = []
            for actor in self.actors:
                params = ray.get(actor.get_model_parameters.remote())
                all_params.append(params)

            # Perform centralized aggregation
            return self.strategy.aggregate(all_params, weights)
        else:
            # For decentralized strategies
            adjacency = self.topology_manager.adjacency_list
            all_aggregated_params = []

            for node_idx, node_actor in enumerate(self.actors):
                # Get neighbors from adjacency list (all other nodes in complete topology)
                neighbors = adjacency[node_idx]

                # Collect parameters from neighbors and self
                local_params_list = []
                local_weights = []

                # Get self parameters
                self_params = ray.get(node_actor.get_model_parameters.remote())
                local_params_list.append(self_params)

                # Add self weight
                if weights is not None:
                    local_weights.append(weights[node_idx])
                else:
                    local_weights.append(1.0)

                # Get neighbor parameters
                for neighbor_idx in neighbors:
                    neighbor_params = ray.get(
                        self.actors[neighbor_idx].get_model_parameters.remote()
                    )
                    local_params_list.append(neighbor_params)

                    if weights is not None:
                        local_weights.append(weights[neighbor_idx])
                    else:
                        local_weights.append(1.0)

                # Normalize weights
                total_weight = sum(local_weights)
                local_weights = [w / total_weight for w in local_weights]

                # Local aggregation
                local_aggregated = self.strategy.aggregate(
                    local_params_list, local_weights
                )
                all_aggregated_params.append(local_aggregated)

            # Combine all local aggregations
            return self._combine_aggregated_params(all_aggregated_params)

    def _coordinate_line_topology(
        self, weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate aggregation for line topology.
        In a line topology, each node communicates with its immediate neighbors.

        :param weights: Optional list of weights for each actor's parameters
        :return: Aggregated model parameters
        """
        # Line topologies generally require decentralized coordination
        adjacency = self.topology_manager.adjacency_list
        all_aggregated_params = []

        for node_idx, node_actor in enumerate(self.actors):
            neighbors = adjacency[node_idx]

            # Collect parameters from neighbors and self
            local_params_list = []
            local_weights = []

            # Get self parameters
            self_params = ray.get(node_actor.get_model_parameters.remote())
            local_params_list.append(self_params)

            # Add self weight
            if weights is not None:
                local_weights.append(weights[node_idx])
            else:
                local_weights.append(1.0)

            # Get neighbor parameters
            for neighbor_idx in neighbors:
                neighbor_params = ray.get(
                    self.actors[neighbor_idx].get_model_parameters.remote()
                )
                local_params_list.append(neighbor_params)

                if weights is not None:
                    local_weights.append(weights[neighbor_idx])
                else:
                    local_weights.append(1.0)

            # Normalize weights
            total_weight = sum(local_weights)
            local_weights = [w / total_weight for w in local_weights]

            # Local aggregation
            local_aggregated = self.strategy.aggregate(local_params_list, local_weights)
            all_aggregated_params.append(local_aggregated)

        # Combine all local aggregations
        return self._combine_aggregated_params(all_aggregated_params)

    def _coordinate_custom_topology(
        self, weights: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Coordinate aggregation for custom topology.
        In a custom topology, the communication pattern is defined by the user.

        :param weights: Optional list of weights for each actor's parameters
        :return: Aggregated model parameters
        """
        # Custom topologies generally require decentralized coordination
        adjacency = self.topology_manager.adjacency_list
        all_aggregated_params = []

        for node_idx, node_actor in enumerate(self.actors):
            neighbors = adjacency[node_idx]

            # Collect parameters from neighbors and self
            local_params_list = []
            local_weights = []

            # Get self parameters
            self_params = ray.get(node_actor.get_model_parameters.remote())
            local_params_list.append(self_params)

            # Add self weight
            if weights is not None:
                local_weights.append(weights[node_idx])
            else:
                local_weights.append(1.0)

            # Get neighbor parameters
            for neighbor_idx in neighbors:
                neighbor_params = ray.get(
                    self.actors[neighbor_idx].get_model_parameters.remote()
                )
                local_params_list.append(neighbor_params)

                if weights is not None:
                    local_weights.append(weights[neighbor_idx])
                else:
                    local_weights.append(1.0)

            # Normalize weights
            total_weight = sum(local_weights)
            local_weights = [w / total_weight for w in local_weights]

            # Local aggregation
            local_aggregated = self.strategy.aggregate(local_params_list, local_weights)
            all_aggregated_params.append(local_aggregated)

        # Combine all local aggregations
        return self._combine_aggregated_params(all_aggregated_params)

    @staticmethod
    def _combine_aggregated_params(
        aggregated_params_list: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Combine all local aggregated parameters into a single set of parameters.

        :param aggregated_params_list: List of aggregated parameters from each node
        :return: Combined model parameters
        """
        if not aggregated_params_list:
            raise ValueError("Empty aggregated parameters list")

        combined_params = {}

        # Get all keys from the first parameter set
        for key in aggregated_params_list[0].keys():
            # Stack parameters along a new axis
            stacked_params = np.stack(
                [params[key] for params in aggregated_params_list], axis=0
            )

            # Average across all parameter sets
            combined_params[key] = np.mean(stacked_params, axis=0)

        return combined_params

    def _determine_coordination_mode(self) -> CoordinationMode:
        """
        Get the coordination mode based on the strategy's characteristics.

        Returns:
            CoordinationMode enum value
        """
        # Get coordination mode from strategy (defaults to DECENTRALIZED in the interface)
        return self.strategy.coordination_mode

    @staticmethod
    def create(
        actors: List[Any],
        topology_manager: TopologyManager,
        strategy: AggregationStrategy,
    ) -> "TopologyCoordinator":
        """
        Factory method to create a TopologyCoordinator instance.

        :param actors: List of actor instances
        :param topology_manager: Topology manager instance
        :param strategy: Aggregation strategy instance
        :return: TopologyCoordinator instance
        """
        return TopologyCoordinator(actors, topology_manager, strategy)
