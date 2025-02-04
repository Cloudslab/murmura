from typing import Dict, Any, List, Optional

import ray

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
