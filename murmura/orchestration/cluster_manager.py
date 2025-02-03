from typing import Dict, Any, List, Optional

import ray

from murmura.orchestration.client_actor import VirtualClientActor


class ClusterManager:
    """
    Manages Ray cluster and virtual client actors
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.actors: List[Any] = []

        if not ray.is_initialized():
            ray.init(address=self.config["address"] if "address" in config else None)

    def create_actors(self, num_actors: int) -> List[Any]:
        """
        Create pool of virtual client actors

        :param num_actors: Number of actors to create
        :return: List of actors
        """
        self.actors = [
            VirtualClientActor.remote(f"client_{i}")  # type: ignore[attr-defined]
            for i in range(num_actors)
        ]
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

    @staticmethod
    def shutdown() -> None:
        """
        Shutdown ray cluster gracefully
        """
        ray.shutdown()
