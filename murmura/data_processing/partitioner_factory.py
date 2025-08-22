from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.data_processing.partitioner import (
    Partitioner,
    DirichletPartitioner,
    IIDPartitioner,
)


class PartitionerFactory:
    """
    Factory for creating partitioner instances based on configuration.
    """

    @staticmethod
    def create(config: OrchestrationConfig) -> Partitioner:
        """
        Create a partitioner instance based on the configuration.

        :param config: Orchestration configuration containing partition strategy
        :return: Initialized partitioner instance
        """
        if config.partition_strategy == "dirichlet":
            return DirichletPartitioner(
                num_partitions=config.num_actors,
                alpha=config.alpha,
                partition_by="label",
                min_partition_size=config.min_partition_size,
                seed=config.data_partitioning_seed,
            )
        elif config.partition_strategy == "iid":
            return IIDPartitioner(
                num_partitions=config.num_actors, 
                shuffle=True,
                seed=config.data_partitioning_seed,
            )
        else:
            raise ValueError(
                f"Unsupported partition strategy {config.partition_strategy}"
            )