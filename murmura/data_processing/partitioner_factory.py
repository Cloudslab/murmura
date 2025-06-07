from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.data_processing.partitioner import (
    Partitioner,
    DirichletPartitioner,
    IIDPartitioner,
)
from murmura.data_processing.attack_partitioner import (
    SensitiveGroupPartitioner,
    TopologyCorrelatedPartitioner,
    ImbalancedSensitivePartitioner,
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
            )
        elif config.partition_strategy == "iid":
            return IIDPartitioner(num_partitions=config.num_actors, shuffle=True)
        elif config.partition_strategy == "sensitive_groups":
            # For MNIST: Group digits 0-4 vs 5-9
            return SensitiveGroupPartitioner(
                num_partitions=config.num_actors,
                sensitive_groups={
                    "low_digits": [0, 1, 2, 3, 4],
                    "high_digits": [5, 6, 7, 8, 9]
                },
                topology_assignment={
                    "low_digits": list(range(config.num_actors // 2)),
                    "high_digits": list(range(config.num_actors // 2, config.num_actors))
                }
            )
        elif config.partition_strategy == "topology_correlated":
            # Infer topology type from config
            topology_type = "ring"  # Default
            if hasattr(config.topology, 'name'):
                if config.topology.name == "star":
                    topology_type = "star"
                elif config.topology.name == "ring":
                    topology_type = "ring"
                elif config.topology.name == "line":
                    topology_type = "line"
            
            return TopologyCorrelatedPartitioner(
                num_partitions=config.num_actors,
                topology_type=topology_type,
                correlation_strength=0.8
            )
        elif config.partition_strategy == "imbalanced_sensitive":
            return ImbalancedSensitivePartitioner(
                num_partitions=config.num_actors,
                rare_class_nodes=[0, 1],  # First two nodes get rare classes
                rare_classes=[0, 1],      # Digits 0 and 1 are rare
                rarity_factor=0.1
            )
        else:
            raise ValueError(
                f"Unsupported partition strategy {config.partition_strategy}"
            )
