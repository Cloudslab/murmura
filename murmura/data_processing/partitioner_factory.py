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
            return PartitionerFactory._create_sensitive_groups_partitioner(config)
        elif config.partition_strategy == "topology_correlated":
            return PartitionerFactory._create_topology_correlated_partitioner(config)
        elif config.partition_strategy == "imbalanced_sensitive":
            return PartitionerFactory._create_imbalanced_sensitive_partitioner(config)
        else:
            raise ValueError(
                f"Unsupported partition strategy {config.partition_strategy}"
            )

    @staticmethod
    def _create_sensitive_groups_partitioner(
        config: OrchestrationConfig,
    ) -> SensitiveGroupPartitioner:
        """Create dataset-aware sensitive groups partitioner."""
        dataset_name = getattr(config, "dataset_name", "unknown").lower()

        if "mnist" in dataset_name:
            # MNIST: 10 classes (digits 0-9)
            return SensitiveGroupPartitioner(
                num_partitions=config.num_actors,
                sensitive_groups={
                    "low_digits": [0, 1, 2, 3, 4],
                    "high_digits": [5, 6, 7, 8, 9],
                },
                topology_assignment={
                    "low_digits": list(range(config.num_actors // 2)),
                    "high_digits": list(
                        range(config.num_actors // 2, config.num_actors)
                    ),
                },
            )
        elif (
            "skin" in dataset_name
            or "lesion" in dataset_name
            or "ham10000" in dataset_name
        ):
            # Skin lesion/HAM10000: 7 classes (diagnostic categories)
            # HAM10000 classes: 0=akiec, 1=bcc, 2=bkl, 3=df, 4=mel, 5=nv, 6=vasc
            return SensitiveGroupPartitioner(
                num_partitions=config.num_actors,
                sensitive_groups={
                    "malignant": [4],  # MEL (Melanoma) - most dangerous
                    "benign_common": [5],  # NV (Melanocytic nevi) - most common
                    "other_conditions": [0, 1, 2, 3, 6],  # AKIEC, BCC, BKL, DF, VASC
                },
                topology_assignment={
                    "malignant": [0],  # Melanoma at specialized node
                    "benign_common": list(range(1, max(2, config.num_actors // 2))),
                    "other_conditions": list(
                        range(max(2, config.num_actors // 2), config.num_actors)
                    ),
                },
            )
        else:
            # Default: assume 10 classes like MNIST
            return SensitiveGroupPartitioner(
                num_partitions=config.num_actors,
                sensitive_groups={
                    "group_a": list(range(5)),
                    "group_b": list(range(5, 10)),
                },
                topology_assignment={
                    "group_a": list(range(config.num_actors // 2)),
                    "group_b": list(range(config.num_actors // 2, config.num_actors)),
                },
            )

    @staticmethod
    def _create_topology_correlated_partitioner(
        config: OrchestrationConfig,
    ) -> TopologyCorrelatedPartitioner:
        """Create topology correlated partitioner."""
        # Infer topology type from config
        topology_type = "ring"  # Default
        if hasattr(config.topology, "name"):
            if config.topology.name == "star":
                topology_type = "star"
            elif config.topology.name == "ring":
                topology_type = "ring"
            elif config.topology.name == "line":
                topology_type = "line"

        return TopologyCorrelatedPartitioner(
            num_partitions=config.num_actors,
            topology_type=topology_type,
            correlation_strength=0.8,
        )

    @staticmethod
    def _create_imbalanced_sensitive_partitioner(
        config: OrchestrationConfig,
    ) -> ImbalancedSensitivePartitioner:
        """Create dataset-aware imbalanced sensitive partitioner."""
        dataset_name = getattr(config, "dataset_name", "unknown").lower()

        if "mnist" in dataset_name:
            # MNIST: Make digits 0 and 1 rare
            return ImbalancedSensitivePartitioner(
                num_partitions=config.num_actors,
                rare_class_nodes=[0, 1],  # First two nodes get rare classes
                rare_classes=[0, 1],  # Digits 0 and 1 are rare
                rarity_factor=0.1,  # 10% go to other nodes, 90% stay at rare nodes
            )
        elif (
            "skin" in dataset_name
            or "lesion" in dataset_name
            or "ham10000" in dataset_name
        ):
            # Skin lesion/HAM10000: Make DF and VASC rare (actually less common conditions)
            # HAM10000 classes: 0=akiec, 1=bcc, 2=bkl, 3=df, 4=mel, 5=nv, 6=vasc
            return ImbalancedSensitivePartitioner(
                num_partitions=config.num_actors,
                rare_class_nodes=[0, 1],  # First two nodes get rare classes
                rare_classes=[3, 6],  # DF (Dermatofibroma) and VASC (Vascular lesions)
                rarity_factor=0.05,  # Very rare (5%)
            )
        else:
            # Default: classes 0 and 1 as rare
            return ImbalancedSensitivePartitioner(
                num_partitions=config.num_actors,
                rare_class_nodes=[0, 1],
                rare_classes=[0, 1],
                rarity_factor=0.1,
            )
