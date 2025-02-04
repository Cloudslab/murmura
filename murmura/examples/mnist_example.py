import argparse

from murmura.config import OrchestrationConfig
from murmura.data_processing.dataset import MDataset, DatasetSource
from murmura.data_processing.partitioner_factory import PartitionerFactory
from murmura.orchestration.cluster_manager import ClusterManager


def main() -> None:
    """
    Orchestrate Learning Process
    """
    parser = argparse.ArgumentParser(
        description="Federated Data Distribution Orchestrator"
    )
    parser.add_argument(
        "--num_actors",
        type=int,
        default=10,
        help="Number of virtual clients (default: 10)",
    )
    parser.add_argument(
        "--partition_strategy",
        choices=["dirichlet", "iid"],
        default="dirichlet",
        help="Partitioning strategy (default: dirichlet)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet alpha parameter (default: 0.5)",
    )
    parser.add_argument(
        "--min_partition_size",
        type=int,
        default=100,
        help="Minimum samples per partition (default: 100)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to partition and distribute",
    )
    args = parser.parse_args()

    # Create configuration from command-line arguments
    config = OrchestrationConfig(
        num_actors=args.num_actors,
        alpha=args.alpha,
        partition_strategy=args.partition_strategy,
        split=args.split,
        min_partition_size=args.min_partition_size,
    )

    # Load MNIST Dataset
    dataset = MDataset.load(
        DatasetSource.HUGGING_FACE, dataset_name="mnist", split=config.split
    )

    # Create appropriate partitioner
    partitioner = PartitionerFactory.create(config)
    partitioner.partition(dataset, config.split)

    # Initialize Ray Cluster
    cluster_manager = ClusterManager(config.model_dump())

    try:
        cluster_manager.create_actors(config.num_actors)
        partitions = dataset.get_partitions(config.split).values()

        distribution_result = cluster_manager.distribute_data(
            data_partitions=list(partitions),
            metadata={
                "split_name": config.split,
                "dataset": config.dataset_name,
            },
        )

        print(f"Data distribution completed: {distribution_result}")

        print(f"\nDistribution Summary ({config.partition_strategy} strategy):")
        print(f" - Total clients: {config.num_actors}")
        print(f" - Total partitions: {len(partitions)}")

    finally:
        cluster_manager.shutdown()


if __name__ == "__main__":
    main()
