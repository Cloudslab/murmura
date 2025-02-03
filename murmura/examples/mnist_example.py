import argparse

from murmura.config import OrchestrationConfig
from murmura.data_processing.dataset import MDataset, DatasetSource
from murmura.data_processing.partitioner import (
    DirichletPartitioner,
    IIDPartitioner,
    Partitioner,
)
from murmura.orchestration.cluster_manager import ClusterManager


def main() -> None:
    """
    Orchestrate Learning Process
    """
    parser = argparse.ArgumentParser(description="Run federated data distribution")
    parser.add_argument(
        "--num_actors", type=int, default=10, help="Number of virtual clients"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Dirichlet alpha parameter"
    )
    parser.add_argument(
        "--min_partition_size",
        type=int,
        default=100,
        help="Minimum samples per partition",
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
        min_partition_size=args.min_partition_size,
    )

    # Load MNIST Dataset
    dataset = MDataset.load(
        DatasetSource.HUGGING_FACE, dataset_name="mnist", split=config.split
    )

    # Create partitions
    if config.partition_strategy == "dirichlet":
        partitioner: Partitioner = DirichletPartitioner(
            num_partitions=config.num_actors,
            alpha=config.alpha,
            partition_by="label",
            min_partition_size=config.min_partition_size,
        )
    else:
        partitioner = IIDPartitioner(num_partitions=config.num_actors)

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

    finally:
        cluster_manager.shutdown()


if __name__ == "__main__":
    main()
