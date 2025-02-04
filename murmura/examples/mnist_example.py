import argparse

from murmura.helper import visualize_network_topology
from murmura.network_management.topology import TopologyConfig
from murmura.orchestration.orchestration_config import OrchestrationConfig
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
        "--num_actors", type=int, default=10, help="Number of virtual clients"
    )
    parser.add_argument(
        "--partition_strategy",
        choices=["dirichlet", "iid"],
        default="dirichlet",
        help="Data partitioning strategy",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Dirichlet concentration parameter"
    )
    parser.add_argument(
        "--min_partition_size",
        type=int,
        default=100,
        help="Minimum samples per partition",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )

    # Topology arguments
    parser.add_argument(
        "--topology",
        type=str,
        default="complete",
        choices=["star", "ring", "complete", "line", "custom"],
        help="Network topology between clients",
    )
    parser.add_argument(
        "--hub_index", type=int, default=0, help="Hub node index for star topology"
    )

    args = parser.parse_args()

    try:
        # Create configuration from command-line arguments
        config = OrchestrationConfig(
            num_actors=args.num_actors,
            partition_strategy=args.partition_strategy,
            alpha=args.alpha,
            min_partition_size=args.min_partition_size,
            split=args.split,
            topology=TopologyConfig(
                topology_type=args.topology, hub_index=args.hub_index
            ),
        )

        # Load MNIST Dataset
        dataset = MDataset.load(
            DatasetSource.HUGGING_FACE,
            dataset_name=config.dataset_name,
            split=config.split,
        )

        # Create partitioner and split data
        partitioner = PartitionerFactory.create(config)
        partitioner.partition(dataset, config.split)

        # Initialize Ray Cluster
        cluster_manager = ClusterManager(config.model_dump())

        try:
            cluster_manager.create_actors(config.num_actors, config.topology)
            partitions = dataset.get_partitions(config.split).values()

            _ = cluster_manager.distribute_data(
                data_partitions=list(partitions),
                metadata={
                    "split": config.split,
                    "dataset": config.dataset_name,
                    "topology": config.topology.topology_type.value,
                },
            )

            # Print summary
            print("\n=== Distribution Summary ===")
            print(f"Strategy: {config.partition_strategy}")
            print(f"Clients: {config.num_actors}")
            print(f"Partitions: {len(partitions)}")
            print(f"Min samples/client: {min(len(p) for p in partitions)}")
            print(f"Max samples/client: {max(len(p) for p in partitions)}")

        finally:
            cluster_manager.shutdown()
            visualize_network_topology(cluster_manager)

    except Exception as e:
        print(f"Learning Process orchestration failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
