import argparse
from statistics import mean

import numpy as np
import torch
import torch.nn as nn

from murmura.aggregation.aggregation_config import AggregationConfig, AggregationStrategyType
from murmura.helper import visualize_network_topology
from murmura.model.pytorch_model import PyTorchModel, TorchModelWrapper
from murmura.network_management.topology import TopologyConfig
from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.data_processing.dataset import MDataset, DatasetSource
from murmura.data_processing.partitioner_factory import PartitionerFactory
from murmura.orchestration.cluster_manager import ClusterManager


class MNISTModel(PyTorchModel):
    """
    Simple CNN model for MNIST classification
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # Ensure input has the right shape (add channel dimension if needed)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # Add channel dimension for grayscale

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


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
    parser.add_argument(
        "--test_split", type=str, default="test", help="Test split to use"
    )
    parser.add_argument(
        "--aggregation_strategy",
        type=str,
        choices=["fedavg", "trimmed_mean"],
        default="fedavg",
        help="Aggregation strategy to use"
    )
    parser.add_argument(
        "--trim_ratio",
        type=float,
        default=0.1,
        help="Trim ratio for trimmed_mean strategy (0.1 = 10% trimmed from each end)"
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

    # Training arguments
    parser.add_argument(
        "--rounds", type=int, default=5, help="Number of federated learning rounds"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of local epochs per round"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate"
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
            aggregation=AggregationConfig(
                strategy_type=args.aggregation_strategy,
                params={"trim_ratio": args.trim_ratio} if args.aggregation_strategy == "trimmed_mean" else None
            )
        )

        print("\n=== Loading MNIST Dataset ===")
        # Load MNIST Dataset
        dataset = MDataset.load(
            DatasetSource.HUGGING_FACE,
            dataset_name=config.dataset_name,
            split=config.split,
        )

        print("\n=== Creating Data Partitions ===")
        # Create partitioner and split data
        partitioner = PartitionerFactory.create(config)
        partitioner.partition(dataset, config.split)

        # Initialize Ray Cluster
        cluster_manager = ClusterManager(config.model_dump())
        cluster_manager.set_aggregation_strategy(config.aggregation)

        try:
            print("\n=== Creating Virtual Clients ===")
            cluster_manager.create_actors(config.num_actors, config.topology)

            print("\n=== Distributing Data Partitions ===")
            partitions = dataset.get_partitions(config.split).values()

            _ = cluster_manager.distribute_data(
                data_partitions=list(partitions),
                metadata={
                    "split": config.split,
                    "dataset": config.dataset_name,
                    "topology": config.topology.topology_type.value,
                },
            )

            print("\n=== Distributing Dataset ===")
            # Distribute the dataset to all clients
            cluster_manager.distribute_dataset(
                dataset,
                feature_columns=["image"],
                label_column="label"
            )

            print("\n=== Creating and Distributing Model ===")
            # Create the MNIST model with PyTorch wrapper
            model = MNISTModel()
            input_shape = (1, 28, 28)  # (channels, height, width)

            global_model = TorchModelWrapper(
                model=model,
                loss_fn=nn.CrossEntropyLoss(),
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs={"lr": args.lr},
                input_shape=input_shape
            )

            # Distribute the model to all clients
            cluster_manager.distribute_model(global_model)

            # Print initial summary
            print("\n=== Federated Learning Setup ===")
            print(f"Strategy: {config.partition_strategy}")
            print(f"Clients: {config.num_actors}")
            print(f"Partitions: {len(partitions)}")
            print(f"Min samples/client: {min(len(p) for p in partitions)}")
            print(f"Max samples/client: {max(len(p) for p in partitions)}")
            print(f"Aggregation strategy: {config.aggregation.strategy_type}")
            print(f"Rounds: {args.rounds}")
            print(f"Local epochs: {args.epochs}")
            print(f"Batch size: {args.batch_size}")
            print(f"Learning rate: {args.lr}")

            # Federated Learning Loop
            print("\n=== Starting Federated Learning ===")

            # For evaluation we'll use global model on the whole test set
            test_dataset = MDataset.load(
                DatasetSource.HUGGING_FACE,
                dataset_name=config.dataset_name,
                split=args.test_split,
            )
            test_dataset = test_dataset.get_split(args.test_split)
            test_images = np.array(test_dataset["image"])
            test_labels = np.array(test_dataset["label"])

            # Evaluate initial model
            initial_metrics = global_model.evaluate(test_images, test_labels)
            print(f"Initial Test Accuracy: {initial_metrics['accuracy'] * 100:.2f}%")

            # Training rounds
            for round_num in range(1, args.rounds + 1):
                print(f"\n--- Round {round_num}/{args.rounds} ---")

                # 1. Local Training
                print("Training on clients...")
                train_metrics = cluster_manager.train_models(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    verbose=False
                )

                # Calculate average training metrics
                avg_train_loss = mean([m["loss"] for m in train_metrics])
                avg_train_acc = mean([m["accuracy"] for m in train_metrics])
                print(f"Avg Training Loss: {avg_train_loss:.4f}")
                print(f"Avg Training Accuracy: {avg_train_acc * 100:.2f}%")

                # 2. Parameter Aggregation
                print("Aggregating model parameters...")
                # Get client data sizes for weighted aggregation
                client_data_sizes = [len(partition) for partition in partitions]

                # Aggregate with weighted averaging based on data size
                # Only pass weights for fedavg since trimmed_mean ignores them
                weights = None
                if config.aggregation.strategy_type == AggregationStrategyType.FEDAVG:
                    weights = client_data_sizes

                aggregated_params = cluster_manager.aggregate_model_parameters(weights=weights)

                # 3. Model Update
                # Update global model
                global_model.set_parameters(aggregated_params)

                # Distribute updated model to clients
                cluster_manager.update_models(aggregated_params)

                # 4. Evaluation
                # Evaluate global model on test set
                test_metrics = global_model.evaluate(test_images, test_labels)
                print(f"Global Model Test Loss: {test_metrics['loss']:.4f}")
                print(f"Global Model Test Accuracy: {test_metrics['accuracy'] * 100:.2f}%")

            # Final evaluation
            print("\n=== Final Model Evaluation ===")
            final_metrics = global_model.evaluate(test_images, test_labels)
            print(f"Final Test Accuracy: {final_metrics['accuracy'] * 100:.2f}%")
            print(f"Accuracy Improvement: {(final_metrics['accuracy'] - initial_metrics['accuracy']) * 100:.2f}%")

            # Visualize the network topology
            print("\n=== Network Topology Visualization ===")
            visualize_network_topology(cluster_manager)

            # Save the final model
            print("\n=== Saving Final Model ===")
            save_path = "mnist_federated_model.pt"
            global_model.save(save_path)
            print(f"Model saved to '{save_path}'")

        finally:
            print("\n=== Shutting Down Cluster ===")
            cluster_manager.shutdown()

    except Exception as e:
        print(f"Learning Process orchestration failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
