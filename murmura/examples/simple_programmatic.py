"""Simple programmatic example of using Murmura."""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from murmura import Network
from murmura.core import Node
from murmura.topology import create_topology
from murmura.aggregation import FedAvgAggregator, UBARAggregator
from murmura.data import DatasetAdapter
from murmura.utils import set_seed, get_device


def create_simple_model():
    """Create a simple MLP model."""
    return nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 2)
    )


def create_synthetic_dataset(num_clients=5, samples_per_client=100, input_dim=10):
    """Create a simple synthetic dataset for demonstration."""
    # Generate random data
    X = torch.randn(num_clients * samples_per_client, input_dim)
    y = torch.randint(0, 2, (num_clients * samples_per_client,))

    # Create dataset
    dataset = TensorDataset(X, y)

    # Create client partitions
    client_partitions = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client
        client_partitions.append(list(range(start_idx, end_idx)))

    return DatasetAdapter(dataset, client_partitions)


def main():
    # Set seed for reproducibility
    set_seed(42)
    device = get_device()

    print(f"Using device: {device}")

    # Create synthetic dataset
    dataset_adapter = create_synthetic_dataset(num_clients=10, samples_per_client=50)

    # Create topology
    topology = create_topology("ring", num_nodes=10)

    # Create nodes
    nodes = []
    for node_id in range(10):
        # Create model
        model = create_simple_model()

        # Get node's data
        train_dataset = dataset_adapter.get_client_data(node_id)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

        # Create aggregator (FedAvg for simplicity)
        aggregator = FedAvgAggregator()

        # Create node
        node = Node(
            node_id=node_id,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            aggregator=aggregator,
            device=device
        )
        nodes.append(node)

    # Create network
    network = Network(nodes=nodes, topology=topology, attack=None)

    # Train
    print("\nStarting training...")
    history = network.train(
        rounds=10,
        local_epochs=2,
        lr=0.01,
        verbose=True
    )

    # Print final results
    print("\n=== Final Results ===")
    print(f"Final Mean Accuracy: {history['mean_accuracy'][-1]:.4f}")
    print(f"Final Std Accuracy: {history['std_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
