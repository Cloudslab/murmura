"""CIFAR-10 dataset adapter with IID and non-IID partitioning."""

from typing import List, Optional
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from murmura.data.adapters import DatasetAdapter
from murmura.data.partitioners import iid_partition, dirichlet_partition


def load_cifar10_adapter(
    num_nodes: int,
    data_path: str = "./data/cifar10",
    partition_method: str = "iid",
    alpha: float = 0.5,
    seed: int = 42,
    split: str = "train",
    normalize: bool = True,
    **kwargs
) -> DatasetAdapter:
    """Load CIFAR-10 as federated dataset.

    Args:
        num_nodes: Number of federated clients
        data_path: Path to download/load CIFAR-10
        partition_method: "iid" for uniform distribution, "dirichlet" for non-IID
        alpha: Dirichlet concentration parameter (lower = more non-IID)
               - alpha -> 0: Each client gets samples from only one class
               - alpha = 0.5: Moderate heterogeneity
               - alpha -> inf: Approaches IID
        seed: Random seed for reproducibility
        split: "train" or "test"
        normalize: Whether to normalize images with CIFAR-10 statistics
        **kwargs: Additional arguments (ignored)

    Returns:
        DatasetAdapter for CIFAR-10
    """
    # Build transforms
    transform_list = [transforms.ToTensor()]
    if normalize:
        # CIFAR-10 normalization values
        transform_list.append(
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010)
            )
        )
    transform = transforms.Compose(transform_list)

    # Load CIFAR-10 dataset
    train = (split.lower() == "train")
    dataset = datasets.CIFAR10(
        root=data_path,
        train=train,
        download=True,
        transform=transform
    )

    # Get labels for partitioning
    labels = np.array(dataset.targets)

    # Create partitions based on method
    if partition_method.lower() == "iid":
        partitions = iid_partition(
            num_samples=len(dataset),
            num_clients=num_nodes,
            seed=seed
        )
    elif partition_method.lower() == "dirichlet":
        partitions = dirichlet_partition(
            labels=labels,
            num_clients=num_nodes,
            alpha=alpha,
            seed=seed
        )
    else:
        raise ValueError(
            f"Unknown partition method: {partition_method}. "
            f"Use 'iid' or 'dirichlet'."
        )

    return DatasetAdapter(dataset=dataset, client_partitions=partitions)


def get_cifar10_info() -> dict:
    """Get information about CIFAR-10 dataset."""
    return {
        "name": "CIFAR-10",
        "num_classes": 10,
        "input_shape": (3, 32, 32),
        "class_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ],
        "train_size": 50000,
        "test_size": 10000,
        "normalization": {
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010)
        }
    }
