"""Wearable dataset adapters for Murmura with Dirichlet partitioning support."""

from typing import Optional, List

from murmura.data.adapters import DatasetAdapter
from murmura.data.partitioners import (
    dirichlet_partition,
    iid_partition,
    natural_partition,
)
from murmura.examples.wearables.datasets import (
    UCIHARDataset,
    PAMAP2Dataset,
    PPGDaLiADataset,
)


def load_wearable_adapter(
    dataset_type: str,
    data_path: str,
    num_nodes: int,
    partition_method: str = "dirichlet",
    alpha: float = 0.5,
    seed: int = 42,
    split: str = "train",
    max_samples: Optional[int] = None,
    **kwargs,
) -> DatasetAdapter:
    """Load a wearable dataset as a Murmura DatasetAdapter with configurable partitioning.

    Args:
        dataset_type: Type of dataset ('uci_har', 'pamap2', 'ppg_dalia')
        data_path: Path to the dataset directory
        num_nodes: Number of nodes/clients to partition data across
        partition_method: Partitioning strategy:
            - 'dirichlet': Non-IID using Dirichlet distribution (default)
            - 'iid': Uniform random partitioning
            - 'natural': Use dataset's natural client/subject IDs
        alpha: Dirichlet concentration parameter (only for 'dirichlet' method)
               Lower values = more non-IID (typical: 0.1-1.0)
        seed: Random seed for reproducibility
        split: 'train' or 'test' (for UCI HAR)
        max_samples: Maximum samples per client (optional truncation)
        **kwargs: Dataset-specific parameters passed to dataset constructor

    Returns:
        DatasetAdapter configured for the wearable dataset

    Raises:
        ValueError: If unknown dataset type or partition method

    Examples:
        # Non-IID UCI HAR with extreme heterogeneity
        adapter = load_wearable_adapter(
            dataset_type='uci_har',
            data_path='wearables_datasets/UCI HAR Dataset',
            num_nodes=10,
            partition_method='dirichlet',
            alpha=0.1,  # Very non-IID
        )

        # IID PAMAP2
        adapter = load_wearable_adapter(
            dataset_type='pamap2',
            data_path='wearables_datasets/PAMAP2_Dataset',
            num_nodes=9,
            partition_method='natural',  # Use subject IDs
        )
    """
    # Load the dataset
    dataset, labels, natural_ids = _load_dataset(
        dataset_type, data_path, split, **kwargs
    )

    # Create partitions based on method
    if partition_method == "dirichlet":
        partitions = dirichlet_partition(
            labels=labels,
            num_clients=num_nodes,
            alpha=alpha,
            seed=seed,
        )
    elif partition_method == "iid":
        partitions = iid_partition(
            num_samples=len(dataset),
            num_clients=num_nodes,
            seed=seed,
        )
    elif partition_method == "natural":
        if natural_ids is None:
            raise ValueError(
                f"Dataset '{dataset_type}' does not support natural partitioning"
            )
        partitions, actual_clients = natural_partition(
            client_ids=natural_ids,
            num_clients=num_nodes,
        )
        if actual_clients < num_nodes:
            print(
                f"Warning: Only {actual_clients} natural clients available, "
                f"requested {num_nodes}"
            )
    else:
        raise ValueError(f"Unknown partition method: {partition_method}")

    # Optionally truncate samples per client
    if max_samples is not None:
        partitions = [indices[:max_samples] for indices in partitions]

    return DatasetAdapter(dataset=dataset, client_partitions=partitions)


def _load_dataset(
    dataset_type: str,
    data_path: str,
    split: str,
    **kwargs,
):
    """Load dataset and return (dataset, labels, natural_ids).

    Returns:
        Tuple of (dataset, labels_array, natural_client_ids_or_None)
    """
    dataset_type = dataset_type.lower().replace("-", "_")

    if dataset_type == "uci_har":
        dataset = UCIHARDataset(
            root=data_path,
            split=split,
            normalize=kwargs.get("normalize", True),
        )
        labels = dataset.get_labels()
        natural_ids = dataset.get_subjects()
        return dataset, labels, natural_ids

    elif dataset_type == "pamap2":
        dataset = PAMAP2Dataset(
            root=data_path,
            subjects=kwargs.get("subjects"),
            activities=kwargs.get("activities"),
            window_size=kwargs.get("window_size", 100),
            window_stride=kwargs.get("window_stride", 50),
            normalize=kwargs.get("normalize", True),
            include_heart_rate=kwargs.get("include_heart_rate", True),
        )
        labels = dataset.get_labels()
        natural_ids = dataset.get_subjects()
        return dataset, labels, natural_ids

    elif dataset_type == "ppg_dalia":
        dataset = PPGDaLiADataset(
            root=data_path,
            subjects=kwargs.get("subjects"),
            activities=kwargs.get("activities"),
            window_size=kwargs.get("window_size", 32),
            window_stride=kwargs.get("window_stride", 16),
            normalize=kwargs.get("normalize", True),
            use_wrist_only=kwargs.get("use_wrist_only", True),
        )
        labels = dataset.get_labels()
        natural_ids = dataset.get_subjects()
        return dataset, labels, natural_ids

    else:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. "
            f"Available: 'uci_har', 'pamap2', 'ppg_dalia'"
        )


def get_wearable_dataset_info(dataset_type: str) -> dict:
    """Get information about a wearable dataset.

    Args:
        dataset_type: Type of dataset ('uci_har', 'pamap2', 'ppg_dalia')

    Returns:
        Dictionary with dataset metadata
    """
    info = {
        "uci_har": {
            "name": "UCI Human Activity Recognition",
            "num_features": 561,
            "num_classes": 6,
            "natural_clients": 30,
            "activities": list(UCIHARDataset.ACTIVITIES.values()),
            "description": "Smartphone-based activity recognition from 30 subjects",
        },
        "pamap2": {
            "name": "PAMAP2 Physical Activity Monitoring",
            "num_features": "variable (window_size * 40)",
            "num_classes": 12,
            "natural_clients": 9,
            "activities": list(PAMAP2Dataset.ACTIVITY_NAMES.values()),
            "description": "IMU-based activity recognition from 9 subjects",
        },
        "ppg_dalia": {
            "name": "PPG-DaLiA Activity Recognition",
            "num_features": "variable (window_size * 6)",
            "num_classes": 7,
            "natural_clients": 15,
            "activities": list(PPGDaLiADataset.ACTIVITY_NAMES.values()),
            "description": "PPG/wearable-based activity recognition from 15 subjects",
        },
    }

    dataset_type = dataset_type.lower().replace("-", "_")
    if dataset_type not in info:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return info[dataset_type]
