"""LEAF dataset adapters for Murmura."""

from typing import List

from murmura.data.adapters import DatasetAdapter
from murmura.examples.leaf.datasets import (
    create_leaf_client_partitions,
    load_leaf_dataset,
)


def _truncate_partitions(partitions: List[List[int]], max_samples: int) -> List[List[int]]:
    """Optionally cap per-client samples for quick tests."""
    if max_samples is None:
        return partitions
    return [indices[:max_samples] for indices in partitions]


def load_leaf_adapter(dataset_type: str, **kwargs) -> DatasetAdapter:
    """Load LEAF dataset as a Murmura DatasetAdapter.

    Args:
        dataset_type: Type of LEAF dataset ('femnist' or 'celeba')
        **kwargs: Dataset-specific parameters (e.g., data_path, split, num_nodes)

    Returns:
        DatasetAdapter configured for LEAF dataset
    """
    data_path = kwargs.get("data_path", f"leaf/data/{dataset_type}/data")
    split = kwargs.get("split", "train")
    max_samples = kwargs.get("max_samples")
    num_nodes = kwargs.get("num_nodes") or kwargs.get("num_clients")
    seed = kwargs.get("seed", 42)

    if num_nodes is None:
        raise ValueError("num_nodes is required for LEAF adapter (pass topology.num_nodes).")

    # Load LEAF dataset (returns train/test and metadata)
    train_dataset, test_dataset, _, _, _ = load_leaf_dataset(
        dataset_name=dataset_type,
        data_path=data_path,
    )

    # Create client partitions for both splits
    train_partitions, test_partitions = create_leaf_client_partitions(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_nodes=num_nodes,
        seed=seed,
    )

    if split == "train":
        dataset = train_dataset
        partitions = train_partitions
    else:
        dataset = test_dataset
        partitions = test_partitions

    partitions = _truncate_partitions(partitions, max_samples)

    return DatasetAdapter(dataset=dataset, client_partitions=partitions)
