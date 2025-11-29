"""LEAF dataset adapters for Murmura."""

from murmura.data.adapters import DatasetAdapter
from murmura.examples.leaf.datasets import load_leaf_dataset, create_leaf_client_partitions


def load_leaf_adapter(dataset_type: str, **kwargs) -> DatasetAdapter:
    """Load LEAF dataset as a Murmura DatasetAdapter.

    Args:
        dataset_type: Type of LEAF dataset ('femnist' or 'celeba')
        **kwargs: Dataset-specific parameters (e.g., data_path, split)

    Returns:
        DatasetAdapter configured for LEAF dataset
    """
    data_path = kwargs.get("data_path", f"leaf/data/{dataset_type}/data")
    split = kwargs.get("split", "train")
    max_samples = kwargs.get("max_samples", None)

    # Load LEAF dataset
    train_dataset, test_dataset = load_leaf_dataset(
        dataset_name=dataset_type,
        data_path=data_path
    )

    # Use train or test based on split
    dataset = train_dataset if split == "train" else test_dataset

    # Create client partitions
    client_partitions = create_leaf_client_partitions(
        dataset, max_samples_per_client=max_samples
    )

    # Wrap in DatasetAdapter
    return DatasetAdapter(
        dataset=dataset,
        client_partitions=client_partitions
    )
