import pytest
import numpy as np
from datasets import Dataset, DatasetDict
from typing import Dict, List

from murmura.data_processing.partitioner import DirichletPartitioner, IIDPartitioner
from murmura.data_processing.dataset import MDataset  # Add MDataset import


@pytest.fixture
def sample_mdataset():
    """Create a sample MDataset for testing."""
    data = {
        "text": ["sample " + str(i) for i in range(100)],
        "label": [i % 5 for i in range(100)],  # 5 classes with 20 samples each
    }
    hf_dataset = Dataset.from_dict(data)
    return MDataset(DatasetDict({"train": hf_dataset}))


def validate_partitions(
    partitions: Dict[int, List[int]], dataset_size: int, num_partitions: int
):
    """Helper function to validate basic partition properties."""
    assert len(partitions) == num_partitions
    all_indices = [idx for indices in partitions.values() for idx in indices]
    assert len(set(all_indices)) == len(all_indices)
    assert set(all_indices) == set(range(dataset_size))


class TestDirichletPartitioner:
    def test_init_with_float_alpha(self):
        """Test initialization with float alpha."""
        partitioner = DirichletPartitioner(
            num_partitions=5, alpha=0.5, partition_by="label"
        )
        assert len(partitioner.alpha) == 5
        assert np.all(partitioner.alpha == 0.5)

    def test_init_with_list_alpha(self):
        """Test initialization with list alpha."""
        alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        partitioner = DirichletPartitioner(
            num_partitions=5, alpha=alpha_list, partition_by="label"
        )
        assert np.array_equal(partitioner.alpha, np.array(alpha_list))

    def test_init_with_invalid_alpha_length(self):
        """Test initialization with invalid alpha length."""
        with pytest.raises(ValueError):
            DirichletPartitioner(
                num_partitions=5,
                alpha=[0.1, 0.2, 0.3],
                partition_by="label",
            )

    def test_init_with_invalid_alpha_values(self):
        """Test initialization with invalid alpha values."""
        with pytest.raises(ValueError):
            DirichletPartitioner(
                num_partitions=3,
                alpha=[0.1, -0.2, 0.3],
                partition_by="label",
            )

    def test_basic_partition(self, sample_mdataset):
        """Test basic partitioning functionality."""
        partitioner = DirichletPartitioner(
            num_partitions=5, alpha=0.5, partition_by="label", seed=42
        )
        partitioner.partition(sample_mdataset, "train")
        partitions = sample_mdataset.get_partitions("train")
        split_size = len(sample_mdataset.get_split("train"))
        validate_partitions(partitions, split_size, 5)

    def test_reproducibility(self):
        """Test if results are reproducible with same seed."""
        # Create independent datasets
        data = {"text": ["text"] * 100, "label": [i % 5 for i in range(100)]}
        ds1 = MDataset(DatasetDict({"train": Dataset.from_dict(data)}))
        ds2 = MDataset(DatasetDict({"train": Dataset.from_dict(data)}))

        partitioner1 = DirichletPartitioner(
            num_partitions=5, alpha=0.5, partition_by="label", seed=42
        )
        partitioner2 = DirichletPartitioner(
            num_partitions=5, alpha=0.5, partition_by="label", seed=42
        )

        partitioner1.partition(ds1, "train")
        partitioner2.partition(ds2, "train")

        for pid in range(5):
            assert ds1.get_partitions("train")[pid] == ds2.get_partitions("train")[pid]


class TestIIDPartitioner:
    def test_basic_partition(self, sample_mdataset):
        """Test basic IID partitioning functionality."""
        partitioner = IIDPartitioner(num_partitions=5, seed=42)
        partitioner.partition(sample_mdataset, "train")
        partitions = sample_mdataset.get_partitions("train")
        split_size = len(sample_mdataset.get_split("train"))
        validate_partitions(partitions, split_size, 5)

    def test_equal_partition_sizes(self, sample_mdataset):
        """Test if IID partitions are roughly equal in size."""
        partitioner = IIDPartitioner(num_partitions=5, seed=42)
        partitioner.partition(sample_mdataset, "train")
        partitions = sample_mdataset.get_partitions("train")
        sizes = [len(indices) for indices in partitions.values()]
        assert max(sizes) - min(sizes) <= 1

    def test_reproducibility(self):
        """Test if IID partitioning is reproducible with same seed."""
        data = {"text": ["text"] * 100, "label": [0] * 100}
        ds1 = MDataset(DatasetDict({"train": Dataset.from_dict(data)}))
        ds2 = MDataset(DatasetDict({"train": Dataset.from_dict(data)}))

        partitioner1 = IIDPartitioner(num_partitions=5, seed=42)
        partitioner2 = IIDPartitioner(num_partitions=5, seed=42)

        partitioner1.partition(ds1, "train")
        partitioner2.partition(ds2, "train")

        for pid in range(5):
            assert ds1.get_partitions("train")[pid] == ds2.get_partitions("train")[pid]

    def test_different_seeds(self):
        """Test if different seeds produce different partitions."""
        data = {"text": ["text"] * 100, "label": [0] * 100}
        ds1 = MDataset(DatasetDict({"train": Dataset.from_dict(data)}))
        ds2 = MDataset(DatasetDict({"train": Dataset.from_dict(data)}))

        partitioner1 = IIDPartitioner(num_partitions=5, seed=42)
        partitioner2 = IIDPartitioner(num_partitions=5, seed=43)

        partitioner1.partition(ds1, "train")
        partitioner2.partition(ds2, "train")

        different = any(
            ds1.get_partitions("train")[pid] != ds2.get_partitions("train")[pid]
            for pid in range(5)
        )
        assert different
