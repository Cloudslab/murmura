import pytest
import numpy as np
from datasets import Dataset
from typing import Dict, List

from murmura.data_processing.partitioner import DirichletPartitioner, IIDPartitioner


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    data = {
        "text": ["sample " + str(i) for i in range(100)],
        "label": [i % 5 for i in range(100)],  # 5 classes with 20 samples each
    }
    return Dataset.from_dict(data)


def validate_partitions(
    partitions: Dict[int, List[int]], dataset_size: int, num_partitions: int
):
    """Helper function to validate basic partition properties."""
    # Check if we have the correct number of partitions
    assert len(partitions) == num_partitions

    # Check if all indices are unique
    all_indices = []
    for indices in partitions.values():
        all_indices.extend(indices)
    assert len(set(all_indices)) == len(all_indices)

    # Check if we have all indices from 0 to dataset_size-1
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
        with pytest.raises(
            ValueError, match="Alpha length must match number of partitions"
        ):
            DirichletPartitioner(
                num_partitions=5,
                alpha=[0.1, 0.2, 0.3],  # Wrong length
                partition_by="label",
            )

    def test_init_with_invalid_alpha_values(self):
        """Test initialization with invalid alpha values."""
        with pytest.raises(ValueError, match="All alpha values must be > 0"):
            DirichletPartitioner(
                num_partitions=3,
                alpha=[0.1, -0.2, 0.3],  # Negative value
                partition_by="label",
            )

    def test_basic_partition(self, sample_dataset):
        """Test basic partitioning functionality."""
        partitioner = DirichletPartitioner(
            num_partitions=5, alpha=0.5, partition_by="label", seed=42
        )
        partitions = partitioner.partition(sample_dataset)
        validate_partitions(partitions, len(sample_dataset), 5)

    def test_reproducibility(self, sample_dataset):
        """Test if results are reproducible with same seed."""
        seed = 42
        partitioner1 = DirichletPartitioner(
            num_partitions=5, alpha=0.5, partition_by="label", seed=seed
        )
        partitioner2 = DirichletPartitioner(
            num_partitions=5, alpha=0.5, partition_by="label", seed=seed
        )

        partitions1 = partitioner1.partition(sample_dataset)
        partitions2 = partitioner2.partition(sample_dataset)

        # Check if partitions are identical
        for pid in range(5):
            assert partitions1[pid] == partitions2[pid]


class TestIIDPartitioner:
    def test_basic_partition(self, sample_dataset):
        """Test basic IID partitioning functionality."""
        partitioner = IIDPartitioner(num_partitions=5, seed=42)
        partitions = partitioner.partition(sample_dataset)
        validate_partitions(partitions, len(sample_dataset), 5)

    def test_equal_partition_sizes(self, sample_dataset):
        """Test if IID partitions are roughly equal in size."""
        partitioner = IIDPartitioner(num_partitions=5, seed=42)
        partitions = partitioner.partition(sample_dataset)

        # Get partition sizes
        sizes = [len(indices) for indices in partitions.values()]
        # Maximum difference should be at most 1 due to integer division
        assert max(sizes) - min(sizes) <= 1

    def test_reproducibility(self, sample_dataset):
        """Test if IID partitioning is reproducible with same seed."""
        seed = 42
        partitioner1 = IIDPartitioner(num_partitions=5, seed=seed)
        partitioner2 = IIDPartitioner(num_partitions=5, seed=seed)

        partitions1 = partitioner1.partition(sample_dataset)
        partitions2 = partitioner2.partition(sample_dataset)

        # Check if partitions are identical
        for pid in range(5):
            assert partitions1[pid] == partitions2[pid]

    def test_different_seeds(self, sample_dataset):
        """Test if different seeds produce different partitions."""
        partitioner1 = IIDPartitioner(num_partitions=5, seed=42)
        partitioner2 = IIDPartitioner(num_partitions=5, seed=43)

        partitions1 = partitioner1.partition(sample_dataset)
        partitions2 = partitioner2.partition(sample_dataset)

        # Check if at least one partition is different
        assert any(partitions1[pid] != partitions2[pid] for pid in range(5))
