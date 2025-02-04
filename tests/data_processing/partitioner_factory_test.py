from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError

from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.data_processing.partitioner import DirichletPartitioner, IIDPartitioner
from murmura.data_processing.partitioner_factory import PartitionerFactory


def test_create_dirichlet_partitioner():
    """Test creating DirichletPartitioner with valid configuration"""
    config = OrchestrationConfig(
        num_actors=5, partition_strategy="dirichlet", alpha=0.5, min_partition_size=100
    )

    partitioner = PartitionerFactory.create(config)

    assert isinstance(partitioner, DirichletPartitioner)
    assert partitioner.num_partitions == 5
    assert isinstance(partitioner.alpha, np.ndarray)
    assert partitioner.alpha.shape == (5,)
    assert np.allclose(partitioner.alpha, 0.5)
    assert partitioner.min_partition_size == 100
    assert partitioner.partition_by == "label"


def test_create_iid_partitioner():
    """Test creating IIDPartitioner with valid configuration"""
    config = OrchestrationConfig(num_actors=10, partition_strategy="iid")

    partitioner = PartitionerFactory.create(config)

    assert isinstance(partitioner, IIDPartitioner)
    assert partitioner.num_partitions == 10
    assert partitioner.shuffle is True


def test_invalid_strategy_through_config():
    """Test invalid partition strategy via config validation"""
    with pytest.raises(ValidationError) as exc_info:
        OrchestrationConfig(num_actors=5, partition_strategy="invalid_strategy")

    assert "partition_strategy" in str(exc_info.value)
    assert "Input should be 'dirichlet' or 'iid'" in str(exc_info.value)


def test_dirichlet_parameters_validation():
    """Test Dirichlet parameter validation through config"""
    with pytest.raises(ValidationError) as exc_info:
        OrchestrationConfig(
            num_actors=5,
            partition_strategy="dirichlet",
            alpha=-0.5,  # Invalid alpha
        )

    assert "alpha" in str(exc_info.value)
    assert "greater than 0" in str(exc_info.value)


def test_iid_partitioner_defaults():
    """Test IIDPartitioner uses correct default values"""
    config = OrchestrationConfig(num_actors=8, partition_strategy="iid")

    partitioner = PartitionerFactory.create(config)

    assert isinstance(partitioner, IIDPartitioner)
    assert partitioner.num_partitions == 8
    assert partitioner.seed is not None


def test_dirichlet_min_partition_size():
    """Test DirichletPartitioner receives min_partition_size"""
    config = OrchestrationConfig(
        num_actors=7, partition_strategy="dirichlet", min_partition_size=50
    )

    partitioner = cast(DirichletPartitioner, PartitionerFactory.create(config))
    assert partitioner.min_partition_size == 50
