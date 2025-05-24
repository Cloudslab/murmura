import pytest
from pydantic import ValidationError

from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.network_management.topology import TopologyConfig, TopologyType


def test_default_initialization():
    """Test default initialization of OrchestrationConfig"""
    config = OrchestrationConfig()

    # Check default values
    assert config.num_actors == 10
    assert config.ray_address is None
    assert config.dataset_name == "mnist"
    assert config.partition_strategy == "dirichlet"
    assert config.alpha == 0.5
    assert config.min_partition_size == 100
    assert config.split == "train"

    # Check nested configs
    assert isinstance(config.topology, TopologyConfig)
    assert config.topology.topology_type == TopologyType.COMPLETE

    assert isinstance(config.aggregation, AggregationConfig)
    assert config.aggregation.strategy_type == AggregationStrategyType.FEDAVG


def test_custom_initialization():
    """Test initialization with custom values"""
    config = OrchestrationConfig(
        num_actors=5,
        ray_address="ray://192.168.1.100:10001",
        dataset_name="cifar10",
        partition_strategy="iid",
        split="test",
        topology=TopologyConfig(topology_type=TopologyType.STAR, hub_index=1),
        aggregation=AggregationConfig(
            strategy_type=AggregationStrategyType.TRIMMED_MEAN
        ),
    )

    assert config.num_actors == 5
    assert config.ray_address == "ray://192.168.1.100:10001"
    assert config.dataset_name == "cifar10"
    assert config.partition_strategy == "iid"
    assert config.split == "test"

    assert config.topology.topology_type == TopologyType.STAR
    assert config.topology.hub_index == 1

    assert config.aggregation.strategy_type == AggregationStrategyType.TRIMMED_MEAN


def test_invalid_num_actors():
    """Test validation for num_actors"""
    with pytest.raises(ValidationError):
        OrchestrationConfig(num_actors=0)

    with pytest.raises(ValidationError):
        OrchestrationConfig(num_actors=-5)


def test_invalid_alpha():
    """Test validation for alpha"""
    with pytest.raises(ValidationError):
        OrchestrationConfig(alpha=0)

    with pytest.raises(ValidationError):
        OrchestrationConfig(alpha=-0.5)


def test_invalid_min_partition_size():
    """Test validation for min_partition_size"""
    with pytest.raises(ValidationError):
        OrchestrationConfig(min_partition_size=0)

    with pytest.raises(ValidationError):
        OrchestrationConfig(min_partition_size=-10)


def test_invalid_partition_strategy():
    """Test validation for partition_strategy"""
    with pytest.raises(ValidationError):
        OrchestrationConfig(partition_strategy="invalid_strategy")


def test_model_dump():
    """Test the model_dump method for creating config dictionaries"""
    config = OrchestrationConfig(
        num_actors=5, dataset_name="cifar10", partition_strategy="iid"
    )

    # Dump the config to dict
    config_dict = config.model_dump()

    # Check dict structure
    assert isinstance(config_dict, dict)
    assert config_dict["num_actors"] == 5
    assert config_dict["dataset_name"] == "cifar10"
    assert config_dict["partition_strategy"] == "iid"

    # Check nested configs are also dumped
    assert "topology" in config_dict
    assert "aggregation" in config_dict

    # Make sure the nested configs are dictionaries
    assert isinstance(config_dict["topology"], dict)
    assert isinstance(config_dict["aggregation"], dict)


def test_compatibility_with_partitioner():
    """Test that the config can be properly used with a partitioner factory"""
    # Create a config for dirichlet partitioning
    config = OrchestrationConfig(
        num_actors=3, partition_strategy="dirichlet", alpha=0.1, min_partition_size=50
    )

    # Check relevant fields for partitioner
    assert config.partition_strategy == "dirichlet"
    assert config.num_actors == 3
    assert config.alpha == 0.1
    assert config.min_partition_size == 50


def test_compatibility_with_topology():
    """Test that the config can be properly used with topology manager"""
    # Create a config with a star topology
    config = OrchestrationConfig(
        topology=TopologyConfig(topology_type=TopologyType.STAR, hub_index=2)
    )

    # Check the topology configuration is properly nested
    assert config.topology.topology_type == TopologyType.STAR
    assert config.topology.hub_index == 2
