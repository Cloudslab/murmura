import pytest
from pydantic import ValidationError

from murmura.orchestration.orchestration_config import OrchestrationConfig
from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.node.resource_config import RayClusterConfig


def test_default_initialization():
    """Test default initialization of OrchestrationConfig"""
    config = OrchestrationConfig(
        feature_columns=["image"], 
        label_column="label"
    )

    # Check default values
    assert config.num_actors == 10
    assert config.ray_address is None
    assert config.dataset_name == "unknown"
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
        ray_cluster=RayClusterConfig(address="ray://192.168.1.100:10001"),
        dataset_name="cifar10",
        partition_strategy="iid",
        split="test",
        feature_columns=["image"],
        label_column="label",
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
        OrchestrationConfig(num_actors=0, feature_columns=["image"], label_column="label")

    with pytest.raises(ValidationError):
        OrchestrationConfig(num_actors=-5, feature_columns=["image"], label_column="label")


def test_invalid_alpha():
    """Test validation for alpha"""
    with pytest.raises(ValidationError):
        OrchestrationConfig(alpha=0, feature_columns=["image"], label_column="label")

    with pytest.raises(ValidationError):
        OrchestrationConfig(alpha=-0.5, feature_columns=["image"], label_column="label")


def test_invalid_min_partition_size():
    """Test validation for min_partition_size"""
    with pytest.raises(ValidationError):
        OrchestrationConfig(min_partition_size=0, feature_columns=["image"], label_column="label")

    with pytest.raises(ValidationError):
        OrchestrationConfig(min_partition_size=-10, feature_columns=["image"], label_column="label")


def test_invalid_partition_strategy():
    """Test validation for partition_strategy"""
    with pytest.raises(ValidationError):
        OrchestrationConfig(partition_strategy="invalid_strategy", feature_columns=["image"], label_column="label")


def test_model_dump():
    """Test the model_dump method for creating config dictionaries"""
    config = OrchestrationConfig(
        num_actors=5, dataset_name="cifar10", partition_strategy="iid", feature_columns=["image"], label_column="label"
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
        num_actors=3, partition_strategy="dirichlet", alpha=0.1, min_partition_size=50, feature_columns=["image"], label_column="label"
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
        topology=TopologyConfig(topology_type=TopologyType.STAR, hub_index=2), feature_columns=["image"], label_column="label"
    )

    # Check the topology configuration is properly nested
    assert config.topology.topology_type == TopologyType.STAR
    assert config.topology.hub_index == 2


def test_config_serialization_deserialization():
    """Test config serialization and deserialization round-trip"""
    config = OrchestrationConfig(
        num_actors=4,
        dataset_name="testset",
        partition_strategy="iid",
        topology=TopologyConfig(topology_type=TopologyType.LINE),
        aggregation=AggregationConfig(strategy_type=AggregationStrategyType.GOSSIP_AVG),
        feature_columns=["image"],
        label_column="label"
    )
    dumped = config.model_dump()
    loaded = OrchestrationConfig.model_validate(dumped)
    assert loaded == config


def test_config_inheritance_and_override():
    """Test config inheritance and override behavior"""
    base = OrchestrationConfig(num_actors=2, feature_columns=["image"], label_column="label")
    override = base.model_copy(update={"num_actors": 5, "partition_strategy": "iid"})
    assert override.num_actors == 5
    assert override.partition_strategy == "iid"
    assert override.topology == base.topology


def test_invalid_config_combinations():
    """Test invalid config combinations (e.g., custom topology with missing adjacency)"""
    with pytest.raises(Exception):
        OrchestrationConfig(
            topology=TopologyConfig(
                topology_type=TopologyType.CUSTOM, adjacency_list=None
            ),
            feature_columns=["image"],
            label_column="label"
        )
    with pytest.raises(Exception):
        OrchestrationConfig(
            topology=TopologyConfig(
                topology_type=TopologyType.CUSTOM, adjacency_list={}
            ),
            feature_columns=["image"],
            label_column="label"
        )


def test_nested_config_validation():
    """Test validation of nested configuration parameters"""
    config = OrchestrationConfig(
        aggregation=AggregationConfig(
            strategy_type=AggregationStrategyType.TRIMMED_MEAN,
            params={"trim_ratio": 0.2},
        ),
        topology=TopologyConfig(topology_type=TopologyType.STAR, hub_index=1),
        feature_columns=["image"],
        label_column="label"
    )
    assert config.aggregation.params["trim_ratio"] == 0.2
    assert config.topology.hub_index == 1


def test_missing_feature_columns():
    """Test validation when feature_columns is missing"""
    with pytest.raises(ValidationError, match="feature_columns must be specified"):
        OrchestrationConfig(label_column="label")


def test_missing_label_column():
    """Test validation when label_column is missing"""
    with pytest.raises(ValidationError, match="label_column must be specified"):
        OrchestrationConfig(feature_columns=["image"])


def test_empty_feature_columns():
    """Test validation when feature_columns is empty"""
    with pytest.raises(ValidationError, match="feature_columns must be a non-empty list"):
        OrchestrationConfig(feature_columns=[], label_column="label")


def test_invalid_feature_columns_type():
    """Test validation when feature_columns is not a list"""
    with pytest.raises(ValidationError):
        OrchestrationConfig(feature_columns="image", label_column="label")


def test_empty_label_column():
    """Test validation when label_column is empty string"""
    with pytest.raises(ValidationError, match="label_column must be a non-empty string"):
        OrchestrationConfig(feature_columns=["image"], label_column="")


def test_label_in_features():
    """Test validation when label_column is included in feature_columns"""
    with pytest.raises(ValidationError, match="label_column .* cannot be included in feature_columns"):
        OrchestrationConfig(feature_columns=["image", "label"], label_column="label")


def test_multinode_validation_actors_per_node():
    """Test validation for actors_per_node configuration"""
    from murmura.node.resource_config import ResourceConfig
    
    with pytest.raises(ValidationError, match="num_actors .* cannot be less than actors_per_node"):
        OrchestrationConfig(
            num_actors=5,
            resources=ResourceConfig(actors_per_node=10),
            feature_columns=["image"],
            label_column="label"
        )


def test_invalid_placement_strategy():
    """Test validation for invalid placement strategy"""
    # Since Pydantic validates at creation time, we'll use a manual validation call
    config = OrchestrationConfig(
        feature_columns=["image"],
        label_column="label"
    )
    
    # Manually set invalid strategy to test validation logic
    original_strategy = config.resources.placement_strategy
    config.resources.placement_strategy = "invalid_strategy"
    
    with pytest.raises(ValueError, match="placement_strategy must be one of"):
        config.validate_multinode_config()
    
    # Restore valid strategy
    config.resources.placement_strategy = original_strategy


def test_training_parameters_validation():
    """Test validation of training parameters"""
    # Valid training parameters
    config = OrchestrationConfig(
        rounds=10,
        epochs=3,
        batch_size=64,
        learning_rate=0.01,
        feature_columns=["image"],
        label_column="label"
    )
    assert config.rounds == 10
    assert config.epochs == 3
    assert config.batch_size == 64
    assert config.learning_rate == 0.01
    
    # Invalid rounds
    with pytest.raises(ValidationError):
        OrchestrationConfig(rounds=0, feature_columns=["image"], label_column="label")
    
    # Invalid epochs
    with pytest.raises(ValidationError):
        OrchestrationConfig(epochs=-1, feature_columns=["image"], label_column="label")
    
    # Invalid batch_size
    with pytest.raises(ValidationError):
        OrchestrationConfig(batch_size=0, feature_columns=["image"], label_column="label")
    
    # Invalid learning_rate
    with pytest.raises(ValidationError):
        OrchestrationConfig(learning_rate=0.0, feature_columns=["image"], label_column="label")


def test_sampling_parameters():
    """Test client and data sampling parameters"""
    config = OrchestrationConfig(
        client_sampling_rate=0.5,
        data_sampling_rate=0.8,
        enable_subsampling_amplification=True,
        feature_columns=["image"],
        label_column="label"
    )
    assert config.client_sampling_rate == 0.5
    assert config.data_sampling_rate == 0.8
    assert config.enable_subsampling_amplification is True
    
    # Test boundary values
    config_min = OrchestrationConfig(
        client_sampling_rate=0.01,
        data_sampling_rate=0.01,
        feature_columns=["image"],
        label_column="label"
    )
    assert config_min.client_sampling_rate == 0.01
    
    config_max = OrchestrationConfig(
        client_sampling_rate=1.0,
        data_sampling_rate=1.0,
        feature_columns=["image"],
        label_column="label"
    )
    assert config_max.data_sampling_rate == 1.0
    
    # Test invalid sampling rates
    with pytest.raises(ValidationError):
        OrchestrationConfig(client_sampling_rate=0.005, feature_columns=["image"], label_column="label")
    
    with pytest.raises(ValidationError):
        OrchestrationConfig(data_sampling_rate=1.5, feature_columns=["image"], label_column="label")


def test_monitoring_parameters():
    """Test monitoring configuration parameters"""
    config = OrchestrationConfig(
        monitor_resources=True,
        health_check_interval=10,
        feature_columns=["image"],
        label_column="label"
    )
    assert config.monitor_resources is True
    assert config.health_check_interval == 10
    
    # Invalid health_check_interval
    with pytest.raises(ValidationError):
        OrchestrationConfig(health_check_interval=0, feature_columns=["image"], label_column="label")


def test_get_resource_requirements():
    """Test get_resource_requirements method"""
    from murmura.node.resource_config import ResourceConfig
    
    config = OrchestrationConfig(
        resources=ResourceConfig(
            cpus_per_actor=2.0,
            gpus_per_actor=0.5,
            memory_per_actor=1024  # MB
        ),
        feature_columns=["image"],
        label_column="label"
    )
    
    requirements = config.get_resource_requirements()
    
    assert requirements["num_cpus"] == 2.0
    assert requirements["num_gpus"] == 0.5
    assert requirements["memory"] == 1024 * 1024 * 1024  # Converted to bytes


def test_get_resource_requirements_none_values():
    """Test get_resource_requirements with None values"""
    from murmura.node.resource_config import ResourceConfig
    
    config = OrchestrationConfig(
        resources=ResourceConfig(
            cpus_per_actor=None,
            gpus_per_actor=None,
            memory_per_actor=None
        ),
        feature_columns=["image"],
        label_column="label"
    )
    
    requirements = config.get_resource_requirements()
    
    assert requirements == {}


def test_get_placement_group_strategy():
    """Test get_placement_group_strategy method"""
    from murmura.node.resource_config import ResourceConfig
    
    strategies = ["spread", "pack", "strict_spread", "strict_pack"]
    expected_mappings = ["SPREAD", "PACK", "STRICT_SPREAD", "STRICT_PACK"]
    
    for strategy, expected in zip(strategies, expected_mappings):
        config = OrchestrationConfig(
            resources=ResourceConfig(placement_strategy=strategy),
            feature_columns=["image"],
            label_column="label"
        )
        assert config.get_placement_group_strategy() == expected


def test_ray_address_backward_compatibility():
    """Test ray_address property for backward compatibility"""
    from murmura.node.resource_config import RayClusterConfig
    
    config = OrchestrationConfig(
        ray_cluster=RayClusterConfig(address="ray://localhost:10001"),
        feature_columns=["image"],
        label_column="label"
    )
    
    assert config.ray_address == "ray://localhost:10001"
    
    # Test None case
    config_none = OrchestrationConfig(
        feature_columns=["image"],
        label_column="label"
    )
    assert config_none.ray_address is None


def test_test_split_parameter():
    """Test test_split parameter"""
    config = OrchestrationConfig(
        test_split="validation",
        feature_columns=["image"],
        label_column="label"
    )
    assert config.test_split == "validation"
    
    # Default value
    config_default = OrchestrationConfig(
        feature_columns=["image"],
        label_column="label"
    )
    assert config_default.test_split == "test"


def test_multi_feature_columns():
    """Test configuration with multiple feature columns"""
    config = OrchestrationConfig(
        feature_columns=["image", "text", "metadata"],
        label_column="category",
    )
    assert len(config.feature_columns) == 3
    assert "image" in config.feature_columns
    assert "text" in config.feature_columns
    assert "metadata" in config.feature_columns


def test_config_comprehensive_example():
    """Test a comprehensive configuration example"""
    from murmura.node.resource_config import ResourceConfig, RayClusterConfig
    
    config = OrchestrationConfig(
        # Core configuration
        num_actors=20,
        topology=TopologyConfig(topology_type=TopologyType.RING),
        aggregation=AggregationConfig(strategy_type=AggregationStrategyType.FEDAVG),
        
        # Dataset configuration
        dataset_name="cifar10",
        partition_strategy="dirichlet",
        alpha=0.3,
        min_partition_size=200,
        split="train",
        test_split="test",
        feature_columns=["image"],
        label_column="label",
        
        # Training parameters
        rounds=20,
        epochs=2,
        batch_size=128,
        learning_rate=0.001,
        
        # Sampling parameters
        client_sampling_rate=0.6,
        data_sampling_rate=0.9,
        enable_subsampling_amplification=True,
        
        # Monitoring
        monitor_resources=True,
        health_check_interval=3,
        
        # Multi-node configuration
        ray_cluster=RayClusterConfig(address="ray://head-node:10001"),
        resources=ResourceConfig(
            cpus_per_actor=1.0,
            gpus_per_actor=0.25,
            memory_per_actor=512,
            actors_per_node=5,
            placement_strategy="spread"
        )
    )
    
    # Validate all fields are set correctly
    assert config.num_actors == 20
    assert config.dataset_name == "cifar10"
    assert config.rounds == 20
    assert config.client_sampling_rate == 0.6
    assert config.monitor_resources is True
    assert config.ray_address == "ray://head-node:10001"
    
    # Test resource requirements
    requirements = config.get_resource_requirements()
    assert requirements["num_cpus"] == 1.0
    assert requirements["num_gpus"] == 0.25
    
    # Test placement strategy
    assert config.get_placement_group_strategy() == "SPREAD"
