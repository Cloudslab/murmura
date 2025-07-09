import pytest
import ray

from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.orchestration.cluster_manager import ClusterManager
from murmura.orchestration.orchestration_config import OrchestrationConfig


@pytest.fixture(scope="module")
def ray_init():
    """Initialize Ray for the tests"""
    ray.init(local_mode=True)
    yield
    ray.shutdown()


@pytest.fixture
def cluster_manager(ray_init):
    """Create a basic cluster manager"""
    config = OrchestrationConfig(
        feature_columns=["image"],
        label_column="label"
    )
    return ClusterManager(config=config)


def test_create_actors(cluster_manager, ray_init):
    """Test actor creation with proper IDs and count"""
    num_actors = 3
    # Use a basic topology (COMPLETE) for actor creation
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    actors = cluster_manager.create_actors(num_actors, topology)

    assert len(actors) == num_actors
    for i, actor in enumerate(actors):
        info = ray.get(actor.get_data_info.remote())
        assert info["client_id"] == f"client_{i}"


def test_distribute_data_equal_partitions(cluster_manager, ray_init):
    """Test distribution with equal actors and partitions"""
    num_actors = 3
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    cluster_manager.create_actors(num_actors, topology)
    partitions = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    metadata = {"split": "train"}

    acks = cluster_manager.distribute_data(partitions, metadata)

    # Expect an acknowledgment per actor
    assert len(acks) == num_actors
    for i, actor in enumerate(cluster_manager.actors):
        info = ray.get(actor.get_data_info.remote())
        # Each partition has 3 items
        assert info["data_size"] == 3
        # Metadata should include the provided metadata with partition_idx overridden per actor
        assert info["metadata"]["split"] == "train"
        assert info["metadata"]["partition_idx"] == i


def test_distribute_data_more_actors_than_partitions(cluster_manager, ray_init):
    """Test round-robin distribution when there are more actors than partitions"""
    # Create 5 actors, but provide only 2 partitions. Distribution will wrap around.
    num_actors = 5
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    cluster_manager.create_actors(num_actors, topology)
    partitions = [[1, 2], [3, 4]]
    acks = cluster_manager.distribute_data(partitions)
    # Expected: actor 0 gets partition 0, actor 1 gets partition 1, actor 2 gets partition 0, etc.
    expected_indices = [0, 1, 0, 1, 0]

    # Verify that each actor's metadata includes the correct partition index and
    # that the acknowledgment message contains the number of items received.
    for idx, expected_idx in enumerate(expected_indices):
        ack = acks[idx]
        # For example, ack might be "received 2 items" (adjust based on your actor's implementation)
        assert f"received {len(partitions[expected_idx])}" in ack
        info = ray.get(cluster_manager.actors[idx].get_data_info.remote())
        assert info["metadata"]["partition_idx"] == expected_idx


def test_distribute_data_metadata_override(cluster_manager, ray_init):
    """Test that metadata provided to distribute_data properly overrides any defaults"""
    num_actors = 2
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    cluster_manager.create_actors(num_actors, topology)
    partitions = [[1], [2]]
    # Here, even if metadata contains a "partition_idx", it should be overridden per actor.
    metadata = {"partition_idx": 99, "source": "mnist"}

    cluster_manager.distribute_data(partitions, metadata)

    for i, actor in enumerate(cluster_manager.actors):
        info = ray.get(actor.get_data_info.remote())
        # The actor's metadata should have "source" preserved and "partition_idx" replaced by the correct index.
        assert info["metadata"]["source"] == "mnist"
        assert info["metadata"]["partition_idx"] == i


def test_get_topology_information_not_initialized(cluster_manager):
    """Test getting topology information when not initialized"""
    # No topology manager yet
    info = cluster_manager.get_topology_information()

    assert info["initialized"] is False


def test_get_compatible_strategies_without_topology(cluster_manager):
    """Test getting compatible strategies without a topology"""
    # No topology set yet
    strategies = cluster_manager.get_compatible_strategies()

    # Should return empty list
    assert strategies == []


def test_aggregate_model_parameters_without_strategy(cluster_manager):
    """Test that aggregating without a strategy raises an error"""
    # Create actors
    topology = TopologyConfig(topology_type=TopologyType.STAR)
    cluster_manager.create_actors(2, topology)

    # Try to aggregate without setting strategy
    with pytest.raises(ValueError, match="Aggregation strategy not set"):
        cluster_manager.aggregate_model_parameters()


def test_aggregate_model_parameters_without_coordinator(cluster_manager):
    """Test that aggregating without a coordinator raises an error"""
    # Set strategy but don't create actors/coordinator
    aggregation_config = AggregationConfig(strategy_type=AggregationStrategyType.FEDAVG)
    cluster_manager.set_aggregation_strategy(aggregation_config)

    # Try to aggregate without coordinator
    with pytest.raises(ValueError, match="Topology coordinator not initialized"):
        cluster_manager.aggregate_model_parameters()


def test_get_cluster_stats(cluster_manager):
    """Test getting cluster statistics"""
    stats = cluster_manager.get_cluster_stats()
    
    assert "cluster_info" in stats
    assert "num_actors" in stats
    assert "placement_strategy" in stats
    assert "has_placement_group" in stats
    assert "resource_config" in stats
    assert stats["num_actors"] == 0  # No actors created yet


def test_set_aggregation_strategy(cluster_manager):
    """Test setting aggregation strategy"""
    aggregation_config = AggregationConfig(strategy_type=AggregationStrategyType.FEDAVG)
    cluster_manager.set_aggregation_strategy(aggregation_config)
    
    assert cluster_manager.aggregation_strategy is not None
    assert cluster_manager.aggregation_strategy.__class__.__name__ == "FedAvg"


def test_get_compatible_strategies_with_topology(cluster_manager):
    """Test getting compatible strategies with a topology"""
    num_actors = 3
    topology = TopologyConfig(topology_type=TopologyType.STAR)
    cluster_manager.create_actors(num_actors, topology)
    
    strategies = cluster_manager.get_compatible_strategies()
    
    assert isinstance(strategies, list)
    assert len(strategies) > 0
    assert "fedavg" in strategies

def test_shutdown():
    """Test cluster shutdown functionality"""
    # If Ray is running, shut it down first.
    if ray.is_initialized():
        ray.shutdown()

    ray.init(local_mode=True)
    assert ray.is_initialized()

    ClusterManager.shutdown_ray()
    assert not ray.is_initialized()
