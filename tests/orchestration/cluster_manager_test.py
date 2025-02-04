import pytest
import ray

from murmura.orchestration.cluster_manager import ClusterManager
from murmura.network_management.topology import TopologyConfig, TopologyType


@pytest.fixture(scope="module")
def ray_init():
    ray.init(local_mode=True)
    yield
    ray.shutdown()


@pytest.fixture
def cluster_manager():
    # Create a cluster manager with a config that points to a local Ray cluster.
    return ClusterManager(config={"ray_address": "auto"})


def test_create_actors(cluster_manager: ClusterManager, ray_init: None):
    """Test actor creation with proper IDs and count"""
    num_actors = 3
    # Use a basic topology (COMPLETE) for actor creation
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    actors = cluster_manager.create_actors(num_actors, topology)

    assert len(actors) == num_actors
    for i, actor in enumerate(actors):
        info = ray.get(actor.get_data_info.remote())
        assert info["client_id"] == f"client_{i}"


def test_distribute_data_equal_partitions(
    cluster_manager: ClusterManager, ray_init: None
):
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
        assert info["metadata"] == {"split": "train", "partition_idx": i}


def test_distribute_data_more_actors_than_partitions(
    cluster_manager: ClusterManager, ray_init: None
):
    """Test round-robin distribution when there are more actors than partitions"""
    # Create 5 actors, but provide only 2 partitions. Distribution will wrap around.
    num_actors = 5
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    cluster_manager.create_actors(num_actors, topology)
    partitions = [[1, 2], [3, 4]]
    acks = cluster_manager.distribute_data(partitions)
    # Expected: actor 0 gets partition 0, actor 1 gets partition 1, actor 2 gets partition 0, etc.
    expected_indices = [0, 1, 0, 1, 0]

    # Verify that each actor’s metadata includes the correct partition index and
    # that the acknowledgment message contains the number of items received.
    for idx, expected_idx in enumerate(expected_indices):
        ack = acks[idx]
        # For example, ack might be "received 2 items" (adjust based on your actor's implementation)
        assert f"received {len(partitions[expected_idx])}" in ack
        info = ray.get(cluster_manager.actors[idx].get_data_info.remote())
        assert info["metadata"]["partition_idx"] == expected_idx


def test_distribute_data_metadata_override(
    cluster_manager: ClusterManager, ray_init: None
):
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
        assert info["metadata"] == {"source": "mnist", "partition_idx": i}


def test_distribute_data_empty_partitions(
    cluster_manager: ClusterManager, ray_init: None
):
    """Test error handling for empty partitions list"""
    num_actors = 2
    topology = TopologyConfig(topology_type=TopologyType.COMPLETE)
    cluster_manager.create_actors(num_actors, topology)

    with pytest.raises(ZeroDivisionError):
        # When the partitions list is empty, a modulo operation will trigger a ZeroDivisionError.
        cluster_manager.distribute_data([], {})


def test_shutdown():
    """Test cluster shutdown functionality"""
    # If Ray is running, shut it down first.
    if ray.is_initialized():
        ray.shutdown()

    ray.init(local_mode=True)
    assert ray.is_initialized()

    ClusterManager.shutdown()
    assert not ray.is_initialized()
