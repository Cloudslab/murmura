import pytest
import ray

from murmura.orchestration.cluster_manager import ClusterManager


@pytest.fixture(scope="module")
def ray_init():
    ray.init(local_mode=True)
    yield
    ray.shutdown()


@pytest.fixture
def cluster_manager():
    return ClusterManager(config={"ray_address": "auto"})


def test_create_actors(cluster_manager: ClusterManager, ray_init: None):
    """Test actor creation with proper IDs and count"""
    num_actors = 3
    actors = cluster_manager.create_actors(num_actors)

    assert len(actors) == num_actors
    for i, actor in enumerate(actors):
        info = ray.get(actor.get_data_info.remote())
        assert info["client_id"] == f"client_{i}"


def test_distribute_data_equal_partitions(
    cluster_manager: ClusterManager, ray_init: None
):
    """Test distribution with equal actors and partitions"""
    num_actors = 3
    cluster_manager.create_actors(num_actors)
    partitions = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    metadata = {"split": "train"}

    acks = cluster_manager.distribute_data(partitions, metadata)

    assert len(acks) == num_actors
    for i, actor in enumerate(cluster_manager.actors):
        info = ray.get(actor.get_data_info.remote())
        assert info["data_size"] == 3
        assert info["metadata"] == {"split": "train", "partition_idx": i}


def test_distribute_data_more_actors_than_partitions(
    cluster_manager: ClusterManager, ray_init: None
):
    """Test round-robin distribution with more actors than partitions"""
    cluster_manager.create_actors(5)
    partitions = [[1, 2], [3, 4]]

    acks = cluster_manager.distribute_data(partitions)
    expected_indices = [0, 1, 0, 1, 0]

    for idx, (ack, expected_idx) in enumerate(zip(acks, expected_indices)):
        assert f"received {len(partitions[expected_idx])}" in ack
        info = ray.get(cluster_manager.actors[idx].get_data_info.remote())
        assert info["metadata"]["partition_idx"] == expected_idx


def test_distribute_data_metadata_override(
    cluster_manager: ClusterManager, ray_init: None
):
    """Test metadata merging with partition_idx"""
    cluster_manager.create_actors(2)
    partitions = [[1], [2]]
    metadata = {"partition_idx": 99, "source": "mnist"}

    cluster_manager.distribute_data(partitions, metadata)

    for i, actor in enumerate(cluster_manager.actors):
        info = ray.get(actor.get_data_info.remote())
        assert info["metadata"] == {"source": "mnist", "partition_idx": i}


def test_distribute_data_empty_partitions(
    cluster_manager: ClusterManager, ray_init: None
):
    """Test error handling for empty partitions list"""
    cluster_manager.create_actors(2)

    with pytest.raises(ZeroDivisionError):
        cluster_manager.distribute_data([], {})


def test_shutdown():
    """Test cluster shutdown functionality"""
    if ray.is_initialized():
        ray.shutdown()

    ray.init(local_mode=True)
    assert ray.is_initialized()

    ClusterManager.shutdown()
    assert not ray.is_initialized()
