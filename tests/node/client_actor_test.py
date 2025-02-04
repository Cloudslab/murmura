import pytest
import ray

from murmura.node.client_actor import VirtualClientActor


@pytest.fixture(scope="module")
def ray_init():
    ray.init(local_mode=True)
    yield
    ray.shutdown()


@pytest.fixture
def client_actor(ray_init):
    actor = VirtualClientActor.remote("client_01")
    yield actor
    ray.kill(actor)


def test_initial_state(client_actor):
    """Test actor initialization with correct default values"""
    info = ray.get(client_actor.get_data_info.remote())

    assert info["client_id"] == "client_01"
    assert info["data_size"] == 0
    assert info["metadata"] == {}


def test_receive_data_with_metadata(client_actor):
    """Test receiving data with metadata"""
    test_data = [1, 2, 3, 4, 5]
    metadata = {"source": "partition_1", "split": "train"}

    response = ray.get(client_actor.receive_data.remote(test_data, metadata))
    info = ray.get(client_actor.get_data_info.remote())

    assert response == "Client client_01 received 5 samples"
    assert info["data_size"] == 5
    assert info["metadata"] == metadata


def test_receive_data_without_metadata(client_actor):
    """Test receiving data without metadata"""
    test_data = [10, 20, 30]

    response = ray.get(client_actor.receive_data.remote(test_data))
    info = ray.get(client_actor.get_data_info.remote())

    assert response == "Client client_01 received 3 samples"
    assert info["data_size"] == 3
    assert info["metadata"] == {}


def test_data_overwrite(client_actor):
    """Test subsequent data receives overwrite previous data"""
    ray.get(client_actor.receive_data.remote([1, 2, 3], {"batch": 1}))
    ray.get(client_actor.receive_data.remote([4, 5], {"batch": 2}))

    info = ray.get(client_actor.get_data_info.remote())

    assert info["data_size"] == 2
    assert info["metadata"]["batch"] == 2


def test_empty_data_partition(client_actor):
    """Test receiving empty data partition"""
    response = ray.get(client_actor.receive_data.remote([]))
    info = ray.get(client_actor.get_data_info.remote())

    assert response == "Client client_01 received 0 samples"
    assert info["data_size"] == 0


def test_large_data_partition(client_actor):
    """Test receiving large data partition"""
    large_data = list(range(1000))

    response = ray.get(client_actor.receive_data.remote(large_data))
    info = ray.get(client_actor.get_data_info.remote())

    assert response == "Client client_01 received 1000 samples"
    assert info["data_size"] == 1000


def test_metadata_types(client_actor):
    """Test different metadata types are handled correctly"""
    complex_metadata = {"numeric": 42, "nested": {"key": "value"}, "list": [1, 2, 3]}

    ray.get(client_actor.receive_data.remote([1, 2], complex_metadata))
    info = ray.get(client_actor.get_data_info.remote())

    assert info["metadata"] == complex_metadata
