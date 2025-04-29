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


def test_get_id(client_actor):
    """Test getting the client ID"""
    client_id = ray.get(client_actor.get_id.remote())
    assert client_id == "client_01"


def test_set_neighbors_and_get_neighbors(client_actor, ray_init):
    """Test setting and getting neighbors"""
    # Create some neighbor actors
    neighbor1 = VirtualClientActor.remote("neighbor_01")
    neighbor2 = VirtualClientActor.remote("neighbor_02")

    # Set neighbors
    ray.get(client_actor.set_neighbours.remote([neighbor1, neighbor2]))

    # Get neighbor IDs
    neighbor_ids = ray.get(client_actor.get_neighbours.remote())

    # Check expected IDs
    assert len(neighbor_ids) == 2
    assert "neighbor_01" in neighbor_ids
    assert "neighbor_02" in neighbor_ids

    # Clean up
    ray.kill(neighbor1)
    ray.kill(neighbor2)
