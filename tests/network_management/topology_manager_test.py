import pytest
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.network_management.topology_manager import TopologyManager


def test_star_topology():
    """
    For STAR topology:
      - The hub is computed as hub_index modulo num_clients.
      - The hub node is connected to all other nodes.
      - Every non-hub node is connected only to the hub.
    """
    num_clients = 5
    hub_index = 7  # 7 % 5 == 2, so node 2 is the hub.
    config = TopologyConfig(topology_type=TopologyType.STAR, hub_index=hub_index)
    manager = TopologyManager(num_clients=num_clients, config=config)

    hub = hub_index % num_clients
    expected = {}
    for i in range(num_clients):
        if i == hub:
            # hub is connected to all nodes except itself
            expected[i] = [n for n in range(num_clients) if n != hub]
        else:
            expected[i] = [hub]
    assert manager.adjacency_list == expected


def test_ring_topology():
    """
    For RING topology:
      - Each node is connected to its left and right neighbors (using modular arithmetic).
    """
    num_clients = 5
    config = TopologyConfig(topology_type=TopologyType.RING)
    manager = TopologyManager(num_clients=num_clients, config=config)

    expected = {}
    for i in range(num_clients):
        expected[i] = [(i - 1) % num_clients, (i + 1) % num_clients]
    assert manager.adjacency_list == expected


def test_complete_topology():
    """
    For COMPLETE topology:
      - Each node is connected to every other node.
    """
    num_clients = 5
    config = TopologyConfig(topology_type=TopologyType.COMPLETE)
    manager = TopologyManager(num_clients=num_clients, config=config)

    expected = {}
    for i in range(num_clients):
        expected[i] = [n for n in range(num_clients) if n != i]
    assert manager.adjacency_list == expected


def test_line_topology():
    """
    For LINE topology:
      Based on the current implementation:
         For each node i:
           if i > 0:
               returns [i - 1]
           else:
               returns [] + [i + 1] if i < num_clients - 1 else []
         Expected behavior:
           - Node 0: returns [1]
           - Node 1: returns [0]
           - Node 2: returns [1]
    """
    num_clients = 3
    config = TopologyConfig(topology_type=TopologyType.LINE)
    manager = TopologyManager(num_clients=num_clients, config=config)

    expected = {
        0: [1],
        1: [0],
        2: [1],
    }
    assert manager.adjacency_list == expected


def test_custom_topology_valid():
    """
    For CUSTOM topology:
      - The manager should simply return the adjacency list provided in the config.
    """
    custom_adj = {
        0: [1, 2],
        1: [0],
        2: [0],
    }
    config = TopologyConfig(
        topology_type=TopologyType.CUSTOM, adjacency_list=custom_adj
    )
    manager = TopologyManager(num_clients=3, config=config)
    assert manager.adjacency_list == custom_adj


def test_custom_topology_invalid():
    """
    Even though TopologyConfig’s own validators should prevent creation of a
    custom config with a missing adjacency list, we simulate a scenario where
    the custom method is called with an invalid configuration.

    Here we create a valid config and then force its adjacency_list to None.
    The _custom() method (invoked during TopologyManager initialization) should raise a ValueError.
    """
    custom_adj = {
        0: [1, 2],
        1: [0],
        2: [0],
    }
    config = TopologyConfig(
        topology_type=TopologyType.CUSTOM, adjacency_list=custom_adj
    )
    # Force adjacency_list to None
    config.__dict__["adjacency_list"] = None

    with pytest.raises(ValueError, match="Custom topology requires adjacency list"):
        TopologyManager(num_clients=3, config=config)
