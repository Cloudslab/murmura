from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from murmura.aggregation.coordination_mode import CoordinationMode
from murmura.aggregation.strategies.fed_avg import FedAvg
from murmura.aggregation.strategies.gossip_avg import GossipAvg
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.network_management.topology_manager import TopologyManager
from murmura.orchestration.topology_coordinator import TopologyCoordinator


@pytest.fixture
def mock_actors():
    """Create mock actors for testing"""
    actor1 = MagicMock()
    actor1.get_model_parameters.remote.return_value = {"layer": np.array([1.0, 2.0])}

    actor2 = MagicMock()
    actor2.get_model_parameters.remote.return_value = {"layer": np.array([3.0, 4.0])}

    actor3 = MagicMock()
    actor3.get_model_parameters.remote.return_value = {"layer": np.array([5.0, 6.0])}

    return [actor1, actor2, actor3]


@pytest.fixture
def star_topology_manager():
    """Create a star topology manager"""
    config = TopologyConfig(topology_type=TopologyType.STAR, hub_index=0)
    return TopologyManager(num_clients=3, config=config)


@pytest.fixture
def line_topology_manager():
    """Create a line topology manager"""
    config = TopologyConfig(topology_type=TopologyType.LINE)
    return TopologyManager(num_clients=3, config=config)


@pytest.fixture
def custom_topology_manager():
    """Create a custom topology manager"""
    config = TopologyConfig(
        topology_type=TopologyType.CUSTOM, adjacency_list={0: [1, 2], 1: [0], 2: [0]}
    )
    return TopologyManager(num_clients=3, config=config)


def test_determination_of_coordination_mode():
    """Test the determination of coordination mode based on strategy"""
    # Create a mock strategy with each coordination mode
    centralized_strategy = MagicMock()
    centralized_strategy.coordination_mode = CoordinationMode.CENTRALIZED

    decentralized_strategy = MagicMock()
    decentralized_strategy.coordination_mode = CoordinationMode.DECENTRALIZED

    # Create a topology manager
    config = TopologyConfig(topology_type=TopologyType.STAR)
    topology_manager = TopologyManager(num_clients=3, config=config)

    # Test with centralized strategy
    coordinator = TopologyCoordinator(
        [MagicMock()] * 3, topology_manager, centralized_strategy
    )
    assert coordinator.coordination_mode == CoordinationMode.CENTRALIZED

    # Test with decentralized strategy
    coordinator = TopologyCoordinator(
        [MagicMock()] * 3, topology_manager, decentralized_strategy
    )
    assert coordinator.coordination_mode == CoordinationMode.DECENTRALIZED


@patch("ray.get")
def test_coordinate_dispatch_to_correct_topology_method(mock_ray_get, mock_actors):
    """Test that coordinate_aggregation dispatches to the correct topology method"""
    # Setup mock ray.get
    mock_ray_get.side_effect = lambda x: {"layer": np.array([1.0, 2.0])}

    # Test for each topology type with compatible strategies
    topology_configs = [
        (TopologyType.STAR, "_coordinate_star_topology", FedAvg()),  # Centralized strategy
        (TopologyType.COMPLETE, "_coordinate_complete_topology", FedAvg()),  # Centralized strategy
        (TopologyType.RING, "_coordinate_ring_topology", GossipAvg()),  # Decentralized strategy
        (TopologyType.LINE, "_coordinate_line_topology", GossipAvg()),  # Decentralized strategy
        (TopologyType.CUSTOM, "_coordinate_custom_topology", GossipAvg()),  # Decentralized strategy
    ]

    for topology_type, method_name, strategy in topology_configs:
        # Create topology config - use adjacency list for CUSTOM
        if topology_type == TopologyType.CUSTOM:
            config = TopologyConfig(
                topology_type=topology_type, adjacency_list={0: [1, 2], 1: [0], 2: [0]}
            )
        else:
            config = TopologyConfig(topology_type=topology_type)

        # Create topology manager and coordinator
        topology_manager = TopologyManager(num_clients=3, config=config)
        coordinator = TopologyCoordinator(mock_actors, topology_manager, strategy)

        # Patch the specific topology method to verify it's called
        patched_method = MagicMock(return_value={"patched": True})
        original_method = getattr(coordinator, method_name)
        setattr(coordinator, method_name, patched_method)

        try:
            # Call coordinate_aggregation
            result = coordinator.coordinate_aggregation()

            # Verify the correct method was called
            patched_method.assert_called_once()
            assert result == {"patched": True}
        finally:
            # Restore original method
            setattr(coordinator, method_name, original_method)


@patch("ray.get")
def test_centralized_and_decentralized_with_complete_topology(
    mock_ray_get, mock_actors
):
    """Test both centralized and decentralized modes with complete topology"""
    # Setup mock ray.get
    mock_ray_get.side_effect = lambda x: {"layer": np.array([1.0, 2.0])}

    # Create complete topology
    config = TopologyConfig(topology_type=TopologyType.COMPLETE)
    topology_manager = TopologyManager(num_clients=3, config=config)

    # Test with centralized strategy
    centralized_strategy = MagicMock()
    centralized_strategy.coordination_mode = CoordinationMode.CENTRALIZED
    centralized_strategy.aggregate.return_value = {"layer": np.array([1.5, 2.5])}

    centralized_coordinator = TopologyCoordinator(
        mock_actors, topology_manager, centralized_strategy
    )

    # Create spy on the aggregate method
    centralized_strategy.aggregate = MagicMock(
        return_value={"layer": np.array([1.5, 2.5])}
    )

    # Run centralized coordination
    centralized_coordinator._coordinate_complete_topology()

    # Verify centralized behavior: should call aggregate once with all parameters
    centralized_strategy.aggregate.assert_called_once()
    args, _ = centralized_strategy.aggregate.call_args
    assert len(args[0]) == 3  # Should have parameters from all 3 actors

    # Test with decentralized strategy
    decentralized_strategy = MagicMock()
    decentralized_strategy.coordination_mode = CoordinationMode.DECENTRALIZED
    decentralized_strategy.aggregate.return_value = {"layer": np.array([1.5, 2.5])}

    decentralized_coordinator = TopologyCoordinator(
        mock_actors, topology_manager, decentralized_strategy
    )

    # Run decentralized coordination
    decentralized_coordinator._coordinate_complete_topology()

    # Verify decentralized behavior: should call aggregate once per node
    assert decentralized_strategy.aggregate.call_count == 3


@patch("ray.get")
def test_line_topology_parameter_collection(
    mock_ray_get, mock_actors, line_topology_manager
):
    """Test parameter collection in line topology"""
    # Setup mock ray.get
    mock_ray_get.side_effect = lambda x: {"layer": np.array([1.0, 2.0])}

    # Create a coordinator with line topology
    strategy = MagicMock()
    strategy.aggregate.return_value = {"layer": np.array([1.5, 2.5])}

    coordinator = TopologyCoordinator(mock_actors, line_topology_manager, strategy)

    # Get the adjacency list
    adjacency = line_topology_manager.adjacency_list

    # Run line coordination
    coordinator._coordinate_line_topology()

    # Verify aggregate was called for each node
    assert strategy.aggregate.call_count == 3

    # Check that the right parameters were collected for each node
    for i, node_actor in enumerate(mock_actors):
        # Each node should collect parameters from itself and neighbors
        neighbors = adjacency[i]
        expected_num_params = 1 + len(neighbors)  # Self + neighbors

        # Check the call arguments for this node's aggregation
        call_args_list = strategy.aggregate.call_args_list[i]
        args, _ = call_args_list
        actual_params = args[0]

        assert len(actual_params) == expected_num_params


@patch("ray.get")
def test_custom_topology_parameter_collection(
    mock_ray_get, mock_actors, custom_topology_manager
):
    """Test parameter collection in custom topology"""
    # Setup mock ray.get
    mock_ray_get.side_effect = lambda x: {"layer": np.array([1.0, 2.0])}

    # Create a coordinator with custom topology
    strategy = MagicMock()
    strategy.aggregate.return_value = {"layer": np.array([1.5, 2.5])}

    coordinator = TopologyCoordinator(mock_actors, custom_topology_manager, strategy)

    # Get the adjacency list
    adjacency = custom_topology_manager.adjacency_list

    # Run custom coordination
    coordinator._coordinate_custom_topology()

    # Verify aggregate was called for each node
    assert strategy.aggregate.call_count == 3

    # Check that the right parameters were collected for each node
    for i, node_actor in enumerate(mock_actors):
        # Each node should collect parameters from itself and neighbors
        neighbors = adjacency[i]
        expected_num_params = 1 + len(neighbors)  # Self + neighbors

        # Check the call arguments for this node's aggregation
        call_args_list = strategy.aggregate.call_args_list[i]
        args, _ = call_args_list
        actual_params = args[0]

        assert len(actual_params) == expected_num_params
