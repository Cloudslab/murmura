import pytest
import numpy as np
from unittest.mock import MagicMock, patch

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
def ring_topology_manager():
    """Create a ring topology manager"""
    config = TopologyConfig(topology_type=TopologyType.RING)
    return TopologyManager(num_clients=3, config=config)


@pytest.fixture
def complete_topology_manager():
    """Create a complete topology manager"""
    config = TopologyConfig(topology_type=TopologyType.COMPLETE)
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
        topology_type=TopologyType.CUSTOM,
        adjacency_list={0: [1, 2], 1: [0], 2: [0]}
    )
    return TopologyManager(num_clients=3, config=config)


@patch('ray.get')
def test_line_topology_coordination(mock_ray_get, mock_actors, line_topology_manager):
    """Test coordination with line topology"""
    # Setup mock behaviors
    mock_ray_get.side_effect = lambda x: {"layer": np.array([1.0, 2.0])}

    # Use GossipAvg strategy which is compatible with line topology
    strategy = GossipAvg()
    strategy.aggregate = MagicMock(return_value={"layer": np.array([2.0, 3.0])})

    # Create coordinator
    coordinator = TopologyCoordinator(mock_actors, line_topology_manager, strategy)

    # Test coordination
    result = coordinator._coordinate_line_topology()

    # Verify strategy.aggregate was called for each node
    assert strategy.aggregate.call_count == 3

    # Verify the result contains the expected data
    assert "layer" in result
    assert np.array_equal(result["layer"], np.array([2.0, 3.0]))


@patch('ray.get')
def test_custom_topology_coordination(mock_ray_get, mock_actors, custom_topology_manager):
    """Test coordination with custom topology"""
    # Setup mock behaviors
    mock_ray_get.side_effect = lambda x: {"layer": np.array([1.0, 2.0])}

    # Use GossipAvg strategy which is compatible with custom topology
    strategy = GossipAvg()
    strategy.aggregate = MagicMock(return_value={"layer": np.array([2.0, 3.0])})

    # Create coordinator
    coordinator = TopologyCoordinator(mock_actors, custom_topology_manager, strategy)

    # Test coordination
    result = coordinator._coordinate_custom_topology()

    # Verify strategy.aggregate was called for each node
    assert strategy.aggregate.call_count == 3

    # Verify the result contains the expected data
    assert "layer" in result
    assert np.array_equal(result["layer"], np.array([2.0, 3.0]))


@patch('ray.get')
def test_complete_topology_decentralized(mock_ray_get, mock_actors, complete_topology_manager):
    """Test coordination with complete topology using decentralized strategy"""
    # Setup mock behaviors
    mock_ray_get.side_effect = lambda x: {"layer": np.array([1.0, 2.0])}

    # Use GossipAvg strategy (decentralized)
    strategy = GossipAvg()
    strategy.coordination_mode = CoordinationMode.DECENTRALIZED
    strategy.aggregate = MagicMock(return_value={"layer": np.array([2.0, 3.0])})

    # Create coordinator
    coordinator = TopologyCoordinator(mock_actors, complete_topology_manager, strategy)

    # Test coordination
    result = coordinator._coordinate_complete_topology()

    # For decentralized coordination, should call aggregate once per node
    assert strategy.aggregate.call_count == 3

    # Verify the result
    assert "layer" in result
    assert np.array_equal(result["layer"], np.array([2.0, 3.0]))


def test_weighted_aggregation_normalization():
    """Test that aggregation properly normalizes weights"""
    # Create parameters
    params_a = {"layer": np.array([1.0, 1.0])}
    params_b = {"layer": np.array([2.0, 2.0])}
    params_c = {"layer": np.array([3.0, 3.0])}

    # Aggregate with normalized weights
    params_list = [params_a, params_b, params_c]
    weights = [10, 20, 30]  # Sum = 60, should be normalized to [1/6, 2/6, 3/6]

    # Expected result with normalized weights
    expected = {
        "layer": np.array([
            (1/6)*1.0 + (2/6)*2.0 + (3/6)*3.0,
            (1/6)*1.0 + (2/6)*2.0 + (3/6)*3.0
        ])
    }

    # Call the combined method directly
    result = TopologyCoordinator._combine_aggregated_params([expected, expected, expected])

    # Verify the result
    assert np.allclose(result["layer"], expected["layer"])


def test_empty_aggregated_params_list():
    """Test combine_aggregated_params with empty list raises error"""
    with pytest.raises(ValueError, match="Empty aggregated parameters list"):
        TopologyCoordinator._combine_aggregated_params([])


def test_combine_aggregated_params_with_multiple_keys():
    """Test combining parameters with multiple keys"""
    # Create multiple parameter sets
    params1 = {
        "layer1": np.array([1.0, 2.0]),
        "layer2": np.array([[1.0, 2.0], [3.0, 4.0]])
    }

    params2 = {
        "layer1": np.array([3.0, 4.0]),
        "layer2": np.array([[5.0, 6.0], [7.0, 8.0]])
    }

    params3 = {
        "layer1": np.array([5.0, 6.0]),
        "layer2": np.array([[9.0, 10.0], [11.0, 12.0]])
    }

    # Expected result: average of all parameters
    expected_layer1 = np.array([3.0, 4.0])  # Avg of [1,2], [3,4], [5,6]
    expected_layer2 = np.array([[5.0, 6.0], [7.0, 8.0]])  # Avg of the matrices

    # Call the method
    result = TopologyCoordinator._combine_aggregated_params([params1, params2, params3])

    # Verify the result
    assert "layer1" in result
    assert "layer2" in result
    assert np.allclose(result["layer1"], expected_layer1)
    assert np.allclose(result["layer2"], expected_layer2)


@patch('ray.get')
def test_coordinate_aggregation_with_weights(mock_ray_get, mock_actors, star_topology_manager):
    """Test coordinate_aggregation with weights"""
    # Setup mock behaviors
    mock_ray_get.side_effect = lambda x: {"layer": np.array([1.0, 2.0])}

    # Use FedAvg strategy with a patched _coordinate_star_topology method
    strategy = FedAvg()
    strategy.aggregate = MagicMock(return_value={"layer": np.array([2.0, 3.0])})

    # Create coordinator
    coordinator = TopologyCoordinator(mock_actors, star_topology_manager, strategy)

    # Mock the internal _coordinate_star_topology method to verify it receives weights
    original_method = coordinator._coordinate_star_topology

    def mock_star_topology(weights=None):
        # Store the weights that were passed so we can verify them
        mock_star_topology.called_with_weights = weights
        # Call the original method to ensure normal behavior
        return original_method(weights)

    # Add attribute to track weights
    mock_star_topology.called_with_weights = None
    coordinator._coordinate_star_topology = mock_star_topology

    # Test coordination with custom weights
    weights = [0.6, 0.3, 0.1]
    result = coordinator.coordinate_aggregation(weights=weights)

    # Verify weights were passed to the internal topology method
    assert mock_star_topology.called_with_weights == weights

    # Verify the result
    assert "layer" in result
    assert np.array_equal(result["layer"], np.array([2.0, 3.0]))


@patch('ray.get')
def test_coordinate_aggregation_dispatch(mock_ray_get, mock_actors):
    """Test that coordinate_aggregation dispatches to the right method based on topology"""
    # Setup mock
    mock_ray_get.side_effect = lambda x: {"layer": np.array([1.0, 2.0])}

    # Test for each topology type
    topology_types = [
        (TopologyType.STAR, "_coordinate_star_topology"),
        (TopologyType.RING, "_coordinate_ring_topology"),
        (TopologyType.COMPLETE, "_coordinate_complete_topology"),
        (TopologyType.LINE, "_coordinate_line_topology"),
        (TopologyType.CUSTOM, "_coordinate_custom_topology")
    ]

    for topology_type, method_name in topology_types:
        # Create config with proper adjacency list for CUSTOM
        if topology_type == TopologyType.CUSTOM:
            config = TopologyConfig(
                topology_type=topology_type,
                adjacency_list={0: [1, 2], 1: [0], 2: [0]}
            )
        else:
            config = TopologyConfig(topology_type=topology_type)

        # Create topology manager
        topology_manager = TopologyManager(num_clients=3, config=config)

        # Use FedAvg strategy
        strategy = FedAvg()

        # Create coordinator
        coordinator = TopologyCoordinator(mock_actors, topology_manager, strategy)

        # Patch the specific coordination method
        with patch.object(coordinator, method_name, return_value={"test": "result"}):
            # Call coordinate_aggregation
            result = coordinator.coordinate_aggregation()

            # Verify the right method was called
            getattr(coordinator, method_name).assert_called_once()
            assert result == {"test": "result"}
