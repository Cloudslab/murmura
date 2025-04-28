import pytest
import ray
import numpy as np
from unittest.mock import MagicMock, patch

from murmura.aggregation.coordination_mode import CoordinationMode
from murmura.aggregation.strategies.fed_avg import FedAvg
from murmura.aggregation.strategies.gossip_avg import GossipAvg
from murmura.network_management.topology import TopologyConfig, TopologyType
from murmura.network_management.topology_manager import TopologyManager
from murmura.orchestration.topology_coordinator import TopologyCoordinator


@pytest.fixture(scope="module")
def ray_init():
    """Initialize Ray for the tests"""
    if not ray.is_initialized():
        ray.init(local_mode=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


class MockActor:
    """Mock actor for tests that doesn't require Ray initialization"""
    def __init__(self, actor_id, params):
        self.actor_id = actor_id
        self.params = params

    def get_model_parameters(self):
        return self.params

    def get_model_parameters_remote(self):
        return self.params


def test_coordination_mode_detection():
    """Test that the coordination mode is correctly detected from the strategy"""
    actors = [MockActor(i, {"layer": np.array([float(i)])}) for i in range(3)]

    # FedAvg uses centralized coordination
    fed_avg = FedAvg()
    topology_config = TopologyConfig(topology_type=TopologyType.STAR)
    topology_manager = TopologyManager(num_clients=3, config=topology_config)

    coordinator = TopologyCoordinator(actors, topology_manager, fed_avg)
    assert coordinator.coordination_mode == CoordinationMode.CENTRALIZED

    # GossipAvg uses decentralized coordination
    gossip_avg = GossipAvg()
    coordinator = TopologyCoordinator(actors, topology_manager, gossip_avg)
    assert coordinator.coordination_mode == CoordinationMode.DECENTRALIZED


def test_create_factory_method():
    """Test the create factory method"""
    actors = [MockActor(i, {"layer": np.array([float(i)])}) for i in range(3)]
    topology_config = TopologyConfig(topology_type=TopologyType.STAR)
    topology_manager = TopologyManager(num_clients=3, config=topology_config)
    strategy = FedAvg()

    coordinator = TopologyCoordinator.create(actors, topology_manager, strategy)
    assert isinstance(coordinator, TopologyCoordinator)
    assert coordinator.actors == actors
    assert coordinator.topology_manager == topology_manager
    assert coordinator.strategy == strategy


@patch('ray.get')
def test_star_topology_coordination(mock_ray_get, ray_init):
    """Test coordination with star topology"""
    # Setup mock behaviors
    mock_ray_get.side_effect = lambda x: x

    # Create actors with fixed parameters for predictable results
    actors = []
    for i in range(3):
        actor = MagicMock()
        actor.get_model_parameters.remote.return_value = {"layer": np.array([float(i)])}
        actors.append(actor)

    # Configure star topology with node 0 as hub
    topology_config = TopologyConfig(topology_type=TopologyType.STAR, hub_index=0)
    topology_manager = TopologyManager(num_clients=3, config=topology_config)

    # Use FedAvg strategy
    strategy = FedAvg()
    # Mock the aggregate method to return a predictable result
    strategy.aggregate = MagicMock(return_value={"layer": np.array([1.0])})

    # Create coordinator
    coordinator = TopologyCoordinator(actors, topology_manager, strategy)

    # Test coordination
    result = coordinator._coordinate_star_topology()

    # Verify the strategy's aggregate method was called with the correct parameters
    strategy.aggregate.assert_called_once()
    # First argument should be list of parameters
    call_args = strategy.aggregate.call_args[0]
    assert len(call_args[0]) == 3  # All three actors' parameters

    # Verify the result
    assert "layer" in result
    assert np.array_equal(result["layer"], np.array([1.0]))


@patch('ray.get')
def test_ring_topology_coordination(mock_ray_get, ray_init):
    """Test coordination with ring topology"""
    # Setup mock behaviors
    mock_ray_get.side_effect = lambda x: x

    # Create actors with fixed parameters for predictable results
    actors = []
    for i in range(3):
        actor = MagicMock()
        actor.get_model_parameters.remote.return_value = {"layer": np.array([float(i)])}
        actors.append(actor)

    # Configure ring topology
    topology_config = TopologyConfig(topology_type=TopologyType.RING)
    topology_manager = TopologyManager(num_clients=3, config=topology_config)

    # Use GossipAvg strategy
    strategy = GossipAvg()
    # Mock the aggregate method
    strategy.aggregate = MagicMock(return_value={"layer": np.array([1.0])})

    # Create coordinator
    coordinator = TopologyCoordinator(actors, topology_manager, strategy)

    # Test coordination
    result = coordinator._coordinate_ring_topology()

    # Verify strategy.aggregate was called for each node
    assert strategy.aggregate.call_count == 3

    # Verify the final result comes from combining local aggregations
    assert "layer" in result


@patch('ray.get')
def test_complete_topology_centralized(mock_ray_get, ray_init):
    """Test coordination with complete topology using centralized strategy"""
    # Setup mock behaviors
    mock_ray_get.side_effect = lambda x: x

    # Create actors
    actors = []
    for i in range(3):
        actor = MagicMock()
        actor.get_model_parameters.remote.return_value = {"layer": np.array([float(i)])}
        actors.append(actor)

    # Configure complete topology
    topology_config = TopologyConfig(topology_type=TopologyType.COMPLETE)
    topology_manager = TopologyManager(num_clients=3, config=topology_config)

    # Use FedAvg strategy (centralized)
    strategy = FedAvg()
    strategy.aggregate = MagicMock(return_value={"layer": np.array([1.0])})

    # Create coordinator
    coordinator = TopologyCoordinator(actors, topology_manager, strategy)

    # Test coordination
    result = coordinator._coordinate_complete_topology()

    # Verify the strategy.aggregate was called once with all parameters
    strategy.aggregate.assert_called_once()
    call_args = strategy.aggregate.call_args[0]
    assert len(call_args[0]) == 3

    # Verify result
    assert "layer" in result
    assert np.array_equal(result["layer"], np.array([1.0]))


@patch('ray.get')
def test_weighted_aggregation(mock_ray_get, ray_init):
    """Test aggregation with custom weights"""
    # Setup mock behaviors
    mock_ray_get.side_effect = lambda x: x

    # Create actors with distinctive parameter values for tracing
    actors = []
    actor_params = []
    for i in range(3):
        actor = MagicMock()
        # Use distinctive values that we can identify in the aggregation
        params = {"layer": np.array([float(i+10)])}  # Values: [10, 11, 12]
        actor.get_model_parameters.remote.return_value = params
        actors.append(actor)
        actor_params.append(params)

    # Configure star topology
    topology_config = TopologyConfig(topology_type=TopologyType.STAR, hub_index=0)
    topology_manager = TopologyManager(num_clients=3, config=topology_config)

    # Use a real FedAvg for actual calculation to trace weight application
    strategy = FedAvg()

    # Create coordinator
    coordinator = TopologyCoordinator(actors, topology_manager, strategy)

    # Spy on the strategy.aggregate method to intercept calls
    with patch.object(strategy, 'aggregate', wraps=strategy.aggregate) as strategy_spy:
        # Test coordination with custom weights - use very distinctive weights
        weights = [0.7, 0.2, 0.1]  # These will be obvious in the result if applied correctly
        result = coordinator._coordinate_star_topology(weights=weights)

        # Verify the strategy.aggregate was called
        strategy_spy.assert_called_once()

        # Get the actual parameters and weights passed
        args, kwargs = strategy_spy.call_args

        # Extract parameters and weights
        if len(args) >= 1:
            passed_params = args[0]
        else:
            assert False, "Parameters not passed to aggregate method"

        if len(args) >= 2:
            passed_weights = args[1]
        elif 'weights' in kwargs:
            passed_weights = kwargs['weights']
        else:
            assert False, "Weights not passed to aggregate method"

        # Now check if weights correctly correspond to parameters
        # First, let's identify if parameters appear to be reordered
        # This requires comparing the passed parameters with our original actor_params

        # Check that we have the right number of parameters and weights
        assert len(passed_params) == 3, "Should have parameters for 3 nodes"
        assert len(passed_weights) == 3, "Should have weights for 3 nodes"

        # Check if the weights sum to 1.0 (normalized)
        assert abs(sum(passed_weights) - 1.0) < 1e-6, "Weights should sum to 1.0"

        # IMPORTANT: Verify weight-parameter correspondence
        # We need to check that if weights are reordered, parameters are reordered in the same way

        # Get the expected weighted result if our weights were applied correctly
        expected_value = 0.7*10 + 0.2*11 + 0.1*12

        # Calculate what weighted result we'd actually get with the passed parameters and weights
        calculated_value = sum(passed_weights[i] * passed_params[i]["layer"][0] for i in range(3))

        # These should be very close (allow for floating point precision)
        assert abs(calculated_value - expected_value) < 1e-6, (
            f"Weight-parameter mismatch. Expected weighted value: {expected_value}, "
            f"Got: {calculated_value}. If parameters were reordered, weights should be "
            f"reordered in the same way to maintain correspondence."
        )


@patch('ray.get')
def test_combine_aggregated_params(mock_ray_get):
    """Test the combine_aggregated_params method"""
    # Create sample aggregated parameters
    agg_params_1 = {"layer": np.array([1.0]), "bias": np.array([0.1])}
    agg_params_2 = {"layer": np.array([2.0]), "bias": np.array([0.2])}
    agg_params_3 = {"layer": np.array([3.0]), "bias": np.array([0.3])}

    aggregated_params_list = [agg_params_1, agg_params_2, agg_params_3]

    # Test the static method
    result = TopologyCoordinator._combine_aggregated_params(aggregated_params_list)

    # Expected: average of all parameters
    assert np.allclose(result["layer"], np.array([2.0]))
    assert np.allclose(result["bias"], np.array([0.2]))


def test_combine_empty_params_list():
    """Test combining an empty list of parameters"""
    with pytest.raises(ValueError, match="Empty aggregated parameters list"):
        TopologyCoordinator._combine_aggregated_params([])


@patch('ray.get')
def test_coordinate_aggregation_dispatches_to_correct_method(mock_ray_get, ray_init):
    """Test that coordinate_aggregation calls the appropriate topology-specific method"""
    # Setup mock behaviors
    mock_ray_get.side_effect = lambda x: x

    # Create actors
    actors = []
    for i in range(3):
        actor = MagicMock()
        actor.get_model_parameters.remote.return_value = {"layer": np.array([float(i)])}
        actors.append(actor)

    # For each topology type, create a coordinator and patch its topology-specific method
    for topology_type in [
        TopologyType.STAR,
        TopologyType.RING,
        TopologyType.COMPLETE,
        TopologyType.LINE,
        TopologyType.CUSTOM
    ]:
        topology_config = TopologyConfig(
            topology_type=topology_type,
            adjacency_list={0: [1, 2], 1: [0], 2: [0]} if topology_type == TopologyType.CUSTOM else None
        )
        topology_manager = TopologyManager(num_clients=3, config=topology_config)
        strategy = FedAvg()

        coordinator = TopologyCoordinator(actors, topology_manager, strategy)

        # Patch the specific topology method
        method_name = f"_coordinate_{topology_type.value}_topology"
        with patch.object(coordinator, method_name, return_value={"patched": True}):
            # Call coordinate_aggregation
            result = coordinator.coordinate_aggregation()

            # Verify the correct method was called
            getattr(coordinator, method_name).assert_called_once()
            assert result == {"patched": True}


def test_unsupported_topology():
    """Test handling of unsupported topology type"""
    actors = [MockActor(i, {"layer": np.array([float(i)])}) for i in range(3)]
    topology_config = TopologyConfig(topology_type=TopologyType.STAR)
    topology_manager = TopologyManager(num_clients=3, config=topology_config)
    strategy = FedAvg()

    coordinator = TopologyCoordinator(actors, topology_manager, strategy)

    # Hack the topology_type to an unsupported value
    coordinator.topology_type = "unsupported_topology"

    with pytest.raises(ValueError, match="Unsupported topology type"):
        coordinator.coordinate_aggregation()
