import numpy as np
import pytest

from murmura.aggregation.strategies.gossip_avg import GossipAvg
from murmura.aggregation.coordination_mode import CoordinationMode


def test_gossip_avg_initialization_default():
    """Test initialization of GossipAvg strategy with default parameters"""
    strategy = GossipAvg()
    assert strategy.mixing_parameter == 0.5
    assert strategy.coordination_mode == CoordinationMode.DECENTRALIZED


def test_gossip_avg_initialization_custom():
    """Test initialization of GossipAvg strategy with custom parameters"""
    strategy = GossipAvg(mixing_parameter=0.7)
    assert strategy.mixing_parameter == 0.7


def test_gossip_avg_invalid_mixing_parameter():
    """Test initialization with invalid mixing parameter values"""
    # Below minimum (0.0)
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        GossipAvg(mixing_parameter=-0.1)

    # Above maximum (1.0)
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        GossipAvg(mixing_parameter=1.1)


def test_gossip_avg_aggregation_equal_weights():
    """Test GossipAvg aggregation with equal weights"""
    strategy = GossipAvg(mixing_parameter=0.5)

    # Create dummy parameters for two clients (self and one neighbor)
    params_self = {"layer1": np.array([1.0, 2.0, 3.0])}
    params_neighbor = {"layer1": np.array([2.0, 3.0, 4.0])}

    params_list = [params_self, params_neighbor]

    # Aggregate with equal weights (default)
    result = strategy.aggregate(params_list)

    # Check results with equal weighting
    expected = 0.5 * params_self["layer1"] + 0.5 * params_neighbor["layer1"]
    np.testing.assert_array_almost_equal(result["layer1"], expected)


def test_gossip_avg_aggregation_custom_weights():
    """Test GossipAvg aggregation with custom weights"""
    strategy = GossipAvg(mixing_parameter=0.5)

    # Create dummy parameters for multiple clients
    params_a = {"layer1": np.array([1.0, 2.0, 3.0])}
    params_b = {"layer1": np.array([2.0, 3.0, 4.0])}
    params_c = {"layer1": np.array([3.0, 4.0, 5.0])}

    params_list = [params_a, params_b, params_c]
    weights = [0.5, 0.3, 0.2]  # Custom weights for each client

    # Aggregate with custom weights
    result = strategy.aggregate(params_list, weights)

    # Calculate expected value
    expected = (
        0.5 * params_a["layer1"] + 0.3 * params_b["layer1"] + 0.2 * params_c["layer1"]
    )
    np.testing.assert_array_almost_equal(result["layer1"], expected)


def test_gossip_avg_empty_params_list():
    """Test GossipAvg with an empty parameters list"""
    strategy = GossipAvg()

    with pytest.raises(ValueError, match="No parameters to aggregate"):
        strategy.aggregate([])


def test_gossip_avg_aggregation_normalizes_weights():
    """Test that GossipAvg normalizes the provided weights"""
    strategy = GossipAvg()

    # Create dummy parameters for two clients
    params_a = {"layer1": np.array([1.0, 2.0, 3.0])}
    params_b = {"layer1": np.array([2.0, 3.0, 4.0])}

    params_list = [params_a, params_b]
    weights = [7, 3]  # Should be normalized to [0.7, 0.3]

    # Aggregate with unnormalized weights
    result = strategy.aggregate(params_list, weights)

    # Calculate expected value with normalized weights
    expected = 0.7 * params_a["layer1"] + 0.3 * params_b["layer1"]

    np.testing.assert_array_almost_equal(result["layer1"], expected)


def test_gossip_avg_stacking_error():
    """Test error handling for inconsistent parameter shapes"""
    strategy = GossipAvg()

    # Parameters with inconsistent shapes
    params_a = {"layer1": np.array([1.0, 2.0])}
    params_b = {"layer1": np.array([3.0, 4.0, 5.0])}  # Different shape

    params_list = [params_a, params_b]

    # Should raise a specific ValueError about stacking
    with pytest.raises(ValueError, match="Error stacking parameters"):
        strategy.aggregate(params_list)
