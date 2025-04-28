import numpy as np
import pytest

from murmura.aggregation.strategies.fed_avg import FedAvg
from murmura.aggregation.coordination_mode import CoordinationMode


def test_fed_avg_initialization():
    """Test initialization of FedAvg strategy"""
    strategy = FedAvg()
    assert strategy is not None
    assert strategy.coordination_mode == CoordinationMode.CENTRALIZED


def test_fed_avg_aggregation_equal_weights():
    """Test FedAvg aggregation with equal weights"""
    strategy = FedAvg()

    # Create dummy parameters for two clients
    params_a = {
        "layer1": np.array([1.0, 2.0, 3.0]),
        "layer2": np.array([[4.0, 5.0], [6.0, 7.0]]),
    }
    params_b = {
        "layer1": np.array([2.0, 3.0, 4.0]),
        "layer2": np.array([[5.0, 6.0], [7.0, 8.0]]),
    }

    params_list = [params_a, params_b]

    # Aggregate with equal weights (default)
    result = strategy.aggregate(params_list)

    # Check that the result contains the expected keys
    assert set(result.keys()) == {"layer1", "layer2"}

    # Check that values are correctly averaged
    np.testing.assert_array_equal(result["layer1"], np.array([1.5, 2.5, 3.5]))
    np.testing.assert_array_equal(result["layer2"], np.array([[4.5, 5.5], [6.5, 7.5]]))


def test_fed_avg_aggregation_custom_weights():
    """Test FedAvg aggregation with custom weights"""
    strategy = FedAvg()

    # Create dummy parameters for two clients
    params_a = {
        "layer1": np.array([1.0, 2.0, 3.0]),
        "layer2": np.array([[4.0, 5.0], [6.0, 7.0]]),
    }
    params_b = {
        "layer1": np.array([2.0, 3.0, 4.0]),
        "layer2": np.array([[5.0, 6.0], [7.0, 8.0]]),
    }

    params_list = [params_a, params_b]
    weights = [0.7, 0.3]  # 70% weight to client A, 30% to client B

    # Aggregate with custom weights
    result = strategy.aggregate(params_list, weights)

    # Check results with weighted average
    expected_layer1 = 0.7 * params_a["layer1"] + 0.3 * params_b["layer1"]
    expected_layer2 = 0.7 * params_a["layer2"] + 0.3 * params_b["layer2"]

    np.testing.assert_array_almost_equal(result["layer1"], expected_layer1)
    np.testing.assert_array_almost_equal(result["layer2"], expected_layer2)


def test_fed_avg_empty_params_list():
    """Test FedAvg with an empty parameters list should raise ValueError"""
    strategy = FedAvg()

    with pytest.raises(ValueError, match="Empty parameters list"):
        strategy.aggregate([])


def test_fed_avg_aggregation_normalizes_weights():
    """Test that FedAvg normalizes the provided weights"""
    strategy = FedAvg()

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


def test_fed_avg_aggregation_different_shapes():
    """Test handling of parameter arrays with different shapes (should raise error)"""
    strategy = FedAvg()

    # Create parameters with inconsistent shapes
    params_a = {"layer1": np.array([1.0, 2.0, 3.0])}
    params_b = {"layer1": np.array([2.0, 3.0, 4.0, 5.0])}  # Different shape

    params_list = [params_a, params_b]

    # Should raise an error when trying to stack arrays of different shapes
    with pytest.raises(Exception):
        strategy.aggregate(params_list)
