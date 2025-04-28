import numpy as np
import pytest

from murmura.aggregation.strategies.trimmed_mean import TrimmedMean
from murmura.aggregation.coordination_mode import CoordinationMode


def test_trimmed_mean_initialization_default():
    """Test initialization of TrimmedMean strategy with default parameters"""
    strategy = TrimmedMean()
    assert strategy.trim_ratio == 0.1
    assert strategy.coordination_mode == CoordinationMode.CENTRALIZED


def test_trimmed_mean_initialization_custom():
    """Test initialization of TrimmedMean strategy with custom parameters"""
    strategy = TrimmedMean(trim_ratio=0.2)
    assert strategy.trim_ratio == 0.2


def test_trimmed_mean_invalid_trim_ratio():
    """Test initialization with invalid trim ratio values"""
    # Below minimum (0.0)
    with pytest.raises(ValueError, match="must be between 0"):
        TrimmedMean(trim_ratio=-0.1)

    # Above maximum (0.5)
    with pytest.raises(ValueError, match="must be between 0"):
        TrimmedMean(trim_ratio=0.5)


def test_trimmed_mean_aggregation():
    """Test basic trimmed mean aggregation"""
    strategy = TrimmedMean(trim_ratio=0.2)

    # Create parameters for 5 clients, with some outliers
    params_a = {"layer1": np.array([1.0, 1.0, 1.0])}  # Normal
    params_b = {"layer1": np.array([2.0, 2.0, 2.0])}  # Normal
    params_c = {"layer1": np.array([3.0, 3.0, 3.0])}  # Normal
    params_d = {"layer1": np.array([10.0, 10.0, 10.0])}  # Outlier (high)
    params_e = {"layer1": np.array([-5.0, -5.0, -5.0])}  # Outlier (low)

    params_list = [params_a, params_b, params_c, params_d, params_e]

    # With trim_ratio=0.2, we should trim 20% from each end (1 client from each end)
    result = strategy.aggregate(params_list)

    # Expected: average of params_a, params_b, params_c (excluding outliers)
    expected = np.mean(
        np.stack([params_a["layer1"], params_b["layer1"], params_c["layer1"]]), axis=0
    )

    np.testing.assert_array_almost_equal(result["layer1"], expected)


def test_trimmed_mean_aggregation_few_clients():
    """Test trimmed mean behavior when there are too few clients to trim"""
    strategy = TrimmedMean(trim_ratio=0.3)

    # Only 2 clients, not enough to trim at 30%
    params_a = {"layer1": np.array([1.0, 1.0, 1.0])}
    params_b = {"layer1": np.array([2.0, 2.0, 2.0])}

    params_list = [params_a, params_b]

    # Should fall back to simple average when not enough clients
    result = strategy.aggregate(params_list)

    expected = np.mean(np.stack([params_a["layer1"], params_b["layer1"]]), axis=0)
    np.testing.assert_array_almost_equal(result["layer1"], expected)


def test_trimmed_mean_empty_params_list():
    """Test TrimmedMean with an empty parameters list"""
    strategy = TrimmedMean()

    with pytest.raises(ValueError, match="Empty parameters list"):
        strategy.aggregate([])


def test_trimmed_mean_exact_sorting():
    """Test that parameters are sorted correctly for trimming"""
    strategy = TrimmedMean(trim_ratio=0.25)  # Trim 25% from each end

    # 4 clients with specific scalar values for easier testing
    params_list = [
        {"val": np.array([5.0])},  # Will be kept
        {"val": np.array([10.0])},  # Will be kept
        {"val": np.array([1.0])},  # Will be trimmed (low)
        {"val": np.array([20.0])},  # Will be trimmed (high)
    ]

    result = strategy.aggregate(params_list)

    # Expected: average of 5.0 and 10.0 (= 7.5)
    assert result["val"][0] == 7.5


def test_trimmed_mean_fallback_method():
    """Test that the fallback weighted average method works correctly"""
    strategy = TrimmedMean(trim_ratio=0.2)

    params_a = {"layer1": np.array([1.0, 2.0, 3.0])}
    params_b = {"layer1": np.array([4.0, 5.0, 6.0])}

    params_list = [params_a, params_b]
    weights = [0.3, 0.7]

    # Test the fallback method directly
    result = strategy._weighted_average(params_list, weights)

    # Expected weighted average
    expected = 0.3 * params_a["layer1"] + 0.7 * params_b["layer1"]
    np.testing.assert_array_almost_equal(result["layer1"], expected)
