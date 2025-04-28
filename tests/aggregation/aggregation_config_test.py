import pytest
from pydantic import ValidationError

from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)


def test_default_initialization():
    """Test default initialization of AggregationConfig"""
    config = AggregationConfig()

    assert config.strategy_type == AggregationStrategyType.FEDAVG
    assert config.params is not None
    assert config.params == {}


def test_initialization_with_strategy():
    """Test initialization with specific strategy type"""
    config = AggregationConfig(strategy_type=AggregationStrategyType.TRIMMED_MEAN)

    assert config.strategy_type == AggregationStrategyType.TRIMMED_MEAN
    assert config.params is not None
    assert "trim_ratio" in config.params
    assert config.params["trim_ratio"] == 0.1  # Default value


def test_initialization_with_params():
    """Test initialization with custom parameters"""
    config = AggregationConfig(
        strategy_type=AggregationStrategyType.TRIMMED_MEAN,
        params={"trim_ratio": 0.2}
    )

    assert config.strategy_type == AggregationStrategyType.TRIMMED_MEAN
    assert config.params["trim_ratio"] == 0.2


def test_param_validation_trimmed_mean():
    """Test validation of parameters for TrimmedMean strategy"""
    # Valid parameter
    config = AggregationConfig(
        strategy_type=AggregationStrategyType.TRIMMED_MEAN,
        params={"trim_ratio": 0.3}
    )
    assert config.params["trim_ratio"] == 0.3

    # Invalid parameter - too low
    with pytest.raises(ValidationError):
        AggregationConfig(
            strategy_type=AggregationStrategyType.TRIMMED_MEAN,
            params={"trim_ratio": -0.1}
        )

    # Invalid parameter - too high
    with pytest.raises(ValidationError):
        AggregationConfig(
            strategy_type=AggregationStrategyType.TRIMMED_MEAN,
            params={"trim_ratio": 0.5}
        )


def test_param_validation_gossip_avg():
    """Test validation of parameters for GossipAvg strategy"""
    # Valid parameter
    config = AggregationConfig(
        strategy_type=AggregationStrategyType.GOSSIP_AVG,
        params={"mixing_parameter": 0.3}
    )
    assert config.params["mixing_parameter"] == 0.3

    # Invalid parameter - too low
    with pytest.raises(ValidationError):
        AggregationConfig(
            strategy_type=AggregationStrategyType.GOSSIP_AVG,
            params={"mixing_parameter": -0.1}
        )

    # Invalid parameter - too high
    with pytest.raises(ValidationError):
        AggregationConfig(
            strategy_type=AggregationStrategyType.GOSSIP_AVG,
            params={"mixing_parameter": 1.1}
        )


def test_defaults_for_missing_params():
    """Test that default parameters are applied when missing"""
    # TrimmedMean with no params
    config = AggregationConfig(strategy_type=AggregationStrategyType.TRIMMED_MEAN)
    assert config.params["trim_ratio"] == 0.1

    # GossipAvg with no params
    config = AggregationConfig(strategy_type=AggregationStrategyType.GOSSIP_AVG)
    assert config.params["mixing_parameter"] == 0.5


def test_unknown_params_preserved():
    """Test that unknown parameters are preserved"""
    config = AggregationConfig(
        strategy_type=AggregationStrategyType.FEDAVG,
        params={"custom_param": "value"}
    )

    assert "custom_param" in config.params
    assert config.params["custom_param"] == "value"
