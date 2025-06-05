import pytest
import numpy as np

from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.aggregation.strategy_factory import AggregationStrategyFactory
from murmura.aggregation.strategies.fed_avg import FedAvg
from murmura.aggregation.strategies.trimmed_mean import TrimmedMean
from murmura.aggregation.strategies.gossip_avg import GossipAvg
from murmura.network_management.topology import TopologyConfig, TopologyType


def test_create_fedavg_strategy():
    """Test creating a FedAvg strategy from config"""
    config = AggregationConfig(strategy_type=AggregationStrategyType.FEDAVG)
    strategy = AggregationStrategyFactory.create(config)

    assert isinstance(strategy, FedAvg)


def test_create_trimmed_mean_strategy():
    """Test creating a TrimmedMean strategy from config with parameters"""
    config = AggregationConfig(
        strategy_type=AggregationStrategyType.TRIMMED_MEAN, params={"trim_ratio": 0.2}
    )
    strategy = AggregationStrategyFactory.create(config)

    assert isinstance(strategy, TrimmedMean)
    assert strategy.trim_ratio == 0.2


def test_create_gossip_avg_strategy():
    """Test creating a GossipAvg strategy from config with parameters"""
    config = AggregationConfig(
        strategy_type=AggregationStrategyType.GOSSIP_AVG,
        params={"mixing_parameter": 0.7},
    )
    strategy = AggregationStrategyFactory.create(config)

    assert isinstance(strategy, GossipAvg)
    assert strategy.mixing_parameter == 0.7


def test_create_unsupported_strategy():
    """Test creating an unsupported strategy type raises ValueError"""
    # Create a config with a strategy type that doesn't exist
    config = AggregationConfig()
    # Hack the strategy_type to an unsupported value
    config.strategy_type = "unsupported_strategy"  # type: ignore

    with pytest.raises(ValueError, match="Unsupported aggregation strategy"):
        AggregationStrategyFactory.create(config)


def test_topology_compatibility_check_compatible():
    """Test strategy creation with compatible topology"""
    strategy_config = AggregationConfig(strategy_type=AggregationStrategyType.FEDAVG)
    topology_config = TopologyConfig(topology_type=TopologyType.STAR)

    # Should create successfully - FedAvg is compatible with STAR topology
    strategy = AggregationStrategyFactory.create(strategy_config, topology_config)
    assert isinstance(strategy, FedAvg)


def test_topology_compatibility_check_incompatible():
    """Test strategy creation with incompatible topology raises error"""
    from murmura.network_management.topology_compatibility import (
        TopologyCompatibilityManager,
    )

    # Create configs
    strategy_config = AggregationConfig(strategy_type=AggregationStrategyType.FEDAVG)
    topology_config = TopologyConfig(topology_type=TopologyType.LINE)

    # Get the original compatibility settings to restore after test
    original_map = TopologyCompatibilityManager._strategy_topology_map.copy()

    try:
        # Should raise ValueError - FedAvg is not compatible with LINE topology
        with pytest.raises(
            ValueError, match="Strategy fedavg is not compatible with topology line"
        ):
            AggregationStrategyFactory.create(strategy_config, topology_config)
    finally:
        # Restore original compatibility settings
        TopologyCompatibilityManager._strategy_topology_map = original_map


@pytest.mark.parametrize(
    "strategy_type,params,should_raise",
    [
        (AggregationStrategyType.FEDAVG, None, False),
        (AggregationStrategyType.TRIMMED_MEAN, {"trim_ratio": 0.2}, False),
        (AggregationStrategyType.TRIMMED_MEAN, {"trim_ratio": -0.1}, True),
        (AggregationStrategyType.TRIMMED_MEAN, {"trim_ratio": 0.5}, True),
        (AggregationStrategyType.GOSSIP_AVG, {"mixing_parameter": 0.7}, False),
        (AggregationStrategyType.GOSSIP_AVG, {"mixing_parameter": -0.1}, True),
        (AggregationStrategyType.GOSSIP_AVG, {"mixing_parameter": 1.1}, True),
    ],
)
def test_strategy_param_validation(strategy_type, params, should_raise):
    """Test parameter validation for all aggregation strategies"""
    if should_raise:
        with pytest.raises(Exception):
            AggregationConfig(strategy_type=strategy_type, params=params)
    else:
        config = AggregationConfig(strategy_type=strategy_type, params=params)
        assert config.strategy_type == strategy_type


def test_aggregation_empty_parameters():
    """Test aggregation with empty parameters for all strategies"""
    for strategy_cls in [FedAvg, TrimmedMean, GossipAvg]:
        strategy = strategy_cls()
        with pytest.raises(Exception):
            strategy.aggregate([])


def test_aggregation_null_weights():
    """Test aggregation with null weights (should default to equal)"""
    params = [{"layer": np.array([1.0, 2.0])}, {"layer": np.array([3.0, 4.0])}]
    for strategy_cls in [FedAvg, TrimmedMean, GossipAvg]:
        strategy = strategy_cls()
        result = strategy.aggregate(params, None)
        assert "layer" in result


def test_aggregation_large_scale():
    """Performance test for large-scale aggregation (marked slow)"""
    params = [{"layer": np.random.rand(1000)} for _ in range(1000)]
    weights = np.random.dirichlet(np.ones(1000)).tolist()
    for strategy_cls in [FedAvg, TrimmedMean, GossipAvg]:
        strategy = strategy_cls()
        result = strategy.aggregate(params, weights)
        assert result["layer"].shape == (1000,)
