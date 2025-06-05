from typing import List, Dict, Any, Optional

import pytest

from murmura.aggregation.strategies.fed_avg import FedAvg
from murmura.aggregation.strategies.gossip_avg import GossipAvg
from murmura.aggregation.strategies.trimmed_mean import TrimmedMean
from murmura.aggregation.strategy_interface import AggregationStrategy
from murmura.network_management.topology import TopologyType
from murmura.network_management.topology_compatibility import (
    TopologyCompatibilityManager,
)


def test_is_compatible():
    """Test compatibility check between strategies and topologies"""
    # FedAvg should be compatible with STAR topology
    assert TopologyCompatibilityManager.is_compatible(FedAvg, TopologyType.STAR)

    # FedAvg should not be compatible with LINE topology
    assert not TopologyCompatibilityManager.is_compatible(FedAvg, TopologyType.LINE)

    # GossipAvg should be compatible with most topologies
    assert TopologyCompatibilityManager.is_compatible(GossipAvg, TopologyType.RING)
    assert TopologyCompatibilityManager.is_compatible(GossipAvg, TopologyType.LINE)


def test_get_compatible_topologies():
    """Test getting all compatible topologies for a strategy"""
    # FedAvg should be compatible with STAR and COMPLETE topologies
    fed_avg_topologies = TopologyCompatibilityManager.get_compatible_topologies(FedAvg)
    assert TopologyType.STAR in fed_avg_topologies
    assert TopologyType.COMPLETE in fed_avg_topologies
    assert TopologyType.LINE not in fed_avg_topologies

    # GossipAvg should be compatible with most topology types
    gossip_avg_topologies = TopologyCompatibilityManager.get_compatible_topologies(
        GossipAvg
    )
    assert TopologyType.RING in gossip_avg_topologies
    assert TopologyType.LINE in gossip_avg_topologies
    assert TopologyType.COMPLETE in gossip_avg_topologies


def test_get_compatible_strategies():
    """Test getting all compatible strategies for a topology type"""
    # STAR topology should be compatible with FedAvg, TrimmedMean, and GossipAvg
    star_strategies = TopologyCompatibilityManager.get_compatible_strategies(
        TopologyType.STAR
    )
    assert FedAvg in star_strategies
    assert TrimmedMean in star_strategies
    assert GossipAvg in star_strategies

    # LINE topology should only be compatible with GossipAvg
    line_strategies = TopologyCompatibilityManager.get_compatible_strategies(
        TopologyType.LINE
    )
    assert FedAvg not in line_strategies
    assert TrimmedMean not in line_strategies
    assert GossipAvg in line_strategies


def test_register_compatibility():
    """Test registering new compatibility settings"""

    # Create a dummy strategy class for testing
    class DummyStrategy(AggregationStrategy):
        def aggregate(
            self,
            parameters_list: List[Dict[str, Any]],
            weights: Optional[List[float]] = None,
        ) -> Dict[str, Any]:
            pass

    # Register compatibility for the dummy strategy
    TopologyCompatibilityManager.register_compatibility(
        DummyStrategy, [TopologyType.STAR, TopologyType.RING]
    )

    try:
        # Verify the compatibility settings were added
        assert TopologyCompatibilityManager.is_compatible(
            DummyStrategy, TopologyType.STAR
        )
        assert TopologyCompatibilityManager.is_compatible(
            DummyStrategy, TopologyType.RING
        )
        assert not TopologyCompatibilityManager.is_compatible(
            DummyStrategy, TopologyType.LINE
        )
    finally:
        # Clean up the test changes
        TopologyCompatibilityManager._strategy_topology_map.pop(DummyStrategy, None)


@pytest.mark.parametrize(
    "strategy,topology,expected",
    [
        (FedAvg, TopologyType.STAR, True),
        (FedAvg, TopologyType.COMPLETE, True),
        (FedAvg, TopologyType.LINE, False),
        (TrimmedMean, TopologyType.STAR, True),
        (TrimmedMean, TopologyType.COMPLETE, True),
        (TrimmedMean, TopologyType.LINE, False),
        (GossipAvg, TopologyType.STAR, True),
        (GossipAvg, TopologyType.COMPLETE, True),
        (GossipAvg, TopologyType.LINE, True),
        (GossipAvg, TopologyType.RING, True),
        (GossipAvg, TopologyType.CUSTOM, True),
    ],
)
def test_strategy_topology_matrix(strategy, topology, expected):
    """Parametrized test for all strategy-topology combinations"""
    assert TopologyCompatibilityManager.is_compatible(strategy, topology) == expected


def test_large_topology_stress():
    """Stress test compatibility manager with large topology configs"""

    class DummyStrategy(AggregationStrategy):
        def aggregate(self, parameters_list, weights=None):
            return {}

    # Register DummyStrategy for a large number of fake topologies
    large_topologies = [
        TopologyType.STAR,
        TopologyType.COMPLETE,
        TopologyType.LINE,
        TopologyType.RING,
        TopologyType.CUSTOM,
    ]
    TopologyCompatibilityManager.register_compatibility(DummyStrategy, large_topologies)
    try:
        for t in large_topologies:
            assert TopologyCompatibilityManager.is_compatible(DummyStrategy, t)
    finally:
        TopologyCompatibilityManager._strategy_topology_map.pop(DummyStrategy, None)


@pytest.mark.parametrize(
    "strategy,topology",
    [
        (FedAvg, TopologyType.LINE),
        (TrimmedMean, TopologyType.LINE),
        (FedAvg, TopologyType.CUSTOM),
    ],
)
def test_incompatible_strategy_topology_negative(strategy, topology):
    """Negative test for incompatible strategy-topology pairs"""
    assert not TopologyCompatibilityManager.is_compatible(strategy, topology)
