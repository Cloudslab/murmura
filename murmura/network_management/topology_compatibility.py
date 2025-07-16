from typing import Dict, Type, Set, List

from murmura.aggregation.strategies.fed_avg import FedAvg
from murmura.aggregation.strategies.gossip_avg import GossipAvg
from murmura.aggregation.strategies.trimmed_mean import TrimmedMean
from murmura.aggregation.strategies.trust_weighted_gossip import TrustWeightedGossip
from murmura.aggregation.strategy_interface import AggregationStrategy
from murmura.network_management.topology import TopologyType


class TopologyCompatibilityManager:
    """
    Manages compatibility between network topologies and aggregation strategies
    """

    _strategy_topology_map: Dict[Type[AggregationStrategy], Set[TopologyType]] = {
        FedAvg: {TopologyType.STAR, TopologyType.COMPLETE},
        TrimmedMean: {TopologyType.STAR, TopologyType.COMPLETE},
        GossipAvg: {
            TopologyType.RING,
            TopologyType.COMPLETE,
            TopologyType.STAR,
            TopologyType.LINE,
            TopologyType.CUSTOM,
        },
        TrustWeightedGossip: {
            TopologyType.RING,
            TopologyType.COMPLETE,
            TopologyType.STAR,
            TopologyType.LINE,
            TopologyType.CUSTOM,
        },
    }

    @classmethod
    def is_compatible(
        cls, strategy_class: Type[AggregationStrategy], topology_type: TopologyType
    ) -> bool:
        """
        Check if the aggregation strategy is compatible with the topology type.

        :param strategy_class: Aggregation strategy class
        :param topology_type: Topology type
        :return: True if compatible, False otherwise
        """
        return topology_type in cls._strategy_topology_map.get(strategy_class, set())

    @classmethod
    def get_compatible_topologies(
        cls, strategy_class: Type[AggregationStrategy]
    ) -> List[TopologyType]:
        """
        Get compatible topologies for a given aggregation strategy.

        :param strategy_class: Aggregation strategy class
        :return: List of compatible topology types
        """
        return list(cls._strategy_topology_map.get(strategy_class, set()))

    @classmethod
    def get_compatible_strategies(
        cls, topology_type: TopologyType
    ) -> List[Type[AggregationStrategy]]:
        """
        Get compatible aggregation strategies for a given topology type.

        :param topology_type: Topology type
        :return: List of compatible aggregation strategy classes
        """
        return [
            strategy
            for strategy, topologies in cls._strategy_topology_map.items()
            if topology_type in topologies
        ]

    @classmethod
    def register_compatibility(
        cls,
        strategy_class: Type[AggregationStrategy],
        topology_types: List[TopologyType],
    ) -> None:
        """
        Register compatibility between an aggregation strategy and a topology type.

        :param strategy_class: Aggregation strategy class
        :param topology_types: List of compatible topology types
        """
        cls._strategy_topology_map[strategy_class] = set(topology_types)
