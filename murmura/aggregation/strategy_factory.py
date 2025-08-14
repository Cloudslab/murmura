from typing import Optional, Type

from murmura.aggregation.aggregation_config import (
    AggregationConfig,
    AggregationStrategyType,
)
from murmura.aggregation.strategies.fed_avg import FedAvg
from murmura.aggregation.strategies.gossip_avg import GossipAvg
from murmura.aggregation.strategies.trimmed_mean import TrimmedMean
from murmura.aggregation.strategies.trust_weighted_gossip import TrustWeightedGossip
from murmura.aggregation.strategy_interface import AggregationStrategy
from murmura.network_management.topology import TopologyConfig
from murmura.network_management.topology_compatibility import (
    TopologyCompatibilityManager,
)


class AggregationStrategyFactory:
    """
    Factory for creating aggregation strategy instances based on configuration
    """

    @staticmethod
    def create(
        config: AggregationConfig, topology_config: Optional[TopologyConfig] = None
    ) -> AggregationStrategy:
        """
        Create an aggregation strategy instance based on the configuration,
        checking compatibility with the topology if provided.

        Args:
            config: Aggregation configuration
            topology_config: Optional topology configuration for compatibility check

        Returns:
            Initialized aggregation strategy instance

        Raises:
            ValueError: If the aggregation strategy is not supported or incompatible with the topology
        """
        strategy_type = config.strategy_type
        params = config.params or {}
        strategy_class: Type[AggregationStrategy]

        if strategy_type == AggregationStrategyType.FEDAVG:
            strategy_class = FedAvg
        elif strategy_type == AggregationStrategyType.TRIMMED_MEAN:
            strategy_class = TrimmedMean
        elif strategy_type == AggregationStrategyType.GOSSIP_AVG:
            strategy_class = GossipAvg
        elif strategy_type == AggregationStrategyType.TRUST_WEIGHTED_GOSSIP:
            strategy_class = TrustWeightedGossip
        else:
            raise ValueError(f"Unsupported aggregation strategy: {strategy_type}")

        # Check compatibility with topology if provided
        if topology_config:
            topology_type = topology_config.topology_type
            if not TopologyCompatibilityManager.is_compatible(
                strategy_class, topology_type
            ):
                compatible_topologies = (
                    TopologyCompatibilityManager.get_compatible_topologies(
                        strategy_class
                    )
                )
                compatible_strategies = (
                    TopologyCompatibilityManager.get_compatible_strategies(
                        topology_type
                    )
                )

                raise ValueError(
                    f"Strategy {strategy_type.value} is not compatible with topology {topology_type.value}. "
                    f"Compatible topologies for this strategy: {[t.value for t in compatible_topologies]}. "
                    f"Compatible strategies for {topology_type.value} topology: "
                    f"{[s.__name__.lower() for s in compatible_strategies]}."
                )

        return strategy_class(**params)
