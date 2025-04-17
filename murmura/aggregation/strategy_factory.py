from murmura.aggregation.aggregation_config import AggregationConfig, AggregationStrategyType
from murmura.aggregation.strategies.fed_avg import FedAvg
from murmura.aggregation.strategies.trimmed_mean import TrimmedMean
from murmura.aggregation.strategy_interface import AggregationStrategy


class AggregationStrategyFactory:
    """
    Factory for creating aggregation strategy instances based on configuration
    """

    @staticmethod
    def create(config: AggregationConfig) -> AggregationStrategy:
        """
        Create an aggregation strategy instance based on the configuration.

        Args:
            config: Aggregation configuration

        Returns:
            Initialized aggregation strategy instance
        """
        strategy_type = config.strategy_type
        params = config.params or {}

        if strategy_type == AggregationStrategyType.FEDAVG:
            return FedAvg()
        elif strategy_type == AggregationStrategyType.TRIMMED_MEAN:
            trim_ratio = params.get("trim_ratio", 0.1)
            return TrimmedMean(trim_ratio=trim_ratio)
        else:
            raise ValueError(f"Unsupported aggregation strategy: {strategy_type}")
