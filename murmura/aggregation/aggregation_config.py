from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, model_validator


class AggregationStrategyType(str, Enum):
    """
    Enumeration of available aggregation strategies.
    """

    FEDAVG = "fedavg"
    TRIMMED_MEAN = "trimmed_mean"
    GOSSIP_AVG = "gossip_avg"
    TRUST_WEIGHTED_GOSSIP = "trust_weighted_gossip"


class AggregationConfig(BaseModel):
    """
    Configuration object for aggregation strategies in distributed learning.
    """

    strategy_type: AggregationStrategyType = Field(
        default=AggregationStrategyType.FEDAVG,
        description="Type of aggregation strategy to use",
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Strategy-specific parameters for the aggregation method",
    )

    @model_validator(mode="after")
    def validate_strategy(self) -> "AggregationConfig":
        """
        Validate the aggregation strategy and its parameters.
        """
        # Initialize empty params dict if None
        if self.params is None:
            self.params = {}

        # Validate strategy-specific parameters
        if self.strategy_type == AggregationStrategyType.TRIMMED_MEAN:
            # Ensure trim_ratio exists and is valid
            trim_ratio = self.params.get("trim_ratio", 0.1)
            if trim_ratio < 0 or trim_ratio >= 0.5:
                raise ValueError("trim_ratio must be in [0, 0.5)")
            self.params["trim_ratio"] = trim_ratio
        elif self.strategy_type == AggregationStrategyType.GOSSIP_AVG:
            mixing_parameter = self.params.get("mixing_parameter", 0.25)
            if mixing_parameter < 0 or mixing_parameter > 1:
                raise ValueError("mixing_parameter must be in [0, 1]")
            self.params["mixing_parameter"] = mixing_parameter
        elif self.strategy_type == AggregationStrategyType.TRUST_WEIGHTED_GOSSIP:
            mixing_parameter = self.params.get("mixing_parameter", 0.25)
            if mixing_parameter < 0 or mixing_parameter > 1:
                raise ValueError("mixing_parameter must be in [0, 1]")
            self.params["mixing_parameter"] = mixing_parameter

        return self
