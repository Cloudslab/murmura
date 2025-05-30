from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, ClassVar

from murmura.aggregation.coordination_mode import CoordinationMode


class AggregationStrategy(ABC):
    """
    Abstract interface for aggregation strategies in distributed learning.

    This interface defines how model parameters are aggregated across multiple clients.
    """

    coordination_mode: ClassVar[CoordinationMode] = CoordinationMode.DECENTRALIZED

    @abstractmethod
    def aggregate(
        self,
        parameters_list: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        round_number: Optional[int] = None,
        sampling_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate model parameters from multiple clients.

        :param parameters_list: List of model parameters from different clients.
        :param weights: Optional list of weights for each client's parameters.
        :param round_number: Optional round number.
        :param sampling_rate: Optional sampling rate.

        :return: Aggregated model parameters.
        """
        pass
