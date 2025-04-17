from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class AggregationStrategy(ABC):
    """
    Abstract interface for aggregation strategies in distributed learning.

    This interface defines how model parameters are aggregated across multiple clients.
    """

    @abstractmethod
    def aggregate(self, parameters_list: List[Dict[str, Any]], weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Aggregate model parameters from multiple clients.

        :param parameters_list: List of model parameters from different clients.
        :param weights: Optional list of weights for each client's parameters.

        :return: Aggregated model parameters.
        """
        pass
