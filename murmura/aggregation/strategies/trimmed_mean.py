from typing import List, Dict, Any, Optional

import numpy as np

from murmura.aggregation.strategy_interface import AggregationStrategy


class TrimmedMean(AggregationStrategy):
    """
    Implements the Trimmed Mean aggregation strategy for Byzantine-robust aggregation.

    This strategy trims the highest and lowest values of each parameter before averaging,
    which provides robustness against malicious clients that may try to poison the model.
    """

    def __init__(self, trim_ratio: float = 0.1) -> None:
        """
        Initialize the TrimmedMean strategy.

        :param trim_ratio: The ratio of values to trim from each end (default is 0.1).
        """
        if trim_ratio < 0 or trim_ratio > 0.5:
            raise ValueError(
                "Trim ratio must be between 0 (inclusive) and 0.5 (exclusive)"
            )

        self.trim_ratio = trim_ratio

    def aggregate(
        self,
        parameters_list: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate parameters using trimmed mean

        :param parameters_list: List of parameter dictionaries from clients
        :param weights: Optional list of weights for each client (ignored in this strategy)

        :return: Aggregated parameters as a dictionary
        """
        if not parameters_list:
            raise ValueError("Empty parameters list")

        num_clients = len(parameters_list)
        if num_clients <= 2:
            # Not enough clients to perform trimming, fall back to simple average
            equal_weights = [1.0 / num_clients] * num_clients
            return self._weighted_average(parameters_list, equal_weights)

        # Calculate how many values to trim from each end
        k = int(num_clients * self.trim_ratio)

        # Initialize with keys from the first set of parameters
        aggregated_params = {}

        for key in parameters_list[0].keys():
            try:
                # Stack parameters along a new axis
                stacked_params = np.stack(
                    [params[key] for params in parameters_list], axis=0
                )

                # For each parameter element, sort values across clients and trim
                # We need to sort along axis 0 (client dimension)
                sorted_params = np.sort(stacked_params, axis=0)

                # Trim k values from each end
                trimmed_params = sorted_params[k : num_clients - k]

                # Average the remaining values
                aggregated_params[key] = np.mean(trimmed_params, axis=0)

            except Exception as e:
                raise ValueError(f"Error aggregating parameter {key}: {e}")

        return aggregated_params

    @staticmethod
    def _weighted_average(
        parameters_list: List[Dict[str, Any]], weights: List[float]
    ) -> Dict[str, Any]:
        """
        Helper method for weighted average when falling back
        """
        aggregated_params = {}

        for key in parameters_list[0].keys():
            try:
                stacked_params = np.stack(
                    [params[key] for params in parameters_list], axis=0
                )
                weighted_params = np.zeros_like(stacked_params[0])

                for i, weight in enumerate(weights):
                    weighted_params += weight * stacked_params[i]

                aggregated_params[key] = weighted_params

            except Exception as e:
                raise ValueError(f"Error aggregating parameter {key}: {e}")

        return aggregated_params
