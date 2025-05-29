from typing import List, Dict, Any, Optional

import numpy as np

from murmura.aggregation.coordination_mode import CoordinationMode
from murmura.aggregation.strategy_interface import AggregationStrategy


class FedAvg(AggregationStrategy):
    """
    Implements the Federated Averaging (FedAvg) algorithm for model aggregation.

    FedAvg computes the weighted average of model parameters from multiple clients.
    """

    coordination_mode = CoordinationMode.CENTRALIZED

    def __init__(self):
        """
        Initialize the FedAvg strategy.
        """
        super().__init__()

    def aggregate(
        self,
        parameters_list: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        round_number: Optional[int] = None,
        sampling_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate model parameters using FedAvg.

        :param parameters_list: List of model parameters from different clients.
        :param weights: Optional list of weights for each client's parameters.
        :param round_number: Optional round number.
        :param sampling_rate: Optional sampling rate.

        :return: Aggregated model parameters.
        """
        if not parameters_list:
            raise ValueError("Empty parameters list")

        # Special case: If there's only one client, just return its parameters
        if len(parameters_list) == 1:
            return parameters_list[0].copy()

        if weights is None:
            weights = [1.0 / len(parameters_list)] * len(parameters_list)
        else:
            total = sum(weights)
            weights = [w / total for w in weights]

        aggregated_params = {}

        for key in parameters_list[0].keys():
            try:
                # Handle 'num_batches_tracked' and other integer parameters specially
                if "num_batches_tracked" in key or any(
                    np.issubdtype(params[key].dtype, np.integer)
                    for params in parameters_list
                ):
                    # For integer parameters, we'll use the maximum value
                    # This is especially appropriate for 'num_batches_tracked'
                    aggregated_params[key] = np.max(
                        [params[key] for params in parameters_list]
                    )
                else:
                    # Normal floating-point parameters use weighted average
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
