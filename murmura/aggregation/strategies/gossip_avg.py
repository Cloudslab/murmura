from typing import List, Dict, Any, Optional

import numpy as np

from murmura.aggregation.coordination_mode import CoordinationMode
from murmura.aggregation.strategy_interface import AggregationStrategy


class GossipAvg(AggregationStrategy):
    """
    Implements the Gossip Averaging algorithm for decentralized model aggregation.

    This algorithm is suitable for non-star topologies where each node
    exchanges parameters with its neighbors and computes local averages.
    """

    coordination_mode = CoordinationMode.DECENTRALIZED

    def __init__(self, mixing_parameter: float = 0.5) -> None:
        """
        Initialize the GossipAvg strategy with a mixing parameter.

        :param mixing_parameter: Controls how much weight to give to neighbours versus local model (default is 0.5
        for equal weighting)
        """
        if mixing_parameter < 0 or mixing_parameter > 1:
            raise ValueError("Mixing parameter must be between 0 and 1.")

        self.mixing_parameter = mixing_parameter

    def aggregate(
            self,
            parameters_list: List[Dict[str, Any]],
            weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate model parameters using the Gossip Averaging algorithm.

        Note: This is called by the TopologyCoordinator which handles the actual communication pattern based on
        topology.

        :param parameters_list: List of model parameters from different actors
        :param weights: Optional list of weights for each actor's parameters
        :return: Aggregated model parameters
        """
        if not parameters_list:
            raise ValueError("No parameters to aggregate.")

        # Special case: If there's only one set of parameters, just return it
        if len(parameters_list) == 1:
            return parameters_list[0].copy()

        # If no weights are provided, assume equal weighting
        if weights is None:
            weights = [1.0 / len(parameters_list)] * len(parameters_list)
        else:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

        aggregated_params = {}

        for key in parameters_list[0].keys():
            try:
                # Handle 'num_batches_tracked' and other integer parameters specially
                if 'num_batches_tracked' in key or any(
                        np.issubdtype(params[key].dtype, np.integer)
                        for params in parameters_list
                ):
                    # For integer parameters, we'll use the maximum value
                    # This is especially appropriate for 'num_batches_tracked'
                    aggregated_params[key] = np.max([
                        params[key] for params in parameters_list
                    ])
                else:
                    stacked_params = np.stack(
                        [params[key] for params in parameters_list], axis=0
                    )
                    weighted_params = np.zeros_like(stacked_params[0])

                    for i, weight in enumerate(weights):
                        weighted_params += weight * stacked_params[i]

                    aggregated_params[key] = weighted_params

            except ValueError as e:
                raise ValueError(f"Error stacking parameters for key '{key}': {e}")

        return aggregated_params
