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
        round_number: Optional[int] = None,
        sampling_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate model parameters using the Gossip Averaging algorithm.

        Note: This is called by the TopologyCoordinator which handles the actual communication pattern based on
        topology.

        :param parameters_list: List of model parameters from different actors
        :param weights: Optional list of weights for each actor's parameters
        :param round_number: Optional round number
        :param sampling_rate: Optional sampling rate

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

        # Apply mixing parameter logic: own_weight * own_params + neighbor_weight * neighbor_aggregation
        own_params = parameters_list[0]  # First parameter set is always own node
        neighbor_params_list = parameters_list[1:]  # Rest are neighbors
        neighbor_weights_list = weights[1:] if len(weights) > 1 else []

        # Log gossip averaging aggregation details
        print(f"GOSSIP_AVG: Aggregating with mixing_parameter={self.mixing_parameter:.3f}")
        print(f"GOSSIP_AVG: Base weights: {weights}")
        print(f"GOSSIP_AVG: Neighbor weights: {neighbor_weights_list}")
        print(f"GOSSIP_AVG: Own weight: {self.mixing_parameter:.3f}, Total neighbor weight: {1-self.mixing_parameter:.3f}")
        
        aggregated_params = {}

        for key in own_params.keys():
            try:
                # Handle 'num_batches_tracked' and other integer parameters specially
                if "num_batches_tracked" in key or any(
                    np.issubdtype(params[key].dtype, np.integer)
                    for params in parameters_list
                ):
                    # For integer parameters, we'll use the maximum value
                    aggregated_params[key] = np.max(
                        [params[key] for params in parameters_list]
                    )
                else:
                    # Apply mixing parameter logic
                    if len(neighbor_params_list) == 0:
                        # No neighbors, just return own parameters
                        aggregated_params[key] = own_params[key]
                    else:
                        # Self component (using mixing parameter)
                        self_component = self.mixing_parameter * own_params[key]
                        
                        # Neighbor component (remaining weight distributed among neighbors)
                        neighbor_weight_total = sum(neighbor_weights_list) if neighbor_weights_list else len(neighbor_params_list)
                        if neighbor_weight_total > 0:
                            # Normalize neighbor weights to sum to 1
                            if neighbor_weights_list:
                                normalized_neighbor_weights = [w / neighbor_weight_total for w in neighbor_weights_list]
                            else:
                                normalized_neighbor_weights = [1.0 / len(neighbor_params_list)] * len(neighbor_params_list)
                            
                            # Aggregate neighbors with normalized weights
                            neighbor_aggregated = np.zeros_like(own_params[key])
                            for i, neighbor_params in enumerate(neighbor_params_list):
                                neighbor_aggregated += normalized_neighbor_weights[i] * neighbor_params[key]
                            
                            neighbor_component = (1 - self.mixing_parameter) * neighbor_aggregated
                        else:
                            neighbor_component = np.zeros_like(own_params[key])
                        
                        aggregated_params[key] = self_component + neighbor_component

            except ValueError as e:
                raise ValueError(f"Error processing parameters for key '{key}': {e}")

        return aggregated_params
