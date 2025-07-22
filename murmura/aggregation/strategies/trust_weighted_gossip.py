from typing import List, Dict, Any, Optional

import numpy as np

from murmura.aggregation.coordination_mode import CoordinationMode
from murmura.aggregation.strategy_interface import AggregationStrategy


class TrustWeightedGossip(AggregationStrategy):
    """
    Implements trust-weighted gossip averaging for decentralized model aggregation.
    
    This strategy applies mixing parameters where a node preserves a portion of its own
    parameters and distributes the remaining weight among neighbors based on trust scores.
    
    Formula: aggregated = mixing_parameter * own_params + (1 - mixing_parameter) * trust_weighted_neighbors
    """

    coordination_mode = CoordinationMode.DECENTRALIZED

    def __init__(self, mixing_parameter: float = 0.5) -> None:
        """
        Initialize the TrustWeightedGossip strategy.

        :param mixing_parameter: Weight for own parameters vs neighbors (0.5 = equal weighting)
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
        trust_scores: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate model parameters using trust-weighted gossip with mixing parameter.

        :param parameters_list: List of model parameters [own_params, neighbor1_params, neighbor2_params, ...]
        :param weights: Optional list of base weights (typically not used in trust-weighted)
        :param round_number: Optional round number
        :param sampling_rate: Optional sampling rate
        :param trust_scores: Dictionary mapping neighbor indices to trust scores {0: 1.0, 1: 0.65, 2: 0.01}
        :return: Aggregated model parameters
        """
        if not parameters_list:
            raise ValueError("No parameters to aggregate.")

        # Special case: If there's only one set of parameters, just return it
        if len(parameters_list) == 1:
            return parameters_list[0].copy()

        # Extract own parameters (first in list) and neighbor parameters
        own_params = parameters_list[0]
        neighbor_params_list = parameters_list[1:]
        
        if not neighbor_params_list:
            # No neighbors, return own parameters
            return own_params.copy()

        # Set up trust scores for neighbors
        if trust_scores is None:
            # Default: equal trust for all neighbors
            trust_scores = {i: 1.0 for i in range(len(neighbor_params_list))}
        
        # Ensure we have trust scores for all neighbors
        neighbor_trust_scores = {}
        for i, _ in enumerate(neighbor_params_list):
            neighbor_trust_scores[i] = trust_scores.get(i, 1.0)

        # Calculate total trust for normalization
        total_trust = sum(neighbor_trust_scores.values())
        if total_trust <= 0:
            # All neighbors have zero trust, return own parameters
            print(f"TRUST_WEIGHTED_GOSSIP: All neighbors have zero trust, using only own parameters")
            return own_params.copy()

        # Log trust-weighted aggregation details
        print(f"TRUST_WEIGHTED_GOSSIP: Aggregating with mixing_parameter={self.mixing_parameter:.3f}")
        print(f"TRUST_WEIGHTED_GOSSIP: Trust scores: {neighbor_trust_scores}")
        neighbor_weights = {i: score / total_trust for i, score in neighbor_trust_scores.items()}
        print(f"TRUST_WEIGHTED_GOSSIP: Normalized neighbor weights: {neighbor_weights}")
        print(f"TRUST_WEIGHTED_GOSSIP: Own weight: {self.mixing_parameter:.3f}, Total neighbor weight: {1-self.mixing_parameter:.3f}")

        aggregated_params = {}

        for key in own_params.keys():
            try:
                # Handle integer parameters specially (num_batches_tracked, etc.)
                if "num_batches_tracked" in key or (
                    hasattr(own_params[key], 'dtype') and 
                    np.issubdtype(own_params[key].dtype, np.integer)
                ):
                    # For integer parameters, use maximum value
                    all_values = [own_params[key]] + [params[key] for params in neighbor_params_list]
                    aggregated_params[key] = np.max(all_values)
                else:
                    # Apply mixing parameter logic
                    # Self component
                    self_component = self.mixing_parameter * own_params[key]
                    
                    # Trust-weighted neighbor component
                    neighbor_aggregated = np.zeros_like(own_params[key])
                    for neighbor_idx, neighbor_params in enumerate(neighbor_params_list):
                        trust_weight = neighbor_trust_scores[neighbor_idx] / total_trust
                        neighbor_aggregated += trust_weight * neighbor_params[key]
                    
                    neighbor_component = (1 - self.mixing_parameter) * neighbor_aggregated
                    
                    aggregated_params[key] = self_component + neighbor_component

            except Exception as e:
                raise ValueError(f"Error processing parameters for key '{key}': {e}")

        return aggregated_params