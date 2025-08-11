from typing import List, Dict, Any, Optional
import logging

import numpy as np

from murmura.aggregation.coordination_mode import CoordinationMode
from murmura.aggregation.strategy_interface import AggregationStrategy


class TrustWeightedGossip(AggregationStrategy):
    """
    Adaptive trust-weighted gossip averaging that preserves network connectivity.
    
    Key innovations:
    1. Uses influence weights instead of raw trust scores
    2. Maintains minimum weights to preserve routing benefits
    3. Adaptive mixing based on trust distribution
    
    Formula: aggregated = adaptive_mix * own_params + (1 - adaptive_mix) * weighted_neighbors
    """

    coordination_mode = CoordinationMode.DECENTRALIZED

    def __init__(
        self, 
        mixing_parameter: float = 0.25,
        adaptive_mixing: bool = True,
        min_neighbor_influence: float = 0.02
    ) -> None:
        """
        Initialize the TrustWeightedGossip strategy.

        :param mixing_parameter: Base weight for own parameters (0.25 = collaborative learning)
        :param adaptive_mixing: Whether to adapt mixing based on trust distribution
        :param min_neighbor_influence: Minimum influence to preserve connectivity
        """
        if mixing_parameter < 0 or mixing_parameter > 1:
            raise ValueError("Mixing parameter must be between 0 and 1.")

        self.base_mixing_parameter = mixing_parameter
        self.adaptive_mixing = adaptive_mixing
        self.min_neighbor_influence = min_neighbor_influence
        self.logger = logging.getLogger("murmura.trust_weighted_gossip")

    def aggregate(
        self,
        parameters_list: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        round_number: Optional[int] = None,
        sampling_rate: Optional[float] = None,
        trust_scores: Optional[Dict[int, float]] = None,
        influence_weights: Optional[Dict[int, float]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate model parameters using adaptive trust-weighted gossip.

        :param parameters_list: List of model parameters [own_params, neighbor1_params, neighbor2_params, ...]
        :param weights: Optional list of base weights (typically not used)
        :param round_number: Optional round number
        :param sampling_rate: Optional sampling rate
        :param trust_scores: Legacy - Dictionary mapping neighbor indices to trust scores
        :param influence_weights: Dictionary mapping neighbor indices to influence weights
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

        # Use influence weights if provided, otherwise fall back to trust scores
        if influence_weights is not None:
            neighbor_weights = influence_weights
        elif trust_scores is not None:
            # Legacy support: convert trust scores to influence weights
            neighbor_weights = {}
            for i in range(len(neighbor_params_list)):
                score = trust_scores.get(i, 1.0)
                # Apply minimum weight to preserve connectivity
                neighbor_weights[i] = max(score, self.min_neighbor_influence)
        else:
            # Default: equal influence for all neighbors
            neighbor_weights = {i: 1.0 for i in range(len(neighbor_params_list))}
        
        # Ensure all neighbors have weights with minimum connectivity
        for i in range(len(neighbor_params_list)):
            if i not in neighbor_weights:
                neighbor_weights[i] = 1.0
            else:
                neighbor_weights[i] = max(neighbor_weights[i], self.min_neighbor_influence)
        
        # Calculate adaptive mixing parameter
        if self.adaptive_mixing and len(neighbor_weights) > 0:
            mixing_parameter = self._compute_adaptive_mixing(neighbor_weights, round_number)
        else:
            mixing_parameter = self.base_mixing_parameter
        
        # Normalize neighbor weights
        total_weight = sum(neighbor_weights.values())
        if total_weight <= 0:
            self.logger.warning("All neighbors have zero weight, using only own parameters")
            return own_params.copy()
        
        normalized_weights = {i: w / total_weight for i, w in neighbor_weights.items()}
        
        # Log aggregation details
        self.logger.info(
            f"TRUST_WEIGHTED_GOSSIP: Round {round_number}, mixing={mixing_parameter:.3f}, "
            f"influence_weights={neighbor_weights}"
        )
        self.logger.debug(
            f"Normalized weights: {normalized_weights}, "
            f"own_weight={mixing_parameter:.3f}, neighbor_weight={1-mixing_parameter:.3f}"
        )

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
                    # Apply adaptive mixing parameter logic
                    # Self component
                    self_component = mixing_parameter * own_params[key]
                    
                    # Trust-weighted neighbor component
                    neighbor_aggregated = np.zeros_like(own_params[key])
                    for neighbor_idx, neighbor_params in enumerate(neighbor_params_list):
                        weight = normalized_weights[neighbor_idx]
                        neighbor_aggregated += weight * neighbor_params[key]
                    
                    neighbor_component = (1 - mixing_parameter) * neighbor_aggregated
                    
                    aggregated_params[key] = self_component + neighbor_component

            except Exception as e:
                raise ValueError(f"Error processing parameters for key '{key}': {e}")

        return aggregated_params
    
    def _compute_adaptive_mixing(self, neighbor_weights: Dict[int, float], round_number: Optional[int]) -> float:
        """
        Compute adaptive mixing parameter based on trust distribution.
        
        When neighbors are untrusted, increase own weight.
        When neighbors are trusted, use balanced mixing.
        """
        if not neighbor_weights:
            return self.base_mixing_parameter
        
        # Calculate trust statistics
        weights = list(neighbor_weights.values())
        mean_weight = np.mean(weights)
        
        # Round-based adaptation factor
        if round_number is not None:
            # Early rounds: more conservative (higher own weight)
            # Later rounds: more collaborative (if neighbors are trusted)
            round_factor = min(1.0, round_number / 20)
        else:
            round_factor = 0.5
        
        # Adapt mixing based on neighbor trustworthiness
        if mean_weight < 0.3:
            # Low trust in neighbors: increase own weight
            adaptive_factor = 0.7 + 0.2 * (1 - round_factor)
        elif mean_weight < 0.6:
            # Medium trust: moderate mixing
            adaptive_factor = 0.5 + 0.1 * (1 - mean_weight)
        else:
            # High trust: allow more neighbor influence
            adaptive_factor = self.base_mixing_parameter * (0.8 + 0.2 * round_factor)
        
        # Ensure bounds
        return np.clip(adaptive_factor, 0.3, 0.8)