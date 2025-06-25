"""
Trust-aware gossip averaging strategy for decentralized federated learning.

This module implements a gossip averaging strategy that incorporates trust
scores from trust monitors to adjust aggregation weights.
"""

from typing import List, Dict, Any, Optional
import logging

import numpy as np
import ray

from murmura.aggregation.coordination_mode import CoordinationMode
from murmura.aggregation.strategy_interface import AggregationStrategy
from murmura.trust.trust_monitor import TrustMonitor


class TrustAwareGossipAvg(AggregationStrategy):
    """
    Trust-aware implementation of the Gossip Averaging algorithm.
    
    This algorithm adjusts aggregation weights based on trust assessments
    from trust monitors, allowing nodes to reduce influence of suspicious
    or malicious neighbors.
    """
    
    coordination_mode = CoordinationMode.DECENTRALIZED
    
    def __init__(
        self,
        mixing_parameter: float = 0.5,
        trust_monitors: Optional[Dict[str, TrustMonitor]] = None,
        use_trust_weights: bool = True,
    ) -> None:
        """
        Initialize the TrustAwareGossipAvg strategy.
        
        Args:
            mixing_parameter: Controls weight given to neighbors vs local model
            trust_monitors: Dictionary mapping node IDs to trust monitor actors
            use_trust_weights: Whether to use trust-based weights
        """
        if mixing_parameter < 0 or mixing_parameter > 1:
            raise ValueError("Mixing parameter must be between 0 and 1.")
        
        self.mixing_parameter = mixing_parameter
        self.trust_monitors = trust_monitors or {}
        self.use_trust_weights = use_trust_weights
        self.logger = logging.getLogger(f"{__name__}.TrustAwareGossipAvg")
    
    def set_trust_monitors(self, trust_monitors: Dict[str, TrustMonitor]) -> None:
        """
        Set trust monitors for trust-aware aggregation.
        
        Args:
            trust_monitors: Dictionary mapping node IDs to trust monitor actors
        """
        self.trust_monitors = trust_monitors
        self.logger.info(f"Set {len(trust_monitors)} trust monitors")
    
    def aggregate(
        self,
        parameters_list: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
        round_number: Optional[int] = None,
        sampling_rate: Optional[float] = None,
        node_id: Optional[str] = None,
        neighbor_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Aggregate model parameters using trust-aware gossip averaging.
        
        Args:
            parameters_list: List of model parameters from different actors
            weights: Optional base weights for each actor's parameters
            round_number: Optional round number
            sampling_rate: Optional sampling rate
            node_id: ID of the node performing aggregation
            neighbor_ids: IDs of neighbors providing parameters
            
        Returns:
            Aggregated model parameters
        """
        if not parameters_list:
            raise ValueError("No parameters to aggregate.")
        
        # Special case: single parameter set
        if len(parameters_list) == 1:
            return parameters_list[0].copy()
        
        # Get trust weights if enabled and possible
        trust_weights = None
        if self.use_trust_weights and node_id and neighbor_ids and node_id in self.trust_monitors:
            try:
                # Get trust monitor for this node
                trust_monitor = self.trust_monitors[node_id]
                
                # Get trust-adjusted weights
                trust_weight_dict = ray.get(
                    trust_monitor.get_trust_weights.remote(neighbor_ids)
                )
                
                # Convert to list in same order as parameters
                trust_weights = []
                for i, neighbor_id in enumerate(neighbor_ids):
                    if neighbor_id in trust_weight_dict:
                        trust_weights.append(trust_weight_dict[neighbor_id])
                    else:
                        # Default weight for unknown neighbors
                        trust_weights.append(1.0)
                
                self.logger.debug(
                    f"Node {node_id} using trust weights: "
                    f"{dict(zip(neighbor_ids, trust_weights))}"
                )
                
            except Exception as e:
                self.logger.warning(
                    f"Failed to get trust weights for node {node_id}: {e}. "
                    "Falling back to uniform weights."
                )
                trust_weights = None
        
        # Combine base weights with trust weights
        if weights is None:
            if trust_weights is not None:
                weights = trust_weights
            else:
                weights = [1.0 / len(parameters_list)] * len(parameters_list)
        else:
            if trust_weights is not None:
                # Multiply base weights by trust weights
                weights = [w * tw for w, tw in zip(weights, trust_weights)]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # All weights are zero (all neighbors excluded)
            self.logger.warning(
                f"All neighbors excluded by trust. Using only local parameters."
            )
            # Return first parameter set (should be local)
            return parameters_list[0].copy()
        
        # Log excluded neighbors
        if neighbor_ids and trust_weights:
            excluded = [
                neighbor_ids[i]
                for i, tw in enumerate(trust_weights)
                if tw == 0.0
            ]
            if excluded:
                self.logger.info(
                    f"Node {node_id} excluded {len(excluded)} neighbors: {excluded}"
                )
        
        # Perform weighted aggregation
        aggregated_params = {}
        
        for key in parameters_list[0].keys():
            try:
                # Handle special parameters
                if "num_batches_tracked" in key or any(
                    np.issubdtype(params[key].dtype, np.integer)
                    for params in parameters_list
                ):
                    # For integer parameters, use maximum
                    aggregated_params[key] = np.max(
                        [params[key] for params in parameters_list]
                    )
                else:
                    # Weighted average for continuous parameters
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
    
    def get_trust_summary(self, node_ids: List[str]) -> Dict[str, Any]:
        """
        Get summary of trust information across nodes.
        
        Args:
            node_ids: List of node IDs to get trust info for
            
        Returns:
            Trust summary dictionary
        """
        summary = {
            "total_nodes": len(node_ids),
            "monitored_nodes": 0,
            "trust_reports": {},
        }
        
        for node_id in node_ids:
            if node_id in self.trust_monitors:
                try:
                    trust_monitor = self.trust_monitors[node_id]
                    report = ray.get(trust_monitor.get_trust_report.remote())
                    summary["trust_reports"][node_id] = report
                    summary["monitored_nodes"] += 1
                except Exception as e:
                    self.logger.error(f"Failed to get trust report for {node_id}: {e}")
        
        return summary