"""
Trust Monitor for decentralized federated learning.

This module implements a Ray actor that monitors trust in model updates
using streaming HSIC to detect potential malicious behavior.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from collections import defaultdict, deque

import numpy as np
import ray

from murmura.trust.hsic import ModelUpdateHSIC
from murmura.trust.adaptive_trust_agent import (
    DatasetIndependentTrustSystem,
    TrustContext,
)


class TrustAction(Enum):
    """Actions that can be taken based on trust assessment."""
    ACCEPT = "accept"
    WARN = "warn"
    DOWNGRADE = "downgrade"
    EXCLUDE = "exclude"


class TrustLevel(Enum):
    """Trust levels for nodes."""
    TRUSTED = "trusted"
    SUSPICIOUS = "suspicious"
    UNTRUSTED = "untrusted"


@ray.remote
class TrustMonitor:
    """
    Ray actor that monitors trust in model updates from neighboring nodes.
    
    This actor runs alongside each client actor and monitors incoming
    model updates for trust drift using HSIC.
    """
    
    def __init__(
        self,
        node_id: str,
        hsic_config: Optional[Dict[str, Any]] = None,
        trust_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Trust Monitor.
        
        Args:
            node_id: ID of the node this monitor is attached to
            hsic_config: Configuration for HSIC algorithm
            trust_config: Configuration for trust policies
        """
        self.node_id = node_id
        self.logger = logging.getLogger(f"murmura.trust.TrustMonitor.{node_id}")
        
        # HSIC configuration
        hsic_config = hsic_config or {}
        self.hsic_monitors: Dict[str, ModelUpdateHSIC] = {}  # One per neighbor
        self.hsic_config = {
            "window_size": hsic_config.get("window_size", 50),
            "kernel_type": hsic_config.get("kernel_type", "rbf"),
            "gamma": hsic_config.get("gamma", 0.1),
            "threshold": hsic_config.get("threshold", 0.1),
            "alpha": hsic_config.get("alpha", 0.9),
            "reduce_dim": hsic_config.get("reduce_dim", True),
            "target_dim": hsic_config.get("target_dim", 100),
            "calibration_rounds": hsic_config.get("calibration_rounds", 5),
            "baseline_percentile": hsic_config.get("baseline_percentile", 95.0),
        }
        
        # Trust configuration
        trust_config = trust_config or {}
        self.trust_config = {
            "warn_threshold": trust_config.get("warn_threshold", 0.15),
            "downgrade_threshold": trust_config.get("downgrade_threshold", 0.3),
            "exclude_threshold": trust_config.get("exclude_threshold", 0.5),
            "reputation_window": trust_config.get("reputation_window", 100),
            "min_samples_for_action": trust_config.get("min_samples_for_action", 10),
            "weight_reduction_factor": trust_config.get("weight_reduction_factor", 0.5),
        }
        
        # Trust state
        self.trust_scores: Dict[str, float] = defaultdict(lambda: 1.0)  # Default trust = 1.0
        self.trust_levels: Dict[str, TrustLevel] = defaultdict(lambda: TrustLevel.TRUSTED)
        self.reputation_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.trust_config["reputation_window"])
        )
        
        # Statistics
        self.update_count: Dict[str, int] = defaultdict(int)
        self.drift_count: Dict[str, int] = defaultdict(int)
        self.last_update_time: Dict[str, float] = {}
        
        # Current model parameters (for comparison)
        self.current_parameters: Optional[Dict[str, np.ndarray]] = None
        
        # Round tracking for calibration
        self.current_round = 0
        
        # Adaptive trust system
        self.adaptive_trust_system = DatasetIndependentTrustSystem()
        
        self.logger.info(f"Trust Monitor initialized for node {node_id} with adaptive agent")
        
    def set_current_parameters(self, parameters: Dict[str, np.ndarray]) -> None:
        """
        Update the current model parameters of the monitored node.
        
        Args:
            parameters: Current model parameters
        """
        self.current_parameters = {k: v.copy() for k, v in parameters.items()}
    
    def set_round_number(self, round_num: int) -> None:
        """
        Set the current round number for tracking.
        
        Args:
            round_num: Current federated learning round number
        """
        self.current_round = round_num
        
        # Update adaptive system context
        if hasattr(self, 'fl_context'):
            self.fl_context['current_round'] = round_num
    
    def set_fl_context(self, 
                      total_rounds: int, 
                      current_accuracy: float = 0.5, 
                      topology: str = 'ring') -> None:
        """
        Set federated learning context for adaptive decisions.
        
        Args:
            total_rounds: Total number of FL rounds
            current_accuracy: Current global model accuracy
            topology: Network topology type
        """
        self.fl_context = {
            'total_rounds': total_rounds,
            'current_accuracy': current_accuracy,
            'topology': topology,
            'current_round': self.current_round
        }
        
    def assess_update(
        self,
        neighbor_id: str,
        neighbor_parameters: Dict[str, np.ndarray],
    ) -> Tuple[TrustAction, float, Dict[str, Any]]:
        """
        Assess trust in a model update from a neighbor.
        
        Args:
            neighbor_id: ID of the neighbor sending the update
            neighbor_parameters: Model parameters from the neighbor
            
        Returns:
            Tuple of (action to take, trust score, detailed statistics)
        """
        if self.current_parameters is None:
            self.logger.warning("No current parameters set, accepting update by default")
            return TrustAction.ACCEPT, 1.0, {}
        
        # Initialize HSIC monitor for this neighbor if needed
        if neighbor_id not in self.hsic_monitors:
            self.hsic_monitors[neighbor_id] = ModelUpdateHSIC(**self.hsic_config)
            self.logger.debug(f"Created HSIC monitor for neighbor {neighbor_id}")
        
        # Get HSIC monitor
        hsic_monitor = self.hsic_monitors[neighbor_id]
        
        # Update HSIC with parameters
        hsic_value, drift_detected, stats = hsic_monitor.update_with_parameters(
            self.current_parameters,
            neighbor_parameters,
            neighbor_id,
        )
        
        # Update counters
        self.update_count[neighbor_id] += 1
        if drift_detected:
            self.drift_count[neighbor_id] += 1
        
        # Get FL context or use defaults
        fl_context = getattr(self, 'fl_context', {})
        
        # Prepare data for adaptive trust assessment
        update_data = {
            'round': self.current_round,
            'total_rounds': fl_context.get('total_rounds', 10),
            'accuracy': fl_context.get('current_accuracy', 0.5),
            'hsic': hsic_value,
            'update_norm': stats.get('update_norm', 0.01),
            'consistency': stats.get('relative_update_norm', 0.8),
            'neighbor_trusts': [self.trust_scores.get(nid, 1.0) for nid in self.trust_scores if nid != neighbor_id],
            'topology': fl_context.get('topology', 'ring'),
        }
        
        # Use adaptive trust system for decision
        adaptive_result = self.adaptive_trust_system.assess_trust(neighbor_id, update_data)
        
        # Extract results
        is_malicious = adaptive_result['malicious']
        confidence = adaptive_result['confidence']
        trust_score = adaptive_result['trust_score']
        
        # Update trust score and reputation
        self.trust_scores[neighbor_id] = trust_score
        self.reputation_history[neighbor_id].append(trust_score)
        
        # Determine action based on adaptive decision
        if is_malicious:
            if confidence > 0.7:
                action = TrustAction.EXCLUDE
            elif confidence > 0.4:
                action = TrustAction.DOWNGRADE
            else:
                action = TrustAction.WARN
        else:
            action = TrustAction.ACCEPT
        
        # Update trust level
        self.trust_levels[neighbor_id] = self._get_trust_level(trust_score)
        
        # Record update time
        self.last_update_time[neighbor_id] = time.time()
        
        # Compile detailed statistics
        detailed_stats = {
            "hsic_value": hsic_value,
            "trust_score": trust_score,
            "trust_level": self.trust_levels[neighbor_id].value,
            "drift_detected": drift_detected,
            "update_count": self.update_count[neighbor_id],
            "drift_count": self.drift_count[neighbor_id],
            "drift_rate": self.drift_count[neighbor_id] / max(1, self.update_count[neighbor_id]),
            "adaptive_decision": is_malicious,
            "adaptive_confidence": confidence,
            "adaptive_threshold": adaptive_result.get('adaptive_threshold', 0.5),
            "adaptive_reasoning": adaptive_result.get('reasoning', 'No reasoning provided'),
            **stats,  # Include HSIC statistics
        }
        
        # Log significant events
        if action != TrustAction.ACCEPT:
            self.logger.warning(
                f"Trust issue with neighbor {neighbor_id}: "
                f"Action={action.value}, HSIC={hsic_value:.4f}, "
                f"Trust={trust_score:.4f}, Level={self.trust_levels[neighbor_id].value}"
            )
        
        return action, trust_score, detailed_stats
    
    def _determine_action(
        self,
        neighbor_id: str,
        hsic_value: float,
        trust_score: float,
    ) -> TrustAction:
        """
        Determine what action to take based on trust assessment.
        
        Args:
            neighbor_id: ID of the neighbor
            hsic_value: Current HSIC value
            trust_score: Current trust score
            
        Returns:
            Action to take
        """
        # Need minimum samples before taking drastic actions
        if self.update_count[neighbor_id] < self.trust_config["min_samples_for_action"]:
            if hsic_value > self.trust_config["exclude_threshold"]:
                return TrustAction.WARN
            return TrustAction.ACCEPT
        
        # Determine action based on HSIC value and trust score
        if hsic_value > self.trust_config["exclude_threshold"] or trust_score < 0.2:
            return TrustAction.EXCLUDE
        elif hsic_value > self.trust_config["downgrade_threshold"] or trust_score < 0.5:
            return TrustAction.DOWNGRADE
        elif hsic_value > self.trust_config["warn_threshold"] or trust_score < 0.7:
            return TrustAction.WARN
        else:
            return TrustAction.ACCEPT
    
    def _get_trust_level(self, trust_score: float) -> TrustLevel:
        """
        Convert trust score to trust level.
        
        Args:
            trust_score: Numerical trust score
            
        Returns:
            Trust level
        """
        if trust_score >= 0.7:
            return TrustLevel.TRUSTED
        elif trust_score >= 0.4:
            return TrustLevel.SUSPICIOUS
        else:
            return TrustLevel.UNTRUSTED
    
    def get_trust_weights(self, neighbor_ids: List[str]) -> Dict[str, float]:
        """
        Get trust-adjusted weights for aggregation.
        
        Args:
            neighbor_ids: List of neighbor IDs
            
        Returns:
            Dictionary mapping neighbor IDs to trust weights
        """
        weights = {}
        
        for neighbor_id in neighbor_ids:
            trust_score = self.trust_scores.get(neighbor_id, 1.0)
            action = self._determine_action(
                neighbor_id,
                self.hsic_monitors.get(neighbor_id, ModelUpdateHSIC()).hsic_history[-1]
                if neighbor_id in self.hsic_monitors and len(self.hsic_monitors[neighbor_id].hsic_history) > 0
                else 0.0,
                trust_score,
            )
            
            if action == TrustAction.EXCLUDE:
                weights[neighbor_id] = 0.0
            elif action == TrustAction.DOWNGRADE:
                weights[neighbor_id] = self.trust_config["weight_reduction_factor"]
            else:
                weights[neighbor_id] = 1.0
        
        # Include self with weight 1.0
        weights[self.node_id] = 1.0
        
        return weights
    
    def get_trust_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive trust report for all neighbors.
        
        Returns:
            Trust report dictionary
        """
        report = {
            "node_id": self.node_id,
            "timestamp": time.time(),
            "neighbors": {},
        }
        
        for neighbor_id in self.trust_scores:
            neighbor_report = {
                "trust_score": self.trust_scores[neighbor_id],
                "trust_level": self.trust_levels[neighbor_id].value,
                "update_count": self.update_count[neighbor_id],
                "drift_count": self.drift_count[neighbor_id],
                "drift_rate": self.drift_count[neighbor_id] / max(1, self.update_count[neighbor_id]),
                "last_update": self.last_update_time.get(neighbor_id, 0),
            }
            
            # Add HSIC statistics if available
            if neighbor_id in self.hsic_monitors:
                hsic_stats = self.hsic_monitors[neighbor_id].get_statistics()
                neighbor_report["hsic_stats"] = hsic_stats
            
            report["neighbors"][neighbor_id] = neighbor_report
        
        return report
    
    def reset_neighbor_trust(self, neighbor_id: str) -> None:
        """
        Reset trust information for a specific neighbor.
        
        Args:
            neighbor_id: ID of the neighbor to reset
        """
        if neighbor_id in self.hsic_monitors:
            self.hsic_monitors[neighbor_id].reset()
        
        self.trust_scores.pop(neighbor_id, None)
        self.trust_levels.pop(neighbor_id, None)
        self.reputation_history.pop(neighbor_id, None)
        self.update_count.pop(neighbor_id, None)
        self.drift_count.pop(neighbor_id, None)
        self.last_update_time.pop(neighbor_id, None)
        
        self.logger.info(f"Reset trust information for neighbor {neighbor_id}")
    
    def update_trust_config(self, trust_config: Dict[str, Any]) -> None:
        """
        Update trust configuration dynamically.
        
        Args:
            trust_config: New trust configuration
        """
        self.trust_config.update(trust_config)
        self.logger.info(f"Updated trust configuration: {trust_config}")
    
    def get_excluded_neighbors(self) -> List[str]:
        """
        Get list of neighbors that should be excluded.
        
        Returns:
            List of neighbor IDs to exclude
        """
        excluded = []
        
        for neighbor_id in self.trust_scores:
            if self.update_count[neighbor_id] >= self.trust_config["min_samples_for_action"]:
                trust_score = self.trust_scores[neighbor_id]
                if self.trust_levels[neighbor_id] == TrustLevel.UNTRUSTED or trust_score < 0.2:
                    excluded.append(neighbor_id)
        
        return excluded