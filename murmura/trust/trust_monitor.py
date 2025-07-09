"""
Trust Monitor for decentralized federated learning.

This module implements a Ray actor that monitors trust in model updates
using robust statistical analysis to detect potential malicious behavior
without relying on HSIC or consensus mechanisms.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from collections import defaultdict, deque

import numpy as np
import ray

from murmura.trust.statistical_trust_detector import RobustStatisticalTrustDetector
from murmura.trust.adaptive_statistical_detector import AdaptiveStatisticalDetector


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
    model updates for trust drift using robust statistical analysis.
    
    CRITICAL: Trust decisions are made WITHOUT any knowledge of ground truth.
    The monitor operates in a completely unsupervised manner, using only
    statistical analysis of parameter patterns. Ground truth labels are
    stored only for validation, testing, and performance evaluation purposes.
    """
    
    def __init__(
        self,
        node_id: str,
        statistical_config: Optional[Dict[str, Any]] = None,
        trust_config: Optional[Dict[str, Any]] = None,
        enable_ensemble_detection: bool = False,
        topology: str = "ring",
    ):
        """
        Initialize the Trust Monitor with robust statistical detection.
        
        Args:
            node_id: ID of the node this monitor is attached to
            statistical_config: Configuration for statistical detection algorithm
            trust_config: Configuration for trust policies
            enable_ensemble_detection: Whether to enable ensemble detection
            topology: Network topology (ring, complete, line) for optimization
        """
        self.node_id = node_id
        self.topology = topology
        self.logger = logging.getLogger(f"murmura.trust.TrustMonitor.{node_id}")
        
        # Statistical detection configuration
        statistical_config = statistical_config or {}
        self.statistical_config = {
            "window_size": statistical_config.get("window_size", 20),
            "min_samples_for_detection": statistical_config.get("min_samples_for_detection", 5),
            "outlier_contamination": statistical_config.get("outlier_contamination", 0.1),
            "enable_adaptive_thresholds": statistical_config.get("enable_adaptive_thresholds", True),
        }
        
        # Initialize adaptive statistical detector for intelligent learning
        use_adaptive = self.statistical_config.get('use_adaptive_detector', True)
        
        if use_adaptive:
            self.statistical_detector = AdaptiveStatisticalDetector(
                node_id=node_id,
                window_size=self.statistical_config.get('window_size', 20),
                learning_rate=self.statistical_config.get('learning_rate', 0.1),
                warmup_rounds=self.statistical_config.get('warmup_rounds', 5),
                topology=topology
            )
            self.logger.info(f"Using AdaptiveStatisticalDetector with warmup={self.statistical_config.get('warmup_rounds', 5)}")
        else:
            # Fallback to original detector
            self.statistical_detector = RobustStatisticalTrustDetector(
                node_id=node_id,
                topology=topology,
                **self.statistical_config
            )
        
        # Trust configuration (simplified for statistical approach)
        trust_config = trust_config or {}
        self.trust_config = {
            "warn_threshold": trust_config.get("warn_threshold", 0.3),
            "downgrade_threshold": trust_config.get("downgrade_threshold", 0.6),
            "exclude_threshold": trust_config.get("exclude_threshold", 0.8),
            "reputation_window": trust_config.get("reputation_window", 50),
            "min_samples_for_action": trust_config.get("min_samples_for_action", 5),
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
        self.detection_count: Dict[str, int] = defaultdict(int)
        self.last_update_time: Dict[str, float] = {}
        
        # Current model parameters (for comparison)
        self.current_parameters: Optional[Dict[str, np.ndarray]] = None
        
        # Round tracking
        self.current_round = 0
        
        # Ensemble detection (optional)
        self.enable_ensemble_detection = enable_ensemble_detection
        self.ensemble_detector = None
        
        if enable_ensemble_detection:
            try:
                from murmura.trust.ensemble_trust_detector import EnsembleTrustDetector
                self.ensemble_detector = EnsembleTrustDetector(num_classes=10)
                self.logger.info(f"Ensemble detection enabled for node {node_id}")
            except ImportError:
                self.logger.warning("Ensemble detection requested but not available")
                self.enable_ensemble_detection = False
        
        # Ground truth tracking (for validation and testing)
        self.ground_truth_labels = {}  # neighbor_id -> "honest" or "malicious"
        
        self.logger.info(f"Robust Statistical Trust Monitor initialized for node {node_id} (topology: {topology})")
    
    
        
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
        Assess trust in a neighbor's parameter update using robust statistical analysis.
        
        Args:
            neighbor_id: ID of the neighbor sending the update
            neighbor_parameters: Model parameters from the neighbor
            
        Returns:
            Tuple of (action to take, trust score, detailed statistics)
        """
        self.logger.debug(f"🔍 Assessing update from {neighbor_id} in round {self.current_round}")
        
        if self.current_parameters is None:
            self.logger.error(f"❌ NO CURRENT PARAMETERS SET for {neighbor_id} in round {self.current_round} - accepting by default")
            return TrustAction.ACCEPT, 1.0, {}
        
        # Update counters
        self.update_count[neighbor_id] += 1
        
        # Perform robust statistical analysis
        is_malicious, suspicion_score, detailed_analysis = self.statistical_detector.analyze_parameter_update(
            neighbor_id=neighbor_id,
            current_params=self.current_parameters,
            neighbor_params=neighbor_parameters,
            round_number=self.current_round
        )
        
        # Convert suspicion score to trust score
        trust_score = max(0.0, 1.0 - suspicion_score)
        confidence = detailed_analysis.get('confidence', 0.5)
        
        # DEBUG: Log detailed statistical analysis results
        self.logger.warning(
            f"🔍 STATISTICAL ANALYSIS {neighbor_id} (Round {self.current_round}): "
            f"malicious={is_malicious}, suspicion={suspicion_score:.4f}, trust={trust_score:.4f}, "
            f"confidence={confidence:.4f}"
        )
        
        # Update trust state
        self.trust_scores[neighbor_id] = trust_score
        self.reputation_history[neighbor_id].append(trust_score)
        
        if is_malicious:
            self.detection_count[neighbor_id] += 1
        
        # Determine action based on statistical analysis
        action = self._determine_action_from_analysis(
            neighbor_id, is_malicious, suspicion_score, confidence
        )
        
        # Update trust level
        self.trust_levels[neighbor_id] = self._get_trust_level(trust_score)
        
        # Record update time
        self.last_update_time[neighbor_id] = time.time()
        
        # Compile comprehensive statistics
        detailed_stats = {
            "method": "robust_statistical",
            "trust_score": trust_score,
            "suspicion_score": suspicion_score,
            "confidence": confidence,
            "is_malicious": is_malicious,
            "trust_level": self.trust_levels[neighbor_id].value,
            "action": action.value,
            "update_count": self.update_count[neighbor_id],
            "detection_count": self.detection_count[neighbor_id],
            "detection_rate": self.detection_count[neighbor_id] / max(1, self.update_count[neighbor_id]),
            "round": self.current_round,
            "timestamp": time.time(),
            "node_id": self.node_id,
            "neighbor_id": neighbor_id,
            "topology": self.topology,
            # Include detailed statistical analysis
            **detailed_analysis
        }
        
        # Log detection events
        if action != TrustAction.ACCEPT:
            self.logger.warning(
                f"🚨 Trust issue detected for {neighbor_id}: "
                f"Action={action.value}, Trust={trust_score:.3f}, "
                f"Suspicion={suspicion_score:.3f}, Confidence={confidence:.3f} "
                f"(Round {self.current_round})"
            )
        elif is_malicious:
            self.logger.info(
                f"⚠️  Malicious behavior detected but action={action.value} for {neighbor_id}: "
                f"Trust={trust_score:.3f}, Suspicion={suspicion_score:.3f}"
            )
        
        return action, trust_score, detailed_stats
    
    def _determine_action_from_analysis(
        self,
        neighbor_id: str,
        is_malicious: bool,
        suspicion_score: float,
        confidence: float
    ) -> TrustAction:
        """Determine action based on statistical analysis results."""
        # Need minimum samples before taking drastic actions
        if self.update_count[neighbor_id] < self.trust_config["min_samples_for_action"]:
            if is_malicious and suspicion_score > 1.5:
                return TrustAction.WARN
            return TrustAction.ACCEPT
        
        # Action determination based on suspicion score and confidence
        if is_malicious:
            if suspicion_score >= self.trust_config["exclude_threshold"] and confidence > 0.7:
                return TrustAction.EXCLUDE
            elif suspicion_score >= self.trust_config["downgrade_threshold"] and confidence > 0.5:
                return TrustAction.DOWNGRADE
            elif suspicion_score >= self.trust_config["warn_threshold"]:
                return TrustAction.WARN
        
        return TrustAction.ACCEPT
    
    def assess_update_with_ensemble(
        self,
        neighbor_id: str,
        neighbor_parameters: Dict[str, np.ndarray],
        predictions: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple[TrustAction, float, Dict[str, Any]]:
        """
        Enhanced assessment combining HSIC with ensemble detection.
        
        Args:
            neighbor_id: ID of the neighbor sending the update
            neighbor_parameters: Model parameters from the neighbor
            predictions: Model predictions (optional, for label distribution analysis)
            labels: True labels (optional, for reference)
            
        Returns:
            Tuple of (action to take, trust score, detailed statistics)
        """
        # Perform standard HSIC-based assessment
        base_action, base_trust_score, base_stats = self.assess_update(neighbor_id, neighbor_parameters)
        
        # If ensemble detection is disabled, return base results
        if not self.enable_ensemble_detection or self.ensemble_detector is None:
            return base_action, base_trust_score, base_stats
        
        # Prepare data for ensemble detection
        detection_data = {
            'current_params': self.current_parameters,
            'neighbor_update': neighbor_parameters,
        }
        
        # Add predictions and labels if available
        if predictions is not None and labels is not None:
            detection_data['predictions'] = predictions
            detection_data['labels'] = labels
        
        # Add previous parameters if we have them (would need to track these)
        # For now, use current parameters as previous (placeholder)
        detection_data['previous_params'] = self.current_parameters
        
        try:
            # Perform ensemble detection
            ensemble_result = self.ensemble_detector.detect_trust_drift(neighbor_id, detection_data)
            
            # Combine ensemble result with base HSIC assessment
            combined_action, combined_trust, combined_stats = self._combine_hsic_ensemble_assessments(
                base_action, base_trust_score, base_stats, ensemble_result
            )
            
            return combined_action, combined_trust, combined_stats
            
        except Exception as e:
            self.logger.error(f"Ensemble detection failed for {neighbor_id}: {e}")
            # Fallback to base assessment
            return base_action, base_trust_score, base_stats
    
    def _combine_hsic_ensemble_assessments(
        self,
        base_action: TrustAction,
        base_trust: float,
        base_stats: Dict[str, Any],
        ensemble_result,
    ) -> Tuple[TrustAction, float, Dict[str, Any]]:
        """
        Combine HSIC-based assessment with ensemble detection result.
        
        Args:
            base_action: Action from HSIC assessment
            base_trust: Trust score from HSIC assessment
            base_stats: Statistics from HSIC assessment
            ensemble_result: Result from ensemble detection
            
        Returns:
            Combined assessment tuple
        """
        # Weight combination: 40% HSIC, 60% ensemble
        hsic_weight = 0.4
        ensemble_weight = 0.6
        
        # Convert base trust to suspicion score
        base_suspicion = 1.0 - base_trust
        
        # Combine suspicion scores
        combined_suspicion = (
            hsic_weight * base_suspicion + 
            ensemble_weight * ensemble_result.overall_suspicion
        )
        
        combined_trust = 1.0 - combined_suspicion
        combined_confidence = (
            hsic_weight * base_stats.get('adaptive_confidence', 0.5) +
            ensemble_weight * ensemble_result.confidence
        )
        
        # Determine action based on combined assessment
        if ensemble_result.is_malicious or combined_suspicion > 0.7:
            if combined_confidence > 0.8:
                combined_action = TrustAction.EXCLUDE
            elif combined_confidence > 0.5:
                combined_action = TrustAction.DOWNGRADE
            else:
                combined_action = TrustAction.WARN
        elif combined_suspicion > 0.4:
            combined_action = TrustAction.WARN
        else:
            combined_action = TrustAction.ACCEPT
        
        # Enhanced statistics
        combined_stats = base_stats.copy()
        combined_stats.update({
            'ensemble_enabled': True,
            'ensemble_suspicion': ensemble_result.overall_suspicion,
            'ensemble_confidence': ensemble_result.confidence,
            'ensemble_malicious': ensemble_result.is_malicious,
            'ensemble_reasoning': ensemble_result.reasoning,
            'individual_detections': [
                {
                    'signal': result.signal_type.value,
                    'suspicion': result.suspicion_score,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning
                }
                for result in ensemble_result.individual_results
            ],
            'combined_suspicion': combined_suspicion,
            'combined_confidence': combined_confidence,
            'hsic_weight': hsic_weight,
            'ensemble_weight': ensemble_weight,
            'detection_method': 'ensemble_enhanced'
        })
        
        # Log enhanced detection details
        if combined_suspicion > 0.5 or ensemble_result.is_malicious:
            signal_details = ", ".join([
                f"{r.signal_type.value}={r.suspicion_score:.2f}" 
                for r in ensemble_result.individual_results if r.suspicion_score > 0.1
            ])
            
            self.logger.warning(
                f"🔥 ENSEMBLE DETECTION - Node {self.node_id} assessing neighbor {neighbor_id}: "
                f"suspicion={combined_suspicion:.3f}, confidence={combined_confidence:.3f} | "
                f"HSIC: {base_trust:.3f}, Ensemble: {ensemble_result.overall_suspicion:.3f} | "
                f"Signals: [{signal_details}]"
            )
        
        return combined_action, combined_trust, combined_stats
    
    
    def _get_trust_level(self, trust_score: float) -> TrustLevel:
        """
        Convert trust score to trust level.
        
        Args:
            trust_score: Numerical trust score
            
        Returns:
            Trust level
        """
        # Adaptive thresholds that work with baseline learning
        if trust_score >= 0.7:  # Standard threshold for trusted
            return TrustLevel.TRUSTED
        elif trust_score >= 0.4:  # Standard threshold for suspicious
            return TrustLevel.SUSPICIOUS
        else:
            return TrustLevel.UNTRUSTED
    
    def get_trust_weights(self, neighbor_ids: List[str]) -> Dict[str, float]:
        """
        Get trust-adjusted weights for aggregation based on statistical analysis.
        
        Args:
            neighbor_ids: List of neighbor IDs
            
        Returns:
            Dictionary mapping neighbor IDs to trust weights
        """
        weights = {}
        
        for neighbor_id in neighbor_ids:
            trust_score = self.trust_scores.get(neighbor_id, 1.0)
            trust_level = self.trust_levels.get(neighbor_id, TrustLevel.TRUSTED)
            
            # Determine weight based on trust level and score
            if trust_level == TrustLevel.UNTRUSTED or trust_score < 0.2:
                weights[neighbor_id] = 0.0  # Exclude
            elif trust_level == TrustLevel.SUSPICIOUS or trust_score < 0.5:
                weights[neighbor_id] = self.trust_config["weight_reduction_factor"]  # Downgrade
            else:
                weights[neighbor_id] = trust_score  # Use trust score as weight
        
        # Include self with weight 1.0
        weights[self.node_id] = 1.0
        
        return weights
    
    def get_trust_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive trust report for all neighbors using statistical analysis.
        
        Returns:
            Trust report dictionary
        """
        report = {
            "node_id": self.node_id,
            "topology": self.topology,
            "method": "robust_statistical",
            "timestamp": time.time(),
            "neighbors": {},
            "summary": {
                "total_neighbors": len(self.trust_scores),
                "total_updates": sum(self.update_count.values()),
                "total_detections": sum(self.detection_count.values()),
                "overall_detection_rate": sum(self.detection_count.values()) / max(1, sum(self.update_count.values()))
            }
        }
        
        for neighbor_id in self.trust_scores:
            neighbor_report = {
                "trust_score": self.trust_scores[neighbor_id],
                "trust_level": self.trust_levels[neighbor_id].value,
                "update_count": self.update_count[neighbor_id],
                "detection_count": self.detection_count[neighbor_id],
                "detection_rate": self.detection_count[neighbor_id] / max(1, self.update_count[neighbor_id]),
                "last_update": self.last_update_time.get(neighbor_id, 0),
                "ground_truth_label": self._get_ground_truth_label(neighbor_id),
            }
            
            # Add statistical detector summary for this neighbor
            neighbor_report["statistical_summary"] = self.statistical_detector.get_detection_summary()
            
            report["neighbors"][neighbor_id] = neighbor_report
        
        return report
    
    def set_ground_truth_label(self, neighbor_id: str, label: str) -> None:
        """
        Set ground truth label for a neighbor (for validation and testing).
        
        Args:
            neighbor_id: ID of the neighbor
            label: Ground truth label ("honest", "malicious", "unknown")
        """
        if label not in ["honest", "malicious", "unknown"]:
            raise ValueError(f"Invalid ground truth label: {label}. Must be 'honest', 'malicious', or 'unknown'")
        
        self.ground_truth_labels[neighbor_id] = label
        self.logger.info(f"Set ground truth for {neighbor_id}: {label}")
    
    def _get_ground_truth_label(self, neighbor_id: str) -> str:
        """
        Get ground truth label for a neighbor.
        
        Args:
            neighbor_id: ID of the neighbor
            
        Returns:
            Ground truth label: "honest", "malicious", or "unknown"
        """
        # Return stored ground truth if available
        if neighbor_id in self.ground_truth_labels:
            return self.ground_truth_labels[neighbor_id]
        
        # Try to infer from neighbor ID patterns (heuristic)
        neighbor_lower = neighbor_id.lower()
        if any(keyword in neighbor_lower for keyword in ["malicious", "attack", "byzantine", "poison"]):
            self.ground_truth_labels[neighbor_id] = "malicious"
            return "malicious"
        elif any(keyword in neighbor_lower for keyword in ["honest", "normal", "good"]):
            self.ground_truth_labels[neighbor_id] = "honest"  
            return "honest"
        else:
            # Default assumption: neighbors are honest unless proven otherwise
            self.ground_truth_labels[neighbor_id] = "unknown"
            return "unknown"

    def export_statistical_data(self) -> Dict[str, Any]:
        """
        Export statistical detection data for analysis and validation.
        
        Returns:
            Dictionary with statistical detection data and summaries
        """
        export_data = {
            "metadata": {
                "node_id": self.node_id,
                "topology": self.topology,
                "method": "robust_statistical",
                "total_neighbors": len(self.trust_scores),
                "total_updates": sum(self.update_count.values()),
                "total_detections": sum(self.detection_count.values()),
                "export_timestamp": time.time()
            },
            "neighbor_data": {},
            "ground_truth_summary": {
                "honest_neighbors": [],
                "malicious_neighbors": [],
                "unknown_neighbors": []
            },
            "statistical_summary": self.statistical_detector.get_detection_summary()
        }
        
        # Export per-neighbor data
        for neighbor_id in self.trust_scores:
            ground_truth = self._get_ground_truth_label(neighbor_id)
            
            export_data["neighbor_data"][neighbor_id] = {
                "ground_truth": ground_truth,
                "trust_score": self.trust_scores[neighbor_id],
                "trust_level": self.trust_levels[neighbor_id].value,
                "update_count": self.update_count[neighbor_id],
                "detection_count": self.detection_count[neighbor_id],
                "detection_rate": self.detection_count[neighbor_id] / max(1, self.update_count[neighbor_id])
            }
            
            # Add to ground truth summary
            if ground_truth == "honest":
                export_data["ground_truth_summary"]["honest_neighbors"].append(neighbor_id)
            elif ground_truth == "malicious":
                export_data["ground_truth_summary"]["malicious_neighbors"].append(neighbor_id)
            else:
                export_data["ground_truth_summary"]["unknown_neighbors"].append(neighbor_id)
        
        return export_data
    
    def reset_neighbor_trust(self, neighbor_id: str) -> None:
        """
        Reset trust information for a specific neighbor.
        
        Args:
            neighbor_id: ID of the neighbor to reset
        """
        # Reset statistical detector data for this neighbor
        self.statistical_detector.reset_neighbor_data(neighbor_id)
        
        # Reset local trust state
        self.trust_scores.pop(neighbor_id, None)
        self.trust_levels.pop(neighbor_id, None)
        self.reputation_history.pop(neighbor_id, None)
        self.update_count.pop(neighbor_id, None)
        self.detection_count.pop(neighbor_id, None)
        self.last_update_time.pop(neighbor_id, None)
        self.ground_truth_labels.pop(neighbor_id, None)
        
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
