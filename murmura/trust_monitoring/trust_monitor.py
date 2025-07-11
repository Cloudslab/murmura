"""
Trust monitoring implementation for detecting malicious behavior in decentralized federated learning.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
from scipy.spatial.distance import cosine  # type: ignore[import-untyped]

from .trust_config import TrustMonitorConfig
from .trust_events import TrustEvent, TrustAnomalyEvent, TrustScoreEvent


@dataclass
class ParameterUpdate:
    """Represents a parameter update from a neighbor."""

    neighbor_id: str
    round_num: int
    parameters: Dict[str, Any]
    parameter_stats: Dict[str, float]  # Precomputed statistics
    reported_loss: Optional[float] = None


class TrustMonitor:
    """
    Trust monitor for detecting malicious behavior using neighbor-relative comparisons.

    This monitor runs as a sidecar on honest nodes to detect progressive attacks
    by analyzing parameter update patterns relative to neighbor behavior.
    """

    def __init__(self, node_id: str, config: TrustMonitorConfig):
        self.node_id = node_id
        self.config = config

        # Trust scores for each neighbor
        self.trust_scores: Dict[str, float] = {}

        # Historical parameter updates for analysis
        self.update_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=config.history_window_size)
        )

        # Statistical baselines for each neighbor
        self.neighbor_baselines: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Detection state
        self.anomaly_counts: Dict[str, int] = defaultdict(int)
        self.consensus_history: deque = deque(maxlen=config.history_window_size)

        # Event callbacks
        self.event_callbacks: List[Callable] = []

        # Logging
        self.logger = logging.getLogger(f"murmura.trust_monitor.{node_id}")

    def add_event_callback(self, callback: Callable) -> None:
        """Add callback for trust events."""
        self.event_callbacks.append(callback)

    def _emit_event(self, event: TrustEvent) -> None:
        """Emit trust event to all registered callbacks."""
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in trust event callback: {e}")

    def _compute_parameter_stats(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Compute statistical fingerprint of parameter update."""
        param_stats = {}

        # Flatten all parameters for global statistics
        all_params = []
        layer_stats = {}

        self.logger.info(
            f"TRUST MONITOR {self.node_id}: Computing parameter stats for {len(parameters)} layers"
        )

        for layer_name, param_tensor in parameters.items():
            if hasattr(param_tensor, "cpu"):
                param_array = param_tensor.cpu().numpy().flatten()
            else:
                param_array = np.array(param_tensor).flatten()

            # Check for NaN or inf values and replace with zeros
            if np.any(np.isnan(param_array)) or np.any(np.isinf(param_array)):
                self.logger.warning(
                    f"Found NaN/inf values in layer {layer_name}, replacing with zeros"
                )
                param_array = np.nan_to_num(
                    param_array, nan=0.0, posinf=0.0, neginf=0.0
                )

            all_params.extend(param_array)

            # Per-layer statistics with NaN protection
            layer_mean = float(np.mean(param_array))
            layer_std = float(np.std(param_array))
            layer_l2_norm = float(np.linalg.norm(param_array))

            # Ensure no NaN values in statistics
            if np.isnan(layer_mean) or np.isnan(layer_std) or np.isnan(layer_l2_norm):
                self.logger.warning(
                    f"Layer {layer_name} produced NaN statistics, using defaults"
                )
                layer_mean = 0.0
                layer_std = 1.0
                layer_l2_norm = 1.0

            layer_stats[f"{layer_name}_mean"] = layer_mean
            layer_stats[f"{layer_name}_std"] = layer_std
            layer_stats[f"{layer_name}_l2_norm"] = layer_l2_norm

            self.logger.info(
                f"TRUST MONITOR {self.node_id}: Layer {layer_name}: mean={layer_mean:.6f}, std={layer_std:.6f}, l2_norm={layer_l2_norm:.6f}, size={len(param_array)}"
            )

        # Global statistics with NaN protection
        all_params_array = np.array(all_params)
        all_params_array = np.nan_to_num(
            all_params_array, nan=0.0, posinf=0.0, neginf=0.0
        )

        global_mean = float(np.mean(all_params_array))
        global_std = float(np.std(all_params_array))
        global_l2_norm = float(np.linalg.norm(all_params_array))
        global_magnitude = float(np.sum(np.abs(all_params_array)))

        # Final NaN check
        if (
            np.isnan(global_mean)
            or np.isnan(global_std)
            or np.isnan(global_l2_norm)
            or np.isnan(global_magnitude)
        ):
            self.logger.warning("Global statistics produced NaN values, using defaults")
            global_mean = 0.0
            global_std = 1.0
            global_l2_norm = 1.0
            global_magnitude = 1.0

        param_stats.update(
            {
                "global_mean": global_mean,
                "global_std": global_std,
                "global_l2_norm": global_l2_norm,
                "global_magnitude": global_magnitude,
                "param_count": len(all_params_array),
            }
        )

        # Add layer-specific stats
        param_stats.update(layer_stats)

        self.logger.info(
            f"TRUST MONITOR {self.node_id}: Global stats: mean={global_mean:.6f}, std={global_std:.6f}, l2_norm={global_l2_norm:.6f}, magnitude={global_magnitude:.6f}, total_params={len(all_params_array)}"
        )

        return param_stats

    def _compute_update_similarity(
        self, update1: ParameterUpdate, update2: ParameterUpdate
    ) -> Dict[str, float]:
        """Compute similarity metrics between two parameter updates."""
        similarities = {}

        # Compare parameter statistics
        stats1 = update1.parameter_stats
        stats2 = update2.parameter_stats

        # Statistical similarity with improved sensitivity and NaN protection
        l2_norm_ratio = min(
            stats1["global_l2_norm"] / max(stats2["global_l2_norm"], 1e-8),
            stats2["global_l2_norm"] / max(stats1["global_l2_norm"], 1e-8),
        )

        magnitude_ratio = min(
            stats1["global_magnitude"] / max(stats2["global_magnitude"], 1e-8),
            stats2["global_magnitude"] / max(stats1["global_magnitude"], 1e-8),
        )

        # Add standard deviation ratio for additional sensitivity
        std_ratio = min(
            stats1["global_std"] / max(stats2["global_std"], 1e-8),
            stats2["global_std"] / max(stats1["global_std"], 1e-8),
        )

        # Ensure no NaN values in similarity ratios
        if np.isnan(l2_norm_ratio):
            l2_norm_ratio = 1.0
        if np.isnan(magnitude_ratio):
            magnitude_ratio = 1.0
        if np.isnan(std_ratio):
            std_ratio = 1.0

        similarities["l2_norm_ratio"] = l2_norm_ratio
        similarities["magnitude_ratio"] = magnitude_ratio
        similarities["std_ratio"] = std_ratio

        self.logger.info(
            f"TRUST MONITOR {self.node_id}: Similarity between {update1.neighbor_id} and {update2.neighbor_id}:"
        )
        self.logger.info(
            f"TRUST MONITOR {self.node_id}:   L2 norms: {stats1['global_l2_norm']:.6f} vs {stats2['global_l2_norm']:.6f}, ratio: {l2_norm_ratio:.6f}"
        )
        self.logger.info(
            f"TRUST MONITOR {self.node_id}:   Magnitudes: {stats1['global_magnitude']:.6f} vs {stats2['global_magnitude']:.6f}, ratio: {magnitude_ratio:.6f}"
        )
        self.logger.info(
            f"TRUST MONITOR {self.node_id}:   Std devs: {stats1['global_std']:.6f} vs {stats2['global_std']:.6f}, ratio: {std_ratio:.6f}"
        )

        # Compute cosine similarity if possible
        try:
            # Flatten parameters for cosine similarity
            params1_list: List[float] = []
            params2_list: List[float] = []

            for layer_name in update1.parameters.keys():
                if layer_name in update2.parameters:
                    p1 = update1.parameters[layer_name]
                    p2 = update2.parameters[layer_name]

                    if hasattr(p1, "cpu"):
                        p1 = p1.cpu().numpy().flatten()
                        p2 = p2.cpu().numpy().flatten()
                    else:
                        p1 = np.array(p1).flatten()
                        p2 = np.array(p2).flatten()

                    # Clean NaN/inf values
                    p1 = np.nan_to_num(p1, nan=0.0, posinf=0.0, neginf=0.0)
                    p2 = np.nan_to_num(p2, nan=0.0, posinf=0.0, neginf=0.0)

                    params1_list.extend(p1.tolist())
                    params2_list.extend(p2.tolist())

            if params1_list and params2_list:
                params1 = np.array(params1_list)
                params2 = np.array(params2_list)

                # Compute cosine similarity with NaN protection
                cosine_sim = 1 - cosine(params1, params2)

                # Check for NaN result
                if np.isnan(cosine_sim):
                    self.logger.warning(
                        "Cosine similarity produced NaN, using default value"
                    )
                    cosine_sim = 0.0

                similarities["cosine_similarity"] = float(cosine_sim)
                self.logger.info(
                    f"TRUST MONITOR {self.node_id}:   Cosine similarity: {cosine_sim:.6f}"
                )
            else:
                similarities["cosine_similarity"] = 0.0
                self.logger.info(
                    f"TRUST MONITOR {self.node_id}:   Cosine similarity: 0.0 (no common parameters)"
                )

        except Exception as e:
            self.logger.warning(f"Could not compute cosine similarity: {e}")
            similarities["cosine_similarity"] = 0.0

        return similarities

    def _detect_anomalies_multi_metric(
        self, neighbor_id: str, current_update: ParameterUpdate
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect anomalies using multi-metric fusion for enhanced sensitivity and robustness."""
        history_length = len(self.update_history[neighbor_id])
        self.logger.info(
            f"TRUST MONITOR {self.node_id}: Multi-metric detection for {neighbor_id}: history length = {history_length}"
        )

        # Need at least 1 historical update to compare against
        if history_length < 1:
            self.logger.debug(
                f"Insufficient history for {neighbor_id} (need 1, have {history_length})"
            )
            return False, 0.0, {}

        # Run independent detection streams
        detectors = {
            'cosine': self._detect_cosine_anomaly,
            'magnitude': self._detect_magnitude_anomaly,
            'distribution': self._detect_distribution_anomaly,
            'dynamics': self._detect_dynamics_anomaly
        }
        
        anomaly_signals = {}
        for detector_name, detector_func in detectors.items():
            try:
                is_anomaly, confidence, evidence = detector_func(neighbor_id, current_update)
                anomaly_signals[detector_name] = {
                    'is_anomaly': is_anomaly,
                    'confidence': confidence,
                    'evidence': evidence
                }
                self.logger.info(
                    f"TRUST MONITOR {self.node_id}: {detector_name} detector: anomaly={is_anomaly}, confidence={confidence:.3f}"
                )
            except Exception as e:
                self.logger.warning(f"Detector {detector_name} failed: {e}")
                anomaly_signals[detector_name] = {
                    'is_anomaly': False,
                    'confidence': 0.0,
                    'evidence': {'error': str(e)}
                }
        
        # Fusion logic: combine signals from multiple detectors
        is_malicious, overall_confidence, fusion_evidence = self._fuse_detection_signals(anomaly_signals)
        
        # Combine all evidence
        combined_evidence = {
            'detection_method': 'multi_metric_fusion',
            'individual_detectors': anomaly_signals,
            'fusion_results': fusion_evidence,
            'neighbor_id': neighbor_id,
            'round': current_update.round_num
        }
        
        self.logger.info(
            f"TRUST MONITOR {self.node_id}: Multi-metric fusion for {neighbor_id}: "
            f"malicious={is_malicious}, confidence={overall_confidence:.3f}"
        )
        
        return bool(is_malicious), float(overall_confidence), combined_evidence

    def _detect_cosine_anomaly(
        self, neighbor_id: str, current_update: ParameterUpdate
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect anomalies in cosine similarity patterns."""
        history = list(self.update_history[neighbor_id])
        num_comparisons = min(3, len(history))
        
        cosine_similarities = []
        for past_update in history[-num_comparisons:]:
            sim_metrics = self._compute_update_similarity(current_update, past_update)
            cosine_similarities.append(sim_metrics["cosine_similarity"])
        
        if not cosine_similarities:
            return False, 0.0, {"error": "No similarity data"}
        
        current_cosine = cosine_similarities[-1]
        if len(cosine_similarities) > 1:
            historical_cosines = cosine_similarities[:-1]
            mean_cosine = float(np.mean(historical_cosines))
            std_cosine = float(np.std(historical_cosines)) + 1e-8
        else:
            mean_cosine = float(current_cosine)
            std_cosine = 0.1
        
        # Detect significant drops in cosine similarity
        cosine_drop = mean_cosine - current_cosine
        z_score = cosine_drop / std_cosine
        
        # Configurable thresholds for cosine detection
        cosine_threshold = self.config.cosine_sensitivity
        is_anomaly = (current_cosine < cosine_threshold) or (cosine_drop > 0.25) or (z_score > 1.0)
        confidence = float(min(1.0, max(
            (cosine_threshold - current_cosine) / cosine_threshold if current_cosine < cosine_threshold else 0.0,
            cosine_drop / 0.5 if cosine_drop > 0.0 else 0.0,
            z_score / 2.0 if z_score > 0.0 else 0.0
        )))
        
        evidence = {
            "current_cosine": float(current_cosine),
            "mean_cosine": float(mean_cosine),
            "cosine_drop": float(cosine_drop),
            "z_score": float(z_score),
            "threshold_violation": current_cosine < 0.75
        }
        
        return bool(is_anomaly), confidence, evidence

    def _detect_magnitude_anomaly(
        self, neighbor_id: str, current_update: ParameterUpdate
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect anomalies in parameter magnitude patterns."""
        history = list(self.update_history[neighbor_id])
        num_comparisons = min(3, len(history))
        
        l2_norms = []
        magnitudes = []
        for past_update in history[-num_comparisons:]:
            sim_metrics = self._compute_update_similarity(current_update, past_update)
            l2_norms.append(sim_metrics["l2_norm_ratio"])
            magnitudes.append(sim_metrics["magnitude_ratio"])
        
        # Add current values
        current_l2 = l2_norms[-1] if l2_norms else 1.0
        current_mag = magnitudes[-1] if magnitudes else 1.0
        
        # Detect unusual magnitude patterns
        if len(l2_norms) > 1:
            l2_deviation = float(np.std(l2_norms))
            mag_deviation = float(np.std(magnitudes))
        else:
            l2_deviation = 0.1
            mag_deviation = 0.1
        
        # Configurable thresholds for magnitude detection
        deviation_threshold = self.config.magnitude_deviation_threshold
        l2_anomaly = current_l2 < 0.5 or current_l2 > 2.0
        mag_anomaly = current_mag < 0.5 or current_mag > 2.0
        variability_anomaly = l2_deviation > deviation_threshold or mag_deviation > deviation_threshold
        
        is_anomaly = l2_anomaly or mag_anomaly or variability_anomaly
        confidence = float(min(1.0, max(
            abs(1.0 - current_l2) if l2_anomaly else 0.0,
            abs(1.0 - current_mag) if mag_anomaly else 0.0,
            (l2_deviation + mag_deviation) / 2.0 if variability_anomaly else 0.0
        )))
        
        evidence = {
            "current_l2_ratio": float(current_l2),
            "current_magnitude_ratio": float(current_mag),
            "l2_deviation": float(l2_deviation),
            "magnitude_deviation": float(mag_deviation),
            "l2_anomaly": l2_anomaly,
            "magnitude_anomaly": mag_anomaly,
            "variability_anomaly": variability_anomaly
        }
        
        return bool(is_anomaly), confidence, evidence

    def _detect_distribution_anomaly(
        self, neighbor_id: str, current_update: ParameterUpdate
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect anomalies in parameter distribution patterns."""
        current_stats = current_update.parameter_stats
        history = list(self.update_history[neighbor_id])
        
        if len(history) < 1:
            return False, 0.0, {"error": "Insufficient history"}
        
        # Collect historical statistics
        historical_means = []
        historical_stds = []
        for past_update in history[-3:]:  # Last 3 updates
            historical_means.append(past_update.parameter_stats.get("global_mean", 0.0))
            historical_stds.append(past_update.parameter_stats.get("global_std", 1.0))
        
        current_mean = current_stats.get("global_mean", 0.0)
        current_std = current_stats.get("global_std", 1.0)
        
        # Detect distribution shifts
        if len(historical_means) > 1:
            mean_baseline = np.mean(historical_means)
            std_baseline = np.mean(historical_stds)
            
            mean_shift = abs(current_mean - mean_baseline) / (abs(mean_baseline) + 1e-8)
            std_shift = abs(current_std - std_baseline) / (std_baseline + 1e-8)
        else:
            mean_shift = 0.0
            std_shift = 0.0
        
        # Configurable thresholds for distribution changes
        shift_threshold = self.config.distribution_shift_threshold
        mean_threshold = shift_threshold
        std_threshold = shift_threshold
        
        mean_anomaly = mean_shift > mean_threshold
        std_anomaly = std_shift > std_threshold
        is_anomaly = mean_anomaly or std_anomaly
        
        confidence = min(1.0, max(
            mean_shift / mean_threshold if mean_anomaly else 0.0,
            std_shift / std_threshold if std_anomaly else 0.0
        ))
        
        evidence = {
            "current_mean": float(current_mean),
            "current_std": float(current_std),
            "mean_shift": float(mean_shift),
            "std_shift": float(std_shift),
            "mean_anomaly": mean_anomaly,
            "std_anomaly": std_anomaly
        }
        
        return is_anomaly, confidence, evidence

    def _detect_dynamics_anomaly(
        self, neighbor_id: str, current_update: ParameterUpdate
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect anomalies in training dynamics (loss progression)."""
        current_loss = current_update.reported_loss
        if current_loss is None:
            return False, 0.0, {"error": "No loss data available"}
        
        history = list(self.update_history[neighbor_id])
        historical_losses = []
        for past_update in history[-5:]:  # Look at last 5 rounds
            if past_update.reported_loss is not None:
                historical_losses.append(past_update.reported_loss)
        
        if len(historical_losses) < 2:
            return False, 0.0, {"error": "Insufficient loss history"}
        
        # Analyze loss trends
        loss_trend = np.polyfit(range(len(historical_losses)), historical_losses, 1)[0]
        expected_loss = historical_losses[-1] + loss_trend
        loss_surprise = abs(current_loss - expected_loss) / (abs(expected_loss) + 1e-8)
        
        # Detect unusual loss patterns
        # 1. Loss increases when it should decrease (potential poisoning)
        # 2. Loss decreases too rapidly (potential overfitting attack)
        # 3. Loss becomes inconsistent with trend
        
        loss_increase_anomaly = (current_loss > expected_loss) and (loss_trend < 0)
        rapid_decrease_anomaly = (current_loss < expected_loss * 0.5) and (loss_trend < 0)
        trend_anomaly = loss_surprise > self.config.dynamics_surprise_threshold
        
        is_anomaly = loss_increase_anomaly or rapid_decrease_anomaly or trend_anomaly
        confidence = min(1.0, loss_surprise) if is_anomaly else 0.0
        
        evidence = {
            "current_loss": float(current_loss),
            "expected_loss": float(expected_loss),
            "loss_trend": float(loss_trend),
            "loss_surprise": float(loss_surprise),
            "loss_increase_anomaly": loss_increase_anomaly,
            "rapid_decrease_anomaly": rapid_decrease_anomaly,
            "trend_anomaly": trend_anomaly
        }
        
        return is_anomaly, confidence, evidence

    def _fuse_detection_signals(
        self, anomaly_signals: Dict[str, Dict[str, Any]]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Fuse multiple detection signals using adaptive logic."""
        # Count active detectors and their confidence levels
        active_detectors = []
        confidence_scores = []
        
        for detector_name, signal in anomaly_signals.items():
            if signal['is_anomaly']:
                active_detectors.append(detector_name)
                confidence_scores.append(signal['confidence'])
        
        num_active = len(active_detectors)
        max_confidence = max(confidence_scores) if confidence_scores else 0.0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Configurable fusion decision logic
        min_detectors = self.config.min_detectors_for_early_warning
        high_conf_threshold = self.config.high_confidence_threshold
        is_malicious = (num_active >= min_detectors) or (num_active >= 1 and max_confidence > high_conf_threshold)
        
        # Overall confidence calculation
        if num_active == 0:
            overall_confidence = 0.0
        elif num_active == 1:
            overall_confidence = max_confidence * 0.7  # Single detector penalty
        elif num_active == 2:
            overall_confidence = avg_confidence * 0.9  # Two detector boost
        else:  # 3+ detectors
            overall_confidence = min(1.0, avg_confidence * 1.1)  # Multiple detector boost
        
        # Attack type inference based on active detectors
        attack_indicators = []
        if 'cosine' in active_detectors:
            attack_indicators.append('gradient_manipulation')
        if 'magnitude' in active_detectors:
            attack_indicators.append('parameter_scaling')
        if 'distribution' in active_detectors:
            attack_indicators.append('distribution_shift')
        if 'dynamics' in active_detectors:
            attack_indicators.append('training_dynamics')
        
        fusion_evidence = {
            "active_detectors": active_detectors,
            "num_active_detectors": num_active,
            "max_confidence": float(max_confidence),
            "avg_confidence": float(avg_confidence),
            "overall_confidence": float(overall_confidence),
            "attack_indicators": attack_indicators,
            "fusion_decision": is_malicious
        }
        
        return is_malicious, overall_confidence, fusion_evidence

    def _detect_anomalies_legacy_cusum(
        self, neighbor_id: str, current_update: ParameterUpdate
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Legacy single-metric CUSUM detection for backward compatibility."""
        # This is a simplified version of the original CUSUM method
        # focusing on cosine similarity only
        is_anomaly, confidence, evidence = self._detect_cosine_anomaly(neighbor_id, current_update)
        
        # Wrap in legacy format
        legacy_evidence = {
            "detection_method": "legacy_cusum_cosine",
            "cosine_evidence": evidence
        }
        
        return is_anomaly, confidence, legacy_evidence

    def _detect_consensus_violations(
        self, round_num: int, all_updates: Dict[str, ParameterUpdate]
    ) -> Dict[str, Tuple[bool, float, Dict[str, Any]]]:
        """Detect neighbors that violate consensus with majority."""
        violations: Dict[str, Tuple[bool, float, Dict[str, Any]]] = {}

        self.logger.info(
            f"TRUST MONITOR {self.node_id}: Consensus detection with {len(all_updates)} updates (min required: {self.config.min_neighbors_for_consensus})"
        )

        if len(all_updates) < self.config.min_neighbors_for_consensus:
            self.logger.debug("Not enough neighbors for consensus detection")
            return violations

        neighbor_ids = list(all_updates.keys())

        # Compute pairwise similarities
        similarity_matrix = {}
        self.logger.info(
            f"TRUST MONITOR {self.node_id}: Computing pairwise similarities for {len(neighbor_ids)} neighbors"
        )
        for i, neighbor1 in enumerate(neighbor_ids):
            for j, neighbor2 in enumerate(neighbor_ids):
                if i != j:
                    sim_metrics = self._compute_update_similarity(
                        all_updates[neighbor1], all_updates[neighbor2]
                    )
                    similarity_matrix[(neighbor1, neighbor2)] = sim_metrics[
                        "cosine_similarity"
                    ]

        # Find consensus by computing average similarity for each neighbor
        consensus_scores = {}
        for neighbor in neighbor_ids:
            similarities_to_others = [
                similarity_matrix.get((neighbor, other), 0.0)
                for other in neighbor_ids
                if other != neighbor
            ]
            consensus_scores[neighbor] = (
                np.mean(similarities_to_others) if similarities_to_others else 0.0
            )
            self.logger.info(
                f"TRUST MONITOR {self.node_id}: Consensus score for {neighbor}: {consensus_scores[neighbor]:.6f} (from {len(similarities_to_others)} comparisons)"
            )

        # Detect outliers using relative thresholds
        if len(consensus_scores) >= 2:
            scores = list(consensus_scores.values())
            scores_array = np.array(scores)
            q1, q3 = np.percentile(scores_array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.0 * iqr  # More sensitive threshold
            upper_bound = q3 + 1.0 * iqr  # More sensitive threshold

            self.logger.info(
                f"TRUST MONITOR {self.node_id}: Consensus outlier detection: Q1={q1:.6f}, Q3={q3:.6f}, IQR={iqr:.6f}, lower_bound={lower_bound:.6f}, upper_bound={upper_bound:.6f}"
            )

            for neighbor, score in consensus_scores.items():
                # Check for both low and high outliers
                is_low_outlier = (
                    score < lower_bound and iqr > 0.01
                )  # More sensitive threshold
                is_high_outlier = (
                    score > upper_bound and iqr > 0.01
                )  # Also check for suspicious consistency

                is_violation = is_low_outlier or is_high_outlier

                if is_low_outlier:
                    violation_score = float((lower_bound - score) / max(iqr, 0.05))
                elif is_high_outlier:
                    violation_score = float((score - upper_bound) / max(iqr, 0.05))
                else:
                    violation_score = 0.0

                self.logger.info(
                    f"TRUST MONITOR {self.node_id}: Neighbor {neighbor}: consensus_score={score:.6f}, is_low_outlier={is_low_outlier}, is_high_outlier={is_high_outlier}, violation_score={violation_score:.6f}"
                )

                evidence = {
                    "consensus_score": float(score),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                    "is_low_outlier": is_low_outlier,
                    "is_high_outlier": is_high_outlier,
                    "all_scores": {n: float(s) for n, s in consensus_scores.items()},
                    "detection_method": "consensus_iqr_bidirectional",
                }

                violations[neighbor] = (is_violation, violation_score, evidence)

        return violations

    def update_trust_scores(
        self, anomalies: Dict[str, Tuple[bool, float, Dict[str, Any]]], round_num: int
    ) -> None:
        """Update trust scores based on detected anomalies."""
        score_changes = {}

        for neighbor_id in self.trust_scores.keys():
            if neighbor_id in anomalies:
                is_anomaly, anomaly_score, evidence = anomalies[neighbor_id]

                if is_anomaly:
                    # Decay trust score
                    old_score = self.trust_scores[neighbor_id]

                    if self.config.enable_exponential_decay:
                        # Exponential decay based on accumulated violations
                        violation_count = self.anomaly_counts.get(neighbor_id, 0)
                        decay = self.config.exponential_decay_base ** (
                            violation_count + 1
                        )
                        self.trust_scores[neighbor_id] *= decay
                    else:
                        # Original linear decay
                        decay = self.config.trust_decay_factor ** (
                            1 + anomaly_score * 0.1
                        )
                        self.trust_scores[neighbor_id] *= decay

                    score_changes[neighbor_id] = (
                        self.trust_scores[neighbor_id] - old_score
                    )

                    # Increment anomaly count
                    self.anomaly_counts[neighbor_id] += 1

                    # Emit anomaly event
                    if self.config.log_trust_events:
                        self.logger.warning(
                            f"Anomaly detected for neighbor {neighbor_id}: "
                            f"score={anomaly_score:.3f}, trust={self.trust_scores[neighbor_id]:.3f}"
                        )

                    # Determine anomaly type based on evidence
                    anomaly_type = evidence.get("detection_method", "unknown")

                    anomaly_event = TrustAnomalyEvent(
                        node_id=self.node_id,
                        round_num=round_num,
                        trust_scores=self.trust_scores.copy(),
                        suspected_neighbor=neighbor_id,
                        anomaly_type=anomaly_type,
                        anomaly_score=anomaly_score,
                        evidence=evidence,
                    )
                    self._emit_event(anomaly_event)

                else:
                    # Recover trust score using polynomial recovery
                    old_score = self.trust_scores[neighbor_id]
                    self.trust_scores[neighbor_id] = self._calculate_trust_recovery(
                        neighbor_id
                    )
                    score_changes[neighbor_id] = (
                        self.trust_scores[neighbor_id] - old_score
                    )
            else:
                # No anomaly detected, slight recovery
                old_score = self.trust_scores[neighbor_id]
                self.trust_scores[neighbor_id] = self._calculate_trust_recovery(
                    neighbor_id
                )
                score_changes[neighbor_id] = self.trust_scores[neighbor_id] - old_score

        # Emit trust score event
        if score_changes:
            score_event = TrustScoreEvent(
                node_id=self.node_id,
                round_num=round_num,
                trust_scores=self.trust_scores.copy(),
                score_changes=score_changes,
                detection_method=self.config.anomaly_detection_method,
            )
            self._emit_event(score_event)

    def process_parameter_updates(
        self,
        round_num: int,
        neighbor_updates: Dict[str, Dict[str, Any]],
        neighbor_losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Process parameter updates from neighbors and detect malicious behavior.

        Args:
            round_num: Current training round
            neighbor_updates: Dict mapping neighbor_id to their parameter updates
            neighbor_losses: Optional dict mapping neighbor_id to their reported training loss

        Returns:
            Dict mapping neighbor_id to current trust score
        """
        if not self.config.enable_trust_monitoring:
            return {neighbor_id: 1.0 for neighbor_id in neighbor_updates.keys()}

        self.logger.info(
            f"TRUST MONITOR {self.node_id}: ========== ROUND {round_num} ==========="
        )
        self.logger.info(
            f"TRUST MONITOR {self.node_id}: Processing {len(neighbor_updates)} neighbors: {list(neighbor_updates.keys())}"
        )

        # Log detailed parameter information for each neighbor
        for neighbor_id, parameters in neighbor_updates.items():
            if parameters:
                param_sizes = {
                    name: np.array(param).size for name, param in parameters.items()
                }
                total_params = sum(param_sizes.values())
                self.logger.info(
                    f"TRUST MONITOR {self.node_id}: Neighbor {neighbor_id} - {len(parameters)} layers, {total_params} total parameters"
                )

                # Log a sample of parameter values for debugging
                sample_layer = next(iter(parameters.keys()))
                sample_param = parameters[sample_layer]
                if hasattr(sample_param, "cpu"):
                    sample_values = sample_param.cpu().numpy().flatten()[:5]
                else:
                    sample_values = np.array(sample_param).flatten()[:5]
                self.logger.info(
                    f"TRUST MONITOR {self.node_id}: Neighbor {neighbor_id} sample values from {sample_layer}: {sample_values}"
                )
            else:
                self.logger.warning(
                    f"TRUST MONITOR {self.node_id}: Neighbor {neighbor_id} has EMPTY parameters!"
                )

        # Initialize trust scores for new neighbors
        for neighbor_id in neighbor_updates.keys():
            if neighbor_id not in self.trust_scores:
                self.trust_scores[neighbor_id] = self.config.initial_trust_score

        # Create ParameterUpdate objects
        current_updates = {}
        for neighbor_id, parameters in neighbor_updates.items():
            param_stats = self._compute_parameter_stats(parameters)
            reported_loss = (
                neighbor_losses.get(neighbor_id) if neighbor_losses else None
            )

            update = ParameterUpdate(
                neighbor_id=neighbor_id,
                round_num=round_num,
                parameters=parameters,
                parameter_stats=param_stats,
                reported_loss=reported_loss,
            )
            current_updates[neighbor_id] = update

            # Don't add to history yet - we need to detect anomalies first
            # to avoid comparing with self

        # Detect anomalies using multiple methods
        all_anomalies = {}

        self.logger.info(
            f"TRUST MONITOR {self.node_id}: Running multi-metric detection on {len(current_updates)} neighbors"
        )

        # 1. Multi-metric or legacy detection for each neighbor
        for neighbor_id, update in current_updates.items():
            if self.config.enable_multi_metric_detection:
                is_anomaly, score, evidence = self._detect_anomalies_multi_metric(
                    neighbor_id, update
                )
            else:
                # Fallback to legacy single-metric detection
                is_anomaly, score, evidence = self._detect_anomalies_legacy_cusum(
                    neighbor_id, update
                )
            self.logger.info(
                f"TRUST MONITOR {self.node_id}: Multi-metric {neighbor_id} - anomaly={is_anomaly}, score={score:.3f}"
            )
            if is_anomaly:
                all_anomalies[neighbor_id] = (is_anomaly, score, evidence)

        self.logger.info(f"TRUST MONITOR {self.node_id}: Running consensus detection")

        # 2. Consensus-based detection
        consensus_violations = self._detect_consensus_violations(
            round_num, current_updates
        )
        for neighbor_id, (
            is_violation,
            score,
            evidence,
        ) in consensus_violations.items():
            self.logger.info(
                f"TRUST MONITOR {self.node_id}: Consensus {neighbor_id} - violation={is_violation}, score={score:.3f}"
            )
            if is_violation:
                # Combine with existing anomaly or create new one
                if neighbor_id in all_anomalies:
                    existing_anomaly, existing_score, existing_evidence = all_anomalies[
                        neighbor_id
                    ]
                    combined_score = max(existing_score, score)
                    combined_evidence = {**existing_evidence, **evidence}
                    all_anomalies[neighbor_id] = (
                        True,
                        combined_score,
                        combined_evidence,
                    )
                else:
                    all_anomalies[neighbor_id] = (is_violation, score, evidence)

        self.logger.info(
            f"TRUST MONITOR {self.node_id}: Total anomalies detected: {len(all_anomalies)}"
        )

        # Update trust scores
        self.update_trust_scores(dict(all_anomalies), round_num)

        # Now add current updates to history for next round
        for neighbor_id, update in current_updates.items():
            self.update_history[neighbor_id].append(update)

        # Log all trust scores regardless of threshold
        self.logger.info(f"TRUST MONITOR {self.node_id}: Updated trust scores:")
        for neighbor_id, score in self.trust_scores.items():
            self.logger.info(f"  {neighbor_id}: {score:.6f}")

        # Use relative thresholding as primary detection method
        suspicious_neighbors = []
        relative_threshold = (
            self.config.suspicion_threshold
        )  # Fallback to absolute threshold

        if len(self.trust_scores) > 1:
            scores = list(self.trust_scores.values())
            relative_threshold = self._calculate_continuous_relative_threshold(scores)

            suspicious_neighbors = [
                neighbor_id
                for neighbor_id, score in self.trust_scores.items()
                if score < relative_threshold
            ]

            if suspicious_neighbors:
                self.logger.warning(
                    f"TRUST MONITOR {self.node_id}: Round {round_num}: Suspicious neighbors (continuous threshold {relative_threshold:.3f}): "
                    f"{[(n, f'{self.trust_scores[n]:.3f}') for n in suspicious_neighbors]}"
                )
            else:
                self.logger.info(
                    f"TRUST MONITOR {self.node_id}: No suspicious neighbors using continuous threshold"
                )
        else:
            # Single neighbor case - use absolute threshold
            suspicious_neighbors = [
                neighbor_id
                for neighbor_id, score in self.trust_scores.items()
                if score < self.config.suspicion_threshold
            ]

            if suspicious_neighbors:
                self.logger.warning(
                    f"TRUST MONITOR {self.node_id}: Round {round_num}: Suspicious neighbors (single neighbor, absolute threshold): "
                    f"{[(n, f'{self.trust_scores[n]:.3f}') for n in suspicious_neighbors]}"
                )
            else:
                self.logger.info(
                    f"TRUST MONITOR {self.node_id}: No suspicious neighbors detected (single neighbor case)"
                )

        return self.trust_scores.copy()

    def _calculate_continuous_relative_threshold(self, scores: List[float]) -> float:
        """Calculate relative threshold using continuous linkage based on IQR."""
        if len(scores) < 2:
            return self.config.suspicion_threshold

        median_score = np.median(scores)
        q1, q3 = np.percentile(scores, [25, 75])
        iqr = q3 - q1

        # If no meaningful variation, fall back to absolute threshold
        if iqr <= 0.01:
            return self.config.suspicion_threshold

        # Continuous scaling factor based on IQR
        # Higher IQR -> more aggressive detection (closer to median-based)
        # Lower IQR -> more conservative detection (closer to q1-based)
        # Scale factor ranges from 0 to 1 over IQR range 0.01 to 0.2
        iqr_factor = min(max((iqr - 0.01) / (0.2 - 0.01), 0.0), 1.0)

        # Two threshold approaches:
        # 1. Median-based (for bimodal distributions): median - 0.5 * iqr
        # 2. Q1-based (for smaller variations): q1 - 0.8 * iqr
        median_based_threshold = median_score - 0.5 * iqr
        q1_based_threshold = q1 - 0.8 * iqr

        # Smooth interpolation between the two approaches
        relative_threshold = (
            1 - iqr_factor
        ) * q1_based_threshold + iqr_factor * median_based_threshold

        self.logger.info(
            f"TRUST MONITOR {self.node_id}: Continuous threshold calculation:"
        )
        self.logger.info(
            f"  Q1={q1:.6f}, Q3={q3:.6f}, IQR={iqr:.6f}, Median={median_score:.6f}"
        )
        self.logger.info(f"  IQR factor={iqr_factor:.6f} (IQR {iqr:.6f} -> factor)")
        self.logger.info(f"  Median-based threshold={median_based_threshold:.6f}")
        self.logger.info(f"  Q1-based threshold={q1_based_threshold:.6f}")
        self.logger.info(f"  Final continuous threshold={relative_threshold:.6f}")

        return float(relative_threshold)

    def _calculate_trust_recovery(self, neighbor_id: str) -> float:
        """Calculate trust recovery using polynomial or linear method."""
        current_score = self.trust_scores[neighbor_id]

        if self.config.enable_polynomial_recovery:
            # Polynomial recovery: faster when trust is low, slower near 1.0
            distance_to_full_trust = 1.0 - current_score

            # Apply polynomial scaling to the distance
            scaled_distance = (
                distance_to_full_trust**self.config.polynomial_recovery_power
            )

            # Recovery step is proportional to the scaled distance
            recovery_step = scaled_distance * (self.config.trust_recovery_factor - 1.0)
            new_score = min(1.0, current_score + recovery_step)

            self.logger.debug(
                f"Polynomial recovery for {neighbor_id}: {current_score:.6f} -> {new_score:.6f} "
                f"(distance: {distance_to_full_trust:.6f}, scaled: {scaled_distance:.6f}, step: {recovery_step:.6f})"
            )

            return new_score
        else:
            # Linear recovery (original method)
            return min(1.0, current_score * self.config.trust_recovery_factor)

    def get_trust_summary(self) -> Dict[str, Any]:
        """Get comprehensive trust monitoring summary using relative thresholds."""
        # Calculate relative threshold for suspicious neighbors
        suspicious_neighbors = []
        relative_threshold = self.config.suspicion_threshold

        if len(self.trust_scores) > 1:
            scores = list(self.trust_scores.values())
            relative_threshold = self._calculate_continuous_relative_threshold(scores)

            suspicious_neighbors = [
                neighbor_id
                for neighbor_id, score in self.trust_scores.items()
                if score < relative_threshold
            ]
        else:
            # Single neighbor case - use absolute threshold
            suspicious_neighbors = [
                neighbor_id
                for neighbor_id, score in self.trust_scores.items()
                if score < self.config.suspicion_threshold
            ]

        return {
            "node_id": self.node_id,
            "trust_scores": self.trust_scores.copy(),
            "anomaly_counts": dict(self.anomaly_counts),
            "suspicious_neighbors": suspicious_neighbors,
            "relative_threshold": float(relative_threshold),
            "monitoring_enabled": self.config.enable_trust_monitoring,
            "total_neighbors": len(self.trust_scores),
        }
