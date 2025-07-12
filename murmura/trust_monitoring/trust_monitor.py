"""
Trust monitoring implementation for detecting malicious behavior in decentralized federated learning.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
from scipy.spatial.distance import cosine

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

    def _detect_anomalies_cusum(
        self, neighbor_id: str, current_update: ParameterUpdate
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """Detect anomalies using CUSUM algorithm on neighbor-relative metrics with training dynamics awareness."""
        history_length = len(self.update_history[neighbor_id])
        self.logger.info(
            f"TRUST MONITOR {self.node_id}: CUSUM detection for {neighbor_id}: history length = {history_length}"
        )

        # Need at least 1 historical update to compare against
        if history_length < 1:
            self.logger.debug(
                f"Insufficient history for {neighbor_id} (need 1, have {history_length})"
            )
            return False, 0.0, {}

        history = list(self.update_history[neighbor_id])

        # Compute similarities with recent history
        similarities = []
        cosine_similarities = []
        # Compare with up to 3 most recent historical updates (excluding current)
        num_comparisons = min(3, history_length)
        self.logger.info(
            f"TRUST MONITOR {self.node_id}: Computing similarities for {neighbor_id} with last {num_comparisons} historical updates"
        )

        # Use historical updates only (not including current)
        comparison_history = history[-num_comparisons:]
        for i, past_update in enumerate(comparison_history):
            sim_metrics = self._compute_update_similarity(current_update, past_update)

            # Enhanced weighting that considers training dynamics
            # Cosine similarity is most important for gradient manipulation detection
            # L2 and magnitude ratios can vary naturally during training
            combined_sim = (
                sim_metrics["cosine_similarity"]
                * 0.6  # Increased weight for cosine similarity
                + sim_metrics["l2_norm_ratio"] * 0.2  # Reduced weight for L2 norm
                + sim_metrics["magnitude_ratio"] * 0.15  # Reduced weight for magnitude
                + sim_metrics["std_ratio"] * 0.05  # Minimal weight for std ratio
            )
            similarities.append(combined_sim)
            cosine_similarities.append(sim_metrics["cosine_similarity"])

            self.logger.info(
                f"TRUST MONITOR {self.node_id}:   Similarity {i}: cosine={sim_metrics['cosine_similarity']:.6f}, l2_ratio={sim_metrics['l2_norm_ratio']:.6f}, mag_ratio={sim_metrics['magnitude_ratio']:.6f}, combined={combined_sim:.6f}"
            )

        # Enhanced CUSUM detection with training dynamics consideration
        if len(similarities) >= 1:
            # For proper anomaly detection, we need to compare current with historical
            # If we have multiple similarities, compare current (last) with historical (all but last)
            if len(similarities) > 1:
                historical_similarities = similarities[:-1]
                historical_cosines = cosine_similarities[:-1]
                mean_sim = float(np.mean(historical_similarities))
                std_sim = float(np.std(historical_similarities)) + 1e-8
                mean_cosine = float(np.mean(historical_cosines))
            else:
                # Only one comparison, use it as both mean and current
                mean_sim = float(similarities[0])
                std_sim = 0.1  # Default std for single comparison
                mean_cosine = float(cosine_similarities[0])

            # Handle NaN values in similarity analysis
            if np.isnan(mean_sim) or np.isnan(std_sim) or np.isnan(mean_cosine):
                self.logger.warning(
                    f"CUSUM analysis for {neighbor_id}: NaN values detected, skipping"
                )
                return (
                    False,
                    0.0,
                    {"detection_method": "cusum_similarity", "error": "NaN values"},
                )

            # Current values are always the last comparison
            current_sim = similarities[-1]
            current_cosine = cosine_similarities[-1]

            if np.isnan(current_sim) or np.isnan(current_cosine):
                self.logger.warning(
                    f"CUSUM analysis for {neighbor_id}: current similarity is NaN, skipping"
                )
                return (
                    False,
                    0.0,
                    {
                        "detection_method": "cusum_similarity",
                        "error": "NaN current similarity",
                    },
                )

            # Multi-criteria detection that balances sensitivity with false positive reduction
            z_score = (mean_sim - current_sim) / std_sim  # Higher when current is lower
            abs_z_score = abs(z_score)

            # Gradient manipulation specific detection criteria
            # 1. Low cosine similarity (< 0.7) indicates parameter manipulation
            # 2. Consistent deviation from recent behavior (z_score > 1.2)
            # 3. Combined similarity drop below threshold

            cosine_threshold = (
                0.7  # Cosine similarity threshold for gradient manipulation
            )
            z_threshold = 1.2  # Reduced from 1.5 for better sensitivity
            combined_threshold = 0.6  # Combined similarity threshold

            # Detection logic with multiple criteria
            # Detect if cosine similarity dropped significantly
            cosine_drop = mean_cosine - current_cosine
            low_cosine_anomaly = (current_cosine < cosine_threshold) or (
                cosine_drop > 0.3
            )

            # Statistical anomaly based on z-score
            statistical_anomaly = abs_z_score > z_threshold

            # Combined similarity drop
            sim_drop = mean_sim - current_sim
            combined_anomaly = (current_sim < combined_threshold) or (sim_drop > 0.2)

            # Gradient manipulation is detected if any strong indicator is present
            # More sensitive: detect if cosine similarity is very low OR there's a significant drop
            is_anomaly = low_cosine_anomaly or (
                statistical_anomaly and (combined_anomaly or current_cosine < 0.5)
            )

            # Calculate confidence score based on multiple factors
            cosine_confidence = (
                max(0.0, (cosine_threshold - current_cosine) / cosine_threshold)
                if current_cosine < cosine_threshold
                else 0.0
            )
            statistical_confidence = (
                min(float(abs_z_score) / z_threshold, 2.0)
                if abs_z_score > z_threshold
                else 0.0
            )
            combined_confidence = (
                max(0.0, (combined_threshold - current_sim) / combined_threshold)
                if current_sim < combined_threshold
                else 0.0
            )

            anomaly_score = float(
                max(
                    cosine_confidence,
                    statistical_confidence * 0.7,
                    combined_confidence * 0.5,
                )
            )

            self.logger.info(
                f"TRUST MONITOR {self.node_id}: CUSUM analysis for {neighbor_id}:"
            )
            self.logger.info(
                f"  mean_sim={mean_sim:.6f}, std_sim={std_sim:.6f}, current_sim={current_sim:.6f}"
            )
            self.logger.info(
                f"  mean_cosine={mean_cosine:.6f}, current_cosine={current_cosine:.6f}"
            )
            self.logger.info(f"  z_score={z_score:.6f}, abs_z_score={abs_z_score:.6f}")
            self.logger.info(
                f"  low_cosine_anomaly={low_cosine_anomaly}, statistical_anomaly={statistical_anomaly}, combined_anomaly={combined_anomaly}"
            )
            self.logger.info(
                f"  is_anomaly={is_anomaly}, anomaly_score={anomaly_score:.6f}"
            )

            evidence = {
                "similarity_scores": similarities,
                "cosine_similarities": cosine_similarities,
                "mean_similarity": float(mean_sim),
                "current_similarity": float(current_sim),
                "mean_cosine": float(mean_cosine),
                "current_cosine": float(current_cosine),
                "z_score": float(z_score),
                "abs_z_score": float(abs_z_score),
                "low_cosine_anomaly": low_cosine_anomaly,
                "statistical_anomaly": statistical_anomaly,
                "combined_anomaly": combined_anomaly,
                "detection_method": "cusum_gradient_manipulation",
            }

            return bool(is_anomaly), anomaly_score, evidence

        return False, 0.0, {}

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
                    # Decay trust score using polynomial decay method
                    old_score = self.trust_scores[neighbor_id]
                    self.trust_scores[neighbor_id] = self._calculate_trust_decay(
                        neighbor_id, anomaly_score
                    )
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
            f"TRUST MONITOR {self.node_id}: Running CUSUM detection on {len(current_updates)} neighbors"
        )

        # 1. CUSUM-based detection for each neighbor
        for neighbor_id, update in current_updates.items():
            is_anomaly, score, evidence = self._detect_anomalies_cusum(
                neighbor_id, update
            )
            self.logger.info(
                f"TRUST MONITOR {self.node_id}: CUSUM {neighbor_id} - anomaly={is_anomaly}, score={score:.3f}"
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

        # Use trust-only ranking-based detection (no thresholds)
        suspicious_neighbors = self._detect_suspicious_neighbors_by_ranking()

        if suspicious_neighbors:
            trust_scores_str = [(n, f'{self.trust_scores[n]:.3f}') for n in suspicious_neighbors]
            self.logger.warning(
                f"TRUST MONITOR {self.node_id}: Round {round_num}: Suspicious neighbors (trust-based ranking): "
                f"{trust_scores_str}"
            )
        else:
            self.logger.info(
                f"TRUST MONITOR {self.node_id}: No suspicious neighbors detected using trust-based ranking"
            )

        return self.trust_scores.copy()

    def _detect_suspicious_neighbors_by_ranking(self) -> List[str]:
        """
        Detect suspicious neighbors using trust-based ranking without thresholds.
        
        Uses a combination of gap detection and adaptive percentile-based detection
        for dataset-agnostic malicious node identification.
        """
        if len(self.trust_scores) < 2:
            # Single neighbor - use conservative absolute threshold as fallback
            suspicious = [
                neighbor_id for neighbor_id, score in self.trust_scores.items()
                if score < 0.5  # Very conservative threshold for single neighbor
            ]
            if suspicious:
                self.logger.info(
                    f"TRUST MONITOR {self.node_id}: Single neighbor detection: "
                    f"flagged {suspicious[0]} with trust score {self.trust_scores[suspicious[0]]:.3f}"
                )
            return suspicious
        
        # Sort neighbors by trust score (lowest first = most suspicious)
        sorted_neighbors = sorted(
            self.trust_scores.items(), 
            key=lambda x: x[1]
        )
        
        neighbor_count = len(sorted_neighbors)
        trust_values = [score for _, score in sorted_neighbors]
        
        # Strategy 1: Gap-based detection (most precise)
        gap_detected = self._detect_by_trust_gap(sorted_neighbors, trust_values)
        if gap_detected:
            self.logger.info(
                f"TRUST MONITOR {self.node_id}: Gap-based detection: "
                f"found {len(gap_detected)} suspicious neighbors with significant trust gap"
            )
            return gap_detected
        
        # Strategy 2: Adaptive percentile-based detection
        percentile_detected = self._detect_by_adaptive_percentile(sorted_neighbors, neighbor_count)
        if percentile_detected:
            self.logger.info(
                f"TRUST MONITOR {self.node_id}: Percentile-based detection: "
                f"flagged bottom {len(percentile_detected)} neighbors"
            )
            return percentile_detected
        
        # No suspicious neighbors detected
        self.logger.info(
            f"TRUST MONITOR {self.node_id}: No suspicious patterns detected in trust scores"
        )
        return []
    
    def _detect_by_trust_gap(self, sorted_neighbors: List[Tuple[str, float]], trust_values: List[float]) -> List[str]:
        """Detect suspicious neighbors based on significant gaps in trust score distribution."""
        if len(trust_values) < 3:  # Need at least 3 neighbors for gap detection
            return []
        
        # Find the largest gap in trust scores
        max_gap = 0.0
        gap_index = -1
        
        for i in range(len(trust_values) - 1):
            gap = trust_values[i + 1] - trust_values[i]
            if gap > max_gap:
                max_gap = gap
                gap_index = i
        
        # Only flag if gap is significant (>= 0.15) and creates clear separation
        if max_gap >= 0.15:
            # Flag all neighbors below the gap
            suspicious = [neighbor_id for neighbor_id, score in sorted_neighbors[:gap_index + 1]]
            
            # Additional validation: ensure we're not flagging too many neighbors
            # Don't flag more than 50% of neighbors to avoid over-detection
            max_flaggable = max(1, len(sorted_neighbors) // 2)
            if len(suspicious) > max_flaggable:
                # Only flag the most suspicious ones
                suspicious = suspicious[:max_flaggable]
            
            return suspicious
        
        return []
    
    def _detect_by_adaptive_percentile(self, sorted_neighbors: List[Tuple[str, float]], neighbor_count: int) -> List[str]:
        """Detect suspicious neighbors using adaptive percentile thresholds."""
        
        # Adaptive percentile based on network topology (neighbor count)
        if neighbor_count >= 9:      # Complete topology (or nearly complete)
            target_percentile = 15   # Bottom 15% (≈1-2 nodes out of 9+)
        elif neighbor_count >= 4:    # Ring topology or similar
            target_percentile = 25   # Bottom 25% (≈1 node out of 4-8)
        else:                        # Line topology (2-3 neighbors)
            target_percentile = 33   # Bottom 33% (≈1 node out of 2-3)
        
        # Calculate how many neighbors to flag
        neighbors_to_flag = max(1, int(neighbor_count * target_percentile / 100))
        
        # But don't flag neighbors with very high trust scores (> 0.8)
        # This prevents false positives when all neighbors are actually honest
        candidates = [
            (neighbor_id, score) for neighbor_id, score in sorted_neighbors[:neighbors_to_flag]
            if score < 0.8  # Only flag if trust score is meaningfully low
        ]
        
        return [neighbor_id for neighbor_id, _ in candidates]


    def _calculate_trust_decay(self, neighbor_id: str, anomaly_score: float) -> float:
        """Calculate trust decay using polynomial, exponential, or linear method."""
        current_score = self.trust_scores[neighbor_id]
        violation_count = self.anomaly_counts.get(neighbor_id, 0)

        if self.config.enable_polynomial_decay:
            # Polynomial decay: more intuitive for trust-weighted aggregation
            # Decay factor gets stronger with more violations
            
            # Base decay factor scaled by violation count and anomaly severity
            base_factor = self.config.polynomial_decay_base_factor
            
            # Calculate violation severity (1 + violation_count)^power
            violation_severity = (1 + violation_count) ** self.config.polynomial_decay_power
            
            # Scale by anomaly score (higher anomaly score = more decay)
            anomaly_multiplier = 1.0 + (anomaly_score * 0.5)  # Scale anomaly impact
            
            # Combined decay factor - gets smaller (more aggressive) with more violations
            decay_factor = base_factor / (violation_severity * anomaly_multiplier)
            
            # Ensure minimum reasonable decay
            decay_factor = max(decay_factor, 0.1)
            
            new_score = current_score * decay_factor
            
            self.logger.debug(
                f"Polynomial decay for {neighbor_id}: {current_score:.6f} -> {new_score:.6f} "
                f"(violations: {violation_count}, anomaly: {anomaly_score:.3f}, "
                f"severity: {violation_severity:.3f}, decay: {decay_factor:.6f})"
            )
            
            return max(0.0, new_score)
            
        elif self.config.enable_exponential_decay:
            # Legacy exponential decay
            decay = self.config.exponential_decay_base ** (violation_count + 1)
            return current_score * decay
        else:
            # Legacy linear decay
            decay = self.config.trust_decay_factor ** (1 + anomaly_score * 0.1)
            return current_score * decay

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
        """Get comprehensive trust monitoring summary using trust-only ranking detection."""
        # Use the new trust-only detection approach
        suspicious_neighbors = self._detect_suspicious_neighbors_by_ranking()
        
        # Calculate trust score statistics for reporting
        trust_stats = {}
        if self.trust_scores:
            scores = list(self.trust_scores.values())
            trust_stats = {
                "min_trust": min(scores),
                "max_trust": max(scores),
                "avg_trust": sum(scores) / len(scores),
                "trust_variance": np.var(scores) if len(scores) > 1 else 0.0
            }

        return {
            "node_id": self.node_id,
            "trust_scores": self.trust_scores.copy(),
            "anomaly_counts": dict(self.anomaly_counts),
            "suspicious_neighbors": suspicious_neighbors,
            "trust_statistics": trust_stats,
            "detection_method": "trust_ranking",
            "monitoring_enabled": self.config.enable_trust_monitoring,
            "total_neighbors": len(self.trust_scores),
        }
