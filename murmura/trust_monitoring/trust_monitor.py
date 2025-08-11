"""
Adaptive Trust Monitor for Decentralized Federated Learning.

This lightweight trust monitor uses pure trust scoring without binary detection,
solving network fragmentation issues and eliminating the need for validation data.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Deque, Set
from collections import defaultdict, deque
from dataclasses import dataclass
from scipy.spatial.distance import cosine
from scipy import stats as scipy_stats

from .trust_config import TrustMonitorConfig
from .trust_events import TrustEvent, TrustScoreEvent, TrustAnomalyEvent


@dataclass
class GradientFingerprint:
    """Lightweight statistical fingerprint of gradient updates."""
    cross_layer_correlation: float
    gradient_entropy: float
    spectral_energy: float
    update_magnitude_distribution: Dict[str, float]
    layer_wise_stats: Dict[str, Dict[str, float]]


@dataclass 
class TrustSignals:
    """Collection of anomaly signals for trust scoring."""
    gradient_divergence: float = 0.0
    pattern_shift: float = 0.0
    consensus_deviation: float = 0.0
    temporal_inconsistency: float = 0.0
    layer_asymmetry: float = 0.0


@dataclass
class TrustStatistics:
    """Anonymized trust statistics for gossip-based detection."""
    min_trust: float
    max_trust: float
    mean_trust: float
    std_trust: float
    node_count: int




class TrustMonitor:
    """
    Adaptive trust monitor using continuous scoring without binary detection.
    
    Key innovations:
    1. No validation data required - purely statistical
    2. Adaptive influence weights preserve network connectivity
    3. Lightweight gradient fingerprinting for anomaly detection
    4. Smooth trust evolution without thresholds
    """
    
    def __init__(self, node_id: str, config: TrustMonitorConfig):
        self.node_id = node_id
        self.config = config
        self.logger = logging.getLogger(f"murmura.trust_monitor.{node_id}")
        
        # Core trust state
        self.trust_scores: Dict[str, float] = {}
        self.trust_velocities: Dict[str, float] = {}
        self.influence_weights: Dict[str, float] = {}
        
        # Gradient fingerprint history for lightweight detection
        self.fingerprint_history: Dict[str, Deque[GradientFingerprint]] = defaultdict(
            lambda: deque(maxlen=config.history_window_size)
        )
        
        # Temporal consistency tracking
        self.update_directions: Dict[str, Deque[np.ndarray]] = defaultdict(
            lambda: deque(maxlen=3)
        )
        
        # Trust trajectory tracking for temporal clustering
        self.trust_trajectories: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=config.history_window_size)
        )
        
        # Gossip trust statistics from other nodes
        self.network_trust_stats: Deque[TrustStatistics] = deque(maxlen=20)
        
        # Network-aware state
        self.network_topology_size: Optional[int] = None
        self.rounds_completed: int = 0
        
        # Note: Detection logic removed - focusing on trust-weighted performance
        
        # Event callbacks
        self.event_callbacks: List[callable] = []
        
        self.logger.info(f"Initialized Adaptive Trust Monitor for node {node_id}")
    
    def add_event_callback(self, callback: callable) -> None:
        """Add callback for trust events."""
        self.event_callbacks.append(callback)
    
    def _emit_event(self, event: TrustEvent) -> None:
        """Emit trust event to all registered callbacks."""
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in trust event callback: {e}")
    
    def set_local_validation_data(self, validation_data: Any, model_template: Any) -> None:
        """Deprecated: No longer needed with lightweight detection."""
        self.logger.info(
            f"TRUST MONITOR {self.node_id}: Validation data not needed for adaptive trust monitoring"
        )
    
    def compute_gradient_fingerprint(self, parameters: Dict[str, Any]) -> GradientFingerprint:
        """
        Compute lightweight statistical fingerprint of gradient update.
        No validation required - pure statistical analysis.
        """
        # 1. Cross-layer correlation (detects broken backprop chains from label flipping)
        cross_correlation = self._compute_cross_layer_correlation(parameters)
        
        # 2. Gradient entropy (detects distribution changes)
        entropy = self._compute_gradient_entropy(parameters)
        
        # 3. Spectral energy distribution (detects manipulation patterns)
        spectral_energy = self._compute_spectral_signature(parameters)
        
        # 4. Update magnitude distribution (detects layer-wise anomalies)
        magnitude_dist = self._compute_magnitude_distribution(parameters)
        
        # 5. Layer-wise statistics for fine-grained analysis
        layer_stats = self._compute_layer_statistics(parameters)
        
        return GradientFingerprint(
            cross_layer_correlation=cross_correlation,
            gradient_entropy=entropy,
            spectral_energy=spectral_energy,
            update_magnitude_distribution=magnitude_dist,
            layer_wise_stats=layer_stats
        )
    
    def _compute_cross_layer_correlation(self, parameters: Dict[str, Any]) -> float:
        """
        Compute correlation between consecutive layers.
        Label flipping breaks the natural correlation pattern.
        """
        layer_names = list(parameters.keys())
        if len(layer_names) < 2:
            return 1.0
        
        correlations = []
        for i in range(len(layer_names) - 1):
            try:
                # Get flattened parameters
                if hasattr(parameters[layer_names[i]], 'cpu'):
                    layer1 = parameters[layer_names[i]].cpu().numpy().flatten()
                    layer2 = parameters[layer_names[i+1]].cpu().numpy().flatten()
                else:
                    layer1 = np.array(parameters[layer_names[i]]).flatten()
                    layer2 = np.array(parameters[layer_names[i+1]]).flatten()
                
                # Skip if layers are too different in size (correlation not meaningful)
                size_ratio = max(len(layer1), len(layer2)) / min(len(layer1), len(layer2))
                if size_ratio > 10:  # Skip if one layer is >10x larger
                    continue
                
                # Take common length for correlation
                min_len = min(len(layer1), len(layer2))
                if min_len > 1000:  # Sample if too large
                    min_len = 1000
                
                # Sample both arrays to same length
                if len(layer1) > min_len:
                    indices = np.random.choice(len(layer1), min_len, replace=False)
                    layer1 = layer1[indices]
                else:
                    layer1 = layer1[:min_len]
                    
                if len(layer2) > min_len:
                    indices = np.random.choice(len(layer2), min_len, replace=False)
                    layer2 = layer2[indices]
                else:
                    layer2 = layer2[:min_len]
                
                # Compute correlation with same-length arrays
                if len(layer1) > 0 and len(layer2) > 0 and len(layer1) == len(layer2):
                    corr = np.corrcoef(layer1, layer2)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            except Exception as e:
                self.logger.debug(f"Correlation computation failed for layers {i}-{i+1}: {e}")
                continue
        
        return float(np.mean(correlations)) if correlations else 0.5
    
    def _compute_gradient_entropy(self, parameters: Dict[str, Any]) -> float:
        """
        Compute entropy of gradient distribution.
        Higher entropy indicates potential label flipping.
        """
        all_gradients = []
        
        for layer_name, params in parameters.items():
            if hasattr(params, 'cpu'):
                param_array = params.cpu().numpy().flatten()
            else:
                param_array = np.array(params).flatten()
            
            # Sample for efficiency
            if len(param_array) > 1000:
                param_array = np.random.choice(param_array, 1000, replace=False)
            
            all_gradients.extend(param_array.tolist())
        
        if not all_gradients:
            return 0.0
        
        # Compute entropy using histogram
        hist, _ = np.histogram(all_gradients, bins=30)
        hist = hist / (hist.sum() + 1e-10)
        
        # Shannon entropy
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        return float(entropy)
    
    def _compute_spectral_signature(self, parameters: Dict[str, Any]) -> float:
        """
        Compute spectral energy concentration.
        Gradient manipulation often changes spectral properties.
        """
        all_params = []
        
        for params in parameters.values():
            if hasattr(params, 'cpu'):
                param_array = params.cpu().numpy().flatten()
            else:
                param_array = np.array(params).flatten()
            
            # Sample for efficiency
            if len(param_array) > 2000:
                param_array = np.random.choice(param_array, 2000, replace=False)
            
            all_params.extend(param_array.tolist())
        
        if len(all_params) < 10:
            return 0.5
        
        # Simple spectral analysis using FFT
        try:
            fft_result = np.fft.fft(all_params[:1024])  # Use power of 2 for efficiency
            power_spectrum = np.abs(fft_result) ** 2
            
            # Measure energy concentration in low frequencies
            total_energy = np.sum(power_spectrum)
            low_freq_energy = np.sum(power_spectrum[:len(power_spectrum)//4])
            
            concentration = low_freq_energy / (total_energy + 1e-10)
            return float(concentration)
        except Exception:
            return 0.5
    
    def _compute_magnitude_distribution(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute distribution of update magnitudes across layers.
        Helps detect layer-specific anomalies.
        """
        magnitudes = {}
        
        for layer_name, params in parameters.items():
            if hasattr(params, 'cpu'):
                param_array = params.cpu().numpy()
            else:
                param_array = np.array(params)
            
            # Compute L2 norm
            magnitude = float(np.linalg.norm(param_array.flatten()))
            magnitudes[layer_name] = magnitude
        
        # Normalize to distribution
        total_magnitude = sum(magnitudes.values()) + 1e-10
        distribution = {k: v/total_magnitude for k, v in magnitudes.items()}
        
        return distribution
    
    def _compute_layer_statistics(self, parameters: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Compute per-layer statistics for fine-grained analysis."""
        layer_stats = {}
        
        for layer_name, params in parameters.items():
            if hasattr(params, 'cpu'):
                param_array = params.cpu().numpy().flatten()
            else:
                param_array = np.array(params).flatten()
            
            # Protect against NaN/Inf
            param_array = np.nan_to_num(param_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            stats = {
                'mean': float(np.mean(param_array)),
                'std': float(np.std(param_array)),
                'skew': float(scipy_stats.skew(param_array)),
                'kurtosis': float(scipy_stats.kurtosis(param_array))
            }
            
            layer_stats[layer_name] = stats
        
        return layer_stats
    
    def compute_trust_signals(
        self, 
        neighbor_id: str, 
        current_fingerprint: GradientFingerprint,
        all_fingerprints: Dict[str, GradientFingerprint],
        own_fingerprint: Optional[GradientFingerprint] = None
    ) -> TrustSignals:
        """
        Compute anomaly signals from gradient fingerprints.
        Uses self-referenced baseline when available for more accurate detection.
        All signals are continuous [0, 1] with no thresholds.
        """
        signals = TrustSignals()
        
        # 1. Gradient divergence - Enhanced with self-referencing baseline
        if own_fingerprint is not None:
            # Self-referenced comparison: compare neighbor against known honest behavior
            correlation_deviation = abs(current_fingerprint.cross_layer_correlation - own_fingerprint.cross_layer_correlation)
            entropy_deviation = abs(current_fingerprint.gradient_entropy - own_fingerprint.gradient_entropy)
            
            # Additional spectral comparison for self-reference
            spectral_deviation = abs(current_fingerprint.spectral_energy - own_fingerprint.spectral_energy)
            
            # Self-referenced divergence score (more sensitive to actual attacks)
            signals.gradient_divergence = min(1.0, (correlation_deviation * 2.0 + entropy_deviation + spectral_deviation) / 3.0)
            
        elif len(self.fingerprint_history[neighbor_id]) >= 2:
            # Fallback to historical comparison when self-reference unavailable
            historical = list(self.fingerprint_history[neighbor_id])
            
            # Compare correlation patterns
            hist_correlations = [fp.cross_layer_correlation for fp in historical]
            correlation_deviation = abs(current_fingerprint.cross_layer_correlation - np.mean(hist_correlations))
            
            # Compare entropy
            hist_entropy = [fp.gradient_entropy for fp in historical]
            entropy_deviation = abs(current_fingerprint.gradient_entropy - np.mean(hist_entropy))
            
            # Normalized divergence score
            signals.gradient_divergence = min(1.0, (correlation_deviation + entropy_deviation) / 2.0)
        
        # 2. Pattern shift detection (for label flipping)
        if len(self.fingerprint_history[neighbor_id]) >= 3:
            # Detect sudden changes in layer-wise distribution
            historical_dists = [fp.update_magnitude_distribution for fp in self.fingerprint_history[neighbor_id]]
            
            # Compare current distribution with historical
            pattern_shifts = []
            for hist_dist in historical_dists[-3:]:
                # Compute JS divergence between distributions
                shift = self._compute_distribution_divergence(
                    current_fingerprint.update_magnitude_distribution,
                    hist_dist
                )
                pattern_shifts.append(shift)
            
            signals.pattern_shift = min(1.0, np.mean(pattern_shifts))
        
        # 3. Consensus deviation - Enhanced with self-referencing
        if own_fingerprint is not None and len(all_fingerprints) >= 2:
            # Self-referenced consensus: compare neighbor against honest node + other neighbors
            honest_correlation = own_fingerprint.cross_layer_correlation
            other_correlations = [fp.cross_layer_correlation for fp in all_fingerprints.values() 
                                 if fp != current_fingerprint]
            
            if other_correlations:
                # Include honest node's correlation in consensus
                all_correlations = [honest_correlation] + other_correlations
                median_correlation = np.median(all_correlations)
                deviation = abs(current_fingerprint.cross_layer_correlation - median_correlation)
                
                # Weight deviation by distance from honest baseline
                honest_deviation = abs(current_fingerprint.cross_layer_correlation - honest_correlation)
                signals.consensus_deviation = min(1.0, (deviation + honest_deviation) / 2.0)
                
        elif len(all_fingerprints) >= 3:
            # Fallback to neighbor-only consensus
            other_correlations = [fp.cross_layer_correlation for fp in all_fingerprints.values() 
                                 if fp != current_fingerprint]
            if other_correlations:
                median_correlation = np.median(other_correlations)
                deviation = abs(current_fingerprint.cross_layer_correlation - median_correlation)
                signals.consensus_deviation = min(1.0, deviation * 2.0)
        
        # 4. Temporal inconsistency (erratic update patterns)
        if neighbor_id in self.update_directions and len(self.update_directions[neighbor_id]) >= 2:
            # Check consistency of update directions
            directions = list(self.update_directions[neighbor_id])
            consistencies = []
            for i in range(len(directions) - 1):
                # Cosine similarity between consecutive update directions
                sim = 1 - cosine(directions[i], directions[i+1])
                consistencies.append(sim)
            
            # Low consistency indicates erratic behavior
            avg_consistency = np.mean(consistencies)
            signals.temporal_inconsistency = max(0.0, 1.0 - avg_consistency)
        
        # 5. Layer asymmetry (characteristic of label flipping)
        layer_stats = current_fingerprint.layer_wise_stats
        if len(layer_stats) >= 2:
            # Check if classifier layers have disproportionate updates
            classifier_layers = [name for name in layer_stats.keys() 
                               if any(term in name.lower() for term in ['fc', 'classifier', 'head', 'output'])]
            feature_layers = [name for name in layer_stats.keys() if name not in classifier_layers]
            
            if classifier_layers and feature_layers:
                classifier_stds = [layer_stats[name]['std'] for name in classifier_layers]
                feature_stds = [layer_stats[name]['std'] for name in feature_layers]
                
                ratio = np.mean(classifier_stds) / (np.mean(feature_stds) + 1e-10)
                # High ratio indicates potential label flipping
                signals.layer_asymmetry = min(1.0, max(0.0, (ratio - 1.0) / 3.0))
        
        return signals
    
    def _compute_distribution_divergence(self, dist1: Dict[str, float], dist2: Dict[str, float]) -> float:
        """Compute Jensen-Shannon divergence between two distributions."""
        # Get common keys
        all_keys = set(dist1.keys()) | set(dist2.keys())
        
        # Create aligned distributions
        p = np.array([dist1.get(k, 0.0) for k in all_keys])
        q = np.array([dist2.get(k, 0.0) for k in all_keys])
        
        # Normalize
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)
        
        # JS divergence
        m = (p + q) / 2
        divergence = 0.5 * np.sum(p * np.log(p / (m + 1e-10) + 1e-10)) + \
                    0.5 * np.sum(q * np.log(q / (m + 1e-10) + 1e-10))
        
        return float(min(1.0, divergence))
    
    def update_trust_scores(
        self,
        neighbor_id: str,
        trust_signals: TrustSignals,
        round_num: int
    ) -> float:
        """
        Update trust score based on anomaly signals.
        Smooth, continuous evolution without thresholds.
        """
        # Initialize if new neighbor
        if neighbor_id not in self.trust_scores:
            self.trust_scores[neighbor_id] = self.config.initial_trust_score
            self.trust_velocities[neighbor_id] = 0.0
        
        current_trust = self.trust_scores[neighbor_id]
        
        # Combine signals with adaptive weighting
        # Early rounds: focus on gradient divergence
        # Later rounds: include more signals as patterns establish
        signal_maturity = min(1.0, round_num / 10)
        
        total_anomaly = (
            0.35 * trust_signals.gradient_divergence +
            0.25 * trust_signals.pattern_shift * signal_maturity +
            0.20 * trust_signals.consensus_deviation * signal_maturity +
            0.10 * trust_signals.temporal_inconsistency +
            0.10 * trust_signals.layer_asymmetry
        )
        
        # Adaptive decay/recovery rates
        # Decay faster for high anomaly, recover slower
        decay_rate = 0.05 + 0.15 * total_anomaly  # [0.05, 0.20]
        recovery_rate = 0.02 * (1.0 - total_anomaly)  # [0, 0.02]
        
        # Net trust change with momentum
        trust_change = -decay_rate + recovery_rate
        
        # Apply momentum from previous velocity
        momentum = 0.3
        smoothed_change = momentum * self.trust_velocities[neighbor_id] + (1 - momentum) * trust_change
        
        # Update trust score
        new_trust = np.clip(current_trust + smoothed_change, 0.01, 1.0)
        
        # Store velocity for next round
        self.trust_velocities[neighbor_id] = smoothed_change
        self.trust_scores[neighbor_id] = new_trust
        
        # Store trust trajectory for temporal clustering
        self.trust_trajectories[neighbor_id].append(new_trust)
        
        self.logger.debug(
            f"Trust update for {neighbor_id}: {current_trust:.3f} -> {new_trust:.3f} "
            f"(anomaly: {total_anomaly:.3f}, change: {smoothed_change:.3f})"
        )
        
        return new_trust
    
    def compute_influence_weight(self, trust_score: float, round_num: int) -> float:
        """
        Convert trust score to influence weight for aggregation.
        Key innovation: Preserves minimum connectivity to prevent network fragmentation.
        """
        # Round-adaptive strictness
        strictness = min(1.0, round_num / 20)
        
        # Minimum weight to preserve network connectivity
        MIN_ROUTING_WEIGHT = 0.05 * (1 - strictness) + 0.02 * strictness
        
        if trust_score > 0.8:
            # Highly trusted: full influence
            return 1.0
        elif trust_score > 0.6:
            # Moderately trusted: proportional influence
            # Maps [0.6, 0.8] to [0.6, 1.0]
            return 0.6 + 2.0 * (trust_score - 0.6)
        elif trust_score > 0.3:
            # Suspicious: reduced influence
            # Maps [0.3, 0.6] to [0.2, 0.6]
            return 0.2 + 1.33 * (trust_score - 0.3)
        else:
            # Highly suspicious: minimal but non-zero influence
            # Maps [0, 0.3] to [MIN_ROUTING_WEIGHT, 0.2]
            return MIN_ROUTING_WEIGHT + (0.2 - MIN_ROUTING_WEIGHT) * (trust_score / 0.3)
    
    def process_parameter_updates(
        self,
        round_num: int,
        neighbor_updates: Dict[str, Dict[str, Any]],
        neighbor_losses: Optional[Dict[str, float]] = None,
        own_parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Process parameter updates and compute influence weights.
        Returns influence weights for aggregation (not just trust scores).
        """
        if not self.config.enable_trust_monitoring:
            return {neighbor_id: 1.0 for neighbor_id in neighbor_updates.keys()}
        
        self.rounds_completed = round_num
        
        # Update network topology awareness
        if self.network_topology_size is None:
            self.network_topology_size = len(neighbor_updates)
        
        self.logger.info(
            f"TRUST MONITOR {self.node_id}: Processing round {round_num} with {len(neighbor_updates)} neighbors"
        )
        
        # Compute own fingerprint for self-referenced baseline
        own_fingerprint = None
        if own_parameters is not None:
            own_fingerprint = self.compute_gradient_fingerprint(own_parameters)
            self._last_own_fingerprint = own_fingerprint  # Store for detection method
            self.logger.debug(
                f"SELF-REFERENCE {self.node_id}: Own fingerprint computed - "
                f"correlation={own_fingerprint.cross_layer_correlation:.3f}, "
                f"entropy={own_fingerprint.gradient_entropy:.3f}, "
                f"spectral={own_fingerprint.spectral_energy:.3f}"
            )
        
        # Compute fingerprints for all neighbors
        current_fingerprints = {}
        for neighbor_id, parameters in neighbor_updates.items():
            fingerprint = self.compute_gradient_fingerprint(parameters)
            current_fingerprints[neighbor_id] = fingerprint
            
            # Store update direction for temporal analysis
            self._store_update_direction(neighbor_id, parameters)
        
        # Compute trust signals and update scores
        for neighbor_id, fingerprint in current_fingerprints.items():
            # Compute anomaly signals with self-referenced baseline
            trust_signals = self.compute_trust_signals(
                neighbor_id, 
                fingerprint,
                current_fingerprints,
                own_fingerprint  # NEW: Pass honest node's fingerprint as baseline
            )
            
            # Update trust score
            new_trust = self.update_trust_scores(neighbor_id, trust_signals, round_num)
            
            # Compute influence weight for aggregation
            influence_weight = self.compute_influence_weight(new_trust, round_num)
            self.influence_weights[neighbor_id] = influence_weight
            
            # Store fingerprint for next round
            self.fingerprint_history[neighbor_id].append(fingerprint)
            
            # Enhanced logging with self-reference comparison
            if own_fingerprint is not None:
                # Compute relative anomaly compared to honest baseline
                relative_anomaly = (
                    trust_signals.gradient_divergence * 0.6 +
                    trust_signals.consensus_deviation * 0.4
                )
                self.logger.info(
                    f"TRUST MONITOR {self.node_id}: {neighbor_id} - "
                    f"trust={new_trust:.3f}, influence={influence_weight:.3f}, "
                    f"relative_anomaly={relative_anomaly:.3f}, "
                    f"signals=(grad:{trust_signals.gradient_divergence:.2f}, "
                    f"pattern:{trust_signals.pattern_shift:.2f}, "
                    f"consensus:{trust_signals.consensus_deviation:.2f})"
                )
            else:
                self.logger.info(
                    f"TRUST MONITOR {self.node_id}: {neighbor_id} - "
                    f"trust={new_trust:.3f}, influence={influence_weight:.3f}, "
                    f"signals=(grad:{trust_signals.gradient_divergence:.2f}, "
                    f"pattern:{trust_signals.pattern_shift:.2f}, "
                    f"consensus:{trust_signals.consensus_deviation:.2f})"
                )
        
        # Emit trust score event if configured
        if self.config.log_trust_events and self.event_callbacks:
            score_changes = {
                nid: self.trust_velocities.get(nid, 0.0) 
                for nid in self.trust_scores.keys()
            }
            
            event = TrustScoreEvent(
                node_id=self.node_id,
                round_num=round_num,
                trust_scores=self.trust_scores.copy(),
                score_changes=score_changes,
                detection_method="adaptive_trust_scoring"
            )
            self._emit_event(event)
        
        # Note: Clustering-based detection removed - focusing on trust-weighted performance metrics
        
        # Return influence weights for aggregation
        return self.influence_weights.copy()
    
    
    
    
    def share_trust_statistics(self) -> TrustStatistics:
        """Generate anonymized trust statistics for gossip sharing."""
        if not self.trust_scores:
            return TrustStatistics(
                min_trust=1.0, max_trust=1.0, mean_trust=1.0, 
                std_trust=0.0, node_count=0
            )
        
        trust_values = list(self.trust_scores.values())
        return TrustStatistics(
            min_trust=float(np.min(trust_values)),
            max_trust=float(np.max(trust_values)),
            mean_trust=float(np.mean(trust_values)),
            std_trust=float(np.std(trust_values)),
            node_count=len(trust_values)
        )
    
    def receive_trust_statistics(self, stats: TrustStatistics) -> None:
        """Receive and store trust statistics from other nodes."""
        self.network_trust_stats.append(stats)
        self.logger.debug(
            f"Received trust stats: mean={stats.mean_trust:.3f}, "
            f"std={stats.std_trust:.3f}, nodes={stats.node_count}"
        )
    
    def _store_update_direction(self, neighbor_id: str, parameters: Dict[str, Any]) -> None:
        """Store normalized update direction for temporal consistency analysis."""
        # Sample parameters to create direction vector
        direction_vector = []
        
        for layer_name, params in list(parameters.items())[:5]:  # Use first 5 layers
            if hasattr(params, 'cpu'):
                param_array = params.cpu().numpy().flatten()
            else:
                param_array = np.array(params).flatten()
            
            # Sample for efficiency
            if len(param_array) > 100:
                param_array = param_array[:100]
            
            direction_vector.extend(param_array.tolist())
        
        if direction_vector:
            # Normalize to unit vector
            direction = np.array(direction_vector)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
                self.update_directions[neighbor_id].append(direction)
    
    def get_trust_summary(self) -> Dict[str, Any]:
        """Get trust monitoring summary focused on performance metrics."""
        trust_values = list(self.trust_scores.values())
        influence_values = list(self.influence_weights.values())
        
        summary = {
            "node_id": self.node_id,
            "rounds_completed": self.rounds_completed,
            "trust_scores": self.trust_scores.copy(),
            "influence_weights": self.influence_weights.copy(),
            "trust_velocities": self.trust_velocities.copy(),
            "monitoring_enabled": self.config.enable_trust_monitoring,
            "total_neighbors": len(self.trust_scores),
        }
        
        # Trust statistics for performance analysis
        if trust_values:
            summary["trust_statistics"] = {
                "mean_trust": float(np.mean(trust_values)),
                "min_trust": float(np.min(trust_values)),
                "max_trust": float(np.max(trust_values)),
                "std_trust": float(np.std(trust_values)),
                "mean_influence": float(np.mean(influence_values)) if influence_values else 1.0,
            }
        
        # Simplified neighbor assessment - no detection, just trust-based influence
        summary["low_trust_neighbors"] = [
            nid for nid, score in self.trust_scores.items() 
            if score < 0.7  # Configurable threshold for performance analysis
        ]
        
        return summary

    def _measure_trust_resource_usage(self, operation_name: str):
        """Context manager for measuring trust monitoring resource usage."""
        # Placeholder implementation - returns a no-op context manager
        class NoOpContextManager:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                return False
        
        return NoOpContextManager()
    
    def get_trust_resource_summary(self) -> Dict[str, Any]:
        """Get trust monitoring resource usage summary."""
        return {
            "status": "no_resource_data",
            "message": "Resource monitoring not implemented in adaptive trust monitor"
        }