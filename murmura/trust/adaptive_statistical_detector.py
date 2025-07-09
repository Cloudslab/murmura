"""
Adaptive Statistical Trust Detector with Online Learning.

This module implements an intelligent trust detection system that:
1. Learns normal behavior patterns online
2. Adapts thresholds based on observed data
3. Uses multiple statistical techniques for robust detection
4. Distinguishes between natural variations and malicious drift

Key Features:
- Online learning with exponential moving averages
- Adaptive threshold calculation using percentile methods
- Multi-signal fusion with confidence weighting
- Asymmetric detection (considers direction of change)
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope


@dataclass
class AdaptiveThresholds:
    """Dynamically learned thresholds for each detection signal."""
    relative_diff_upper: float = 0.1  # Will be learned
    relative_diff_lower: float = 0.01  # Minimum expected change
    cosine_sim_threshold: float = 0.95  # Will be adapted
    magnitude_ratio_upper: float = 1.5  # Will be learned
    magnitude_ratio_lower: float = 0.67  # Will be learned
    
    # Percentiles for adaptive threshold calculation
    normal_percentile: float = 95.0  # Top 5% considered anomalous
    extreme_percentile: float = 99.0  # Top 1% considered highly anomalous


class OnlineStatistics:
    """Maintains online statistics with exponential moving average."""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha  # Learning rate
        self.mean = None
        self.variance = None
        self.count = 0
        
    def update(self, value: float):
        """Update statistics with new observation."""
        if self.mean is None:
            self.mean = value
            self.variance = 0.0
        else:
            # Exponential moving average
            delta = value - self.mean
            self.mean += self.alpha * delta
            self.variance = (1 - self.alpha) * (self.variance + self.alpha * delta * delta)
        self.count += 1
        
    @property
    def std(self) -> float:
        """Return standard deviation."""
        return np.sqrt(self.variance) if self.variance is not None else 0.0
        
    def z_score(self, value: float) -> float:
        """Calculate z-score for a value."""
        if self.std > 1e-8:
            return (value - self.mean) / self.std
        return 0.0


class AdaptiveStatisticalDetector:
    """
    Adaptive statistical detector that learns normal behavior patterns.
    
    This detector uses online learning to adapt to the specific characteristics
    of each neighbor and the overall network behavior.
    """
    
    def __init__(
        self,
        node_id: str,
        window_size: int = 20,
        learning_rate: float = 0.1,
        warmup_rounds: int = 5,
        topology: str = "ring"
    ):
        """
        Initialize adaptive detector.
        
        Args:
            node_id: ID of the node this detector belongs to
            window_size: Size of sliding window for pattern analysis
            learning_rate: Rate of adaptation for online learning
            warmup_rounds: Number of rounds before making detection decisions
            topology: Network topology
        """
        self.node_id = node_id
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.warmup_rounds = warmup_rounds
        self.topology = topology
        
        # Online statistics for each neighbor
        self.neighbor_stats: Dict[str, Dict[str, OnlineStatistics]] = defaultdict(
            lambda: {
                'relative_diff': OnlineStatistics(learning_rate),
                'cosine_similarity': OnlineStatistics(learning_rate),
                'magnitude_ratio': OnlineStatistics(learning_rate),
                'param_norm': OnlineStatistics(learning_rate)
            }
        )
        
        # Per-neighbor baseline learning (only during warmup)
        self.neighbor_baselines: Dict[str, Dict[str, List]] = defaultdict(
            lambda: {
                'relative_diff': [],
                'cosine_similarity': [],
                'magnitude_ratio': [],
                'param_norm': []
            }
        )
        
        # Per-neighbor thresholds (learned from clean baselines)
        self.neighbor_thresholds: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                'relative_diff_upper': 0.05,  # Conservative defaults
                'cosine_sim_lower': 0.95,
                'magnitude_ratio_upper': 1.2,
                'magnitude_ratio_lower': 0.8
            }
        )
        
        # Global baseline for multivariate analysis (warmup only)
        self.baseline_features: List[np.ndarray] = []
        self.baseline_learned = False
        
        # Track which neighbors have learned thresholds
        self.thresholds_learned: Dict[str, bool] = defaultdict(bool)
        
        # Round counter
        self.round_count = 0
        
        # Detection state
        self.detection_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10)
        )
        
        self.logger = logging.getLogger(f"murmura.trust.AdaptiveDetector.{node_id}")
        
    def analyze_parameter_update(
        self,
        neighbor_id: str,
        current_params: Dict[str, np.ndarray],
        neighbor_params: Dict[str, np.ndarray],
        round_number: int
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Analyze parameter update with robust baseline learning.
        
        Returns:
            Tuple of (is_malicious, suspicion_score, detailed_analysis)
        """
        self.round_count = round_number
        
        # Extract features
        features = self._extract_features(current_params, neighbor_params)
        
        # Update online statistics
        self._update_statistics(neighbor_id, features)
        
        # During warmup: collect features from all neighbors (assuming most are honest)
        if self.round_count <= self.warmup_rounds:
            self._collect_baseline_features(neighbor_id, features)
            is_malicious = False
            suspicion_score = 0.0
            confidence = 0.0
            detection_details = {'status': 'warmup', 'round': self.round_count}
            
            # Return early during warmup - don't run detection logic
            return is_malicious, suspicion_score, {
                'neighbor_id': neighbor_id,
                'round': round_number,
                'is_malicious': is_malicious,
                'suspicion_score': suspicion_score,
                'confidence': confidence,
                'features': features,
                'detection_details': detection_details,
                'neighbor_thresholds': {},
                'warmup_status': f"{self.round_count}/{self.warmup_rounds}",
                'method': 'adaptive_statistical_robust'
            }
        
        # After warmup: learn robust thresholds and start detection
        elif self.round_count > self.warmup_rounds:
            # Check if thresholds have been learned for this neighbor
            if not self.thresholds_learned[neighbor_id]:
                # Learn thresholds using robust statistical methods (assuming majority are honest)
                self._learn_robust_thresholds_from_baselines(neighbor_id)
                self.thresholds_learned[neighbor_id] = True
            
            # Learn global baseline once (not per neighbor)
            if not self.baseline_learned:
                self._learn_robust_global_baseline()
                self.baseline_learned = True
            
            # Start detection
            detection_results = self._detect_anomalies_robust(neighbor_id, features)
            is_malicious, suspicion_score, confidence = self._fuse_signals_robust(detection_results)
            detection_details = detection_results
        
        # Update detection history
        self.detection_history[neighbor_id].append({
            'round': round_number,
            'suspicious': is_malicious,
            'score': suspicion_score
        })
        
        # Compile analysis
        thresholds = self.neighbor_thresholds[neighbor_id] if neighbor_id in self.neighbor_thresholds else {}
        analysis = {
            'neighbor_id': neighbor_id,
            'round': round_number,
            'is_malicious': is_malicious,
            'suspicion_score': suspicion_score,
            'confidence': confidence,
            'features': features,
            'detection_details': detection_details,
            'neighbor_thresholds': thresholds,
            'warmup_status': f"{self.round_count}/{self.warmup_rounds}",
            'method': 'adaptive_statistical_robust'
        }
        
        if is_malicious:
            self.logger.warning(
                f"🎯 ROBUST DETECTION: {neighbor_id} flagged "
                f"(suspicion: {suspicion_score:.3f}, confidence: {confidence:.3f})"
            )
        
        return is_malicious, suspicion_score, analysis
    
    def _extract_features(
        self,
        current_params: Dict[str, np.ndarray],
        neighbor_params: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Extract statistical features from parameter comparison."""
        features = {}
        
        # Calculate norms and differences
        current_flat = np.concatenate([p.flatten() for p in current_params.values()])
        neighbor_flat = np.concatenate([p.flatten() for p in neighbor_params.values()])
        
        current_norm = np.linalg.norm(current_flat)
        neighbor_norm = np.linalg.norm(neighbor_flat)
        diff_norm = np.linalg.norm(neighbor_flat - current_flat)
        
        # Relative difference
        features['relative_diff'] = diff_norm / max(current_norm, 1e-8)
        
        # Cosine similarity
        if current_norm > 1e-8 and neighbor_norm > 1e-8:
            features['cosine_similarity'] = np.dot(current_flat, neighbor_flat) / (current_norm * neighbor_norm)
        else:
            features['cosine_similarity'] = 1.0
        
        # Magnitude ratio
        features['magnitude_ratio'] = neighbor_norm / max(current_norm, 1e-8)
        
        # Parameter norm
        features['param_norm'] = neighbor_norm
        
        # Layer-wise analysis
        layer_diffs = []
        for layer_name in current_params:
            if layer_name in neighbor_params:
                layer_diff = np.linalg.norm(
                    neighbor_params[layer_name].flatten() - current_params[layer_name].flatten()
                )
                layer_diffs.append(layer_diff)
        
        features['layer_diff_std'] = np.std(layer_diffs) if layer_diffs else 0.0
        features['layer_diff_max'] = np.max(layer_diffs) if layer_diffs else 0.0
        
        return features
    
    def _update_statistics(self, neighbor_id: str, features: Dict[str, float]):
        """Update online statistics for the neighbor."""
        stats = self.neighbor_stats[neighbor_id]
        
        stats['relative_diff'].update(features['relative_diff'])
        stats['cosine_similarity'].update(features['cosine_similarity'])
        stats['magnitude_ratio'].update(features['magnitude_ratio'])
        stats['param_norm'].update(features['param_norm'])
    
    def _collect_baseline_features(self, neighbor_id: str, features: Dict[str, float]):
        """Collect clean baseline features during warmup (assumes no attacks during warmup)."""
        baselines = self.neighbor_baselines[neighbor_id]
        
        baselines['relative_diff'].append(features['relative_diff'])
        baselines['cosine_similarity'].append(features['cosine_similarity'])
        baselines['magnitude_ratio'].append(features['magnitude_ratio'])
        baselines['param_norm'].append(features['param_norm'])
        
        # Store feature vector for global baseline
        feature_vector = np.array([
            features['relative_diff'],
            features['cosine_similarity'],
            features['magnitude_ratio'],
            features['layer_diff_std']
        ])
        self.baseline_features.append(feature_vector)
    
    def _learn_robust_thresholds_from_baselines(self, neighbor_id: str):
        """Learn per-neighbor thresholds using robust statistics (handles contaminated data)."""
        baselines = self.neighbor_baselines[neighbor_id]
        thresholds = self.neighbor_thresholds[neighbor_id]
        
        # Learn relative difference threshold using robust percentile method
        if baselines['relative_diff']:
            data = np.array(baselines['relative_diff'])
            # Use 90th percentile as threshold (assuming <10% malicious during warmup)
            thresholds['relative_diff_upper'] = np.percentile(data, 90)
            thresholds['relative_diff_upper'] = max(thresholds['relative_diff_upper'], 0.01)  # Minimum sensitivity
        
        # Learn cosine similarity threshold using robust percentile method
        if baselines['cosine_similarity']:
            data = np.array(baselines['cosine_similarity'])
            # Use 10th percentile as lower threshold (assuming <10% malicious during warmup)
            thresholds['cosine_sim_lower'] = np.percentile(data, 10)
            thresholds['cosine_sim_lower'] = min(thresholds['cosine_sim_lower'], 0.99)  # Maximum sensitivity
        
        # Learn magnitude ratio thresholds using robust method
        if baselines['magnitude_ratio']:
            data = np.array(baselines['magnitude_ratio'])
            # Use median ± robust MAD (Median Absolute Deviation)
            median_ratio = np.median(data)
            mad = np.median(np.abs(data - median_ratio))
            # Use 3*MAD as robust alternative to 3*std
            thresholds['magnitude_ratio_upper'] = median_ratio + 3.0 * mad
            thresholds['magnitude_ratio_lower'] = median_ratio - 3.0 * mad
            # Ensure reasonable bounds
            thresholds['magnitude_ratio_upper'] = max(thresholds['magnitude_ratio_upper'], 1.05)
            thresholds['magnitude_ratio_lower'] = min(thresholds['magnitude_ratio_lower'], 0.95)
        
        self.logger.info(
            f"Learned robust thresholds for {neighbor_id}: "
            f"rel_diff={thresholds['relative_diff_upper']:.4f}, "
            f"cos_sim={thresholds['cosine_sim_lower']:.4f}, "
            f"mag_ratio=({thresholds['magnitude_ratio_lower']:.3f}, {thresholds['magnitude_ratio_upper']:.3f})"
        )
    
    def _learn_robust_global_baseline(self):
        """Learn robust global baseline model for multivariate analysis."""
        if len(self.baseline_features) >= 15:
            try:
                from sklearn.covariance import EllipticEnvelope
                # Use higher contamination estimate since we don't know ground truth
                self.baseline_detector = EllipticEnvelope(
                    contamination=0.15,  # Assume up to 15% contamination during warmup
                    support_fraction=0.8  # Be more conservative
                )
                X = np.array(self.baseline_features)
                self.baseline_detector.fit(X)
                self.logger.info(f"Learned robust global baseline from {len(self.baseline_features)} samples")
            except Exception as e:
                self.logger.warning(f"Failed to learn robust global baseline: {e}")
                self.baseline_detector = None
        else:
            self.baseline_detector = None
    
    def _detect_anomalies_robust(
        self,
        neighbor_id: str,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Detect anomalies using robust per-neighbor thresholds."""
        results = {}
        stats = self.neighbor_stats[neighbor_id]
        thresholds = self.neighbor_thresholds[neighbor_id]
        
        # 1. Relative difference detection (using learned neighbor-specific threshold)
        rel_diff_z = stats['relative_diff'].z_score(features['relative_diff'])
        rel_diff_anomalous = features['relative_diff'] > thresholds['relative_diff_upper']
        
        # Normalized score: 0 = normal, 1 = highly anomalous
        rel_diff_score = min(1.0, features['relative_diff'] / max(thresholds['relative_diff_upper'], 0.001))
        
        results['relative_diff'] = {
            'anomalous': rel_diff_anomalous,
            'value': features['relative_diff'],
            'threshold': thresholds['relative_diff_upper'],
            'z_score': rel_diff_z,
            'score': rel_diff_score
        }
        
        # 2. Cosine similarity detection (using learned neighbor-specific threshold)
        cos_sim_z = stats['cosine_similarity'].z_score(features['cosine_similarity'])
        cos_sim_anomalous = features['cosine_similarity'] < thresholds['cosine_sim_lower']
        
        # Normalized score: higher difference = more suspicious
        cos_sim_score = max(0.0, min(1.0, (thresholds['cosine_sim_lower'] - features['cosine_similarity']) * 10))
        
        results['cosine_similarity'] = {
            'anomalous': cos_sim_anomalous,
            'value': features['cosine_similarity'],
            'threshold': thresholds['cosine_sim_lower'],
            'z_score': cos_sim_z,
            'score': cos_sim_score
        }
        
        # 3. Magnitude ratio detection (using learned neighbor-specific thresholds)
        mag_ratio_anomalous = (
            features['magnitude_ratio'] > thresholds['magnitude_ratio_upper'] or
            features['magnitude_ratio'] < thresholds['magnitude_ratio_lower']
        )
        
        # Normalized score: deviation from normal range
        if features['magnitude_ratio'] > thresholds['magnitude_ratio_upper']:
            mag_ratio_score = min(1.0, (features['magnitude_ratio'] - thresholds['magnitude_ratio_upper']) / 0.5)
        elif features['magnitude_ratio'] < thresholds['magnitude_ratio_lower']:
            mag_ratio_score = min(1.0, (thresholds['magnitude_ratio_lower'] - features['magnitude_ratio']) / 0.5)
        else:
            mag_ratio_score = 0.0
        
        results['magnitude_ratio'] = {
            'anomalous': mag_ratio_anomalous,
            'value': features['magnitude_ratio'],
            'upper_threshold': thresholds['magnitude_ratio_upper'],
            'lower_threshold': thresholds['magnitude_ratio_lower'],
            'score': mag_ratio_score
        }
        
        # 4. Multivariate anomaly detection (using clean baseline)
        if hasattr(self, 'baseline_detector') and self.baseline_detector is not None:
            try:
                current_vector = np.array([[
                    features['relative_diff'],
                    features['cosine_similarity'],
                    features['magnitude_ratio'],
                    features['layer_diff_std']
                ]])
                
                anomaly_score = self.baseline_detector.decision_function(current_vector)[0]
                is_outlier = self.baseline_detector.predict(current_vector)[0] == -1
                
                # Normalize anomaly score to 0-1 range (more negative = more anomalous)
                normalized_score = max(0, min(1, -anomaly_score / 5.0))
                
                results['multivariate'] = {
                    'anomalous': is_outlier,
                    'score': normalized_score,
                    'method': 'baseline_elliptic_envelope'
                }
            except Exception as e:
                results['multivariate'] = {'anomalous': False, 'score': 0.0, 'error': str(e)}
        else:
            results['multivariate'] = {'anomalous': False, 'score': 0.0, 'status': 'baseline_not_learned'}
        
        # 5. Pattern consistency check
        if len(self.detection_history[neighbor_id]) >= 3:
            recent_suspicious = sum(
                1 for d in list(self.detection_history[neighbor_id])[-3:]
                if d['suspicious']
            )
            results['pattern_consistency'] = {
                'anomalous': recent_suspicious >= 2,
                'suspicious_count': recent_suspicious,
                'window': 3,
                'score': recent_suspicious / 3.0
            }
        else:
            results['pattern_consistency'] = {'anomalous': False, 'suspicious_count': 0, 'score': 0.0}
        
        return results
    
    def _fuse_signals_robust(self, detection_results: Dict[str, Any]) -> Tuple[bool, float, float]:
        """Fuse multiple detection signals with robust scoring."""
        # Weight for each signal (based on reliability)
        weights = {
            'relative_diff': 0.35,      # Most reliable for parameter drift
            'cosine_similarity': 0.30,  # Good for direction changes
            'magnitude_ratio': 0.20,    # Useful for scaling attacks
            'multivariate': 0.10,       # Additional confirmation
            'pattern_consistency': 0.05 # Temporal consistency
        }
        
        # Calculate weighted suspicion score
        total_score = 0.0
        total_weight = 0.0
        anomaly_count = 0
        signal_scores = {}
        
        for signal, weight in weights.items():
            if signal in detection_results:
                result = detection_results[signal]
                score = result.get('score', 0.0)
                
                # Ensure score is in [0, 1] range
                score = max(0.0, min(1.0, score))
                signal_scores[signal] = score
                
                if result.get('anomalous', False):
                    anomaly_count += 1
                
                # Weight the score contribution
                total_score += weight * score
                total_weight += weight
        
        # Normalize suspicion score
        if total_weight > 0:
            suspicion_score = total_score / total_weight
        else:
            suspicion_score = 0.0
        
        # Ensure suspicion score is in [0, 1]
        suspicion_score = max(0.0, min(1.0, suspicion_score))
        
        # Calculate confidence based on signal agreement and strength
        num_signals = len([s for s in weights.keys() if s in detection_results])
        if num_signals > 0:
            confidence = anomaly_count / num_signals
            # Boost confidence if multiple strong signals agree
            if anomaly_count >= 2 and suspicion_score > 0.5:
                confidence = min(1.0, confidence * 1.2)
        else:
            confidence = 0.0
        
        # Robust decision logic
        is_malicious = False
        
        # Strong single signal
        if suspicion_score > 0.6:
            is_malicious = True
        # Multiple moderate signals
        elif anomaly_count >= 2 and suspicion_score > 0.4:
            is_malicious = True
        # Multiple weak signals with high consistency
        elif anomaly_count >= 3 and suspicion_score > 0.3:
            is_malicious = True
        
        return is_malicious, float(suspicion_score), float(confidence)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get detector summary."""
        return {
            'node_id': self.node_id,
            'round_count': self.round_count,
            'warmup_complete': self.round_count >= self.warmup_rounds,
            'neighbors_tracked': len(self.neighbor_stats),
            'neighbor_thresholds': dict(self.neighbor_thresholds),
            'detection_history_summary': {
                neighbor_id: {
                    'total_checks': len(history),
                    'suspicious_count': sum(1 for h in history if h['suspicious']),
                    'avg_score': np.mean([h['score'] for h in history]) if history else 0.0
                }
                for neighbor_id, history in self.detection_history.items()
            }
        }
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """Get detection summary (compatibility method for trust monitor)."""
        return self.get_summary()
    
    def reset_neighbor_data(self, neighbor_id: str) -> None:
        """Reset all data for a specific neighbor (compatibility method for trust monitor)."""
        # Remove neighbor from all tracking structures
        if neighbor_id in self.neighbor_stats:
            del self.neighbor_stats[neighbor_id]
        
        if neighbor_id in self.detection_history:
            del self.detection_history[neighbor_id]
        
        # Note: We don't reset metric_history as it's used for global threshold adaptation